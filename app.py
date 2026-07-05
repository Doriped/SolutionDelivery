# -*- coding: utf-8 -*-
"""
Assistant radiologue virtuel — Démo web (Semaine 5).

Architecture (slide « Architecture cible ») : le CLASSIFIEUR léger (DenseNet-121, amélioration S4)
décide la CLASSE + la confiance ; le VLM médical (MedGemma) rédige l'ANALYSE spécifique à l'image
(signes visuels, justification, qualité, limitations). Garde-fous + sortie JSON + journalisation SQLite.

    ./env/Scripts/python.exe app.py
    -> ouvrir http://127.0.0.1:7860

⚠️ Prototype pédagogique — ne remplace pas un diagnostic médical.
"""
import os, json, time, sqlite3, datetime
import numpy as np
import torch
import pydicom
from PIL import Image
from torchvision import transforms, models
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import gradio as gr

# --------------------------------------------------------------------------- Config
CLASSES   = ["normal", "suspected_opacity", "uncertain"]
LIBELLE   = {"normal": "Normal", "suspected_opacity": "Suspicion d'opacité", "uncertain": "Incertain"}
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEUIL     = 0.60                       # sous ce seuil de confiance -> 'uncertain' + relecture
MODELE    = "classifieur_leger/densenet121_best.pt"
MODEL_ID  = "google/medgemma-1.5-4b-it"
DB        = "journal_inferences.sqlite"
CHARGER_EN_4BIT = False                # True si mémoire GPU insuffisante (plus lent mais ~4 Go)

# --------------------------------------------------------------------------- Classifieur (décision)
clf = models.densenet121(weights=None)
clf.classifier = torch.nn.Linear(clf.classifier.in_features, len(CLASSES))
clf.load_state_dict(torch.load(MODELE, map_location=DEVICE))
clf.to(DEVICE).eval()

tf_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------------- VLM (analyse rédigée)
load_dotenv()
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
print("Chargement de MedGemma (peut prendre ~1 min)...")
if CHARGER_EN_4BIT:
    _bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    vlm = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=_bnb, device_map="auto")
else:
    vlm = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
vlm.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("MedGemma prêt sur", vlm.device)

# --------------------------------------------------------------------------- Prétraitement
def charger_image(chemin):
    """DICOM (.dcm) ou image classique -> PIL RGB normalisée (min-max)."""
    if chemin.lower().endswith(".dcm"):
        dcm = pydicom.dcmread(chemin)
        px = dcm.pixel_array.astype(float)
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            px = px.max() - px
        px = px - px.min()
        px = px / (px.max() + 1e-8)
        px = (px * 255).astype(np.uint8)
        return Image.fromarray(px).convert("RGB")
    return Image.open(chemin).convert("RGB")

def extraire_json(texte):
    if not isinstance(texte, str):
        return None
    d = texte.find("{")
    if d == -1:
        return None
    prof = 0
    for i in range(d, len(texte)):
        if texte[i] == "{": prof += 1
        elif texte[i] == "}":
            prof -= 1
            if prof == 0:
                try: return json.loads(texte[d:i + 1])
                except json.JSONDecodeError: return None
    return None

# --------------------------------------------------------------------------- Analyse VLM (spécifique image)
# Repli par classe si le VLM échoue (dégradation gracieuse, jamais "indisponible").
JUST_TEMPLATE = {
    "normal": {"image_quality": "n/a", "visual_evidence": "Champs pulmonaires clairs, pas d'opacité franche.",
               "justification": "Aspect compatible avec une radiographie normale.",
               "limitations": "Description générique (analyse VLM indisponible)."},
    "suspected_opacity": {"image_quality": "n/a", "visual_evidence": "Opacité pulmonaire suspecte.",
               "justification": "Aspect évocateur d'un processus pneumonique ; corrélation clinique recommandée.",
               "limitations": "Description générique (analyse VLM indisponible)."},
    "uncertain": {"image_quality": "n/a", "visual_evidence": "Signes non concluants ou anomalie atypique.",
               "justification": "Éléments insuffisants pour trancher ; relecture recommandée.",
               "limitations": "Description générique (analyse VLM indisponible)."},
}

def analyser_vlm(image, classe, conf):
    """MedGemma décrit l'image en cohérence avec la classe décidée par le classifieur."""
    consigne = (
        f"Tu es un assistant radiologue. Un classifieur a analysé cette radiographie thoracique "
        f"frontale et l'a classée '{classe}' (confiance {conf:.2f}). "
        "Sois CONCIS : UNE seule phrase courte par champ, pas de listes. "
        "Décris factuellement les signes visuels de CETTE image, donne une justification clinique "
        "brève cohérente avec la classe, et évalue la qualité de l'image. "
        "Réponds UNIQUEMENT par un JSON valide, sans markdown :\n"
        '{ "image_quality": "<bonne|moyenne|mauvaise>", '
        '"visual_evidence": "<une phrase>", '
        '"justification": "<une phrase>", '
        '"limitations": "<une phrase>" }'
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": consigne}]}]
    messages[0]["content"][0]["image"] = image
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(vlm.device)

    # PREFILL : on force la réponse à commencer par le JSON. MedGemma-1.5 émet sinon un bloc de
    # "réflexion" (<unused94>thought...) qui épuise le budget de tokens avant d'écrire le JSON.
    prefill = '{\n  "image_quality":'
    pf = processor.tokenizer(prefill, add_special_tokens=False, return_tensors="pt").input_ids.to(vlm.device)
    input_ids = torch.cat([inputs["input_ids"], pf], dim=-1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(pf)], dim=-1)
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "token_type_ids" in extra:      # allonger pour matcher input_ids (prefill = texte -> 0)
        extra["token_type_ids"] = torch.cat([extra["token_type_ids"], torch.zeros_like(pf)], dim=-1)

    with torch.inference_mode():
        out = vlm.generate(input_ids=input_ids, attention_mask=attn,
                           max_new_tokens=160, do_sample=False, **extra)   # plafond de sécurité
    texte = prefill + processor.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    d = extraire_json(texte)
    if not isinstance(d, dict) or "visual_evidence" not in d:      # repli gracieux par classe
        return dict(JUST_TEMPLATE.get(classe, JUST_TEMPLATE["uncertain"]))
    return d

# --------------------------------------------------------------------------- Journalisation
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS inferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT, horodatage TEXT, fichier TEXT,
        predicted_class TEXT, confidence REAL, mode_degrade INTEGER, latence_s REAL)""")
    con.commit(); con.close()

def journaliser(fichier, sortie):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO inferences (horodatage, fichier, predicted_class, confidence, mode_degrade, latence_s) "
                "VALUES (?,?,?,?,?,?)",
                (datetime.datetime.now().isoformat(timespec="seconds"), os.path.basename(fichier),
                 sortie["predicted_class"], sortie["confidence"], int(sortie["mode_degrade"]), sortie["latence_s"]))
    con.commit(); con.close()

# --------------------------------------------------------------------------- Pipeline /predict
def predire(fichier):
    if not fichier:
        return None, "<div class='carte'><div class='warn-info'>Aucun fichier fourni.</div></div>", {}
    t0 = time.time()
    image = charger_image(fichier)

    # 1) CLASSIFIEUR : décision de classe + confiance
    x = tf_eval(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prob = torch.softmax(clf(x), 1)[0].cpu().numpy()
    idx = int(prob.argmax())
    classe, conf = CLASSES[idx], float(prob[idx])

    # 2) GARDE-FOU : doute -> uncertain + relecture
    mode_degrade = conf < SEUIL
    classe_finale = "uncertain" if mode_degrade else classe

    # 3) VLM : analyse spécifique à l'image, cohérente avec la classe
    a = analyser_vlm(image, classe_finale, conf)

    # 4) Assemblage de la sortie JSON
    sortie = {
        "predicted_class": classe_finale,
        "confidence": round(conf, 2),
        "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, prob)},
        "image_quality": a.get("image_quality"),
        "visual_evidence": a.get("visual_evidence"),
        "justification": a.get("justification"),
        "limitations": a.get("limitations"),
        "mode_degrade": mode_degrade,
        "warning": ("Avertissement: confiance faible, relecture médicale requise" if mode_degrade
                    else "Avertissement: aide IA, ne remplace pas un radiologue"),
        "latence_s": round(time.time() - t0, 2),
    }
    journaliser(fichier, sortie)                                  # 100% des sorties journalisées
    return image, rendu_html(sortie), sortie

# --------------------------------------------------------------------------- Rendu visuel
COULEUR = {                              # (couleur claire texte/barre, fond teinté sombre) par classe
    "normal":            ("#4ade80", "rgba(34,197,94,.14)"),
    "suspected_opacity": ("#fb923c", "rgba(249,115,22,.14)"),
    "uncertain":         ("#60a5fa", "rgba(59,130,246,.16)"),
}

def _svg(paths, color="#64748b"):
    return (f'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-2px;margin-right:8px;flex-shrink:0">{paths}</svg>')

_ALERT = '<path d="M10.3 3.9l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3l-8-14a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
IC_PHOTO = _svg('<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/>')
IC_EYE   = _svg('<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="2.5"/>')
IC_BULB  = _svg('<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10c.7.7 1 1.6 1 2.5h6c0-.9.3-1.8 1-2.5a6 6 0 0 0-4-10z"/>')
IC_ALERT = _svg(_ALERT)
IC_CLOCK = _svg('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>')
IC_INFO  = _svg('<circle cx="12" cy="12" r="9"/><path d="M12 8h.01"/><path d="M11 12h1v4h1"/>', "#94a3b8")
IC_WARN  = _svg(_ALERT, "#fca5a5")

def rendu_html(s):
    """Construit une carte de résultat lisible à partir de la sortie JSON."""
    classe = s["predicted_class"]
    coul, coul_bg = COULEUR.get(classe, COULEUR["uncertain"])
    conf_pct = int(round(s["confidence"] * 100))

    barres = ""
    for c in CLASSES:
        p = s["probabilities"][c]
        cc = COULEUR[c][0]
        gras = "font-weight:600;color:#f1f5f9" if c == classe else ""
        barres += (f'<div class="pb-row"><span class="pb-lab" style="{gras}">{LIBELLE[c]}</span>'
                   f'<div class="pb-track"><div class="pb-fill" style="width:{p*100:.0f}%;background:{cc}"></div></div>'
                   f'<span class="pb-val" style="{gras}">{p*100:.0f}%</span></div>')

    if s["mode_degrade"]:
        banniere = f'<div class="warn-alert">{IC_WARN}{s["warning"]}</div>'
    else:
        banniere = f'<div class="warn-info">{IC_INFO}{s["warning"]}</div>'

    def det(icon, titre, val):
        return (f'<div class="det"><div class="det-t">{icon}{titre}</div>'
                f'<div class="det-v">{val or "—"}</div></div>')

    return (
        f'<div class="carte">'
        f'<div class="verdict" style="background:{coul_bg};border-left:6px solid {coul}">'
        f'<span class="v-classe" style="color:{coul}">{LIBELLE[classe]}</span>'
        f'<span class="v-conf">Confiance <b>{conf_pct}%</b></span></div>'
        f'{banniere}'
        f'<div class="probs">{barres}</div>'
        f'<div class="dets">'
        + det(IC_PHOTO, "Qualité de l'image", s["image_quality"])
        + det(IC_EYE, "Signes visuels", s["visual_evidence"])
        + det(IC_BULB, "Justification", s["justification"])
        + det(IC_ALERT, "Limitations", s["limitations"])
        + f'</div><div class="meta">{IC_CLOCK}{s["latence_s"]} s · analyse journalisée (SQLite)</div></div>'
    )

# --------------------------------------------------------------------------- Interface
CSS = """
#titre {text-align:center; padding:8px 0 2px;}
#titre h1 {margin:0; font-size:1.7rem; font-weight:600; letter-spacing:.3px;}
#titre p {margin:5px 0 0; color:#64748b; font-size:.9rem;}
.carte {background:#1e293b; color:#e2e8f0; border-radius:14px; overflow:hidden;
        border:0.5px solid #334155; font-family:system-ui,-apple-system,sans-serif;}
.verdict {display:flex; justify-content:space-between; align-items:center; padding:15px 20px;}
.v-classe {font-size:1.45rem; font-weight:600;}
.v-conf {font-size:.95rem; color:#cbd5e1;}
.v-conf b {color:#f8fafc; font-weight:600;}
.warn-alert {display:flex; align-items:center; background:#3a1d1d; color:#fca5a5; padding:10px 20px; font-size:.85rem;}
.warn-info {display:flex; align-items:center; background:#0f172a; color:#94a3b8; padding:10px 20px; font-size:.85rem;}
.probs {padding:15px 20px 8px;}
.pb-row {display:flex; align-items:center; gap:10px; margin:8px 0;}
.pb-lab {width:150px; font-size:.85rem; color:#94a3b8;}
.pb-track {flex:1; background:#334155; border-radius:5px; height:10px; overflow:hidden;}
.pb-fill {height:100%; border-radius:5px; transition:width .4s ease;}
.pb-val {width:44px; text-align:right; font-size:.85rem; color:#94a3b8; font-variant-numeric:tabular-nums;}
.dets {padding:8px 20px 14px; display:grid; gap:13px;}
.det-t {display:flex; align-items:center; font-weight:600; font-size:.88rem; color:#f1f5f9; margin-bottom:3px;}
.det-v {font-size:.9rem; color:#94a3b8; line-height:1.5;}
.meta {display:flex; align-items:center; padding:10px 20px; background:#0f172a; color:#64748b; font-size:.78rem;}
#pied {text-align:center; color:#64748b; font-size:.82rem; margin-top:10px;}
"""

PLACEHOLDER = ('<div class="carte"><div class="warn-info">Charge une radiographie et clique '
               'Analyser pour obtenir le compte-rendu.</div></div>')

init_db()
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
               css=CSS, title="PneumoniX") as demo:
    gr.HTML('<div id="titre"><h1>PneumoniX</h1>'
            '<p>Le classifieur décide la classe · MedGemma rédige l\'analyse · '
            'prototype pédagogique, ne remplace pas un diagnostic médical</p></div>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            entree = gr.File(label="Radiographie (.dcm / .png / .jpg)",
                             file_types=[".dcm", ".png", ".jpg", ".jpeg"], type="filepath")
            bouton = gr.Button("Analyser", variant="primary", size="lg")
            apercu = gr.Image(label="Radiographie", type="pil", height=360)
        with gr.Column(scale=5):
            resultat = gr.HTML(PLACEHOLDER)
            with gr.Accordion("Voir le JSON brut (contrat de sortie)", open=False):
                json_brut = gr.JSON()
    gr.HTML(f'<div id="pied">Garde-fou : confiance &lt; {SEUIL:.2f} → routé vers « incertain » '
            '(relecture humaine) · chaque analyse est journalisée dans <code>journal_inferences.sqlite</code></div>')
    bouton.click(predire, inputs=entree, outputs=[apercu, resultat, json_brut])

if __name__ == "__main__":
    demo.launch(share = True)   # share=True pour un lien public temporaire
