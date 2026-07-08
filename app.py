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
import pandas as pd
import cv2
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

# --------------------------------------------------------------------------- Grad-CAM (explicabilité)
# On capture les activations et les gradients de la dernière couche conv du DenseNet (sortie 7x7x1024).
# La carte de chaleur montre OÙ le classifieur a fondé sa décision -> le VLM dit quoi, Grad-CAM dit où.
_gc = {}                                 # tampon partagé activations/gradients
_COUCHE_CIBLE = clf.features             # bloc convolutif final (avant pooling + classifieur)

def _hook_forward(module, entree, sortie):
    _gc["act"] = sortie
    if sortie.requires_grad:             # sous inference_mode (prédiction), pas de gradient -> on saute
        sortie.register_hook(lambda g: _gc.__setitem__("grad", g))

_COUCHE_CIBLE.register_forward_hook(_hook_forward)

def carte_chaleur(image, classe_idx, alpha=0.45):
    """Grad-CAM pour la classe décidée -> overlay PIL (radio + zones chaudes), ou None si échec."""
    try:
        x = tf_eval(image).unsqueeze(0).to(DEVICE)
        clf.zero_grad(set_to_none=True)
        logits = clf(x)                                    # forward AVEC gradient (déclenche les hooks)
        logits[0, classe_idx].backward()                   # rétropropage le score de la classe visée
        act, grad = _gc["act"][0], _gc["grad"][0]          # [1024,7,7] chacun
        poids = grad.mean(dim=(1, 2))                      # importance de chaque canal
        cam = torch.relu((poids[:, None, None] * act).sum(0))
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()

        base = np.array(image.convert("RGB"))
        H, W = base.shape[:2]
        heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (W, H))), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)       # cv2 -> RGB pour PIL
        overlay = np.uint8((1 - alpha) * base + alpha * heat)
        return Image.fromarray(overlay)
    except Exception as e:
        print("Grad-CAM indisponible:", e)
        return None

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
        return None, None, "<div class='carte'><div class='warn-info'>Aucun fichier fourni.</div></div>", {}
    t0 = time.time()
    image = charger_image(fichier)

    # 1) CLASSIFIEUR : décision de classe + confiance
    x = tf_eval(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prob = torch.softmax(clf(x), 1)[0].cpu().numpy()
    idx = int(prob.argmax())
    classe, conf = CLASSES[idx], float(prob[idx])

    # 1bis) GRAD-CAM : où le classifieur a regardé pour décider cette classe
    heat = carte_chaleur(image, idx)

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
    return image, heat, rendu_html(sortie), sortie

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

# --------------------------------------------------------------------------- File de triage (worklist)
# Priorisation d'un lot : SEUL le classifieur léger tourne (instantané) -> on trie toute la file ;
# le VLM (lourd) reste pour l'analyse détaillée d'un cas isolé. C'est le sens de « classifieur léger ».
PRIORITE = {"suspected_opacity": 0, "uncertain": 1, "normal": 2}          # 0 = plus urgent
BADGE    = {"suspected_opacity": "🔴 Urgent", "uncertain": "🟡 À relire", "normal": "🟢 Normal"}
ACTION   = {"suspected_opacity": "Lecture prioritaire",
            "uncertain": "Relecture humaine requise", "normal": "File standard"}

def _classer(image):
    """Classifieur léger seul -> (classe finale après garde-fou, confiance, mode_degrade)."""
    x = tf_eval(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prob = torch.softmax(clf(x), 1)[0].cpu().numpy()
    idx = int(prob.argmax())
    conf = float(prob[idx])
    mode_degrade = conf < SEUIL
    classe = "uncertain" if mode_degrade else CLASSES[idx]
    return classe, conf, mode_degrade

def trier_file(fichiers):
    """Trie un lot de radios par priorité clinique (urgent -> relecture -> standard)."""
    if not fichiers:
        return "<div class='carte'><div class='warn-info'>Aucun fichier fourni.</div></div>", pd.DataFrame()
    lignes = []
    for chemin in fichiers:
        t0 = time.time()
        try:
            classe, conf, mode_degrade = _classer(charger_image(chemin))
        except Exception as e:
            print("Triage: échec sur", chemin, ":", e)
            continue
        journaliser(chemin, {"predicted_class": classe, "confidence": round(conf, 2),   # nourrit la supervision
                             "mode_degrade": mode_degrade, "latence_s": round(time.time() - t0, 3)})
        lignes.append({"_tier": PRIORITE[classe], "_conf": conf,
                       "Priorité": BADGE[classe], "Fichier": os.path.basename(chemin),
                       "Classe": LIBELLE[classe], "Confiance": f"{conf*100:.0f}%",
                       "Action recommandée": ACTION[classe]})
    if not lignes:
        return "<div class='carte'><div class='warn-info'>Aucune image exploitable.</div></div>", pd.DataFrame()

    df = pd.DataFrame(lignes).sort_values(["_tier", "_conf"], ascending=[True, False]).reset_index(drop=True)
    n, n_urg = len(df), int((df["_tier"] == 0).sum())
    n_rel, n_norm = int((df["_tier"] == 1).sum()), int((df["_tier"] == 2).sum())
    df.insert(0, "Rang", range(1, n + 1))
    df = df.drop(columns=["_tier", "_conf"])

    def kpi(v, lib, coul="#f7f2e9"):
        return f"<div class='kpi'><div class='kpi-v' style='color:{coul}'>{v}</div><div class='kpi-l'>{lib}</div></div>"
    resume = ("<div class='kpis'>" + kpi(n, "Radios triées")
              + kpi(n_urg, "Lecture prioritaire", "#fb7185")
              + kpi(n_rel, "Relecture humaine", "#fbbf24")
              + kpi(n_norm, "File standard", "#4ade80") + "</div>")
    return resume, df

# --------------------------------------------------------------------------- Tableau de bord (journal SQLite)
def _lire_journal():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM inferences ORDER BY id", con)
    con.close()
    return df

def stats_dashboard():
    """Agrège le journal SQLite -> KPIs + graphiques + dernières analyses (données live)."""
    df = _lire_journal()
    vide_c = pd.DataFrame({"Classe": [], "Nombre": []})
    vide_g = pd.DataFrame({"Statut": [], "Nombre": []})
    if df.empty:
        return ("<div class='carte'><div class='warn-info'>Aucune analyse journalisée pour l'instant.</div></div>",
                vide_c, vide_g, pd.DataFrame())

    total       = len(df)
    n_relecture = int(df["mode_degrade"].sum())
    taux        = 100 * n_relecture / total
    conf_moy    = 100 * df["confidence"].mean()
    lat_moy     = df["latence_s"].mean()

    def kpi(val, lib):
        return f"<div class='kpi'><div class='kpi-v'>{val}</div><div class='kpi-l'>{lib}</div></div>"
    kpis = ("<div class='kpis'>"
            + kpi(total, "Analyses réalisées")
            + kpi(f"{taux:.0f}%", "Routées en relecture")
            + kpi(f"{conf_moy:.0f}%", "Confiance moyenne")
            + kpi(f"{lat_moy:.1f}s", "Latence moyenne")
            + "</div>")

    rep = df["predicted_class"].value_counts().reindex(CLASSES, fill_value=0)
    df_classes = pd.DataFrame({"Classe": [LIBELLE[c] for c in rep.index], "Nombre": rep.values})

    df_garde = pd.DataFrame({"Statut": ["Validées auto", "Routées en relecture"],
                             "Nombre": [total - n_relecture, n_relecture]})

    recent = df.sort_values("id", ascending=False).head(10)[
        ["horodatage", "fichier", "predicted_class", "confidence", "mode_degrade", "latence_s"]].copy()
    recent["predicted_class"] = recent["predicted_class"].map(LIBELLE)
    recent["mode_degrade"]    = recent["mode_degrade"].map({1: "Oui", 0: "—"})
    recent.columns = ["Horodatage", "Fichier", "Classe", "Confiance", "Relecture", "Latence (s)"]
    return kpis, df_classes, df_garde, recent

# --------------------------------------------------------------------------- Interface
CSS = """
/* Fond global sombre et chaud (cf. maquette de référence) */
.gradio-container {background: radial-gradient(1100px 520px at 50% -12%, #241a10, #100c08) !important;}
#titre {text-align:center; padding:8px 0 2px;}
#titre h1 {margin:0; font-size:1.6rem; font-weight:600; letter-spacing:.3px; color:#f5efe4;}
#titre p {margin:5px 0 0; color:#9a9080; font-size:.9rem;}
.carte {background:#1c1813; color:#ece7dd; border-radius:14px; overflow:hidden;
        border:0.5px solid #3a3327; font-family:system-ui,-apple-system,sans-serif;}
.verdict {display:flex; justify-content:space-between; align-items:center; padding:15px 20px;}
.v-classe {font-size:1.45rem; font-weight:600;}
.v-conf {font-size:.95rem; color:#cdc5b6;}
.v-conf b {color:#f7f2e9; font-weight:600;}
.warn-alert {display:flex; align-items:center; background:#3a1d1d; color:#fca5a5; padding:10px 20px; font-size:.85rem;}
.warn-info {display:flex; align-items:center; background:#14110c; color:#9a9080; padding:10px 20px; font-size:.85rem;}
.probs {padding:15px 20px 8px;}
.pb-row {display:flex; align-items:center; gap:10px; margin:8px 0;}
.pb-lab {width:150px; font-size:.85rem; color:#9a9080;}
.pb-track {flex:1; background:#3a3327; border-radius:5px; height:10px; overflow:hidden;}
.pb-fill {height:100%; border-radius:5px; transition:width .4s ease;}
.pb-val {width:44px; text-align:right; font-size:.85rem; color:#9a9080; font-variant-numeric:tabular-nums;}
.dets {padding:8px 20px 14px; display:grid; gap:13px;}
.det-t {display:flex; align-items:center; font-weight:600; font-size:.88rem; color:#f1ece2; margin-bottom:3px;}
.det-v {font-size:.9rem; color:#9a9080; line-height:1.5;}
.meta {display:flex; align-items:center; padding:10px 20px; background:#14110c; color:#8a8272; font-size:.78rem;}
#pied {text-align:center; color:#8a8272; font-size:.82rem; margin-top:10px;}
.kpis {display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 16px;}
.kpi {background:#1c1813; border:0.5px solid #3a3327; border-radius:12px; padding:16px 12px; text-align:center;}
.kpi-v {font-size:1.85rem; font-weight:600; color:#f7f2e9; font-variant-numeric:tabular-nums;}
.kpi-l {font-size:.78rem; color:#9a9080; margin-top:5px;}
/* Barre latérale (panneau ouvrable) */
.sb-logo {font-size:1.35rem; font-weight:700; color:#f6a94a; padding:8px 6px 0;}
.sb-sub {font-size:.78rem; color:#9a9080; padding:2px 6px 14px; border-bottom:1px solid #2a2419; margin-bottom:8px;}
.nav {width:100% !important; text-align:left !important; justify-content:flex-start !important;
      background:transparent !important; border:0 !important; box-shadow:none !important;
      color:#cdc5b6 !important; font-weight:500 !important; padding-left:10px !important;}
.nav:hover {background:#241d14 !important; color:#f6a94a !important;}
/* Colonne du bouton Analyser : centrage vertical face à l'import */
.col-btn {justify-content:center !important;}
"""

PLACEHOLDER = ('<div class="carte"><div class="warn-info">Charge une radiographie et clique '
               'Analyser pour obtenir le compte-rendu.</div></div>')

# --------------------------------------------------------------------------- Guide
try:
    with open("GUIDE_DEMO.md", encoding="utf-8") as _f:
        GUIDE_TXT = _f.read()
except Exception:
    GUIDE_TXT = "_Guide indisponible (GUIDE_DEMO.md introuvable)._"

# Thème sombre chaud (orange/pierre) + on force le mode sombre au chargement.
THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="stone").set(
    body_background_fill_dark="#100c08",
    background_fill_primary_dark="#1c1813",
    background_fill_secondary_dark="#14110c",
    block_background_fill_dark="#1c1813",
    border_color_primary_dark="#3a3327",
)
JS_DARK = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
        return;
    }
    // Retire le « - ou - » et « Cliquez pour télécharger » sous « Déposez le fichier ici »
    const nettoyer = () => {
        document.querySelectorAll('.wrap').forEach(w => {
            const ou = w.querySelector('.or');
            if (ou) {
                let n = ou.nextSibling;
                while (n) { const s = n.nextSibling; n.remove(); n = s; }
                ou.remove();
            }
        });
    };
    setTimeout(nettoyer, 300);
    setTimeout(nettoyer, 1000);
}
"""

init_db()
with gr.Blocks(theme=THEME, css=CSS, title="PneumoniX", js=JS_DARK) as demo:

    # --------------------------------------------------------------- Panneau ouvrable (navigation)
    with gr.Sidebar(open=True):
        gr.HTML("<div class='sb-logo'>PneumoniX</div>"
                "<div class='sb-sub'>Assistant radiologue virtuel</div>")
        nav_analyse = gr.Button("Analyse", elem_classes="nav")
        nav_triage  = gr.Button("File de triage", elem_classes="nav")
        nav_suivi   = gr.Button("Suivi & supervision", elem_classes="nav")
        nav_guide   = gr.Button("Guide d'utilisation", elem_classes="nav")

    # =============================================================== PAGE : ANALYSE
    with gr.Column(visible=True) as page_analyse:
        gr.HTML('<div id="titre"><h1>PneumoniX — Analyse</h1></div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                entree = gr.File(label="Radiographie (.dcm / .png / .jpg)",
                                 file_types=[".dcm", ".png", ".jpg", ".jpeg"], type="filepath")
            with gr.Column(scale=1, min_width=140, elem_classes="col-btn"):
                bouton = gr.Button("Analyser", variant="primary", size="lg")

        with gr.Accordion("Analyse", open=False) as acc_analyse:       # radio (gauche) + compte-rendu (droite)
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    apercu = gr.Image(label="Radiographie", type="pil", height=430)
                with gr.Column(scale=1):
                    resultat = gr.HTML(PLACEHOLDER)

        with gr.Accordion("Carte d'attention — Grad-CAM", open=False) as acc_gradcam:  # explicabilité
            carte = gr.Image(label="Zones qui ont pesé dans la décision", type="pil", height=430)
            gr.HTML("<div style='color:#8a8272;font-size:.8rem;padding:4px'>"
                    "Rouge = régions qui ont le plus pesé dans la décision du classifieur.</div>")

        with gr.Accordion("Contrat de sortie — JSON", open=False) as acc_json:  # sortie structurée
            json_brut = gr.JSON()

        gr.HTML('<div id="pied">Chaque analyse est journalisée dans '
                '<code>journal_inferences.sqlite</code> (traçabilité).</div>')

    # =============================================================== PAGE : FILE DE TRIAGE
    with gr.Column(visible=False) as page_triage:
        gr.HTML('<div id="titre"><h1>File de triage</h1>'
                '<p>Priorisation automatique d\'un lot de radios — le classifieur léger trie tout le lot, '
                'le VLM reste pour l\'analyse détaillée</p></div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                entree_lot = gr.File(label="Radios à trier (.dcm / .png / .jpg — sélection multiple)",
                                     file_types=[".dcm", ".png", ".jpg", ".jpeg"],
                                     type="filepath", file_count="multiple")
            with gr.Column(scale=1, min_width=140, elem_classes="col-btn"):
                bouton_tri = gr.Button("Trier la file", variant="primary", size="lg")
        resume_tri = gr.HTML()
        table_tri  = gr.Dataframe(label="File d'attente priorisée (cas urgents en haut)",
                                  interactive=False, wrap=True)

    # =============================================================== PAGE : SUIVI & SUPERVISION
    with gr.Column(visible=False) as page_suivi:
        gr.HTML('<div id="titre"><h1>Suivi & supervision du service</h1>'
                '<p>Activité, sécurité (garde-fou) et traçabilité — données live du journal d\'audit</p></div>')
        btn_refresh = gr.Button("Rafraîchir", size="sm")
        kpi_html = gr.HTML()
        with gr.Row():
            plot_classes = gr.BarPlot(x="Classe", y="Nombre", color="Classe",
                                      title="Répartition des décisions", height=300)
            plot_garde = gr.BarPlot(x="Statut", y="Nombre", color="Statut",
                                    title="Impact du garde-fou de confiance", height=300)
        table_recent = gr.Dataframe(label="10 dernières analyses", interactive=False, wrap=True)

    # =============================================================== PAGE : GUIDE
    with gr.Column(visible=False) as page_guide:
        gr.HTML('<div id="titre"><h1>Guide d\'utilisation</h1></div>')
        gr.Markdown(GUIDE_TXT)

    # --------------------------------------------------------------- Navigation & actions
    pages   = [page_analyse, page_triage, page_suivi, page_guide]
    dash_out = [kpi_html, plot_classes, plot_garde, table_recent]

    def _aff(i):                                     # n'affiche que la page i
        return [gr.update(visible=(j == i)) for j in range(len(pages))]

    nav_analyse.click(lambda: _aff(0), outputs=pages)
    nav_triage.click(lambda: _aff(1), outputs=pages)
    nav_suivi.click(lambda: _aff(2), outputs=pages).then(stats_dashboard, outputs=dash_out)
    nav_guide.click(lambda: _aff(3), outputs=pages)

    bouton.click(predire, inputs=entree, outputs=[apercu, carte, resultat, json_brut]).then(
        lambda: (gr.update(open=True), gr.update(open=True)),   # déplie les sections après analyse
        outputs=[acc_analyse, acc_gradcam])
    bouton_tri.click(trier_file, inputs=entree_lot, outputs=[resume_tri, table_tri])
    btn_refresh.click(stats_dashboard, outputs=dash_out)
    demo.load(stats_dashboard, outputs=dash_out)     # état initial du suivi au démarrage

if __name__ == "__main__":
    demo.launch(share=True)   # share=True pour un lien public temporaire
