---
title: pneumonix
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
---
# PneumoniX — Assistant radiologue virtuel

Prototype **pédagogique** de classification de radiographies thoraciques frontales en 3 classes
(`normal` / `suspected_opacity` / `uncertain`), avec sortie **JSON structurée**, avertissement
systématique et traçabilité.

> ⚠️ Usage pédagogique uniquement — ne remplace pas un diagnostic médical.

## Démarche

Projet mené selon la marche à suivre du cahier des charges : **baseline → amélioration légère →
démonstrateur web**, avec le fine-tuning réservé à une éventuelle validation ultérieure.

| Étape | Contenu |
|-------|---------|
| S1 | Cadrage, données RSNA, prétraitement |
| S2 | Baseline VLM (MedGemma) + contrat de sortie JSON |
| S3 | Comparaison de prompts (baseline / few-shot / CoT) |
| S4 | **Amélioration légère** : classifieur léger (DenseNet-121) + garde-fous de confiance |
| S5 | Démonstrateur web (`app.py`) |
| S6 | Analyse d'erreurs |

## Résultat

Le fine-tuning du VLM (LoRA langage, CLAHE, LoRA vision) dégradait les résultats. Le levier
gagnant est un **classifieur supervisé léger** entraîné sur les labels RSNA :

| Système | Macro-F1 (300 cas) | `uncertain` F1 |
|---------|:---:|:---:|
| Baseline VLM (few-shot) | 0.50 | 0.00 |
| **Classifieur léger + garde-fous** | **0.68** | **0.55** |

Cible du projet (Macro-F1 ≥ 0.68) atteinte, de façon reproductible.

## Architecture de l'application

```
Upload radio -> prétraitement -> [ classifieur (classe + confiance) + MedGemma (analyse rédigée) ]
             -> garde-fou (confiance < 0.60 -> uncertain) -> JSON -> interface web -> journal SQLite
```

Le **classifieur** décide la classe (instantané) ; **MedGemma** rédige l'analyse spécifique à
l'image ; un **garde-fou** route les cas peu confiants vers `uncertain` (relecture humaine).

## Structure du dépôt

| Fichier | Rôle |
|---------|------|
| `assistant_radiologue_v2.ipynb` | Notebook principal, documenté (S1 → S6) |
| `app.py` | Application web PneumoniX (Gradio) |
| `GUIDE_DEMO.md` | Guide utilisateur de la démo |
| `classifieur_leger/densenet121_best.pt` | Modèle entraîné (DenseNet-121) |
| `notebook.ipynb` | Archive des expériences de fine-tuning (résultats négatifs) |

## Utilisation

Prérequis : environnement Python avec `torch`, `torchvision`, `transformers`, `gradio`, `pydicom`,
`opencv-python-headless`, et un token Hugging Face (`.env` : `HF_TOKEN=...`) pour MedGemma.

Le dataset [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
n'est pas versionné : le placer dans `rsna-pneumonia-detection-challenge/`.

Lancer la démo web :

```bash
python app.py
```

Puis ouvrir http://127.0.0.1:7860 (voir `GUIDE_DEMO.md`).
