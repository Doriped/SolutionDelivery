# Guide utilisateur — Démo « Assistant radiologue virtuel » (S5)

Prototype **pédagogique** de classification de radiographies thoraciques frontales.
⚠️ *Ne remplace pas un diagnostic médical. Toute sortie porte un avertissement.*

## Architecture

L'application combine **les deux modèles** (architecture cible du projet) :

| Composant | Rôle |
|---|---|
| **Classifieur léger** (DenseNet-121, S4) | décide la **classe** + la **confiance** (rapide, fiable) |
| **MedGemma** (VLM) | rédige l'**analyse spécifique à l'image** : signes visuels, justification, qualité, limitations |
| **Garde-fou** | confiance < 0.60 → route vers `uncertain` + avertissement « relecture requise » |
| **SQLite** | journalise 100 % des analyses (`journal_inferences.sqlite`) |

Le classifieur *décide*, MedGemma *explique* — chacun sur ce qu'il fait le mieux.

## 1. Lancer la démo

```bash
./env/Scripts/python.exe app.py
```

Au démarrage, MedGemma se charge (~1 min). Une URL locale s'affiche (`http://127.0.0.1:7860`).
Pour un lien public temporaire : `demo.launch(share=True)` (dernière ligne de `app.py`).

> **Mémoire GPU insuffisante ?** Mettre `CHARGER_EN_4BIT = True` en haut de `app.py`
> (MedGemma en 4-bit, ~4 Go, un peu plus lent).

## 2. Utiliser

1. **Charge** une radiographie : `.dcm`, `.png` ou `.jpg`.
2. Clique **Analyser**.
3. Résultat : image analysée + **probabilités par classe** + **JSON structuré** spécifique à l'image
   (classe, confiance, `visual_evidence`, `justification`, `limitations`, avertissement, latence).

## 3. Consulter le journal (SQLite)

```bash
./env/Scripts/python.exe -c "import sqlite3; \
print(sqlite3.connect('journal_inferences.sqlite').execute('SELECT * FROM inferences').fetchall())"
```

## 4. Cibles du projet

| Cible | Statut |
|---|---|
| 1 API `/predict` | ✅ `predire()` |
| JSON valide ≥ 95 % | ✅ (repli si le VLM échoue) |
| Avertissement présent | ✅ 100 % |
| Sorties journalisées | ✅ SQLite |
| Latence < 10 s | ~classifieur < 0.1 s + justification VLM ~10 s |

## 5. Deux profils de déploiement

| Profil | Analyse image (VLM) | Matériel serveur | Latence | Usage |
|---|:---:|---|:---:|---|
| **Complet** (défaut) | ✅ oui | **GPU** (~8 Go, ou 4-bit) | ~10 s | démo riche, fidèle à l'architecture |
| **Léger** | ❌ (justification templatée) | **CPU** suffit | < 0.1 s | déploiement Hugging Face Spaces / machine modeste |

Le **client** (celui qui utilise la démo) n'a besoin que d'un **navigateur** : tout le calcul se fait
sur le **serveur** (la machine qui exécute `app.py`). Pour le profil léger sur CPU, désactiver le VLM
et revenir à une justification par classe.
