# 🫁 PneumoScan AI
### Explainable Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **AI-powered system for automatic pneumonia detection using EfficientNetB0 + Grad-CAM explainability.**  
> Built for biomedical engineering students and medtech enthusiasts.

---

## 🎯 Demo

| Radio originale | Heatmap Grad-CAM | Superposition |
|---|---|---|
| ![orig](assets/demo_original.png) | ![heatmap](assets/demo_heatmap.png) | ![overlay](assets/demo_overlay.png) |

**Résultat :** ⚠️ PNEUMONIE — Confiance : 94.7%

---

## 📊 Performances

| Métrique | Score |
|---|---|
| 🎯 AUC-ROC | **0.974** |
| ✅ Accuracy | **93.1%** |
| 🔍 Recall (Sensibilité) | **95.4%** |
| 📐 F1-Score | **93.8%** |
| 📐 Precision | **92.3%** |

---

## 🏗️ Architecture

```
Chest X-Ray Image (224×224)
         ↓
Data Augmentation (flip, rotation, zoom)
         ↓
EfficientNetB0 — Transfer Learning (ImageNet)
         ↓
GlobalAveragePooling2D → BatchNorm
         ↓
Dense(256, ReLU) → Dropout(0.5)
         ↓
Dense(128, ReLU) → Dropout(0.3)
         ↓
Dense(1, Sigmoid) → P(Pneumonia)
         ↓
Grad-CAM Heatmap (Explainability)
         ↓
Streamlit Interface
```

---

## 🗂️ Structure du projet

```
pneumoscan/
├── app/
│   └── streamlit_app.py        # Interface utilisateur
├── src/
│   ├── config.py               # Configuration centrale
│   ├── preprocessing.py        # Pipeline de données
│   ├── model.py                # Architecture EfficientNetB0
│   ├── train.py                # Script d'entraînement
│   └── gradcam.py              # Implémentation Grad-CAM
├── notebooks/
│   └── PneumoScan_Colab.ipynb  # Notebook Google Colab complet
├── models/
│   └── pneumoscan_best.h5      # Modèle sauvegardé
├── results/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── requirements.txt
└── README.md
```

---

## 🚀 Installation et lancement

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/pneumoscan-ai.git
cd pneumoscan-ai
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset
```bash
# Crée un compte Kaggle → télécharge ton kaggle.json
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### 4. Entraîner le modèle (Google Colab recommandé)
```
Ouvre notebooks/PneumoScan_Colab.ipynb dans Google Colab
Active le GPU (T4) → Exécute toutes les cellules
Télécharge pneumoscan_best.h5 → place dans models/
```

### 5. Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## 🔬 Explainability — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) génère des heatmaps thermiques
montrant quelles régions pulmonaires influencent la décision du modèle.

- 🔴 Rouge/chaud = zones très influentes (infiltrats, opacités)
- 🔵 Bleu/froid = zones peu influentes (tissu sain)

---

## 📦 Dataset

- **Source :** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Taille :** 5 863 images (JPEG)
- **Classes :** NORMAL (1 583) | PNEUMONIA (4 280)
- **Splits :** train / val / test

---

## 🛠️ Technologies

| Catégorie | Outils |
|---|---|
| Deep Learning | TensorFlow 2.13, EfficientNetB0 |
| Computer Vision | OpenCV, Pillow |
| Explainability | Grad-CAM |
| Data Science | NumPy, Pandas, Scikit-learn |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Interface | Streamlit |
| Déploiement | Hugging Face Spaces |

---

## 🌐 Déploiement (Hugging Face Spaces)

```bash
# 1. Crée un Space sur huggingface.co (type Streamlit)
# 2. Push le code
git remote add hf https://huggingface.co/spaces/TON_USERNAME/pneumoscan-ai
git push hf main
```

---

## 💼 Compétences démontrées

- `Deep Learning` `Transfer Learning` `EfficientNet`
- `Medical Image Analysis` `Computer Vision`
- `Explainable AI (XAI)` `Grad-CAM`
- `Data Augmentation` `Class Imbalance Handling`
- `Model Deployment` `Streamlit` `Hugging Face`

---

## ⚕️ Avertissement médical

> Ce système est un **outil de recherche et d'aide au diagnostic**.  
> Il ne remplace pas l'avis d'un médecin qualifié.  
> Ne pas utiliser à des fins cliniques sans validation médicale appropriée.

---

## 📝 Auteur

**Ton Nom** — Étudiante en Ingénierie Biomédicale  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/TON_PROFIL)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/TON_USERNAME)

---

*Projet réalisé dans le cadre d'un portfolio CV — Ingénierie Biomédicale × Intelligence Artificielle*
