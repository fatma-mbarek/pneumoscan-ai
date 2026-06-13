# 🫁 PneumoScan AI
### Explainable Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **AI-powered system for automatic pneumonia detection using DenseNet121 + Grad-CAM explainability.**  
> Built for biomedical engineering students and medtech enthusiasts.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| 🎯 AUC-ROC | **0.93** |
| ✅ Accuracy | **~90%** |
| 🔍 Recall | **~95%** |
| 📐 F1-Score | **~92%** |

---

## 🏗️ Architecture

```
Chest X-Ray Image (224×224)
         ↓
Data Augmentation (flip, rotation, zoom)
         ↓
DenseNet121 — Transfer Learning (ImageNet)
         ↓
GlobalAveragePooling2D → BatchNorm
         ↓
Dense(512, ReLU) → Dropout(0.4)
         ↓
Dense(256, ReLU) → Dropout(0.3)
         ↓
Dense(1, Sigmoid) → P(Pneumonia)
         ↓
Grad-CAM Heatmap (Explainability)
         ↓
Streamlit Interface
```

---

## 🗂️ Project Structure

```
pneumoscan/
├── app/
│   └── streamlit_app.py        # Streamlit web interface
├── src/
│   ├── config.py               # Central configuration
│   ├── preprocessing.py        # Data pipeline
│   ├── model.py                # DenseNet121 architecture
│   ├── train.py                # Training script
│   └── gradcam.py              # Grad-CAM implementation
├── notebooks/
│   └── PneumoScan_Colab.ipynb  # Complete Google Colab notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the repo
```bash
git clone https://github.com/fatma-mbarek/pneumoscan-ai.git
cd pneumoscan-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### 4. Train the model (Google Colab recommended)
```
Open notebooks/PneumoScan_Colab.ipynb in Google Colab
Enable GPU (T4) → Run all cells
Download pneumoscan_best.h5 → place in models/
```

### 5. Launch Streamlit app
```bash
python -m streamlit run app/streamlit_app.py
```

---

## 🔬 Explainability — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) generates thermal heatmaps
showing which lung regions influence the model's decision.

- 🔴 Red/warm = highly influential zones (infiltrates, opacities)
- 🔵 Blue/cold = less influential zones (healthy tissue)

---

## 📦 Dataset

- **Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,863 images (JPEG)
- **Classes:** NORMAL (1,583) | PNEUMONIA (4,280)
- **Splits:** train / val / test

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow 2.17, DenseNet121 |
| Computer Vision | OpenCV, Pillow |
| Explainability | Grad-CAM |
| Data Science | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Interface | Streamlit |

---

## 💼 Skills Demonstrated

- `Deep Learning` `Transfer Learning` `DenseNet121`
- `Medical Image Analysis` `Computer Vision`
- `Explainable AI (XAI)` `Grad-CAM`
- `Data Augmentation` `Class Imbalance Handling`
- `Model Deployment` `Streamlit`

---

## ⚕️ Medical Disclaimer

> This system is a **research and diagnostic aid tool**.  
> It does not replace the opinion of a qualified physician.  
> Do not use for clinical purposes without appropriate medical validation.

---

## 📝 Author

**Fatma Mbarek** — Biomedical Engineering Student  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/fatma-mbarek)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/fatmambarek?)

