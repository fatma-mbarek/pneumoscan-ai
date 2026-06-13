# ============================================================
#  PneumoScan AI — Streamlit Interface (English)
#  Run with : streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.preprocessing import preprocess_single_image
from src.gradcam import generate_gradcam_figure, get_confidence_level
from src.model import load_trained_model

# ════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #070b12; }
    .stApp { background-color: #070b12; }
    .header-box {
        background: linear-gradient(135deg, #0d1320, #111827);
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-pneumonia {
        background: rgba(239,68,68,0.1);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-normal {
        background: rgba(16,185,129,0.1);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .disclaimer {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #f59e0b;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return load_trained_model()


def get_severity(confidence):
    """Estimates pneumonia severity based on confidence score."""
    if confidence >= 90:
        return "🔴 Severe", "#ef4444", "Hospitalization recommended — seek urgent medical attention."
    elif confidence >= 75:
        return "🟠 Moderate", "#f59e0b", "Medical consultation recommended within 24 hours."
    else:
        return "🟡 Mild", "#eab308", "Monitoring recommended — consult a doctor."


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🫁 PneumoScan AI")
    st.markdown("**Automatic Pneumonia Detection**  \nExplainable by Grad-CAM")
    st.divider()

    st.markdown("### 📌 About")
    st.markdown("""
    This system uses **DenseNet121** with Transfer Learning
    to classify chest X-ray images.

    **Grad-CAM** generates heatmaps showing exactly
    which lung regions influence the decision.
    """)
    st.divider()

    st.markdown("### 🔬 Model")
    st.markdown("""
    - Architecture : `DenseNet121`
    - Dataset : Kaggle Chest X-Ray (5,863 images)
    - Metrics : AUC, F1, Recall, Precision
    """)
    st.divider()

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <div class="disclaimer">
    This system is a diagnostic aid tool. 
    It does not replace the opinion of a qualified physician.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MAIN HEADER
# ════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-box">
    <h1 style="color:#00e5ff; font-size:2.5rem; margin:0;">🫁 PneumoScan AI</h1>
    <p style="color:#64748b; margin-top:0.5rem;">
        Explainable AI System for Pneumonia Detection · DenseNet121 + Grad-CAM
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ════════════════════════════════════════════════════════════

col_upload, col_info = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Drag and drop a chest X-ray (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Expected format: frontal chest X-ray in grayscale"
    )

with col_info:
    st.markdown("### 💡 How does it work?")
    st.markdown("""
    1. **Upload** your chest X-ray
    2. The AI analyzes the **visual patterns** of the lungs
    3. Result: **Normal** or **Pneumonia** + confidence score
    4. **Grad-CAM** highlights the zones that influenced the decision
    """)

# ════════════════════════════════════════════════════════════
#  ANALYSIS
# ════════════════════════════════════════════════════════════

if uploaded_file is not None:

    st.divider()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── Load model ─────────────────────────────────────────
    with st.spinner("🔄 Loading AI model..."):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"❌ Model not found: {e}")
            st.info("💡 Train the model first with `python src/train.py`")
            st.stop()

    # ── Prediction ─────────────────────────────────────────
    with st.spinner("🧠 Analyzing X-ray..."):
        img_array = preprocess_single_image(tmp_path)
        prediction = float(model.predict(img_array, verbose=0)[0][0])
        confidence = prediction * 100 if prediction >= 0.65 else (1 - prediction) * 100
        label = "PNEUMONIA" if prediction >= 0.65 else "NORMAL"

    # ── Result ─────────────────────────────────────────────
    st.markdown("### 📋 Diagnostic Result")

    col_res, col_conf, col_level = st.columns(3)

    with col_res:
        if label == "PNEUMONIA":
            st.markdown("""
            <div class="result-pneumonia">
                <h2 style="color:#ef4444; margin:0;">⚠️ PNEUMONIA</h2>
                <p style="color:#64748b; margin:0;">Detected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <h2 style="color:#10b981; margin:0;">✅ NORMAL</h2>
                <p style="color:#64748b; margin:0;">No pneumonia detected</p>
            </div>
            """, unsafe_allow_html=True)

    with col_conf:
        st.metric("Confidence Score", f"{confidence:.1f}%")
        st.progress(confidence / 100)

    with col_level:
        level_label, level_desc = get_confidence_level(confidence)
        st.markdown(f"**{level_label}**")
        st.caption(level_desc)

    # ── Severity Score ─────────────────────────────────────
    if label == "PNEUMONIA":
        st.markdown("### 🏥 Estimated Severity")
        severity_label, severity_color, severity_desc = get_severity(confidence)
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.08); border:2px solid {severity_color};
                    border-radius:10px; padding:1rem; text-align:center; margin-top:1rem;">
            <h2 style="color:{severity_color}; margin:0;">{severity_label}</h2>
            <p style="color:#94a3b8; margin:0.5rem 0 0;">{severity_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Grad-CAM ───────────────────────────────────────────
    st.markdown("### 🔍 Explainability — Grad-CAM")
    st.caption("**Red/warm** zones indicate the lung regions that most influenced the model's decision.")

    with st.spinner("🎨 Generating Grad-CAM heatmap..."):
        try:
            fig = generate_gradcam_figure(model, img_array, tmp_path, prediction, confidence)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM unavailable: {e}")
            st.image(tmp_path, caption="Uploaded X-ray", width=300)

    # ── Medical disclaimer ─────────────────────────────────
    st.markdown("""
    <div class="disclaimer" style="margin-top:1.5rem;">
    ⚕️ <strong>Medical Disclaimer:</strong> This result is provided for informational purposes only.
    It does not constitute a medical diagnosis. Always consult a qualified physician for
    X-ray interpretation and clinical decision-making.
    </div>
    """, unsafe_allow_html=True)

    os.unlink(tmp_path)

else:
    st.info("👆 Upload a chest X-ray to start the analysis.")

    st.markdown("---")
    st.markdown("### 📊 Model Performance (on test set)")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🎯 AUC-ROC", "~0.93"),
        ("✅ Accuracy", "~90%"),
        ("🔍 Recall", "~95%"),
        ("📐 F1-Score", "~92%"),
    ]
    for col, (name, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#64748b; margin:0; font-size:0.85rem;">{name}</p>
                <h3 style="color:#00e5ff; margin:0;">{val}</h3>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#64748b; font-size:0.8rem;">
    PneumoScan AI · Biomedical Engineering · DenseNet121 + Grad-CAM · 
    Dataset: Kaggle Chest X-Ray Images
</p>
""", unsafe_allow_html=True)