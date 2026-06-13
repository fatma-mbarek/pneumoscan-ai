# ============================================================
#  PneumoScan AI — Grad-CAM : Explicabilité Visuelle
#  Génère une heatmap sur la radio pour montrer où l'IA regarde
# ============================================================

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.config import *

def get_gradcam_heatmap(model, img_array, layer_name="conv5_block16_1_conv"):
    """
    Calcule la heatmap Grad-CAM pour une image donnée.
    
    Principe :
      1. Propagation avant → obtenir les activations de la dernière couche conv
      2. Calculer le gradient de la prédiction par rapport à ces activations
      3. Pondérer les activations par leurs gradients → carte d'importance
      4. ReLU → garder uniquement les zones positives (contribuant à la classe)
    
    Args:
        model      : modèle Keras chargé
        img_array  : np.array shape (1, 224, 224, 3), normalisé [0,1]
        layer_name : dernière couche conv d'EfficientNetB0
    
    Returns:
        heatmap : np.array (H, W) valeurs entre 0 et 1
    """

    # Modèle intermédiaire : input → [activations conv, prédiction]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        # Score pour la classe PNEUMONIA (sortie sigmoid)
        loss = predictions[:, 0]

    # Gradients de la prédiction par rapport aux activations conv
    grads = tape.gradient(loss, conv_outputs)

    # Importance de chaque feature map = moyenne des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondérer les activations
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalisation [0, 1]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Superpose la heatmap sur l'image originale.
    
    Args:
        original_img_path : chemin vers l'image originale
        heatmap           : np.array (H, W) de get_gradcam_heatmap
        alpha             : intensité de la heatmap (0=invisible, 1=opaque)
        colormap          : palette de couleurs (JET = bleu→rouge)
    
    Returns:
        superimposed : np.array (H, W, 3) RGB — image avec heatmap
    """
    # Charger et redimensionner l'image originale
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionner la heatmap à la taille de l'image
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    # Convertir en image couleur
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superposition
    superimposed = (alpha * heatmap_colored + (1 - alpha) * img_rgb).astype(np.uint8)

    return superimposed, img_rgb


def generate_gradcam_figure(model, img_array, img_path, prediction, confidence):
    """
    Génère une figure matplotlib complète avec :
    - Image originale
    - Heatmap seule
    - Superposition Grad-CAM
    - Score de confiance
    
    Returns:
        fig : matplotlib Figure (utilisée par Streamlit)
    """
    heatmap = get_gradcam_heatmap(model, img_array)
    overlay, original = overlay_heatmap(img_path, heatmap)

    label  = CLASSES[int(prediction >= 0.5)]
    color  = "#ef4444" if label == "PNEUMONIA" else "#10b981"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1320")

    titles = ["📷 Original X-Ray", "🌡️ Grad-CAM Heatmap", "🔍 Overlay"]
    images = [original, plt.cm.jet(heatmap)[:, :, :3], overlay]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img if len(img.shape) == 3 else img, cmap=None)
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.axis("off")
        ax.set_facecolor("#0d1320")

    verdict = f"{'⚠️ PNEUMONIA' if label == 'PNEUMONIA' else '✅ NORMAL'}  —  Confidence : {confidence:.1f}%"
    fig.suptitle(verdict, fontsize=14, fontweight="bold", color=color, y=1.02)
    plt.tight_layout()

    return fig


def get_confidence_level(confidence):
    if confidence >= 90:
        return "🟢 High confidence", "The prediction is very reliable."
    elif confidence >= 70:
        return "🟡 Moderate confidence", "Probable result — confirmation recommended."
    else:
        return "🔴 Low confidence", "Uncertain result — medical advice is essential."