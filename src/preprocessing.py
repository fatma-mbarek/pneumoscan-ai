# ============================================================
#  PneumoScan AI — Pipeline de Prétraitement
#  Étape 1 : préparer les images pour l'entraînement
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from src.config import *


def get_class_weights(train_dir):
    """
    Calcule les poids de classe pour gérer le déséquilibre
    (plus d'images PNEUMONIA que NORMAL dans le dataset).
    
    Returns: dict {0: w_normal, 1: w_pneumonia}
    """
    normal_count    = len(os.listdir(os.path.join(train_dir, "NORMAL")))
    pneumonia_count = len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))
    total           = normal_count + pneumonia_count

    weight_normal    = total / (2.0 * normal_count)
    weight_pneumonia = total / (2.0 * pneumonia_count)

    print(f"📊 Dataset : {normal_count} NORMAL | {pneumonia_count} PNEUMONIA")
    print(f"⚖️  Poids   : NORMAL={weight_normal:.2f} | PNEUMONIA={weight_pneumonia:.2f}")

    return {0: weight_normal, 1: weight_pneumonia}


def build_generators(train_dir, val_dir, test_dir, batch_size=BATCH_SIZE):
    """
    Construit les générateurs d'images Keras avec augmentation pour le train.
    
    Augmentation : retournement horizontal, rotation ±15°,
                   zoom 10%, décalage 10% — images médicales réalistes.
    """

    # ── TRAIN : avec augmentation ──────────────────────────
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        fill_mode="nearest",
    )

    # ── VAL / TEST : sans augmentation ────────────────────
    eval_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    val_data = eval_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_data = eval_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=1,           # batch=1 pour analyse image par image
        class_mode="binary",
        shuffle=False,
    )

    print(f"\n✅ Train : {train_data.samples} images")
    print(f"✅ Val   : {val_data.samples} images")
    print(f"✅ Test  : {test_data.samples} images")

    return train_data, val_data, test_data


def preprocess_single_image(image_path):
    """
    Prétraite une seule image pour la prédiction en inférence.
    Utilisé par Streamlit.
    
    Args: image_path (str) — chemin vers l'image
    Returns: np.array shape (1, 224, 224, 3)
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMG_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = arr / 255.0                 # normalisation
    arr = np.expand_dims(arr, axis=0) # (224,224,3) → (1,224,224,3)
    return arr
