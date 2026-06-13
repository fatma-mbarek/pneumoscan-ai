# ============================================================
#  PneumoScan AI — Architecture du Modèle
#  EfficientNetB0 + Transfer Learning + Fine-tuning
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from src.config import *


def build_model(freeze_base=True):
    """
    Construit le modèle PneumoScan basé sur EfficientNetB0.
    
    Phase 1 (freeze_base=True)  → entraîne uniquement la tête de classification
    Phase 2 (freeze_base=False) → fine-tune les dernières couches d'EfficientNet
    
    Architecture :
        EfficientNetB0 (pré-entraîné ImageNet)
            ↓
        GlobalAveragePooling2D
            ↓
        BatchNormalization
            ↓
        Dense(256, ReLU) + Dropout(0.5)
            ↓
        Dense(128, ReLU) + Dropout(0.3)
            ↓
        Dense(1, Sigmoid)  → probabilité PNEUMONIA
    """

    # ── Base : EfficientNetB0 pré-entraîné ─────────────────
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,          # on enlève la tête ImageNet
        input_shape=INPUT_SHAPE,
    )
    base.trainable = not freeze_base  # geler ou libérer

    if not freeze_base:
        # Fine-tuning : geler tout sauf les 30 dernières couches
        for layer in base.layers[:-30]:
            layer.trainable = False
        print(f"🔓 Fine-tuning : {sum(l.trainable for l in base.layers)} couches libérées")

    # ── Tête de classification médicale ────────────────────
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=base.input, outputs=output, name="PneumoScan_AI")

    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile avec métriques médicales importantes.
    AUC est cruciale en médecine (plus fiable que l'accuracy seule).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(model_path=MODEL_PATH):
    """
    Callbacks pour un entraînement robuste :
    - ModelCheckpoint : sauvegarde le meilleur modèle (AUC max)
    - EarlyStopping   : arrête si pas d'amélioration pendant 7 epochs
    - ReduceLROnPlateau : divise le LR par 2 si plateau (patience=3)
    """
    return [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def load_trained_model(model_path=MODEL_PATH):
    """Charge le modèle sauvegardé pour inférence ou évaluation."""
    print(f"📂 Chargement du modèle : {model_path}")
    return tf.keras.models.load_model(model_path)
