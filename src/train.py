# ============================================================
#  PneumoScan AI — Script d'Entraînement Complet
#  Lance ce script sur Google Colab (GPU recommandé)
#
#  Usage : python src/train.py
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from src.config import *
from src.preprocessing import build_generators, get_class_weights
from src.model import build_model, compile_model, get_callbacks, load_trained_model


# ════════════════════════════════════════════════════════════
#  1. CHARGER LES DONNÉES
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("  🫁  PneumoScan AI — Démarrage de l'entraînement")
print("=" * 55)

train_data, val_data, test_data = build_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)
class_weights = get_class_weights(TRAIN_DIR)


# ════════════════════════════════════════════════════════════
#  2. PHASE 1 — Entraînement tête de classification (base gelée)
# ════════════════════════════════════════════════════════════

print("\n📌 PHASE 1 : Entraînement tête (base EfficientNet gelée)")
print("-" * 55)

model = build_model(freeze_base=True)
model = compile_model(model, learning_rate=LEARNING_RATE)
model.summary()

history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weights,
    callbacks=get_callbacks(),
    verbose=1,
)


# ════════════════════════════════════════════════════════════
#  3. PHASE 2 — Fine-tuning (dernières couches libérées)
# ════════════════════════════════════════════════════════════

print("\n🔓 PHASE 2 : Fine-tuning (30 dernières couches libérées)")
print("-" * 55)

# Recharger le meilleur modèle de la phase 1
model = load_trained_model()

# Libérer les 30 dernières couches de la base
for layer in model.layers:
    if hasattr(layer, 'layers'):           # c'est EfficientNet
        for sub in layer.layers[-30:]:
            sub.trainable = True

model = compile_model(model, learning_rate=FINE_LR)

history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=get_callbacks(),
    verbose=1,
)

print("\n✅ Entraînement terminé. Meilleur modèle sauvegardé.")


# ════════════════════════════════════════════════════════════
#  4. ÉVALUATION MÉDICALE COMPLÈTE
# ════════════════════════════════════════════════════════════

print("\n📊 ÉVALUATION SUR LE JEU DE TEST")
print("-" * 55)

model = load_trained_model()

# Prédictions
y_pred_proba = model.predict(test_data, verbose=1).ravel()
y_pred       = (y_pred_proba >= 0.5).astype(int)
y_true       = test_data.classes

# ── Métriques ─────────────────────────────────────────────
print("\n📋 Rapport de classification :")
print(classification_report(y_true, y_pred, target_names=CLASSES))

auc = roc_auc_score(y_true, y_pred_proba)
print(f"🎯 AUC-ROC : {auc:.4f}")


# ── Matrice de confusion ───────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES
)
plt.title("Matrice de Confusion — PneumoScan AI", fontsize=13, fontweight="bold")
plt.ylabel("Vraie étiquette")
plt.xlabel("Prédiction")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("💾 Matrice sauvegardée.")


# ── Courbe ROC ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#00e5ff", lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs (Recall)")
plt.title("Courbe ROC — PneumoScan AI", fontsize=13, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
plt.show()
print("💾 Courbe ROC sauvegardée.")


# ── Sauvegarder l'historique ───────────────────────────────
full_history = {
    "phase1_accuracy":     history_phase1.history.get("accuracy", []),
    "phase1_val_accuracy": history_phase1.history.get("val_accuracy", []),
    "phase1_auc":          history_phase1.history.get("auc", []),
    "phase1_val_auc":      history_phase1.history.get("val_auc", []),
    "phase2_accuracy":     history_phase2.history.get("accuracy", []),
    "phase2_val_accuracy": history_phase2.history.get("val_accuracy", []),
    "phase2_auc":          history_phase2.history.get("auc", []),
    "phase2_val_auc":      history_phase2.history.get("val_auc", []),
}
with open(HISTORY_PATH, "w") as f:
    json.dump(full_history, f, indent=2)

print(f"\n✅ Historique sauvegardé : {HISTORY_PATH}")
print("🎉 Pipeline complet terminé !")
