# ============================================================
#  PneumoScan AI — Configuration Centrale
#  Modifie uniquement ce fichier pour adapter le projet
# ============================================================

import os

# ── Chemins ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
TRAIN_DIR      = os.path.join(DATA_DIR, "chest_xray", "train")
VAL_DIR        = os.path.join(DATA_DIR, "chest_xray", "val")
TEST_DIR       = os.path.join(DATA_DIR, "chest_xray", "test")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Modèle ─────────────────────────────────────────────────
MODEL_NAME     = "EfficientNetB0"
IMG_SIZE       = (224, 224)
CHANNELS       = 3
INPUT_SHAPE    = (*IMG_SIZE, CHANNELS)
CLASSES        = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES    = len(CLASSES)

# ── Entraînement ───────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS_FROZEN  = 10      # couches EfficientNet gelées
EPOCHS_FINE    = 10      # fine-tuning dégelé
LEARNING_RATE  = 1e-4
FINE_LR        = 1e-5
SEED           = 42

# ── Sauvegarde ─────────────────────────────────────────────
MODEL_PATH     = os.path.join(MODEL_DIR, "pneumoscan_best.h5")
HISTORY_PATH   = os.path.join(RESULTS_DIR, "training_history.json")
