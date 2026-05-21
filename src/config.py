import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
REAL_WORLD_DIR = DATA_DIR / "real_world_unseen_data"
REAL_WORLD_TRAIN_DIR = DATA_DIR / "real_world_train"
REAL_WORLD_TEST_DIR = DATA_DIR / "real_world_test"


MODEL_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

ADAPTATION_MODEL_PATH = MODEL_DIR / "tomato_resnet18_adapted.pth"

ADAPTATION_EPOCHS = 8
ADAPTATION_LEARNING_RATE = 1e-5

MODEL_PATH = MODEL_DIR / "tomato_resnet18.pth"
CLASS_MAP_PATH = MODEL_DIR / "class_to_idx.json"

METRICS_DIR = MODEL_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 12

HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

USE_CLASS_WEIGHTS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")