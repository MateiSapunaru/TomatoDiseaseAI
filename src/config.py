import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "valid"

MODEL_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "tomato_resnet18.pth"
CLASS_MAP_PATH = MODEL_DIR / "class_to_idx.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
