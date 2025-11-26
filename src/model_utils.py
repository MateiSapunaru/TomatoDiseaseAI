import json
from typing import Dict, Tuple

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src import config



def get_transforms(train: bool = True):
    """
    Same transforms used for training and inference.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])



def create_dataloaders() -> Tuple[Dict, Dict]:
    """
    Create ImageFolder datasets and dataloaders for train/valid.
    THIS is the function train.py tries to import.
    """

    train_dataset = datasets.ImageFolder(
        config.TRAIN_DIR,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.ImageFolder(
        config.VAL_DIR,
        transform=get_transforms(train=False),
    )

    image_datasets = {"train": train_dataset, "val": val_dataset}

    dataloaders = {
        phase: DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=config.NUM_WORKERS,
        )
        for phase, ds in image_datasets.items()
    }

    return image_datasets, dataloaders



def create_model(num_classes: int) -> nn.Module:
    """
    ResNet18 with frozen backbone and new head.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model



def save_model(model: nn.Module, path=config.MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def save_class_mapping(class_to_idx: Dict[str, int], path=config.CLASS_MAP_PATH):
    with open(path, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"[INFO] Class mapping saved to {path}")


def load_class_mapping(path=config.CLASS_MAP_PATH) -> Dict[int, str]:
    with open(path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return idx_to_class


def load_trained_model(path=config.MODEL_PATH) -> Tuple[nn.Module, Dict[int, str]]:
    """
    Used by the API to load model + mapping.
    """
    idx_to_class = load_class_mapping()
    num_classes = len(idx_to_class)

    model = create_model(num_classes)
    state_dict = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()

    return model, idx_to_class



@torch.no_grad()
def predict_pil_image(model, image: Image.Image, idx_to_class):
    """
    Predict class of a PIL image.
    Returns (class_name, confidence).
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    prob, pred_idx = torch.max(probs, 1)

    pred_idx = pred_idx.item()
    class_name = idx_to_class[pred_idx]
    return class_name, float(prob.item())
