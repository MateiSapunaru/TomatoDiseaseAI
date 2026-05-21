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
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.02,
            ),

            transforms.RandomPerspective(
                distortion_scale=0.10,
                p=0.15,
            ),

            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ], p=0.15),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloaders() -> Tuple[Dict, Dict]:
    image_datasets = {
        "train": datasets.ImageFolder(
            config.TRAIN_DIR,
            transform=get_transforms(train=True),
        ),

        "val": datasets.ImageFolder(
            config.VAL_DIR,
            transform=get_transforms(train=False),
        ),
    }

    if config.TEST_DIR.exists():
        image_datasets["test"] = datasets.ImageFolder(
            config.TEST_DIR,
            transform=get_transforms(train=False),
        )

    if config.REAL_WORLD_DIR.exists():
        image_datasets["real_world"] = datasets.ImageFolder(
            config.REAL_WORLD_DIR,
            transform=get_transforms(train=False),
        )

    if config.REAL_WORLD_TRAIN_DIR.exists():
        image_datasets["real_world_train"] = datasets.ImageFolder(
            config.REAL_WORLD_TRAIN_DIR,
            transform=get_transforms(train=False),
        )

    if config.REAL_WORLD_TEST_DIR.exists():
        image_datasets["real_world_test"] = datasets.ImageFolder(
            config.REAL_WORLD_TEST_DIR,
            transform=get_transforms(train=False),
        )

    dataloaders = {
        phase: DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=config.NUM_WORKERS,
        )
        for phase, dataset in image_datasets.items()
    }

    return image_datasets, dataloaders


def create_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_layer4(model: nn.Module):
    for param in model.layer4.parameters():
        param.requires_grad = True


def get_trainable_parameters(model: nn.Module):
    return [
        param
        for param in model.parameters()
        if param.requires_grad
    ]


def compute_class_weights(train_dataset) -> torch.Tensor:
    class_counts = torch.zeros(len(train_dataset.classes))

    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = class_counts.sum()

    class_weights = (
        total_samples
        / (len(class_counts) * class_counts)
    )

    return class_weights.to(config.DEVICE)


def save_model(model: nn.Module, path=config.MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def save_class_mapping(
    class_to_idx: Dict[str, int],
    path=config.CLASS_MAP_PATH,
):
    with open(path, "w") as file:
        json.dump(class_to_idx, file, indent=4)

    print(f"[INFO] Class mapping saved to {path}")


def load_class_mapping(
    path=config.CLASS_MAP_PATH,
) -> Dict[int, str]:
    with open(path, "r") as file:
        class_to_idx = json.load(file)

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    return idx_to_class


def load_trained_model(
    path=config.MODEL_PATH,
):
    idx_to_class = load_class_mapping()
    num_classes = len(idx_to_class)

    model = create_model(num_classes)

    state_dict = torch.load(
        path,
        map_location=config.DEVICE,
    )

    model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    model.eval()

    return model, idx_to_class


@torch.no_grad()
def predict_pil_image(
    model,
    image: Image.Image,
    idx_to_class,
):
    transform = get_transforms(train=False)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(config.DEVICE)

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, pred_idx = torch.max(
        probabilities,
        dim=1,
    )

    pred_idx = pred_idx.item()

    class_name = idx_to_class[pred_idx]

    return (
        class_name,
        float(confidence.item()),
    )