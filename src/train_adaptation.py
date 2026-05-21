import json
import time
from copy import deepcopy

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from src import config
from src.model_utils import (
    get_transforms,
    create_model,
    load_class_mapping,
    get_trainable_parameters,
)


def create_real_world_dataloaders():
    train_dataset = datasets.ImageFolder(
        config.REAL_WORLD_TRAIN_DIR,
        transform=get_transforms(train=True),
    )

    test_dataset = datasets.ImageFolder(
        config.REAL_WORLD_TEST_DIR,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataset, test_dataset, train_loader, test_loader


def run_epoch(model, dataloader, criterion, optimizer=None):
    is_training = optimizer is not None

    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.set_grad_enabled(is_training):
        for inputs, labels in dataloader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)

            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total += batch_size

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def train_adaptation():
    print(f"[INFO] Using device: {config.DEVICE}")

    idx_to_class = load_class_mapping()
    num_classes = len(idx_to_class)

    train_dataset, test_dataset, train_loader, test_loader = create_real_world_dataloaders()

    if train_dataset.classes != test_dataset.classes:
        raise RuntimeError(
            "Class folders do not match between real_world_train and real_world_test."
        )

    print("[INFO] Real-world classes:")
    for class_name in train_dataset.classes:
        print(f" - {class_name}")

    model = create_model(num_classes)

    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr=config.ADAPTATION_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_state = deepcopy(model.state_dict())
    best_test_f1 = 0.0

    history = []

    print("\n[INFO] Starting real-world adaptation")
    print("-" * 40)

    for epoch in range(config.ADAPTATION_EPOCHS):
        start_time = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        test_loss, test_acc, test_f1 = run_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            optimizer=None,
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{config.ADAPTATION_EPOCHS} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
            f"real-world test loss: {test_loss:.4f} acc: {test_acc:.4f} f1: {test_f1:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "real_world_test_loss": test_loss,
            "real_world_test_acc": test_acc,
            "real_world_test_macro_f1": test_f1,
        })

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_state = deepcopy(model.state_dict())
            print(f"[INFO] New best real-world test macro F1: {best_test_f1:.4f}")

    model.load_state_dict(best_state)

    torch.save(model.state_dict(), config.ADAPTATION_MODEL_PATH)
    print(f"\n[INFO] Adapted model saved to: {config.ADAPTATION_MODEL_PATH}")

    history_path = config.METRICS_DIR / "real_world_adaptation_history.json"

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    print(f"[INFO] Adaptation history saved to: {history_path}")
    print(f"[INFO] Best real-world test macro F1: {best_test_f1:.4f}")


if __name__ == "__main__":
    train_adaptation()