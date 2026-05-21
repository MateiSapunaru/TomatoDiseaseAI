import time
import json
from copy import deepcopy

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src import config
from src.model_utils import (
    create_dataloaders,
    create_model,
    save_model,
    save_class_mapping,
    compute_class_weights,
    get_trainable_parameters,
    unfreeze_layer4,
)



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


def train_phase(
    model,
    phase_name,
    dataloaders,
    criterion,
    learning_rate,
    num_epochs,
    best_state,
    best_val_f1,
):
    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    epochs_without_improvement = 0
    history = []

    print(f"\n[INFO] Starting phase: {phase_name}")
    print("-" * 40)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
        )

        scheduler.step(val_f1)

        elapsed = time.time() - start_time

        print(
            f"{phase_name} | "
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        history.append({
            "phase": phase_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"[INFO] New best validation macro F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    return best_state, best_val_f1, history


def train_model():
    image_datasets, dataloaders = create_dataloaders()

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print("[INFO] Classes:")
    for class_name in class_names:
        print(f" - {class_name}")

    save_class_mapping(image_datasets["train"].class_to_idx)

    model = create_model(num_classes)
    model = model.to(config.DEVICE)

    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(image_datasets["train"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("[INFO] Using class weights.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[INFO] Using standard CrossEntropyLoss.")

    best_state = deepcopy(model.state_dict())
    best_val_f1 = 0.0
    full_history = []

    best_state, best_val_f1, head_history = train_phase(
        model=model,
        phase_name="head",
        dataloaders=dataloaders,
        criterion=criterion,
        learning_rate=config.HEAD_LEARNING_RATE,
        num_epochs=config.HEAD_EPOCHS,
        best_state=best_state,
        best_val_f1=best_val_f1,
    )

    full_history.extend(head_history)

    print("\n[INFO] Unfreezing layer4 for fine-tuning.")
    unfreeze_layer4(model)

    best_state, best_val_f1, fine_tune_history = train_phase(
        model=model,
        phase_name="fine_tune_layer4",
        dataloaders=dataloaders,
        criterion=criterion,
        learning_rate=config.FINE_TUNE_LEARNING_RATE,
        num_epochs=config.FINE_TUNE_EPOCHS,
        best_state=best_state,
        best_val_f1=best_val_f1,
    )

    full_history.extend(fine_tune_history)

    model.load_state_dict(best_state)
    save_model(model)

    history_path = config.METRICS_DIR / "training_history.json"

    with open(history_path, "w") as file:
        json.dump(full_history, file, indent=4)

    print(f"[INFO] Training history saved to {history_path}")
    print(f"[INFO] Best validation macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    print(f"[INFO] Using device: {config.DEVICE}")
    train_model()