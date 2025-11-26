# src/train.py
import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    PROJECT_ROOT,
)
from src.model_utils import (
    create_dataloaders,
    create_model,
    save_model,
    save_class_mapping,
)


def train_model():
    image_datasets, dataloaders = create_dataloaders()

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print("[INFO] Classes:", class_names)

    # Save class->idx mapping for later API use
    save_class_mapping(image_datasets["train"].class_to_idx)

    model = create_model(num_classes)
    model = model.to(DEVICE)

    # Only params that require grad (the new head)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params_to_update,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = model.state_dict()

    # ---------- NEW: history dict (does NOT affect training) ----------
    metrics_history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    # ------------------------------------------------------------------

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        epoch_results = {}  # temporary storage for this epoch

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            start = time.time()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total += batch_size

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            elapsed = time.time() - start

            print(f"{phase}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  Time: {elapsed:.1f}s")

            # ---- NEW: store results in epoch_results (still no behaviour change) ----
            epoch_results[phase] = {"loss": epoch_loss, "acc": epoch_acc}
            # -------------------------------------------------------------------------

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_state = model.state_dict()

        print(f"[INFO] Best val acc so far: {best_acc:.4f}")

        # ---- NEW: push this epoch into history ----
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(epoch_results["train"]["loss"])
        metrics_history["train_acc"].append(epoch_results["train"]["acc"])
        metrics_history["val_loss"].append(epoch_results["val"]["loss"])
        metrics_history["val_acc"].append(epoch_results["val"]["acc"])
        # -------------------------------------------

    model.load_state_dict(best_state)
    save_model(model)

    # ---- NEW: save history to artifacts/metrics/training_history.json ----
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(metrics_history, f, indent=4)
    print(f"[INFO] Training history saved to {history_path}")
    # ----------------------------------------------------------------------


if __name__ == "__main__":
    train_model()
