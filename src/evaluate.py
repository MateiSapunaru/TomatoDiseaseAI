import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

from src.config import DEVICE, PROJECT_ROOT
from src.model_utils import create_dataloaders, load_trained_model



METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)



def collect_predictions(split: str = "val"):
    """
    Run trained model on validation set and gather:
    - true labels
    - predicted labels
    - predicted probabilities
    - class names
    """
    image_datasets, dataloaders = create_dataloaders()
    dataloader = dataloaders[split]

    model, idx_to_class = load_trained_model()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        class_names,
    )



def save_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    """
    Save normalized confusion matrix to PNG.
    """
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = METRICS_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def save_f1_plot(f1_per_class, class_names, filename="f1_per_class.png"):
    """
    Save F1-per-class bar plot.
    """
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, f1_per_class)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")

    fig.tight_layout()

    path = METRICS_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def save_precision_recall_plots(precision, recall, class_names):
    """
    Save precision and recall bar charts as two PNGs.
    """
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, precision)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision")
    fig.tight_layout()
    path_p = METRICS_DIR / "precision_per_class.png"
    plt.savefig(path_p, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path_p}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, recall)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    fig.tight_layout()
    path_r = METRICS_DIR / "recall_per_class.png"
    plt.savefig(path_r, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path_r}")


def save_roc_curves(y_true, y_proba, class_names):
    """
    Save multi-class ROC curves (one-vs-rest) to PNG.
    """
    n_classes = len(class_names)

    y_true_onehot = np.zeros((y_true.shape[0], n_classes))
    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(fontsize="small", loc="lower right")

    path = METRICS_DIR / "roc_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def save_training_curves():
    """
    Load training_history.json and save loss/accuracy curves.
    """
    history_path = METRICS_DIR / "training_history.json"
    if not history_path.exists():
        print(f"[WARN] Training history not found at {history_path}. "
              f"Run training again to generate it.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    path_loss = METRICS_DIR / "training_loss_curves.png"
    plt.savefig(path_loss, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path_loss}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train Acc")
    ax.plot(epochs, val_acc, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    path_acc = METRICS_DIR / "training_accuracy_curves.png"
    plt.savefig(path_acc, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path_acc}")



def main():
    y_true, y_pred, y_proba, class_names = collect_predictions(split="val")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    clf_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )

    with open(METRICS_DIR / "classification_report.txt", "w") as f:
        f.write(clf_report)
    print(f"[SAVED] {METRICS_DIR / 'classification_report.txt'}")

    with open(METRICS_DIR / "metrics_summary.txt", "w") as f:
        f.write(f"Validation accuracy: {acc:.4f}\n")
        f.write(f"Macro F1 score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1 score: {weighted_f1:.4f}\n")
    print(f"[SAVED] {METRICS_DIR / 'metrics_summary.txt'}")

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names)

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names))
    )
    save_f1_plot(f1_per_class, class_names)
    save_precision_recall_plots(precision, recall, class_names)

    save_roc_curves(y_true, y_proba, class_names)

    save_training_curves()


if __name__ == "__main__":
    main()
