import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

from src import config
from src.model_utils import create_dataloaders, create_model, load_class_mapping


def load_model(model_type: str):
    idx_to_class = load_class_mapping()
    num_classes = len(idx_to_class)

    model = create_model(num_classes)

    if model_type == "base":
        model_path = config.MODEL_PATH
    elif model_type == "adapted":
        model_path = config.ADAPTATION_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    model.eval()

    return model, idx_to_class


def collect_predictions(split: str, model_type: str):
    image_datasets, dataloaders = create_dataloaders()

    if split not in dataloaders:
        available_splits = ", ".join(dataloaders.keys())
        raise ValueError(
            f"Split '{split}' was not found. Available splits: {available_splits}"
        )

    dataloader = dataloaders[split]

    model, idx_to_class = load_model(model_type)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

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


def save_confusion_matrix(y_true, y_pred, class_names, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
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

    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def save_f1_plot(f1_per_class, class_names, output_dir: Path):
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, f1_per_class)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")

    fig.tight_layout()

    path = output_dir / "f1_per_class.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def save_precision_recall_plots(precision, recall, class_names, output_dir: Path):
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, precision)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision")

    fig.tight_layout()

    precision_path = output_dir / "precision_per_class.png"
    plt.savefig(precision_path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {precision_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, recall)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")

    fig.tight_layout()

    recall_path = output_dir / "recall_per_class.png"
    plt.savefig(recall_path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {recall_path}")


def save_roc_curves(y_true, y_proba, class_names, output_dir: Path):
    n_classes = len(class_names)

    y_true_onehot = np.zeros((y_true.shape[0], n_classes))
    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC={roc_auc:.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(fontsize="small", loc="lower right")

    path = output_dir / "roc_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def save_metrics_summary(acc, macro_f1, weighted_f1, output_dir: Path):
    path = output_dir / "metrics_summary.txt"

    with open(path, "w") as file:
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Macro F1 score: {macro_f1:.4f}\n")
        file.write(f"Weighted F1 score: {weighted_f1:.4f}\n")

    print(f"[SAVED] {path}")


def save_classification_report(y_true, y_pred, class_names, output_dir: Path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    path = output_dir / "classification_report.txt"

    with open(path, "w") as file:
        file.write(report)

    print(f"[SAVED] {path}")


def evaluate(split: str, model_type: str):
    output_dir = config.METRICS_DIR / split / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating split: {split}")
    print(f"[INFO] Model type: {model_type}")
    print(f"[INFO] Device: {config.DEVICE}")
    print(f"[INFO] Output directory: {output_dir}")

    y_true, y_pred, y_proba, class_names = collect_predictions(split, model_type)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    print("\nEvaluation results")
    print("-" * 30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    save_metrics_summary(acc, macro_f1, weighted_f1, output_dir)
    save_classification_report(y_true, y_pred, class_names, output_dir)

    save_confusion_matrix(y_true, y_pred, class_names, output_dir)
    save_f1_plot(f1_per_class, class_names, output_dir)
    save_precision_recall_plots(precision, recall, class_names, output_dir)
    save_roc_curves(y_true, y_proba, class_names, output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "val",
            "test",
            "real_world",
            "real_world_train",
            "real_world_test",
        ],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "adapted"],
        help="Model checkpoint to evaluate.",
    )

    args = parser.parse_args()
    evaluate(args.split, args.model)


if __name__ == "__main__":
    main()