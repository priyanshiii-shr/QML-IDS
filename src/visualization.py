"""
visualization.py
==================
Shared plotting utilities: confusion matrix heatmaps, ROC curves, and a
side-by-side bar chart comparing classical vs quantum metrics. All plots
are saved as PNG files into results/figures/ so they can be dropped
directly into a paper or presentation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from src.config import FIGURES_DIR


def plot_confusion_matrix(cm: np.ndarray, model_name: str, filename: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["normal", "attack"], yticklabels=["normal", "attack"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualization] Saved confusion matrix plot to {path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, filename: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualization] Saved ROC curve plot to {path}")


def plot_combined_roc_curves(roc_data: list, filename: str = "combined_roc_curve.png"):
    """
    roc_data: list of tuples (y_true, y_proba, model_name)
    Plots multiple ROC curves on a single chart for direct comparison.
    """
    plt.figure(figsize=(6, 5))
    for y_true, y_proba, model_name in roc_data:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: Classical vs Quantum")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualization] Saved combined ROC curve plot to {path}")


def plot_metric_comparison_bar(comparison_df, filename: str = "metric_comparison_bar.png"):
    """Grouped bar chart comparing Accuracy/Precision/Recall/F1/ROC-AUC across models."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = comparison_df["Model"].tolist()

    x = np.arange(len(metric_cols))
    width = 0.8 / len(models)

    plt.figure(figsize=(9, 5))
    for i, model in enumerate(models):
        values = comparison_df.loc[comparison_df["Model"] == model, metric_cols].values.flatten()
        plt.bar(x + i * width, values, width, label=model)

    plt.xticks(x + width * (len(models) - 1) / 2, metric_cols)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Classical vs Quantum Model — Metric Comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualization] Saved metric comparison bar chart to {path}")


def plot_training_loss(history: list, filename: str = "quantum_training_loss.png"):
    """Plot the quantum model's training loss curve across epochs."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Quantum Model Training Loss")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualization] Saved training loss plot to {path}")
