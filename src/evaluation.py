"""
evaluation.py
===============
Shared evaluation utilities used identically for BOTH the classical and
quantum models, ensuring a fair, apples-to-apples comparison.

Computes: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
Also provides a helper to persist metrics to JSON and to build a combined
comparison table (CSV) across models -- directly usable as a results table
in a research paper.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import METRICS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> dict:
    """Compute the full standard metric suite for a binary classifier."""
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def print_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    print(f"\n===== Classification Report: {model_name} =====")
    print(classification_report(y_true, y_pred, target_names=["normal", "attack"], zero_division=0))


def save_metrics(metrics: dict, filename: str):
    path = os.path.join(METRICS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[evaluation] Saved metrics to {path}")
    return path


def build_comparison_table(metrics_list: list, filename: str = "comparison_table.csv") -> pd.DataFrame:
    """Build a tidy comparison DataFrame across multiple models and save as CSV."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1_score"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(METRICS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[evaluation] Saved comparison table to {path}")
    print(df.to_string(index=False))
    return df
