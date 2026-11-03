"""Evaluation utilities for binary classification (IsCanceled)."""

from pathlib import Path
from typing import Optional, Union

import matplotlib
from sklearn.metrics import precision_recall_curve, roc_curve
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
) -> dict[str, float]:
    """Compute required Section 8 classification metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    output_path: Optional[Union[str, Path]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Generate and optionally save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    labels = ["Not Canceled (0)", "Canceled (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)
    return cm


def plot_roc_pr_curves(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    model_name: str,
    output_dir: Union[str, Path] = "outputs",
) -> None:
    """Generate and save ROC and PR curves for a given model."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve: {model_name}",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path / f"roc_{model_name}.png")
    plt.close(fig)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR (AP = {pr_auc_val:.4f})", color="green", lw=2)
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall Curve: {model_name}",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path / f"pr_{model_name}.png")
    plt.close(fig)

def format_metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert multiple model metric dictionaries into a comparison DataFrame."""
    df = pd.DataFrame.from_dict(results, orient="index")
    columns_order = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    available_cols = [c for c in columns_order if c in df.columns]
    return df[available_cols].round(4)