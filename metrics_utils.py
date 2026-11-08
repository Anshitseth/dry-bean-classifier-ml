"""Shared metric computation used by both training and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba, labels=None) -> dict:
    """Macro-averaged metrics for a multi-class problem.

    `labels` lists every class id the model knows about (column order of
    `y_proba`), so AUC still works if an uploaded file misses some classes.
    """
    y_true = np.asarray(y_true)
    present = np.unique(y_true)
    if labels is None:
        labels = np.arange(y_proba.shape[1])

    auc = np.nan
    if len(present) > 1:
        # Keep only columns of classes present in y_true, then renormalise rows.
        idx = [i for i, c in enumerate(labels) if c in present]
        sub = y_proba[:, idx]
        sub = sub / sub.sum(axis=1, keepdims=True)
        auc = roc_auc_score(y_true, sub, multi_class="ovr", average="macro",
                            labels=present)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
