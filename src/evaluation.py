"""
Evaluation metrics, error analysis, and comparative benchmarking tools.
Provides standardized calculations of classification metrics, confusion matrices, and ROC/PR curves.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary anomaly detection.

    Args:
        y_true: Ground truth binary labels (0=Normal, 1=Anomaly).
        y_pred: Predicted discrete labels (0=Normal, 1=Anomaly).
        y_score: Continuous prediction scores or probabilities for ROC/PR calculation.
        model_name: Identifier for the model.

    Returns:
        Dictionary of computed metrics and confusion counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_score)) if y_score is not None else np.nan
    pr_auc = (
        float(average_precision_score(y_true, y_score))
        if y_score is not None
        else np.nan
    )

    return {
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_samples": len(y_true),
        "prevalence": float(np.mean(y_true)),
    }


def compute_roc_pr_curves(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute False Positive Rate, True Positive Rate, Precision, and Recall arrays.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, and F1 across a sequence of decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    records = []
    for tau in thresholds:
        preds = (y_score >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        records.append({
            "threshold": round(float(tau), 3),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return pd.DataFrame(records)


def analyze_prediction_errors(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    sample_indices: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Decompose misclassifications into set-theoretic categories across models.
    """
    if sample_indices is None:
        sample_indices = list(range(len(y_true)))

    true_anomaly_set = set([sample_indices[i] for i, y in enumerate(y_true) if y == 1])
    results = {}

    for name, preds in predictions_dict.items():
        detected = set([sample_indices[i] for i, p in enumerate(preds) if p == 1])
        tp_set = true_anomaly_set.intersection(detected)
        fn_set = true_anomaly_set.difference(detected)
        fp_set = detected.difference(true_anomaly_set)

        results[name] = {
            "tp_indices": sorted(list(tp_set)),
            "fn_indices": sorted(list(fn_set)),
            "fp_indices": sorted(list(fp_set)),
        }

    return results
