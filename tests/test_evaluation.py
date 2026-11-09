"""
Unit tests for evaluation metrics, ROC/PR curves, and error analysis.
"""

import numpy as np
import pytest
from src.evaluation import (
    evaluate_model_performance,
    compute_roc_pr_curves,
    compute_threshold_sweep,
    analyze_prediction_errors,
)


def test_evaluate_model_performance():
    """Verify evaluation metric calculations against synthetic ground truth."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.85, 0.4])

    metrics = evaluate_model_performance(y_true, y_pred, y_score, model_name="TestModel")
    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["roc_auc"] > 0.5


def test_analyze_prediction_errors():
    """Verify set-theoretic decomposition of errors."""
    y_true = np.array([0, 0, 1, 1])
    preds_dict = {
        "model_A": np.array([0, 1, 1, 0]),
        "model_B": np.array([0, 0, 1, 1]),
    }
    sample_indices = ["idx0", "idx1", "idx2", "idx3"]

    decomp = analyze_prediction_errors(y_true, preds_dict, sample_indices=sample_indices)
    assert decomp["model_A"]["tp_indices"] == ["idx2"]
    assert decomp["model_A"]["fn_indices"] == ["idx3"]
    assert decomp["model_A"]["fp_indices"] == ["idx1"]
    assert decomp["model_B"]["tp_indices"] == ["idx2", "idx3"]
    assert len(decomp["model_B"]["fn_indices"]) == 0
    assert len(decomp["model_B"]["fp_indices"]) == 0
