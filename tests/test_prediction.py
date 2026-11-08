"""
Unit tests for the model prediction pipeline and error handling.
"""

import os
import pytest
import pandas as pd
from src.prediction import AnomalyDetectorPipeline
from src.feature_engineering import create_synthetic_demo_features


@pytest.fixture(scope="module")
def pipeline():
    return AnomalyDetectorPipeline(models_dir="models")


def test_pipeline_loaded_models(pipeline):
    """Verify that all 4 models are loaded in the pipeline."""
    for model_id in ["random_forest", "logistic_regression", "isolation_forest", "local_outlier_factor"]:
        assert pipeline.is_model_available(model_id), f"Model {model_id} should be available."


def test_prediction_output_shapes(pipeline):
    """Test prediction output structure across all 4 models."""
    n_samples = 6
    demo_df = create_synthetic_demo_features(n_samples=n_samples, random_state=42)

    for model_id in pipeline.AVAILABLE_MODELS:
        res = pipeline.predict(demo_df, model_id=model_id)
        assert len(res) == n_samples
        assert "anomaly_prediction" in res.columns
        assert "anomaly_status" in res.columns
        assert "anomaly_score_or_prob" in res.columns
        assert set(res["anomaly_prediction"].unique()).issubset({0, 1})
        assert set(res["anomaly_status"].unique()).issubset({"Normal", "Anomaly"})


def test_missing_column_error(pipeline):
    """Test that missing required columns raises ValueError."""
    bad_df = pd.DataFrame({"dummy_column": [1, 2, 3]})
    with pytest.raises(ValueError) as excinfo:
        pipeline.predict(bad_df, model_id="random_forest")
    assert "Missing 64 required KPI feature columns" in str(excinfo.value)


def test_empty_dataframe_error(pipeline):
    """Test that empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError) as excinfo:
        pipeline.predict(empty_df, model_id="random_forest")
    assert "empty" in str(excinfo.value).lower()
