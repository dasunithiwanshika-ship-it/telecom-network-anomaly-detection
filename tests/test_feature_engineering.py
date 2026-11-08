"""
Unit tests for feature engineering logic and extraction.
"""

import numpy as np
import pytest
from src.data_loader import NUMERIC_KPIS, FEATURE_COLUMNS
from src.feature_engineering import (
    extract_sample_features,
    extract_batch_features,
    create_synthetic_demo_features,
)


def test_extract_sample_features():
    """Verify feature extraction from a synthetic 128-step multi-KPI dictionary."""
    synthetic_sample = {
        "KPIs": {kpi: np.linspace(10.0, 20.0, 128) for kpi in NUMERIC_KPIS}
    }

    feats = extract_sample_features(synthetic_sample)
    assert len(feats) == 64

    for kpi in NUMERIC_KPIS:
        assert np.isclose(feats[f"{kpi}_mean"], 15.0)
        assert np.isclose(feats[f"{kpi}_min"], 10.0)
        assert np.isclose(feats[f"{kpi}_max"], 20.0)
        assert feats[f"{kpi}_std"] > 0.0


def test_missing_kpi_raises_error():
    """Ensure KeyError is raised when a required KPI channel is missing."""
    incomplete_sample = {
        "KPIs": {"RSRP": np.zeros(128)} # Missing 15 other KPIs
    }
    with pytest.raises(KeyError):
        extract_sample_features(incomplete_sample)


def test_synthetic_demo_features():
    """Verify synthetic demonstration generator produces valid 64-feature DataFrame."""
    n = 7
    df = create_synthetic_demo_features(n_samples=n, random_state=42)
    assert df.shape == (n, 64)
    assert list(df.columns) == FEATURE_COLUMNS
    assert not df.isna().any().any()
