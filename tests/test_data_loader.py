"""
Unit tests for data loading and feature columns integrity.
"""

import os
import pytest
import pandas as pd
from src.data_loader import (
    NUMERIC_KPIS,
    FEATURE_COLUMNS,
    load_processed_features,
    get_stratified_split,
)


def test_kpi_definitions():
    """Verify that 16 numerical KPIs are defined producing exactly 64 feature columns."""
    assert len(NUMERIC_KPIS) == 16
    assert len(FEATURE_COLUMNS) == 64
    for kpi in NUMERIC_KPIS:
        for stat in ["mean", "std", "min", "max"]:
            assert f"{kpi}_{stat}" in FEATURE_COLUMNS


def test_load_processed_features():
    """Verify that preprocessed features file loads correctly with expected dimensions and class distribution."""
    pickle_path = "data/processed/telecom_features_64.pkl"
    if not os.path.exists(pickle_path):
        pytest.skip(f"Pickle file not found at {pickle_path}")

    X, y = load_processed_features(pickle_path)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (32000, 64)
    assert len(y) == 32000

    # Target distribution verification
    val_counts = y.value_counts().to_dict()
    assert val_counts[0] == 30765  # Normal
    assert val_counts[1] == 1235   # Anomalies
    assert round(y.mean(), 4) == 0.0386


def test_stratified_split():
    """Verify deterministic stratified partition split properties."""
    pickle_path = "data/processed/telecom_features_64.pkl"
    if not os.path.exists(pickle_path):
        pytest.skip("Dataset file not found")

    X, y = load_processed_features(pickle_path)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y, test_size=0.20, random_state=42)

    assert len(X_train) == 25600
    assert len(X_test) == 6400
    assert sum(y_train == 1) == 988
    assert sum(y_test == 1) == 247
    assert sum(y_test == 0) == 6153
