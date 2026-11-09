"""
Unit tests for data validation, cleaning, and preprocessing.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import FEATURE_COLUMNS
from src.preprocessing import validate_feature_schema, clean_feature_data, TelecomPreprocessor
from src.feature_engineering import create_synthetic_demo_features


def test_validate_feature_schema():
    """Test schema validation on valid, incomplete, and empty dataframes."""
    valid_df = create_synthetic_demo_features(n_samples=5)
    is_valid, missing = validate_feature_schema(valid_df)
    assert is_valid is True
    assert len(missing) == 0

    # Incomplete dataframe
    bad_df = valid_df.drop(columns=["RSRP_mean", "UL_BLER_max"])
    is_valid_bad, missing_bad = validate_feature_schema(bad_df)
    assert is_valid_bad is False
    assert set(missing_bad) == {"RSRP_mean", "UL_BLER_max"}


def test_clean_feature_data():
    """Test handling of NaN, Inf, and non-numeric string values."""
    df = create_synthetic_demo_features(n_samples=5)
    df.loc[0, "RSRP_mean"] = np.nan
    df.loc[1, "UL_BLER_mean"] = np.inf
    # Cast to object first to simulate raw CSV string ingestion
    df["UL_SNR_mean"] = df["UL_SNR_mean"].astype(object)
    df.loc[2, "UL_SNR_mean"] = "corrupted_str_val"

    cleaned = clean_feature_data(df)
    assert not cleaned.isna().any().any()
    assert not np.isinf(cleaned.values).any()
    assert isinstance(cleaned.loc[2, "UL_SNR_mean"], (float, np.floating))


def test_telecom_preprocessor():
    """Test TelecomPreprocessor fitting and transform."""
    df = create_synthetic_demo_features(n_samples=20)
    preprocessor = TelecomPreprocessor()
    assert preprocessor.is_fitted is False

    scaled = preprocessor.fit_transform(df)
    assert preprocessor.is_fitted is True
    assert scaled.shape == (20, 64)
    # Check that means are near 0
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-7)
