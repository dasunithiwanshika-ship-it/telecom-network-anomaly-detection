"""
Preprocessing and feature scaling pipeline for telecom KPI anomaly detection models.
Ensures identical scaling parameters are applied to test sets and live inference inputs.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .data_loader import FEATURE_COLUMNS


def validate_feature_schema(
    df: pd.DataFrame, expected_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that an input DataFrame strictly contains the required 64 features.

    Args:
        df: Input DataFrame to validate.
        expected_columns: Expected list of feature column names (defaults to FEATURE_COLUMNS).

    Returns:
        (is_valid, missing_columns)
    """
    if expected_columns is None:
        expected_columns = FEATURE_COLUMNS

    missing = [c for c in expected_columns if c not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, infinities, and data types for input feature matrices.

    Args:
        df: Input feature DataFrame.

    Returns:
        Cleaned DataFrame with numeric types and imputed values where necessary.
    """
    cleaned = df.copy()
    # Replace infinite values with NaN then fill with median
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    if cleaned.isna().any().any():
        cleaned = cleaned.fillna(cleaned.median())
    return cleaned.astype(float)


class TelecomPreprocessor:
    """
    Preprocessor wrapper maintaining StandardScaler state for models requiring normalization
    (e.g., Logistic Regression and Local Outlier Factor).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "TelecomPreprocessor":
        is_valid, missing = validate_feature_schema(X)
        if not is_valid:
            raise ValueError(f"Cannot fit preprocessor. Missing columns: {missing[:5]}...")
        X_clean = clean_feature_data(X[FEATURE_COLUMNS])
        self.scaler.fit(X_clean)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TelecomPreprocessor must be fitted before calling transform().")
        is_valid, missing = validate_feature_schema(X)
        if not is_valid:
            raise ValueError(f"Cannot transform. Missing columns: {missing[:5]}...")
        X_clean = clean_feature_data(X[FEATURE_COLUMNS])
        return self.scaler.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
