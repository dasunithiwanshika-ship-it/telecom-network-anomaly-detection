"""
Data loading and dataset validation utilities for telecom network anomaly detection.
Provides verified access to TelecomTS features and stratified data partitions.
"""

import os
import pickle
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_KPIS: List[str] = [
    "RSRP",
    "DL_BLER",
    "DL_MCS",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_SNR",
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "UL_NumberOfPackets",
    "DL_NumberOfPackets",
]

FEATURE_COLUMNS: List[str] = [
    f"{kpi}_{stat}"
    for kpi in NUMERIC_KPIS
    for stat in ["mean", "std", "min", "max"]
]


def load_processed_features(
    pickle_path: str = "data/processed/telecom_features_64.pkl"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load precomputed 64 KPI statistical summary features and ground-truth anomaly labels.

    Args:
        pickle_path: Path to the processed features pickle file.

    Returns:
        X (pd.DataFrame): 32,000 x 64 DataFrame of engineered numerical features.
        y (pd.Series): 32,000 binary anomaly indicators (0=Normal, 1=Anomaly).
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Processed features file not found at '{pickle_path}'. "
            "Please ensure feature engineering has been executed."
        )

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
    elif isinstance(data, dict):
        X = data["X"]
        y = data["y"]
    else:
        raise ValueError(f"Unexpected format in {pickle_path}: {type(data)}")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="anomaly")

    return X, y


def get_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate reproducible stratified train/test split matching Day 5-10 experiments.

    Args:
        X: Feature DataFrame.
        y: Target label Series.
        test_size: Proportion of dataset to include in test split (default: 0.20).
        random_state: Deterministic random seed (default: 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
