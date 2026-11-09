"""
Feature engineering module for telecom KPI time-series data.
Transforms raw 128-timestep multidimensional KPI arrays into 64 statistical moments.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from .data_loader import NUMERIC_KPIS, FEATURE_COLUMNS


def extract_sample_features(sample: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract 64 statistical summary moments from a single TelecomTS sample trace.

    Args:
        sample: Dictionary containing 'KPIs' dict with 16 numerical 128-step arrays.

    Returns:
        Dictionary mapping '{kpi}_{moment}' to float values.
    """
    kpis = sample.get("KPIs", sample)
    features: Dict[str, float] = {}

    for kpi in NUMERIC_KPIS:
        if kpi not in kpis:
            raise KeyError(f"Missing required numerical KPI '{kpi}' in sample.")
        values = np.asarray(kpis[kpi], dtype=float)
        if len(values) == 0:
            raise ValueError(f"KPI array for '{kpi}' is empty.")

        features[f"{kpi}_mean"] = float(values.mean())
        features[f"{kpi}_std"] = float(values.std())
        features[f"{kpi}_min"] = float(values.min())
        features[f"{kpi}_max"] = float(values.max())

    return features


def extract_batch_features(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract 64 statistical moments from a batch of TelecomTS sample traces.

    Args:
        samples: List of trace sample dictionaries.

    Returns:
        pd.DataFrame of shape (N, 64) with columns sorted according to FEATURE_COLUMNS.
    """
    rows = [extract_sample_features(s) for s in samples]
    df = pd.DataFrame(rows)
    return df[FEATURE_COLUMNS]


def create_synthetic_demo_features(n_samples: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic sample features strictly for interface demonstration and pipeline validation.
    
    DISCLAIMER: These synthetic rows are mathematically simulated demonstrations matching
    the 64 TelecomTS feature schema; they do NOT represent live SLTMobitel or measured operational traces.
    """
    rng = np.random.RandomState(random_state)
    data = {}

    for col in FEATURE_COLUMNS:
        if "RSRP" in col:
            base = rng.uniform(-120.0, -80.0, size=n_samples)
        elif "BLER" in col:
            base = rng.uniform(0.0001, 0.15, size=n_samples)
        elif "MCS" in col:
            base = rng.uniform(5.0, 28.0, size=n_samples)
        elif "SNR" in col:
            base = rng.uniform(5.0, 30.0, size=n_samples)
        elif "Bytes" in col:
            base = rng.uniform(1000.0, 150000.0, size=n_samples)
        elif "Buffer" in col:
            base = rng.uniform(50.0, 800.0, size=n_samples)
        elif "Utilization" in col:
            base = rng.uniform(5.0, 95.0, size=n_samples)
        elif "NumberOfPackets" in col:
            base = rng.uniform(10.0, 500.0, size=n_samples)
        else:
            base = rng.uniform(1.0, 100.0, size=n_samples)
        data[col] = base

    return pd.DataFrame(data, columns=FEATURE_COLUMNS)
