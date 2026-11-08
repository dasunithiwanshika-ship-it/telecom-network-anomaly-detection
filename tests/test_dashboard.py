"""
Unit tests for Streamlit application structure and asset presence.
"""

import os
import py_compile
import pandas as pd
import pytest


def test_app_syntax():
    """Verify that app/streamlit_app.py compiles without syntax errors."""
    app_path = os.path.join("app", "streamlit_app.py")
    assert os.path.exists(app_path), "app/streamlit_app.py must exist"
    py_compile.compile(app_path, doraise=True)


def test_sample_csv_exists_and_valid():
    """Verify that app/sample_telecom_kpi_features.csv exists and has 64 columns."""
    sample_csv = os.path.join("app", "sample_telecom_kpi_features.csv")
    assert os.path.exists(sample_csv), "sample CSV must exist in app/"

    df = pd.read_csv(sample_csv)
    assert df.shape[1] == 64
    assert df.shape[0] >= 5


def test_benchmark_results_exist():
    """Verify that results tables required by the dashboard are present and populated."""
    comp_file = os.path.join("results", "unified_4model_comparison_table.csv")
    assert os.path.exists(comp_file)
    df_comp = pd.read_csv(comp_file)
    assert len(df_comp) == 4
    assert set(df_comp["Model Name"]).issubset({
        "Logistic Regression",
        "Random Forest",
        "Isolation Forest",
        "Local Outlier Factor",
    })
