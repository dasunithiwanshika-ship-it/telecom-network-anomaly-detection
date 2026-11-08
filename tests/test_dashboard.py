"""
Unit tests for Streamlit application structure, components, and asset presence.
"""

import os
from pathlib import Path
import py_compile
import pandas as pd
import pytest

APP_DIR = Path("app")
PAGES_DIR = APP_DIR / "pages"
COMPONENTS_DIR = APP_DIR / "components"


def test_app_syntax():
    """Verify that app/streamlit_app.py compiles without syntax errors."""
    app_path = APP_DIR / "streamlit_app.py"
    assert app_path.exists(), "app/streamlit_app.py must exist"
    py_compile.compile(str(app_path), doraise=True)


def test_pages_and_components_syntax():
    """Verify that all page and component modules compile cleanly."""
    for py_file in PAGES_DIR.glob("*.py"):
        py_compile.compile(str(py_file), doraise=True)

    for py_file in COMPONENTS_DIR.glob("*.py"):
        py_compile.compile(str(py_file), doraise=True)


def test_sample_csv_exists_and_valid():
    """Verify that app/sample_telecom_kpi_features.csv exists and has 64 columns."""
    sample_csv = APP_DIR / "sample_telecom_kpi_features.csv"
    assert sample_csv.exists(), "sample CSV must exist in app/"

    df = pd.read_csv(sample_csv)
    assert df.shape[1] == 64
    assert df.shape[0] >= 5


def test_benchmark_results_exist():
    """Verify that results tables required by the dashboard are present and populated."""
    comp_file = Path("results") / "unified_4model_comparison_table.csv"
    assert comp_file.exists()
    df_comp = pd.read_csv(comp_file)
    assert len(df_comp) == 4
    assert set(df_comp["Model Name"]).issubset({
        "Logistic Regression",
        "Random Forest",
        "Isolation Forest",
        "Local Outlier Factor",
    })


def test_normal_baseline_stats_exist():
    """Verify that precomputed normal baseline stats artifact exists."""
    stats_file = Path("models") / "normal_baseline_stats.joblib"
    assert stats_file.exists()
