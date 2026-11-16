"""
Unit tests for Streamlit application structure, components, report generation, and asset presence.
"""

from pathlib import Path
import py_compile
import pandas as pd
import pytest

from app.components.metric_cards import get_feature_meta, TELECOM_KPI_METADATA
from app.components.sidebar import NAV_OPTIONS
from app.pages.reports import _build_markdown_report, _build_html_report
from src.feature_engineering import create_synthetic_demo_features
from src.prediction import AnomalyDetectorPipeline

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


def test_navigation_options_and_kpi_metadata():
    """Verify that all 6 end-to-end workflow navigation sections and 16 telecom KPI metadata entries are defined."""
    assert len(NAV_OPTIONS) == 6
    assert len(TELECOM_KPI_METADATA) == 16
    meta = get_feature_meta("UL_NPRB_min")
    assert meta["base_kpi"] == "UL_NPRB"
    assert meta["moment"] == "min"
    assert "PRB" in meta["friendly_name"]


def test_report_generation_pipeline():
    """Verify that the automated diagnostic report generator produces valid Markdown and HTML."""
    pipeline = AnomalyDetectorPipeline()
    demo_df = create_synthetic_demo_features(n_samples=10, random_state=42)
    results = pipeline.predict(demo_df, model_id="random_forest", threshold=0.30)
    results["Record ID"] = [f"TRACE-{i:04d}" for i in range(len(results))]
    results["Severity Level"] = "🔴 CRITICAL ALERT"
    results["Primary KPI Deviation"] = "UL_NPRB (-2.5σ)"

    md_text = _build_markdown_report(results, "Unit Test Batch", "Random Forest", 0.30)
    assert "EXECUTIVE TELEMETRY SUMMARY" in md_text
    assert "Unit Test Batch" in md_text

    html_text = _build_html_report(md_text, "Unit Test Batch", len(results), 2)
    assert "<!DOCTYPE html>" in html_text
    assert "Telecom AI Intelligence Platform" in html_text


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
