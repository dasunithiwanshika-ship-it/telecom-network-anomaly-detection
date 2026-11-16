"""
NEXVANTA NIMS — Telecom Network Intelligence Platform.
Executive Operational Dashboard for Cellular Anomaly Detection,
Alert Triage, and Predictive Network Operations.
"""

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# Add project root to sys.path so src imports work reliably
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.prediction import AnomalyDetectorPipeline
from app.components.styles import apply_custom_theme
from app.components.sidebar import render_sidebar, NAV_OPTIONS

from app.pages import (
    welcome,
    overview,
    data_management,
    detection,
    kpi_monitoring,
    investigation,
    reports,
    research_centre,
)

st.set_page_config(
    page_title="NEXVANTA NIMS — Telecom Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()


@st.cache_resource
def get_pipeline():
    """Initializes and caches the prediction pipeline."""
    return AnomalyDetectorPipeline()


def main():
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Error initializing detection pipeline: {e}")
        return

    selected_page = render_sidebar(pipeline)

    if "01" in selected_page or "Portal" in selected_page or "Welcome" in selected_page:
        welcome.render(pipeline)
    elif "02" in selected_page or "Overview" in selected_page:
        overview.render(pipeline)
    elif "03" in selected_page or "Ingestion" in selected_page or "Data" in selected_page:
        data_management.render(pipeline)
    elif "04" in selected_page or "Detection" in selected_page:
        detection.render(pipeline)
    elif "05" in selected_page or "Alert" in selected_page:
        kpi_monitoring.render(pipeline)
    elif "06" in selected_page or "Investigation" in selected_page or "Export" in selected_page:
        investigation.render(pipeline)
    else:
        overview.render(pipeline)


if __name__ == "__main__":
    main()
