"""
Telecom AI Intelligence Platform (v2.0 Enterprise NOC Edition)
Designed for Telecom Network Engineers to monitor cellular KPIs, detect anomalies,
investigate root-cause KPI deviations, and export diagnostic reports without programming.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so src imports work reliably
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.prediction import AnomalyDetectorPipeline
from app.components.styles import apply_custom_theme
from app.components.sidebar import render_sidebar

# Import page modules
from app.pages import (
    welcome,
    overview,
    data_management,
    detection,
    investigation,
    kpi_monitoring,
    reports,
    research_centre,
    about,
)

# -------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Telecom AI Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark enterprise styling
apply_custom_theme()


@st.cache_resource
def get_pipeline():
    """Initializes and caches the prediction pipeline."""
    return AnomalyDetectorPipeline()


def main():
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Critical error initializing ML Pipeline: {e}")
        return

    # Render enterprise sidebar navigation
    selected_page = render_sidebar(pipeline)

    # Route to selected page module
    if selected_page == "🏠 Welcome & Quick Start":
        welcome.render(pipeline)
    elif selected_page == "🌐 Network Overview":
        overview.render(pipeline)
    elif selected_page == "📂 Data Upload & Validation":
        data_management.render(pipeline)
    elif selected_page == "⚡ AI Detection Workspace":
        detection.render(pipeline)
    elif selected_page == "🔬 Anomaly Investigation":
        investigation.render(pipeline)
    elif selected_page == "📊 KPI Subsystem Monitor":
        kpi_monitoring.render(pipeline)
    elif selected_page == "📑 Reports & Export":
        reports.render(pipeline)
    elif selected_page == "🧪 Research & Model Lab":
        research_centre.render()
    elif selected_page == "ℹ️ About & System Guide":
        about.render()
    else:
        welcome.render(pipeline)


if __name__ == "__main__":
    main()
