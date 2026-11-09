"""
Telecom AI Intelligence Platform
Clean, simple, and intuitive interface for cellular KPI anomaly detection and investigation.
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

from app.pages import (
    overview,
    detection,
    investigation,
    kpi_monitoring,
    research_centre,
)

st.set_page_config(
    page_title="Telecom AI Monitor",
    page_icon="📡",
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
        st.error(f"Error loading AI models: {e}")
        return

    selected_page = render_sidebar(pipeline)

    if selected_page == "🏠 Overview":
        overview.render(pipeline)
    elif selected_page == "⚡ Run Detection":
        detection.render(pipeline)
    elif selected_page == "🔍 Inspect Alert":
        investigation.render(pipeline)
    elif selected_page == "📊 KPIs & Reports":
        kpi_monitoring.render(pipeline)
    elif selected_page == "📚 Research Lab":
        research_centre.render()
    else:
        overview.render(pipeline)


if __name__ == "__main__":
    main()
