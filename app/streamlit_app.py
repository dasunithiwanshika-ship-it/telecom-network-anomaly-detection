"""
Telecom AI Intelligence Platform
Enterprise Hybrid AI/ML Application for Network Anomaly Detection & KPI Intelligence.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so src imports work reliably
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.prediction import AnomalyDetectorPipeline
from app.components.styles import apply_custom_styles
from app.components.sidebar import render_sidebar

# Import page modules
from app.pages import (
    overview,
    detection,
    kpi_monitoring,
    investigation,
    ai_insights,
    model_intelligence,
    research_centre,
    data_management,
    about
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
apply_custom_styles()

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
    if selected_page == "🌐 Network Overview":
        overview.render()
    elif selected_page == "⚡ Anomaly Detection":
        detection.render(pipeline)
    elif selected_page == "📊 KPI Monitoring":
        kpi_monitoring.render()
    elif selected_page == "🔬 Anomaly Investigation":
        investigation.render(pipeline)
    elif selected_page == "💡 AI Insights":
        ai_insights.render()
    elif selected_page == "🧠 Model Intelligence":
        model_intelligence.render()
    elif selected_page == "📚 Research Centre":
        research_centre.render()
    elif selected_page == "🗄️ Data Management":
        data_management.render()
    elif selected_page == "ℹ️ About & Documentation":
        about.render()
    else:
        overview.render()

if __name__ == "__main__":
    main()
