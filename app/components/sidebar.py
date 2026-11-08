"""
Enterprise NOC Sidebar Component for Telecom AI Intelligence Platform.
"""

from typing import Dict, Any, List
import streamlit as st


def render_sidebar(pipeline) -> str:
    """
    Renders the unified enterprise sidebar navigation and status monitors.

    Returns:
        The selected page title string.
    """
    with st.sidebar:
        # Platform Brand Header
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #14B8A6, #0D9488); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">
                        📡
                    </div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.2px;">TELECOM AI</div>
                        <div style="font-size: 0.72rem; color: #14B8A6; font-weight: 600; letter-spacing: 0.5px;">HYBRID INTELLIGENCE PLATFORM</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational Mode Badge
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 6px; padding: 8px 12px; margin-bottom: 18px;">
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">OPERATIONAL MODE</div>
                <div style="display: flex; align-items: center; gap: 6px; margin-top: 3px;">
                    <span style="height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.82rem; color: #F8FAFC; font-weight: 600;">Offline Research / Ingestion</span>
                </div>
                <div style="font-size: 0.7rem; color: #64748B; margin-top: 2px;">Dataset: AliMaatouk/TelecomTS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;'>PLATFORM NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        nav_options = [
            "🌐 Network Overview",
            "⚡ Anomaly Detection",
            "📊 KPI Monitoring",
            "🔬 Anomaly Investigation",
            "💡 AI Insights",
            "🧠 Model Intelligence",
            "📚 Research Centre",
            "🗄️ Data Management",
            "ℹ️ About & Documentation",
        ]

        selected_page = st.radio(
            label="Navigation Menu",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Model Status
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>AI MODEL ENGINE STATUS</div>",
            unsafe_allow_html=True,
        )

        model_labels = {
            "random_forest": "Random Forest (Supervised)",
            "logistic_regression": "Logistic Regression (Supervised)",
            "isolation_forest": "Isolation Forest (Unsupervised)",
            "local_outlier_factor": "Local Outlier Factor (Unsupervised)",
        }

        for mid, mname in model_labels.items():
            avail = pipeline.is_model_available(mid)
            dot_color = "#10B981" if avail else "#EF4444"
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; padding: 3px 0;">
                    <span style="color: #94A3B8;">{mname.split(' (')[0]}</span>
                    <span style="color: {dot_color}; font-weight: 600;">{'● Ready' if avail else '○ Offline'}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Current Workspace"):
            if "last_results" in st.session_state:
                del st.session_state["last_results"]
            if "selected_record_idx" in st.session_state:
                del st.session_state["selected_record_idx"]
            st.rerun()

        st.caption("AI/ML Research Prototype | v2.0 Enterprise")

    return selected_page
