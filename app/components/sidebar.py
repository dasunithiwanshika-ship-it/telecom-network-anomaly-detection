"""
Enterprise NOC Sidebar Component for Telecom AI Intelligence Platform.
Designed for intuitive navigation by Telecom Network Engineers.
"""

from typing import List
import streamlit as st

NAV_OPTIONS: List[str] = [
    "🏠 Welcome & Quick Start",
    "🌐 Network Overview",
    "📂 Data Upload & Validation",
    "⚡ AI Detection Workspace",
    "🔬 Anomaly Investigation",
    "📊 KPI Subsystem Monitor",
    "📑 Reports & Export",
    "🧪 Research & Model Lab",
    "ℹ️ About & System Guide",
]


def render_sidebar(pipeline) -> str:
    """
    Renders the unified enterprise sidebar navigation and operational monitors.

    Returns:
        The selected page title string.
    """
    # Handle programmatic navigation requests from action buttons
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in NAV_OPTIONS:
            st.session_state["nav_page"] = target

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in NAV_OPTIONS:
        st.session_state["nav_page"] = NAV_OPTIONS[0]

    with st.sidebar:
        # Platform Brand Header
        st.markdown(
            """
            <div style="padding: 8px 0 14px 0; border-bottom: 1px solid #1E293B; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #14B8A6, #0D9488); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        📡
                    </div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.2px;">TELECOM AI</div>
                        <div style="font-size: 0.7rem; color: #14B8A6; font-weight: 600; letter-spacing: 0.5px;">NETWORK INTELLIGENCE NOC</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active Session Telemetry Status Box
        active_df = st.session_state.get("active_dataset_df")
        active_name = st.session_state.get("active_dataset_name", "No batch loaded")
        last_results = st.session_state.get("last_results")

        if active_df is not None:
            ds_badge = f"<span style='color:#10B981;font-weight:700;'>● {len(active_df):,} Cell Windows</span>"
        else:
            ds_badge = "<span style='color:#38BDF8;font-weight:600;'>○ Baseline Ready (32k)</span>"

        if last_results is not None:
            n_anom = int(last_results["anomaly_prediction"].sum())
            alert_color = "#EF4444" if n_anom > 0 else "#10B981"
            alert_text = f"{n_anom:,} Flagged Alerts" if n_anom > 0 else "0 Alerts (All Healthy)"
        else:
            alert_color = "#94A3B8"
            alert_text = "Not Run Yet"

        st.markdown(
            f"""
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 12px; margin-bottom: 16px;">
                <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">WORKSPACE STATUS</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 5px;">
                    <span style="color: #94A3B8;">Data Buffer:</span>
                    {ds_badge}
                </div>
                <div style="font-size: 0.72rem; color: #64748B; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    Source: {active_name}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 6px; padding-top: 6px; border-top: 1px solid #1E293B;">
                    <span style="color: #94A3B8;">Active Alerts:</span>
                    <span style="color: {alert_color}; font-weight: 700;">{alert_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.74rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>OPERATIONS & NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            label="Navigation Menu",
            options=NAV_OPTIONS,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Model Engine Readiness
        st.markdown(
            "<div style='font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>DETECTION ENGINES</div>",
            unsafe_allow_html=True,
        )

        model_labels = {
            "random_forest": "High-Precision (RF)",
            "logistic_regression": "Fast Screening (LR)",
            "isolation_forest": "Pattern Discovery (IF)",
            "local_outlier_factor": "Density Scanner (LOF)",
        }

        for mid, mname in model_labels.items():
            avail = pipeline.is_model_available(mid)
            dot_color = "#10B981" if avail else "#EF4444"
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.76rem; padding: 2px 0;">
                    <span style="color: #94A3B8;">{mname}</span>
                    <span style="color: {dot_color}; font-weight: 600;">{'● Online' if avail else '○ Offline'}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session Data", use_container_width=True, help="Clear uploaded data and detection results to start fresh."):
            for k in ["active_dataset_df", "active_dataset_name", "last_results", "last_model_id", "last_threshold", "selected_record_id"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        st.caption("Telecom AI Platform v2.0 | NOC Edition")

    return selected_page
