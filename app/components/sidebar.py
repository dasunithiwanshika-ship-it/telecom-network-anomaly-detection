"""
Clean, Simple Sidebar Navigation for Telecom AI Platform.
"""

from typing import List
import streamlit as st

NAV_OPTIONS: List[str] = [
    "🏠 Overview",
    "⚡ Run Detection",
    "🔍 Inspect Alert",
    "📊 KPIs & Reports",
    "📚 Research Lab",
]


def render_sidebar(pipeline=None) -> str:
    """Renders a clean, uncluttered sidebar navigation."""
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in NAV_OPTIONS:
            st.session_state["nav_page"] = target

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in NAV_OPTIONS:
        st.session_state["nav_page"] = NAV_OPTIONS[0]

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background-color: #14B8A6; color: #0B1220; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 800;">
                        📡
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">Telecom AI</div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">Network Anomaly Monitor</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            label="Menu",
            options=NAV_OPTIONS,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Simple 1-line status indicator
        last_results = st.session_state.get("last_results")
        if last_results is not None:
            n_anom = int(last_results["anomaly_prediction"].sum())
            st.caption(f"🟢 **Scan Active:** {len(last_results)} rows ({n_anom} alerts)")
            if st.button("🔄 Clear Results", use_container_width=True):
                for k in ["active_dataset_df", "active_dataset_name", "last_results", "selected_record_id"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        else:
            st.caption("⚪ **Status:** Ready to scan data")

    return selected_page
