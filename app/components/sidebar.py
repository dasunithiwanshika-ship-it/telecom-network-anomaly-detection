"""
Executive Sidebar Navigation for Telecom AI Platform.
Features clean typography, minimal geometric indicators, and direct access
to the 6-step engineering workflow.
"""

from typing import List
from datetime import datetime
import streamlit as st


NAV_OPTIONS: List[str] = [
    "01 — Portal Access",
    "02 — Network Overview",
    "03 — Data Ingestion",
    "04 — AI Detection",
    "05 — Alert Management",
    "06 — Investigation & Export",
]


def render_sidebar(pipeline=None) -> str:
    """Renders the executive NOC sidebar navigation."""
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in NAV_OPTIONS:
            st.session_state["nav_page"] = target

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in NAV_OPTIONS:
        st.session_state["nav_page"] = NAV_OPTIONS[0]

    with st.sidebar:
        # Refined Header
        st.markdown(
            """
            <div style="padding: 2px 0 16px 0; border-bottom: 1px solid #162032; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: #0284C7; color: #FFFFFF; border-radius: 6px;
                                width: 32px; height: 32px; display: flex; align-items: center;
                                justify-content: center; font-size: 0.88rem; font-weight: 800; letter-spacing: 0.5px;">
                        NOC
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.02em;">
                            NEXVANTA <span style="color: #38BDF8; font-size: 0.8rem; font-weight: 600;">NIMS</span>
                        </div>
                        <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">
                            Telecom Intelligence
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational Persona
        st.markdown(
            """
            <div style="font-size: 0.68rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                OPERATIONAL ROLE
            </div>
            """,
            unsafe_allow_html=True,
        )
        current_persona = st.session_state.get("user_persona", "NOC Engineer (Standard)")
        persona_choice = st.selectbox(
            "Select Persona",
            ["NOC Engineer (Standard)", "AI/ML Specialist (Advanced)"],
            index=0 if "NOC" in current_persona else 1,
            label_visibility="collapsed",
            key="persona_selector",
        )
        st.session_state["user_persona"] = persona_choice

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Workflow Navigation
        st.markdown(
            """
            <div style="font-size: 0.68rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                WORKFLOW STAGES
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            label="Workflow Stages",
            options=NAV_OPTIONS,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #162032; margin: 18px 0;'>", unsafe_allow_html=True)

        # System Status Box
        now_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
            <div style="background: #080C14; border: 1px solid #162032; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.7rem; font-weight: 700; color: #10B981; letter-spacing: 0.5px;">
                        ● LIVE TELEMETRY
                    </span>
                    <span style="font-size: 0.72rem; color: #64748B; font-family: monospace;">{now_time}</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                    Active Model: <strong style="color: #38BDF8;">Random Forest</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_results = st.session_state.get("last_results")
        if last_results is not None:
            n_anom = int(last_results["anomaly_prediction"].sum())
            st.caption(f"Telemetry: {len(last_results)} windows ({n_anom} flagged)")
            if st.button("Reset Telemetry Session", use_container_width=True):
                for k in ["active_dataset_df", "active_dataset_name", "last_results", "selected_record_id"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        else:
            st.caption("Engine ready for telemetry ingestion.")

    return selected_page
