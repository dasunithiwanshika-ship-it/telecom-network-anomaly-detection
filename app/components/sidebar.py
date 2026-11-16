"""
Royal Violet Sidebar Navigation for Telecom AI Platform.
Features clean Inter typography, violet lightning-bolt emblem, and direct access
to the 6-stage operational engineering workflow.
"""

from typing import List
from datetime import datetime
import streamlit as st
from .metric_cards import render_html


NAV_OPTIONS: List[str] = [
    "01 — Portal Access",
    "02 — Network Overview",
    "03 — Data Ingestion",
    "04 — AI Detection",
    "05 — Alert Management",
    "06 — Investigation & Export",
]


def render_sidebar(pipeline=None) -> str:
    """Renders the Royal Violet sidebar navigation."""
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in NAV_OPTIONS:
            st.session_state["nav_page"] = target
        else:
            for opt in NAV_OPTIONS:
                if target[:2] in opt or any(word in opt for word in target.split()):
                    st.session_state["nav_page"] = opt
                    break

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in NAV_OPTIONS:
        st.session_state["nav_page"] = NAV_OPTIONS[0]

    with st.sidebar:
        # Royal Violet Lightning Emblem Header
        render_html(
            """
            <div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(147, 18, 255, 0.18); margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #A020FF 0%, #7A1CF8 100%); color: #F7F7F8; border-radius: 10px;
                                width: 34px; height: 34px; display: flex; align-items: center;
                                justify-content: center; box-shadow: 0 0 18px rgba(147, 18, 255, 0.55);">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #F7F7F8; letter-spacing: -0.02em;">
                            NEXVANTA <span style="color: #B76EFF; font-size: 0.78rem; font-weight: 600;">AI</span>
                        </div>
                        <div style="font-size: 0.68rem; color: #8B8988; text-transform: uppercase; letter-spacing: 0.6px;">
                            Telecom Intelligence
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        # Operational Persona
        render_html(
            """
            <div style="font-size: 0.68rem; font-weight: 600; color: #8B8988; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;">
                OPERATIONAL ROLE
            </div>
            """
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

        render_html("<div style='height: 10px;'></div>")

        # Workflow Navigation
        render_html(
            """
            <div style="font-size: 0.68rem; font-weight: 600; color: #8B8988; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;">
                WORKFLOW STAGES
            </div>
            """
        )

        selected_page = st.radio(
            label="Workflow Stages",
            options=NAV_OPTIONS,
            key="nav_page",
            label_visibility="collapsed",
        )

        render_html("<hr style='border: none; border-top: 1px solid rgba(147, 18, 255, 0.16); margin: 18px 0;'>")

        # System Status Box
        now_time = datetime.now().strftime("%H:%M:%S")
        render_html(
            f"""
            <div style="background: rgba(11, 9, 18, 0.9); border: 1px solid rgba(147, 18, 255, 0.22); border-radius: 10px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.7rem; font-weight: 700; color: #45B558; letter-spacing: 0.5px;">
                        ● LIVE TELEMETRY
                    </span>
                    <span style="font-size: 0.72rem; color: #8B8988; font-family: monospace;">{now_time}</span>
                </div>
                <div style="font-size: 0.75rem; color: #8B8988; margin-top: 4px;">
                    Engine: <strong style="color: #D8B4FE;">Random Forest</strong>
                </div>
            </div>
            """
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
