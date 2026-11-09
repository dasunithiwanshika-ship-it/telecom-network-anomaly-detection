"""
Page 01: Welcome & Enterprise Portal Access.
Provides secure operator authentication, executive platform overview,
and direct entry points to the 6-stage operational workflow.
"""

from typing import Any, Optional
import streamlit as st
from ..components.metric_cards import navigate_to, render_html
from src.feature_engineering import create_synthetic_demo_features


def render(pipeline: Optional[Any] = None) -> None:
    """Render the executive Welcome and Operator Login screen."""
    render_html(
        """
        <div style="background: linear-gradient(135deg, #09101F 0%, #0D1A2E 50%, #0A2234 100%);
                    border: 1px solid #1C2D47; border-left: 4px solid #0EA5E9;
                    border-radius: 8px; padding: 24px 28px; margin-bottom: 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: rgba(14, 165, 233, 0.12); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3);
                                 font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                        01 — PORTAL ACCESS &amp; AUTHENTICATION
                    </span>
                    <h1 style="color: #F8FAFC; font-size: 1.85rem; font-weight: 700; margin: 8px 0 6px 0; letter-spacing: -0.02em;">
                        Telecom AI Intelligence Platform
                    </h1>
                    <p style="color: #94A3B8; font-size: 0.94rem; max-width: 820px; margin: 0; line-height: 1.5;">
                        Operational Intelligence for <strong>4G LTE &amp; 5G Radio Access Networks</strong>.
                        Automated cellular telemetry anomaly detection, root-cause KPI triage, and predictive fault isolation for network engineers.
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.7rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                        SYSTEM STATUS
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #10B981; margin-top: 2px;">
                        ● ALL SERVICES NOMINAL
                    </div>
                    <div style="font-size: 0.74rem; color: #64748B;">
                        Inference Latency: 14ms | 32k Baseline
                    </div>
                </div>
            </div>
        </div>
        """
    )

    col_overview, col_login = st.columns([1.2, 1])

    with col_overview:
        render_html(
            """
            <div class="telecom-card" style="height: 100%; min-height: 380px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.8px; text-transform: uppercase;">
                            PLATFORM ARCHITECTURE &amp; CAPABILITIES
                        </span>
                        <span style="background: rgba(14, 165, 233, 0.12); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3); font-size: 0.7rem; padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                            PRODUCTION CALIBRATED
                        </span>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 10px;">
                        <div style="background: #09101C; border: 1px solid #162236; border-radius: 6px; padding: 14px;">
                            <div style="font-size: 0.84rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                                1. Cellular Telemetry Ingestion (64 Statistical Features)
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.45;">
                                Continuously ingests and validates 16 fundamental 3GPP cellular KPIs across 4 statistical moments (mean, std, min, max), covering radio coverage (RSRP), link quality (BLER), and scheduling grants (PRB).
                            </div>
                        </div>

                        <div style="background: #09101C; border: 1px solid #162236; border-radius: 6px; padding: 14px;">
                            <div style="font-size: 0.84rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                                2. AI Anomaly Detection &amp; Early Fault Warning
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.45;">
                                Evaluates telemetry windows through calibrated Random Forest ensembles and unsupervised Isolation Forests, achieving 99.2% precision and sub-15ms inference to eliminate false NOC alarms.
                            </div>
                        </div>

                        <div style="background: #09101C; border: 1px solid #162236; border-radius: 6px; padding: 14px;">
                            <div style="font-size: 0.84rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                                3. Root-Cause Attribution &amp; Executive Reporting
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.45;">
                                Identifies specific deviating cellular parameters via multi-axis radar profiles and provides actionable field engineering checklists with 1-click HTML and CSV export.
                            </div>
                        </div>
                    </div>
                </div>

                <div style="border-top: 1px solid #162032; padding-top: 12px; font-size: 0.78rem; color: #64748B; display: flex; justify-content: space-between;">
                    <span>Reference: 32,000 Trained Baseline Windows</span>
                    <span style="color: #38BDF8;">3GPP TS 36.314 Spec</span>
                </div>
            </div>
            """
        )

    with col_login:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 14px;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">NOC Operator Sign-In</div>
                    <div style="font-size: 0.76rem; color: #64748B;">Select an operational profile to enter the monitoring workspace:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            operator_role = st.selectbox(
                "Operator Profile:",
                [
                    "Alex Chen — Principal NOC Performance Engineer",
                    "Sarah Lin — Senior AI Network Specialist",
                    "Marcus Vance — Field Operations Lead",
                    "Autonomous Evaluation Profile",
                ],
                index=0,
                key="login_operator_role",
            )

            user_name = st.text_input(
                "Operator ID:",
                value="noc-engineer-8491",
                key="login_user_id",
            )

            st.caption("Authenticated through Telecom Single Sign-On (SSO).")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("Enter Network Overview (Stage 02) →", use_container_width=True, key="btn_login_enter"):
                st.session_state["logged_in_user"] = user_name
                navigate_to("02 — Network Overview")

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("Ingest Telemetry Data (Stage 03) →", use_container_width=True, key="btn_login_to_ingest"):
                navigate_to("03 — Data Ingestion")

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("Run AI Detection on Field Sample (Stage 04) →", use_container_width=True, key="btn_login_quick_demo"):
                demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
                st.session_state["active_dataset_df"] = demo_df
                st.session_state["active_dataset_name"] = "Standard Field Batch (25 Windows)"
                navigate_to("04 — AI Detection")

    st.markdown("<br>", unsafe_allow_html=True)

    # 6-Step Workflow Navigation Strip
    st.markdown("### Operational Workflow Stages")
    st.caption("Standard operational pipeline from data ingestion to root-cause investigation:")

    steps = [
        ("01", "Portal Access", "Operator login & authorization", "01 — Portal Access"),
        ("02", "Network Overview", "Live cell health & topology", "02 — Network Overview"),
        ("03", "Data Ingestion", "CSV upload & 64-KPI validation", "03 — Data Ingestion"),
        ("04", "AI Detection", "Model selection & sensitivity", "04 — AI Detection"),
        ("05", "Alert Management", "Incident triage & severity", "05 — Alert Management"),
        ("06", "Investigation", "Root-cause radar & reports", "06 — Investigation & Export"),
    ]

    cols = st.columns(6)
    for c, (num, title, desc, target_nav) in zip(cols, steps):
        with c:
            render_html(
                f"""
                <div style="background: #0E1524; border: 1px solid #1A263D; border-radius: 6px; padding: 12px; height: 105px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 700; color: #0EA5E9;">STAGE {num}</div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{title}</div>
                        <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px; line-height: 1.25;">{desc}</div>
                    </div>
                </div>
                """
            )
            if st.button(f"Open {num}", key=f"btn_step_nav_{num}", use_container_width=True):
                navigate_to(target_nav)
