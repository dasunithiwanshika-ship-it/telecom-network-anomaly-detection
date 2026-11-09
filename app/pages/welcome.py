"""
Page 01: Welcome & Enterprise NOC Login Portal.
Designed for professional telecom network operations center (NOC) environments.
Provides secure operator authentication, infrastructure grid telemetry,
and rapid workflow entry points.
"""

from typing import Any, Optional
import streamlit as st
from ..components.metric_cards import navigate_to
from src.feature_engineering import create_synthetic_demo_features


def render(pipeline: Optional[Any] = None) -> None:
    """Render the executive Welcome and Operator Login screen."""
    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

    col_visual, col_login = st.columns([1.25, 1])

    with col_visual:
        st.markdown(
            """
            <div class="telecom-card" style="height: 100%; min-height: 400px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.8px; text-transform: uppercase;">
                            CELLULAR INFRASTRUCTURE TOPOLOGY
                        </span>
                        <span style="background: rgba(16, 185, 129, 0.12); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); font-size: 0.7rem; padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                            LIVE STATUS
                        </span>
                    </div>
                    
                    <div style="background: #09101C; border: 1px solid #162236; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em;">
                            Radio Access Network Monitoring Hierarchy
                        </div>
                        <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">
                            eNodeB / gNodeB RAN Telemetry • Sub-Band Carrier Aggregation
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 18px; text-align: left;">
                            <div style="background: #0C1524; border: 1px solid #19273D; border-radius: 6px; padding: 10px;">
                                <div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase; font-weight: 600;">MAC Scheduler</div>
                                <div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-top: 2px;">PRB / MCS</div>
                                <div style="font-size: 0.7rem; color: #10B981; margin-top: 2px;">Grants Active</div>
                            </div>
                            <div style="background: #0C1524; border: 1px solid #19273D; border-radius: 6px; padding: 10px;">
                                <div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Radio Quality</div>
                                <div style="color: #0EA5E9; font-weight: 700; font-size: 0.95rem; margin-top: 2px;">RSRP / SNR</div>
                                <div style="font-size: 0.7rem; color: #10B981; margin-top: 2px;">Nominal Bound</div>
                            </div>
                            <div style="background: #0C1524; border: 1px solid #19273D; border-radius: 6px; padding: 10px;">
                                <div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Link Quality</div>
                                <div style="color: #F59E0B; font-weight: 700; font-size: 0.95rem; margin-top: 2px;">UL/DL BLER</div>
                                <div style="font-size: 0.7rem; color: #F59E0B; margin-top: 2px;">HARQ Tracked</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div style="border-top: 1px solid #162032; padding-top: 10px; font-size: 0.8rem; color: #64748B; display: flex; justify-content: space-between;">
                    <span>32,000 Reference Telemetry Windows</span>
                    <span style="color: #38BDF8;">3GPP TS 36.314 Spec</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_login:
        st.markdown(
            """
            <div class="telecom-card">
                <div style="margin-bottom: 14px;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">NOC Operator Sign-In</div>
                    <div style="font-size: 0.76rem; color: #64748B;">Enter credentials or select an operational profile</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        operator_role = st.selectbox(
            "Operator Profile:",
            [
                "Alex Chen — Principal NOC Performance Engineer",
                "Sarah Lin — Senior AI Network Specialist",
                "Marcus Vance — Field Technician Lead",
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

        pin_code = st.text_input(
            "Security PIN / Token:",
            value="••••••••••••",
            type="password",
            key="login_pin",
        )

        st.caption("Authenticated through Telecom Single Sign-On (SSO).")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            if st.button("Enter NOC Portal", use_container_width=True, key="btn_login_enter"):
                st.session_state["logged_in_user"] = user_name
                st.success(f"Session established for {operator_role.split('—')[0].strip()}.")
                navigate_to("02 — Network Overview")

        with col_in2:
            if st.button("Load Field Sample", use_container_width=True, key="btn_login_quick_demo"):
                demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
                st.session_state["active_dataset_df"] = demo_df
                st.session_state["active_dataset_name"] = "Standard Field Batch (25 Windows)"
                navigate_to("04 — AI Detection")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6-Step Workflow Navigation Strip
    st.markdown("### End-to-End Operational Workflow")
    st.caption("Standard operational pipeline from data ingestion to root-cause investigation:")

    steps = [
        ("01", "Portal Access", "Operator login & authorization", "01 — Portal Access"),
        ("02", "Network Overview", "Live cell health & gauges", "02 — Network Overview"),
        ("03", "Data Ingestion", "CSV upload & 64-KPI validation", "03 — Data Ingestion"),
        ("04", "AI Detection", "Model selection & sensitivity", "04 — AI Detection"),
        ("05", "Alert Management", "Incident triage & severity", "05 — Alert Management"),
        ("06", "Investigation", "Root-cause radar & reports", "06 — Investigation & Export"),
    ]

    cols = st.columns(6)
    for c, (num, title, desc, target_nav) in zip(cols, steps):
        with c:
            st.markdown(
                f"""
                <div style="background: #0E1524; border: 1px solid #1A263D; border-radius: 6px; padding: 12px; height: 105px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 700; color: #0EA5E9;">STAGE {num}</div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{title}</div>
                        <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px; line-height: 1.25;">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {num}", key=f"btn_step_nav_{num}", use_container_width=True):
                navigate_to(target_nav)
