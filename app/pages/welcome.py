"""
Welcome Screen: Telecom AI Intelligence Platform Landing & Quick-Start Hub.
Designed specifically for Telecom Network Engineers to operate without ML or coding knowledge.
"""

import streamlit as st
from ..components.metric_cards import (
    render_metric_card,
    render_academic_disclaimer,
    navigate_to,
)
from src.feature_engineering import create_synthetic_demo_features


def render(pipeline=None):
    """Render the Welcome Screen and Operational Quick-Start Hub."""
    # Main Welcome Hero Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0F172A 0%, #172235 55%, #0D2D3A 100%);
                    border: 1px solid #1E293B; border-left: 6px solid #14B8A6;
                    border-radius: 14px; padding: 32px 36px; margin-bottom: 24px;
                    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);">
            <div style="display: inline-block; background-color: rgba(20, 184, 166, 0.15);
                        color: #14B8A6; border: 1px solid rgba(20, 184, 166, 0.4);
                        font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
                        padding: 4px 12px; border-radius: 999px; margin-bottom: 12px;">
                📡 ENTERPRISE TELECOM NOC DIAGNOSTICS
            </div>
            <h1 style="color: #F8FAFC; font-size: 2.2rem; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.5px;">
                Telecom AI Intelligence Platform
            </h1>
            <p style="color: #CBD5E1; font-size: 1.05rem; line-height: 1.6; max-width: 880px; margin: 0;">
                Automated cellular network KPI monitoring, fault-risk detection, and root-cause triage for
                <strong>4G LTE &amp; 5G Radio Access Networks</strong>. Upload cell-site telemetry, detect abnormal radio
                and scheduling behavior in seconds, inspect deviating KPIs against normal baselines, and export engineering reports—with zero programming required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary Launch Actions
    st.markdown("### 🚀 Quick-Start Operations")
    st.caption("Select an action below to enter the workspace immediately:")

    c_act1, c_act2, c_act3 = st.columns(3)

    with c_act1:
        st.markdown(
            """
            <div class="telecom-card" style="min-height: 155px;">
                <h3 style="color: #14B8A6;">🌐 Network Overview</h3>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5;">
                    View overall network health, monitored radio and MAC subsystems, and baseline cellular KPI status across 32,000 reference traces.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🌐 Enter Network Overview →", use_container_width=True, key="btn_welcome_overview"):
            navigate_to("🌐 Network Overview")

    with c_act2:
        st.markdown(
            """
            <div class="telecom-card" style="min-height: 155px;">
                <h3 style="color: #38BDF8;">📂 Upload KPI Telemetry</h3>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5;">
                    Upload a CSV batch of cellular KPI measurements (or load a built-in field test sample) with automatic column validation and data preview.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📂 Open Data Upload & Validation →", use_container_width=True, key="btn_welcome_upload"):
            navigate_to("📂 Data Upload & Validation")

    with c_act3:
        st.markdown(
            """
            <div class="telecom-card" style="min-height: 155px;">
                <h3 style="color: #F59E0B;">⚡ Instant Demo Scan</h3>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5;">
                    Load a 25-window cellular test batch and immediately open the AI Detection Workspace to experience alert triage and KPI investigation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Launch AI Detection Workspace →", use_container_width=True, key="btn_welcome_detect"):
            if "active_dataset_df" not in st.session_state or st.session_state["active_dataset_df"] is None:
                demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
                st.session_state["active_dataset_df"] = demo_df
                st.session_state["active_dataset_name"] = "Standard Field Sample (25 Cell Windows)"
            navigate_to("⚡ AI Detection Workspace")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4-Step How It Works Workflow for Telecom Engineers
    st.markdown("### 🛠️ How to Operate This Platform (4-Step Workflow)")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            """
            <div class="step-card" style="min-height: 175px;">
                <div style="margin-bottom: 8px;"><span class="step-badge">STEP 1</span> <strong style="color:#F8FAFC;">Load KPI Data</strong></div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Upload your cell-window CSV file or select a pre-loaded sample dataset in <strong>Data Upload &amp; Validation</strong>. The system automatically checks all 64 KPI statistical columns.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="step-card" style="min-height: 175px;">
                <div style="margin-bottom: 8px;"><span class="step-badge">STEP 2</span> <strong style="color:#F8FAFC;">Run AI Detection</strong></div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    In <strong>AI Detection Workspace</strong>, choose a detection engine (such as the Recommended High-Precision Detector), set alert sensitivity, and click <strong>RUN AI DETECTION</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="step-card" style="min-height: 175px;">
                <div style="margin-bottom: 8px;"><span class="step-badge">STEP 3</span> <strong style="color:#F8FAFC;">Inspect Alerts</strong></div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Review flagged cell windows in the <strong>Anomaly Alerts Table</strong>. Filter by severity and click any record to open its dedicated <strong>Anomaly Investigation</strong> view.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
            <div class="step-card" style="min-height: 175px;">
                <div style="margin-bottom: 8px;"><span class="step-badge">STEP 4</span> <strong style="color:#F8FAFC;">Export Reports</strong></div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Compare anomalous KPIs against normal cell baselines, read the automated engineering triage guidance, and download CSV results or printable diagnostic reports.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Monitored Telecom Subsystems Overview
    st.markdown("### 📡 Monitored Cellular Subsystems (16 KPIs / 64 Telemetry Metrics)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Radio & Signal", "RSRP & UL SNR", "Coverage power & uplink clarity", "#14B8A6")
    with m2:
        render_metric_card("Link Error Rates", "UL & DL BLER", "Block CRC error ratios [0–1]", "#38BDF8")
    with m3:
        render_metric_card("Modulation & PRBs", "MCS & PRB Util %", "Spectral efficiency & grant load", "#F59E0B")
    with m4:
        render_metric_card("Traffic & Queues", "Bytes, Pkts & Buffer", "Throughput & UE backlog health", "#10B981")

    render_academic_disclaimer()
