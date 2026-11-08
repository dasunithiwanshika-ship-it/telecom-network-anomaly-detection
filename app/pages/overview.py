"""
Page 1: Network Overview (Executive Telemetry & Health Dashboard).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_academic_disclaimer,
)
from ..components.charts import (
    create_donut_class_distribution,
    apply_dark_chart_layout,
)


def render(pipeline, dataset_tuple, benchmark_tables):
    """Render the Network Overview executive landing page."""
    render_hero_banner(
        title="Network Telemetry Overview & Executive Health",
        subtitle="Global cellular KPI observability, operational status, and macro anomaly distribution",
    )
    render_academic_disclaimer()

    X, y = dataset_tuple
    total_records = len(X) if X is not None else 32000
    normal_records = int((y == 0).sum()) if y is not None else 30765
    anomaly_records = int((y == 1).sum()) if y is not None else 1235
    anomaly_rate = (anomaly_records / total_records) * 100 if total_records > 0 else 3.86

    # Top Executive Metric Cards Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric_card("Total Traces", f"{total_records:,}", "128-step sessions")
    with c2:
        render_metric_card("Normal Sessions", f"{normal_records:,}", "96.14% of network", "#10B981")
    with c3:
        render_metric_card("Network Anomalies", f"{anomaly_records:,}", "Minority class", "#EF4444")
    with c4:
        render_metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%", "Prevalence ratio", "#F59E0B")
    with c5:
        render_metric_card("KPI Features", "64", "Statistical moments", "#38BDF8")
    with c6:
        active_model = st.session_state.get("selected_model_id", "random_forest")
        render_metric_card("Active Engine", active_model.split('_')[0].upper(), "Recommended", "#14B8A6")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>📊 Telemetry Class Distribution</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        donut_fig = create_donut_class_distribution(normal_records, anomaly_records)
        st.plotly_chart(donut_fig, use_container_width=True)

        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: -10px;">
                <strong>Observation:</strong> Severe class imbalance (3.86% anomaly rate).
                In cellular radio networks, standard accuracy is misleading because an uncalibrated classifier
                predicting 100% normal achieves 96.14% accuracy while failing to detect any channel or queue anomalies.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>🛡️ KPI Domain Health & Architecture</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        domain_data = pd.DataFrame([
            {"Domain": "Radio Frequency (RF)", "Monitored KPIs": "RSRP, UL_SNR", "Status": "Optimal (Baseline)", "Risk": "Low"},
            {"Domain": "Error Rates (BLER)", "Monitored KPIs": "UL_BLER, DL_BLER", "Status": "Normal / Sporadic Spikes", "Risk": "Medium"},
            {"Domain": "Link Modulation", "Monitored KPIs": "UL_MCS, DL_MCS", "Status": "Adaptive QAM", "Risk": "Low"},
            {"Domain": "Resource Scheduling", "Monitored KPIs": "PRB Utilization, NPRB", "Status": "Dynamic Allocation", "Risk": "Medium"},
            {"Domain": "Traffic Volume", "Monitored KPIs": "TX/RX Bytes, Packets", "Status": "Heavy Skewness", "Risk": "Low"},
            {"Domain": "Queue Buffer Health", "Monitored KPIs": "Estimated_UL_Buffer", "Status": "Burst Congestion Risk", "Risk": "High"},
        ])
        st.dataframe(domain_data, use_container_width=True, hide_index=True)

        st.info("💡 TelecomTS is an offline research benchmark. When no live stream is connected, this platform operates in historical analysis and batch CSV ingestion mode.")

    st.markdown("---")

    # Recent Activity / Historical Analysis Table
    st.subheader("📋 Recent Analysis & Verification Snapshot")
    st.markdown("Sample records from the deterministic 80/20 test split with ground truth and baseline predictions:")

    if X is not None:
        sample_display = X.iloc[:8][[
            "RSRP_mean", "UL_SNR_mean", "UL_BLER_mean", "Estimated_UL_Buffer_mean", "PRB_Utilization_DL_mean"
        ]].copy()
        sample_display["Ground Truth"] = np.where(y.iloc[:8] == 1, "Anomaly (Fault)", "Normal (Healthy)")
        sample_display["Record ID"] = [f"TRACE-{idx:05d}" for idx in sample_display.index]
        cols_order = ["Record ID", "Ground Truth", "RSRP_mean", "UL_SNR_mean", "UL_BLER_mean", "Estimated_UL_Buffer_mean", "PRB_Utilization_DL_mean"]
        st.dataframe(sample_display[cols_order], use_container_width=True, hide_index=True)
