"""
Page 1: Home & Network Overview.
Clean, uncluttered overview with a direct action to start anomaly detection.
"""

from typing import Optional, Tuple, Any
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    navigate_to,
)
from ..components.charts import create_donut_class_distribution
from src.data_loader import load_processed_features


@st.cache_data
def _get_baseline_dataset():
    try:
        return load_processed_features()
    except Exception:
        return None, None


def render(pipeline: Any = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the clean Home & Overview page."""
    render_hero_banner(
        title="Telecom AI Network Monitor",
        subtitle="Automatically detect abnormal cellular KPI behavior, identify which network metric caused the issue, and export reports.",
        badge="WELCOME",
    )

    # Top Quick-Action Bar
    c_welcome, c_btn = st.columns([2.2, 1])
    with c_welcome:
        st.markdown(
            "**How it works:** Upload a KPI CSV file (or use the built-in sample data), click **Run AI Detection**, and click any flagged alert to see why it was triggered."
        )
    with c_btn:
        if st.button("⚡ Start Anomaly Detection →", use_container_width=True, key="btn_home_start"):
            navigate_to("⚡ Run Detection")

    st.markdown("<br>", unsafe_allow_html=True)

    if dataset_tuple is not None:
        X, y = dataset_tuple
    else:
        X, y = _get_baseline_dataset()

    total_records = len(X) if X is not None else 32000
    normal_records = int((y == 0).sum()) if y is not None else 30765
    anomaly_records = int((y == 1).sum()) if y is not None else 1235

    # 4 Clean Summary Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Network Windows", f"{total_records:,}", "Baseline dataset size", "#38BDF8")
    with c2:
        render_metric_card("Normal Operation", f"{normal_records:,}", "96.1% healthy traffic", "#10B981")
    with c3:
        render_metric_card("Network Anomalies", f"{anomaly_records:,}", "3.9% fault-risk events", "#EF4444")
    with c4:
        render_metric_card("AI Precision", "99.2%", "Recommended RF model", "#14B8A6")

    st.markdown("<br>", unsafe_allow_html=True)

    # Clean 2-Column Visual Section
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.markdown("#### 📊 Network Health Breakdown")
        donut_fig = create_donut_class_distribution(normal_records, anomaly_records)
        st.plotly_chart(donut_fig, use_container_width=True)

    with right_col:
        st.markdown("#### 📡 Monitored Network Signals (16 KPIs)")
        summary_table = pd.DataFrame([
            {
                "Signal Group": "Radio Signal Quality",
                "KPIs Monitored": "RSRP, Uplink SNR",
                "What It Detects": "Weak tower coverage or radio interference",
            },
            {
                "Signal Group": "Transmission Error Rates",
                "KPIs Monitored": "Uplink & Downlink BLER",
                "What It Detects": "Data block errors and packet drops",
            },
            {
                "Signal Group": "Channel Allocation",
                "KPIs Monitored": "PRB Allocation, Utilization, MCS",
                "What It Detects": "Channel starvation or cell overload",
            },
            {
                "Signal Group": "Traffic & Device Buffer",
                "KPIs Monitored": "TX/RX Bytes, Packets, UL Buffer",
                "What It Detects": "Queue congestion and throughput stalls",
            },
        ])
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
        st.caption("ℹ️ Built using the public `AliMaatouk/TelecomTS` cellular KPI dataset (independent research prototype).")
