"""
Page 3: KPI Monitoring & Domain Analytics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_academic_disclaimer,
)
from ..components.charts import (
    create_kpi_distribution_plot,
    apply_dark_chart_layout,
)
from src.data_loader import NUMERIC_KPIS


KPI_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "RSRP": {
        "category": "Radio Quality",
        "unit": "dBm",
        "description": "Reference Signal Received Power: measures the downlink signal power received from the serving base station. Typical ranges span -125 dBm (poor edge coverage) to -75 dBm (strong cell center coverage).",
    },
    "UL_SNR": {
        "category": "Signal Quality",
        "unit": "dB",
        "description": "Uplink Signal-to-Noise Ratio: indicates the quality of the uplink radio channel by comparing signal strength to noise and RF interference.",
    },
    "UL_BLER": {
        "category": "Error Rates",
        "unit": "Ratio [0-1]",
        "description": "Uplink Block Error Rate: ratio of transmitted uplink blocks with uncorrectable CRC errors. Elevated BLER indicates severe interference or poor physical channel conditions.",
    },
    "DL_BLER": {
        "category": "Error Rates",
        "unit": "Ratio [0-1]",
        "description": "Downlink Block Error Rate: proportion of received downlink transport blocks containing errors. Network schedulers aim for ~10% target BLER on first transmissions.",
    },
    "UL_MCS": {
        "category": "Modulation & Coding",
        "unit": "Index [0-28]",
        "description": "Uplink Modulation and Coding Scheme: dynamically assigned by the scheduler. Higher values (e.g. 64-QAM) achieve higher throughput; lower values (QPSK) maximize robustness.",
    },
    "DL_MCS": {
        "category": "Modulation & Coding",
        "unit": "Index [0-28]",
        "description": "Downlink Modulation and Coding Scheme: indicates the downlink efficiency index chosen based on Channel Quality Indicator (CQI) reports.",
    },
    "UL_NPRB": {
        "category": "Scheduling & Allocation",
        "unit": "Count",
        "description": "Number of Physical Resource Blocks allocated in the uplink. Indicates scheduler grant size assigned to the subscriber.",
    },
    "PRBs_DL_Current": {
        "category": "Scheduling & Allocation",
        "unit": "Count",
        "description": "Downlink Physical Resource Blocks allocated to the user equipment in the active scheduling window.",
    },
    "PRBs_UL_Current": {
        "category": "Scheduling & Allocation",
        "unit": "Count",
        "description": "Uplink Physical Resource Blocks currently allocated to the subscriber in the scheduling interval.",
    },
    "PRB_Utilization_DL": {
        "category": "Scheduling & Allocation",
        "unit": "%",
        "description": "Downlink PRB Utilization: percentage of total cell downlink spectrum resources utilized during the observation trace.",
    },
    "PRB_Utilization_UL": {
        "category": "Scheduling & Allocation",
        "unit": "%",
        "description": "Uplink PRB Utilization: fraction of cell uplink resource bandwidth consumed during the transmission interval.",
    },
    "TX_Bytes": {
        "category": "Traffic Volume & Throughput",
        "unit": "Bytes",
        "description": "Transmitted data payload volume: measures subscriber uplink traffic generation across the 128-timestep interval.",
    },
    "RX_Bytes": {
        "category": "Traffic Volume & Throughput",
        "unit": "Bytes",
        "description": "Received data payload volume: measures downlink data received across the 128-timestep trace.",
    },
    "UL_NumberOfPackets": {
        "category": "Traffic Volume & Throughput",
        "unit": "Packets",
        "description": "Number of IP packets transmitted in the uplink direction across the 12.7-second window.",
    },
    "DL_NumberOfPackets": {
        "category": "Traffic Volume & Throughput",
        "unit": "Packets",
        "description": "Number of IP packets received in the downlink direction across the observation trace.",
    },
    "Estimated_UL_Buffer": {
        "category": "Buffer Health",
        "unit": "KB",
        "description": "Estimated Uplink Buffer Occupancy: packet data volume queued in user equipment awaiting transmission grants. Extreme buffer values denote queue congestion.",
    },
}


def render(pipeline, dataset_tuple, benchmark_tables):
    """Render the KPI Monitoring and Domain Analytics section."""
    render_hero_banner(
        title="Cellular KPI Monitoring & Telemetry Analytics",
        subtitle="Empirical statistical distribution and anomaly separation across 16 core cellular KPIs",
    )
    render_academic_disclaimer()

    X, y = dataset_tuple
    if X is None or y is None:
        st.error("Processed features file not found. Ensure data preprocessing has executed.")
        return

    # Categorized KPI Explorer Controls
    col_kpi, col_stat = st.columns([1.2, 1])

    with col_kpi:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>Select Cellular KPI Channel</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        categories = sorted(list(set(d["category"] for d in KPI_DEFINITIONS.values())))
        selected_cat = st.selectbox("Filter by Functional Domain:", ["All Categories"] + categories)

        available_kpis = [
            k for k in NUMERIC_KPIS
            if selected_cat == "All Categories" or KPI_DEFINITIONS.get(k, {}).get("category") == selected_cat
        ]

        selected_kpi = st.selectbox(
            "Select KPI Channel to Inspect:",
            options=available_kpis,
            index=0,
            format_func=lambda x: f"{x} ({KPI_DEFINITIONS.get(x, {}).get('category', '')})",
        )

    with col_stat:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>Select Statistical Moment</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_moment = st.radio(
            "Engineered Moment:",
            ["mean", "std", "min", "max"],
            index=0,
            horizontal=True,
        )

        kpi_info = KPI_DEFINITIONS.get(selected_kpi, {})
        st.markdown(f"**Domain:** `{kpi_info.get('category', 'General')}` | **Unit:** `{kpi_info.get('unit', '-')}`")
        st.caption(kpi_info.get("description", ""))

    feature_col = f"{selected_kpi}_{selected_moment}"

    if feature_col not in X.columns:
        st.error(f"Feature column '{feature_col}' not found in dataset.")
        return

    st.markdown("---")

    # Statistical Comparison Cards
    normal_vals = X.loc[y == 0, feature_col]
    anomaly_vals = X.loc[y == 1, feature_col]

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_metric_card("Normal Mean", f"{normal_vals.mean():.2f}", f"σ = {normal_vals.std():.2f}", "#10B981")
    with s2:
        render_metric_card("Anomaly Mean", f"{anomaly_vals.mean():.2f}", f"σ = {anomaly_vals.std():.2f}", "#EF4444")
    with s3:
        diff = anomaly_vals.mean() - normal_vals.mean()
        z_diff = diff / (normal_vals.std() + 1e-9)
        render_metric_card("Mean Delta (Dev.)", f"{diff:+.2f}", f"{z_diff:+.1f} normal σ", "#F59E0B")
    with s4:
        render_metric_card("Full Value Range", f"[{X[feature_col].min():.1f}, {X[feature_col].max():.1f}]", kpi_info.get("unit", "-"), "#38BDF8")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Distribution Charts
    c_left, c_right = st.columns(2)

    plot_df = pd.DataFrame({
        feature_col: X[feature_col].values,
        "Class": np.where(y.values == 1, "Anomaly", "Normal"),
    })

    with c_left:
        st.markdown(f"#### 📊 Boxplot Comparison: `{feature_col}`")
        box_fig = create_kpi_distribution_plot(plot_df, feature_col, class_col="Class")
        st.plotly_chart(box_fig, use_container_width=True)

    with c_right:
        st.markdown(f"#### 📈 Distribution Overlay: `{feature_col}`")
        hist_fig = px.histogram(
            plot_df,
            x=feature_col,
            color="Class",
            barmode="overlay",
            marginal="box",
            color_discrete_map={"Normal": "#10B981", "Anomaly": "#EF4444"},
            title=f"Histogram Density Overlay ({kpi_info.get('unit', '')})",
            nbins=35,
        )
        hist_fig = apply_dark_chart_layout(hist_fig, height=320)
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 Percentile Breakdown for `{feature_col}`")

    pct_df = pd.DataFrame({
        "Percentile": ["Min (0%)", "10th %", "25th %", "Median (50%)", "75th %", "90th %", "Max (100%)"],
        "Normal Population": [
            f"{normal_vals.min():.2f}",
            f"{normal_vals.quantile(0.10):.2f}",
            f"{normal_vals.quantile(0.25):.2f}",
            f"{normal_vals.median():.2f}",
            f"{normal_vals.quantile(0.75):.2f}",
            f"{normal_vals.quantile(0.90):.2f}",
            f"{normal_vals.max():.2f}",
        ],
        "Anomaly Population": [
            f"{anomaly_vals.min():.2f}",
            f"{anomaly_vals.quantile(0.10):.2f}",
            f"{anomaly_vals.quantile(0.25):.2f}",
            f"{anomaly_vals.median():.2f}",
            f"{anomaly_vals.quantile(0.75):.2f}",
            f"{anomaly_vals.quantile(0.90):.2f}",
            f"{anomaly_vals.max():.2f}",
        ],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)
