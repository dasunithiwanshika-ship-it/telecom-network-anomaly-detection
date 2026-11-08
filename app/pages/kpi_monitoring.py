"""
Page 6: KPI Subsystem Monitor & Telemetry Distribution Explorer.
Allows Telecom Network Engineers to inspect normal operating ranges vs. anomaly signatures
across all 16 cellular KPIs and 7 functional subsystems.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_academic_disclaimer,
    TELECOM_KPI_METADATA,
    MOMENT_LABELS,
)
from ..components.charts import (
    create_kpi_distribution_plot,
    apply_dark_chart_layout,
)
from src.data_loader import NUMERIC_KPIS, load_processed_features


@st.cache_data
def _load_kpi_baseline():
    try:
        return load_processed_features()
    except Exception:
        return None, None


def render(pipeline: Any = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the KPI Subsystem Monitor section."""
    render_hero_banner(
        title="Cellular KPI Subsystem Monitor",
        subtitle="Inspect normal operating envelopes vs. anomalous behavior across all 16 radio, modulation, scheduling, and buffer KPIs.",
        badge="TELEMETRY SUBSYSTEM MONITOR",
    )

    if dataset_tuple is not None:
        X, y = dataset_tuple
    else:
        X, y = _load_kpi_baseline()

    if X is None or y is None:
        st.error("Baseline telemetry dataset not found. Please ensure `data/processed/` is available.")
        return

    # Categorized KPI Explorer Controls
    col_kpi, col_stat = st.columns([1.2, 1])

    with col_kpi:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">SUBSYSTEM</span>
                    <strong style="color: #F8FAFC;">Select Cellular Subsystem &amp; KPI Channel</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        subsystems = sorted(list(set(d["subsystem"] for d in TELECOM_KPI_METADATA.values())))
        selected_sub = st.selectbox("1. Filter by Telecom Subsystem:", ["All 7 Subsystems"] + subsystems)

        available_kpis = [
            k for k in NUMERIC_KPIS
            if selected_sub == "All 7 Subsystems" or TELECOM_KPI_METADATA.get(k, {}).get("subsystem") == selected_sub
        ]

        selected_kpi = st.selectbox(
            "2. Select KPI Channel to Inspect:",
            options=available_kpis,
            index=0,
            format_func=lambda x: f"{TELECOM_KPI_METADATA.get(x, {}).get('friendly', x)} [{x}]",
        )

    with col_stat:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">STATISTIC</span>
                    <strong style="color: #F8FAFC;">Select Window Statistic (12.8s Window)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_moment = st.radio(
            "Window Measurement Type:",
            ["mean", "std", "min", "max"],
            index=0,
            horizontal=True,
            format_func=lambda m: f"{MOMENT_LABELS.get(m, m)} ({m})",
            help="Each 12.8-second cell trace (128 samples) is summarized by its Average (mean), Fluctuation (std), Minimum (min), and Peak (max).",
        )

        kpi_info = TELECOM_KPI_METADATA.get(selected_kpi, {})
        st.markdown(
            f"**Subsystem:** `{kpi_info.get('subsystem', 'General')}` | **Unit:** `{kpi_info.get('unit', '-')}`"
        )
        st.caption(f"📖 **Meaning:** {kpi_info.get('meaning', '')}")
        st.caption(f"🔧 **Engineering Action:** {kpi_info.get('action', '')}")

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
        render_metric_card("Healthy Baseline Avg", f"{normal_vals.mean():.2f}", f"5th–95th: [{normal_vals.quantile(0.05):.1f}, {normal_vals.quantile(0.95):.1f}]", "#10B981")
    with s2:
        render_metric_card("Anomalous Windows Avg", f"{anomaly_vals.mean():.2f}", f"5th–95th: [{anomaly_vals.quantile(0.05):.1f}, {anomaly_vals.quantile(0.95):.1f}]", "#EF4444")
    with s3:
        diff = anomaly_vals.mean() - normal_vals.mean()
        z_diff = diff / (normal_vals.std() + 1e-9)
        render_metric_card("Average Shift in Faults", f"{diff:+.2f}", f"{z_diff:+.2f}σ from healthy mean", "#F59E0B")
    with s4:
        render_metric_card("Measurement Unit", kpi_info.get("unit", "-"), f"Range: [{X[feature_col].min():.1f}, {X[feature_col].max():.1f}]", "#38BDF8")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Distribution Charts
    c_left, c_right = st.columns(2)

    plot_df = pd.DataFrame({
        feature_col: X[feature_col].values,
        "Operating State": np.where(y.values == 1, "Anomaly", "Normal"),
    })

    with c_left:
        st.markdown(f"#### 📊 Healthy vs. Anomaly Range: `{kpi_info.get('friendly', selected_kpi)}`")
        box_fig = create_kpi_distribution_plot(plot_df, feature_col, class_col="Operating State")
        st.plotly_chart(box_fig, use_container_width=True)

    with c_right:
        st.markdown(f"#### 📈 Value Distribution Overlay ({kpi_info.get('unit', '')})")
        hist_fig = px.histogram(
            plot_df,
            x=feature_col,
            color="Operating State",
            barmode="overlay",
            marginal="box",
            color_discrete_map={"Normal": "#10B981", "Anomaly": "#EF4444"},
            title=f"Histogram Density: {feature_col}",
            nbins=35,
        )
        hist_fig = apply_dark_chart_layout(hist_fig, height=320)
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 Operating Envelope & Percentile Reference Table (`{feature_col}`)")

    pct_df = pd.DataFrame({
        "Operating Percentile": [
            "Minimum (0%)",
            "5th Percentile (Lower Normal Bound)",
            "25th Percentile (Q1)",
            "Typical Baseline (Median 50%)",
            "75th Percentile (Q3)",
            "95th Percentile (Upper Normal Bound)",
            "Maximum (100%)",
        ],
        "Healthy Cell Windows (N=30,765)": [
            f"{normal_vals.min():.3f}",
            f"{normal_vals.quantile(0.05):.3f}",
            f"{normal_vals.quantile(0.25):.3f}",
            f"{normal_vals.median():.3f}",
            f"{normal_vals.quantile(0.75):.3f}",
            f"{normal_vals.quantile(0.95):.3f}",
            f"{normal_vals.max():.3f}",
        ],
        "Anomalous Cell Windows (N=1,235)": [
            f"{anomaly_vals.min():.3f}",
            f"{anomaly_vals.quantile(0.05):.3f}",
            f"{anomaly_vals.quantile(0.25):.3f}",
            f"{anomaly_vals.median():.3f}",
            f"{anomaly_vals.quantile(0.75):.3f}",
            f"{anomaly_vals.quantile(0.95):.3f}",
            f"{anomaly_vals.max():.3f}",
        ],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    render_academic_disclaimer()
