"""
Page 4: KPIs & Reports.
Clean 2-tab view combining the interactive KPI Explorer and 1-click Report Downloads.
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    TELECOM_KPI_METADATA,
)
from ..components.charts import (
    create_kpi_distribution_plot,
    apply_dark_chart_layout,
)
from src.data_loader import NUMERIC_KPIS, load_processed_features
from . import reports


@st.cache_data
def _load_kpi_baseline():
    try:
        return load_processed_features()
    except Exception:
        return None, None


def render(pipeline: Any = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the clean KPIs & Reports page."""
    render_hero_banner(
        title="Network KPIs & Reports",
        subtitle="Explore how individual telecom KPIs behave during normal vs. anomalous conditions, or download analysis reports.",
        badge="KPIS & REPORTS",
    )

    tab_kpi, tab_rep = st.tabs([
        "📊 1. Explore Network KPIs",
        "📑 2. Download Reports & CSV",
    ])

    with tab_kpi:
        if dataset_tuple is not None:
            X, y = dataset_tuple
        else:
            X, y = _load_kpi_baseline()

        if X is None or y is None:
            st.error("Processed KPI dataset not found.")
            return

        c_sel1, c_sel2 = st.columns([1.8, 1])
        with c_sel1:
            selected_kpi = st.selectbox(
                "Select a Network KPI to Inspect:",
                options=NUMERIC_KPIS,
                format_func=lambda k: f"{TELECOM_KPI_METADATA.get(k, {}).get('friendly', k)} ({k})",
            )
        with c_sel2:
            selected_moment = st.selectbox(
                "Statistic:",
                options=["mean", "max", "min", "std"],
                format_func=lambda m: {"mean": "Average (Mean)", "max": "Peak (Max)", "min": "Minimum (Min)", "std": "Fluctuation (Std)"}[m],
            )

        kpi_info = TELECOM_KPI_METADATA.get(selected_kpi, {})
        st.caption(f"📖 **What it measures:** {kpi_info.get('meaning', '')}")

        feature_col = f"{selected_kpi}_{selected_moment}"
        normal_vals = X.loc[y == 0, feature_col]
        anomaly_vals = X.loc[y == 1, feature_col]

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Normal Average", f"{normal_vals.mean():.2f}", f"Typical range: [{normal_vals.quantile(0.05):.1f}, {normal_vals.quantile(0.95):.1f}]", "#10B981")
        with m2:
            render_metric_card("Anomaly Average", f"{anomaly_vals.mean():.2f}", f"Anomaly range: [{anomaly_vals.quantile(0.05):.1f}, {anomaly_vals.quantile(0.95):.1f}]", "#EF4444")
        with m3:
            render_metric_card("Unit & Subsystem", kpi_info.get("unit", "-"), kpi_info.get("subsystem", ""), "#38BDF8")

        plot_df = pd.DataFrame({
            feature_col: X[feature_col].values,
            "State": np.where(y.values == 1, "Anomaly", "Normal"),
        })

        ch1, ch2 = st.columns(2)
        with ch1:
            box_fig = create_kpi_distribution_plot(plot_df, feature_col, class_col="State")
            st.plotly_chart(box_fig, use_container_width=True)
        with ch2:
            hist_fig = px.histogram(
                plot_df,
                x=feature_col,
                color="State",
                barmode="overlay",
                color_discrete_map={"Normal": "#10B981", "Anomaly": "#EF4444"},
                title=f"Distribution: {feature_col}",
                nbins=30,
            )
            hist_fig = apply_dark_chart_layout(hist_fig, height=320)
            st.plotly_chart(hist_fig, use_container_width=True)

    with tab_rep:
        reports.render(pipeline, show_header=False)
