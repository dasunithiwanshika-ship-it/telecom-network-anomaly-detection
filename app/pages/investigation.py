"""
Page 4: Anomaly Investigation Workspace (Diagnostic Root-Cause Exploration).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_engineering_notice,
)
from ..components.charts import create_record_deviation_waterfall
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS


def render(pipeline, dataset_tuple, benchmark_tables):
    """Render the Anomaly Investigation diagnostic workspace."""
    render_hero_banner(
        title="Anomaly Investigation & Root-Cause Workspace",
        subtitle="In-depth diagnostic triage: identify anomalous feature deviations, z-scores, and KPI anomalies",
    )
    render_engineering_notice()

    X, y = dataset_tuple

    # Selection: Choose source of record
    st.markdown("### 1. Select Trace Record for Forensic Inspection")

    source_options = ["🔍 Recent Detection Batch (Session State)", "📚 Historical Research Benchmark Anomalies"]
    if "last_results" not in st.session_state:
        source_options = ["📚 Historical Research Benchmark Anomalies"]

    chosen_source = st.radio("Select Telemetry Pool:", source_options, horizontal=True)

    target_df = None
    if "Recent" in chosen_source and "last_results" in st.session_state:
        target_df = st.session_state["last_results"]
    elif X is not None:
        # Default to historical anomalous traces from test set
        anomalies_idx = X.index[y == 1]
        target_df = X.loc[anomalies_idx].copy()
        target_df["Record ID"] = [f"HIST-ANOMALY-{idx:05d}" for idx in anomalies_idx]
        target_df["anomaly_status"] = "Anomaly (Historical Ground Truth)"
        target_df["anomaly_score_or_prob"] = 1.0
        target_df["score_type"] = "Verified Historical Target Label"

    if target_df is None or len(target_df) == 0:
        st.warning("No records available to investigate. Run detection in the Anomaly Detection workspace first.")
        return

    # Record selector dropdown
    record_ids = list(target_df["Record ID"] if "Record ID" in target_df.columns else target_df.index)
    selected_record_id = st.selectbox(
        "Choose Specific Record to Triage:",
        options=record_ids,
        index=0,
    )

    # Locate chosen row
    if "Record ID" in target_df.columns:
        record_row = target_df[target_df["Record ID"] == selected_record_id].iloc[0]
    else:
        record_row = target_df.loc[selected_record_id]

    status = record_row.get("anomaly_status", "Anomaly")
    score_val = record_row.get("anomaly_score_or_prob", 1.0)
    score_type = record_row.get("score_type", "Anomaly Metric")

    # Diagnostic Summary Cards
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Record Identifier", str(selected_record_id), "Telemetry trace", "#38BDF8")
    with c2:
        val_col = "#EF4444" if "Anomaly" in str(status) else "#10B981"
        render_metric_card("Diagnostic Status", str(status), "AI Classification", val_col)
    with c3:
        render_metric_card("Anomaly Score / Prob", f"{float(score_val):.4f}", str(score_type), "#F59E0B")
    with c4:
        engine_name = st.session_state.get("last_model_id", "random_forest").split('_')[0].upper()
        render_metric_card("Diagnosing Model", engine_name, "Evaluator engine", "#14B8A6")

    st.markdown("---")

    # Core Investigation Section: Z-Score Statistical Deviations
    st.markdown("### 2. Forensic KPI Deviation Analysis")
    st.markdown(
        r"""
        To discover **why** this record was flagged, we measure how many standard deviations ($\sigma$)
        each engineered KPI moment departs from the verified normal cellular baseline ($N=30,765$ normal traces):
        $$z_j = \frac{x_j - \mu_{\text{normal}, j}}{\sigma_{\text{normal}, j}}$$
        """
    )

    try:
        explanation_df = pipeline.explain_record(record_row, top_k=8)

        # Plot Waterfall chart of deviations
        chart_col, table_col = st.columns([1.1, 1])

        with chart_col:
            dev_fig = create_record_deviation_waterfall(explanation_df)
            st.plotly_chart(dev_fig, use_container_width=True)

        with table_col:
            st.markdown("#### 📋 Top Deviating KPI Attributes")
            display_exp = explanation_df[[
                "Feature", "Observed Value", "Normal Median", "Normal 5th-95th Range", "Z-Score (Dev.)", "Status"
            ]]
            st.dataframe(display_exp, use_container_width=True, hide_index=True)

        # Natural Language Triage Synthesis
        st.markdown("### 3. Engineering Triage Summary")
        top_feature = explanation_df.iloc[0]
        st.markdown(
            f"""
            <div class="telecom-card">
                <h3>🔍 Diagnostic Findings</h3>
                <ul>
                    <li><strong>Primary Anomaly Driver:</strong> <code>{top_feature['Feature']}</code> exhibited an observed value of 
                    <strong>{top_feature['Observed Value']}</strong>, which is <strong>{top_feature['Z-Score (Dev.)']:+.2f} standard deviations</strong> 
                    away from normal cellular operations (Normal median: {top_feature['Normal Median']}).</li>
                    <li><strong>Cellular Impact:</strong> Telemetry values in this range represent a <strong>{top_feature['Status']}</strong> condition.</li>
                    <li><strong>Verification Recommendation:</strong> Inspect base-station physical layer logs and scheduling grants for this sector to determine if channel fading or burst queueing was present.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Could not compute record explanation: {e}")
