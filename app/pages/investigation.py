"""
Page 3: Inspect Alert (Clean Record Investigation View).
Shows clearly why a record was flagged, comparing its KPI values against normal network ranges.
"""

from typing import Any, Optional, Tuple
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    navigate_to,
    get_feature_meta,
    TELECOM_KPI_METADATA,
)
from ..components.charts import create_record_deviation_waterfall
from src.data_loader import load_processed_features, NUMERIC_KPIS
from src.prediction import AnomalyDetectorPipeline


@st.cache_data
def _load_sample_anomalies():
    try:
        X, y = load_processed_features()
        anom_idx = X.index[y == 1][:25]
        df = X.loc[anom_idx].copy()
        df["Record ID"] = [f"SAMPLE-ANOM-{i:03d}" for i in range(len(df))]
        df["anomaly_status"] = "Anomaly"
        df["Severity Level"] = "🔴 Anomaly Alert"
        df["anomaly_score_or_prob"] = 0.98
        return df
    except Exception:
        return None


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the clean Record Investigation screen."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Inspect Network Record",
        subtitle="See which KPI measurements went outside the normal range and what to check.",
        badge="ALERT DETAILS",
    )

    # Use session detection results if available; otherwise fall back to sample anomalies
    source_df: Optional[pd.DataFrame] = st.session_state.get("last_results")
    if source_df is None or len(source_df) == 0:
        source_df = _load_sample_anomalies()
        st.caption("ℹ️ Showing sample anomaly records. Run a scan in **⚡ Run Detection** to inspect your own data.")

    if source_df is None or len(source_df) == 0:
        st.info("No records loaded yet. Click below to run detection first.")
        if st.button("⚡ Go to Run Detection →"):
            navigate_to("⚡ Run Detection")
        return

    record_ids = list(source_df["Record ID"] if "Record ID" in source_df.columns else source_df.index)
    default_idx = 0
    saved_id = st.session_state.get("selected_record_id")
    if saved_id in record_ids:
        default_idx = record_ids.index(saved_id)
    elif "anomaly_prediction" in source_df.columns:
        anom_rows = source_df[source_df["anomaly_prediction"] == 1]
        if len(anom_rows) > 0:
            first_id = anom_rows["Record ID"].iloc[0]
            if first_id in record_ids:
                default_idx = record_ids.index(first_id)

    top_c1, top_c2 = st.columns([2.5, 1])
    with top_c1:
        selected_id = st.selectbox(
            "Select Record:",
            options=record_ids,
            index=default_idx,
            format_func=lambda rid: (
                f"{rid} ({source_df.loc[source_df['Record ID']==rid, 'Severity Level'].iloc[0]})"
                if "Severity Level" in source_df.columns
                else str(rid)
            ),
            label_visibility="collapsed",
        )
        st.session_state["selected_record_id"] = selected_id
    with top_c2:
        if st.button("⬅️ Back to Detection", use_container_width=True):
            navigate_to("⚡ Run Detection")

    record_row = source_df[source_df["Record ID"] == selected_id].iloc[0]
    status_label = str(record_row.get("Severity Level", record_row.get("anomaly_status", "Anomaly")))
    score_val = float(record_row.get("anomaly_score_or_prob", 1.0))

    try:
        exp_df = pipeline.explain_record(record_row, top_k=6)
    except Exception as e:
        st.error(f"Could not compute KPI comparison: {e}")
        return

    # Add friendly labels
    friendly_list = []
    status_list = []
    action_list = []
    for _, erow in exp_df.iterrows():
        meta = get_feature_meta(str(erow["Feature"]))
        friendly_list.append(meta["friendly_name"])
        action_list.append(meta["action"])
        z = float(erow["Z-Score (Dev.)"])
        if z > 2.0:
            status_list.append("🔴 High (Above Normal)")
        elif z < -2.0:
            status_list.append("🔴 Low (Below Normal)")
        elif abs(z) > 1.2:
            status_list.append("🟠 Slightly Unusual")
        else:
            status_list.append("🟢 Normal")

    exp_df["KPI Name"] = friendly_list
    exp_df["Simple Status"] = status_list
    exp_df["Action"] = action_list

    top_kpi = exp_df.iloc[0]

    # 3 Clean Summary Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Selected Record", str(selected_id), f"Risk Score: {score_val:.3f}", "#38BDF8")
    with c2:
        col = "#10B981" if "Normal" in status_label else "#EF4444"
        render_metric_card("Status", status_label, "AI Classification", col)
    with c3:
        render_metric_card("Primary KPI Issue", str(top_kpi["Feature"]), str(top_kpi["KPI Name"]), "#F59E0B")

    # Plain-English Explanation Box
    st.markdown(
        f"""
        <div class="telecom-card">
            <h3>💡 Why Was This Record Flagged?</h3>
            <p style="color: #E2E8F0; font-size: 0.92rem; margin-bottom: 6px;">
                • <strong>Main Deviation:</strong> <strong>{top_kpi['KPI Name']}</strong> recorded a value of
                <strong>{top_kpi['Observed Value']}</strong>, whereas normal network traffic stays within
                <strong>{top_kpi['Normal 5th-95th Range']}</strong> (normal average: {top_kpi['Normal Median']}).
            </p>
            <p style="color: #14B8A6; font-size: 0.9rem; margin-bottom: 0;">
                • <strong>Recommended Check:</strong> {top_kpi['Action']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-Side Chart & Simple Comparison Table
    ch_col, tb_col = st.columns([1, 1.1])
    with ch_col:
        fig = create_record_deviation_waterfall(exp_df)
        st.plotly_chart(fig, use_container_width=True)

    with tb_col:
        st.markdown("#### 📋 Top 6 KPIs Compared to Normal")
        simple_table = exp_df[[
            "KPI Name",
            "Observed Value",
            "Normal 5th-95th Range",
            "Simple Status",
        ]].copy()
        simple_table.columns = ["KPI Measurement", "Observed Value", "Normal Range", "Status"]
        st.dataframe(simple_table, use_container_width=True, hide_index=True, height=280)

    # Optional expander for all 16 KPIs
    with st.expander("📡 View All 16 Network KPIs for This Record"):
        all_rows = []
        for kpi in NUMERIC_KPIS:
            meta = TELECOM_KPI_METADATA.get(kpi, {})
            all_rows.append({
                "KPI": meta.get("friendly", kpi),
                "Code": kpi,
                "Unit": meta.get("unit", "-"),
                "Average": round(float(record_row.get(f"{kpi}_mean", 0.0)), 3),
                "Min": round(float(record_row.get(f"{kpi}_min", 0.0)), 3),
                "Max": round(float(record_row.get(f"{kpi}_max", 0.0)), 3),
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)
