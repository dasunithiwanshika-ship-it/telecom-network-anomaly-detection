"""
Page 5: Anomaly Investigation & Root-Cause Diagnostic Screen.
Dedicated detail view for inspecting an individual cell window:
  - KPI measurements across all 7 telecom subsystems
  - Anomaly risk score & severity badge
  - Comparison with normal cellular baseline (5th-95th percentile & median)
  - Plain-English AI explanations and NOC troubleshooting actions
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_engineering_notice,
    navigate_to,
    get_feature_meta,
    TELECOM_KPI_METADATA,
)
from ..components.charts import create_record_deviation_waterfall
from src.data_loader import load_processed_features, NUMERIC_KPIS
from src.prediction import AnomalyDetectorPipeline


@st.cache_data
def _load_historical_anomalies():
    try:
        X, y = load_processed_features()
        anom_idx = X.index[y == 1][:50]
        df = X.loc[anom_idx].copy()
        df["Record ID"] = [f"HIST-ANOM-{idx:05d}" for idx in anom_idx]
        df["anomaly_status"] = "Anomaly"
        df["Severity Level"] = "🔴 CRITICAL ALERT (Verified Historical)"
        df["anomaly_score_or_prob"] = 0.99
        df["score_type"] = "Historical Benchmark Fault"
        return df
    except Exception:
        return None


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the dedicated Anomaly Investigation detail screen."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Anomaly Investigation & Root-Cause Triage",
        subtitle="Detailed cell-window diagnostic view: compare observed KPI measurements against normal network baselines and review AI fault attribution.",
        badge="INCIDENT FORENSICS",
    )

    # Top Navigation & Record Source Selector
    has_session_results = "last_results" in st.session_state and st.session_state["last_results"] is not None

    top_c1, top_c2, top_c3 = st.columns([1.4, 1.4, 0.8])

    with top_c1:
        source_choices = []
        if has_session_results:
            source_choices.append("🚨 Active Detection Batch Alerts")
        source_choices.append("📚 Reference Historical Network Faults (50 Verified Samples)")

        chosen_source = st.selectbox(
            "Telemetry Source Pool:",
            options=source_choices,
            index=0,
            help="Inspect records from your recent AI Detection run or explore verified historical faults from the TelecomTS dataset.",
        )

    if "Active Detection" in chosen_source and has_session_results:
        source_df: pd.DataFrame = st.session_state["last_results"]
    else:
        source_df = _load_historical_anomalies()

    if source_df is None or len(source_df) == 0:
        st.warning("No telemetry records available to inspect. Please run a scan in the **AI Detection Workspace** first.")
        if st.button("⚡ Go to AI Detection Workspace →"):
            navigate_to("⚡ AI Detection Workspace")
        return

    record_ids = list(source_df["Record ID"] if "Record ID" in source_df.columns else source_df.index)
    default_idx = 0
    saved_rec_id = st.session_state.get("selected_record_id")
    if saved_rec_id in record_ids:
        default_idx = record_ids.index(saved_rec_id)
    elif "anomaly_prediction" in source_df.columns:
        # Default to first flagged anomaly if available
        anom_rows = source_df[source_df["anomaly_prediction"] == 1]
        if len(anom_rows) > 0 and "Record ID" in anom_rows.columns:
            first_anom_id = anom_rows["Record ID"].iloc[0]
            if first_anom_id in record_ids:
                default_idx = record_ids.index(first_anom_id)

    with top_c2:
        selected_record_id = st.selectbox(
            "Select Cell Window Record to Investigate:",
            options=record_ids,
            index=default_idx,
            format_func=lambda rid: (
                f"{rid} — {source_df.loc[source_df['Record ID']==rid, 'Severity Level'].iloc[0]}"
                if "Severity Level" in source_df.columns
                else str(rid)
            ),
        )
        st.session_state["selected_record_id"] = selected_record_id

    with top_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Alerts", use_container_width=True, key="btn_inv_back"):
            navigate_to("⚡ AI Detection Workspace")

    # Locate the selected record row
    if "Record ID" in source_df.columns:
        record_row = source_df[source_df["Record ID"] == selected_record_id].iloc[0]
    else:
        record_row = source_df.loc[selected_record_id]

    status = str(record_row.get("anomaly_status", "Anomaly"))
    severity = str(record_row.get("Severity Level", "🔴 CRITICAL ALERT" if "Anomaly" in status else "🟢 NORMAL"))
    score_val = float(record_row.get("anomaly_score_or_prob", 1.0))
    score_type = str(record_row.get("score_type", "Anomaly Risk Score"))

    # Compute KPI deviations against normal cellular population
    try:
        explanation_df = pipeline.explain_record(record_row, top_k=10)
    except Exception as e:
        st.error(f"Could not compute baseline comparison: {e}")
        return

    # Enrich explanation_df with friendly telecom names & subsystem metadata
    friendly_names = []
    subsystems = []
    units = []
    actions = []
    plain_statuses = []

    for _, erow in explanation_df.iterrows():
        fmeta = get_feature_meta(str(erow["Feature"]))
        friendly_names.append(fmeta["friendly_name"])
        subsystems.append(fmeta["subsystem"])
        units.append(fmeta["unit"])
        actions.append(fmeta["action"])
        z = float(erow["Z-Score (Dev.)"])
        if z >= 3.0:
            plain_statuses.append("🔴 Severely Above Normal")
        elif z >= 1.5:
            plain_statuses.append("🟠 Above Normal Range")
        elif z <= -3.0:
            plain_statuses.append("🔴 Severely Below Normal")
        elif z <= -1.5:
            plain_statuses.append("🟠 Below Normal Range")
        else:
            plain_statuses.append("🟢 Within Normal Limits")

    explanation_df["Telecom Measurement"] = friendly_names
    explanation_df["Subsystem"] = subsystems
    explanation_df["Unit"] = units
    explanation_df["Recommended Action"] = actions
    explanation_df["Operational Status"] = plain_statuses

    top_driver = explanation_df.iloc[0]
    second_driver = explanation_df.iloc[1] if len(explanation_df) > 1 else top_driver

    # ------------------------------------------------------------------
    # SECTION 1: INCIDENT SUMMARY METRIC CARDS
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Investigated Record", str(selected_record_id), "12.8s Cell Telemetry Window", "#38BDF8")
    with c2:
        badge_col = "#EF4444" if "Anomaly" in status or "CRITICAL" in severity or "HIGH" in severity else "#10B981"
        render_metric_card("Alert Status", severity, status, badge_col)
    with c3:
        render_metric_card("AI Risk Score", f"{score_val:.4f}", score_type, "#F59E0B")
    with c4:
        render_metric_card("Primary Fault Domain", str(top_driver["Subsystem"]), f"Driver: {top_driver['Feature']}", "#14B8A6")

    # ------------------------------------------------------------------
    # SECTION 2: AUTOMATED TELECOM ENGINEERING DIAGNOSTIC BRIEF
    # ------------------------------------------------------------------
    st.markdown("### 🛠️ 1. Automated Engineering Diagnostic Summary")

    is_healthy = "Normal" in status and "NORMAL" in severity
    if is_healthy:
        st.markdown(
            f"""
            <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-left: 5px solid #10B981;
                        border-radius: 10px; padding: 18px 22px; margin-bottom: 20px;">
                <div style="font-weight: 700; color: #10B981; font-size: 1.05rem; margin-bottom: 6px;">
                    🟢 Healthy Cell Window — No Fault Signatures Detected
                </div>
                <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.55;">
                    Record <strong>{selected_record_id}</strong> exhibits normal radio signal quality, expected block error rates, and balanced resource scheduling.
                    Its largest fluctuation is in <strong>{top_driver['Telecom Measurement']}</strong> (<code>{top_driver['Feature']}</code> = {top_driver['Observed Value']}),
                    which remains consistent with healthy baseline operations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #172235; border: 1px solid #EF4444; border-left: 5px solid #EF4444;
                        border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
                <div style="font-weight: 700; color: #EF4444; font-size: 1.08rem; margin-bottom: 10px;">
                    🚨 AI Root-Cause Triage Brief for {selected_record_id}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; color: #E2E8F0; font-size: 0.88rem; line-height: 1.55;">
                    <div style="background-color: #111827; padding: 14px; border-radius: 8px; border: 1px solid #1E293B;">
                        <div style="color: #38BDF8; font-weight: 700; margin-bottom: 4px;">📌 What Deviated From Normal Baseline?</div>
                        <ul style="margin: 0; padding-left: 18px; color: #CBD5E1;">
                            <li><strong>Primary Driver:</strong> <strong>{top_driver['Telecom Measurement']}</strong> (<code>{top_driver['Feature']}</code>) recorded a value of <strong>{top_driver['Observed Value']}</strong> vs. the normal baseline median of <strong>{top_driver['Normal Median']}</strong> (Normal 5th–95th range: <code>{top_driver['Normal 5th-95th Range']}</code>).</li>
                            <li style="margin-top: 4px;"><strong>Secondary Driver:</strong> <strong>{second_driver['Telecom Measurement']}</strong> (<code>{second_driver['Feature']}</code>) recorded <strong>{second_driver['Observed Value']}</strong> vs. normal median <strong>{second_driver['Normal Median']}</strong>.</li>
                        </ul>
                    </div>
                    <div style="background-color: #111827; padding: 14px; border-radius: 8px; border: 1px solid #1E293B;">
                        <div style="color: #14B8A6; font-weight: 700; margin-bottom: 4px;">🔧 Recommended NOC Engineering Checks</div>
                        <ul style="margin: 0; padding-left: 18px; color: #CBD5E1;">
                            <li><strong>Primary Subsystem ({top_driver['Subsystem']}):</strong> {top_driver['Recommended Action']}</li>
                            <li style="margin-top: 4px;"><strong>Secondary Subsystem ({second_driver['Subsystem']}):</strong> {second_driver['Recommended Action']}</li>
                        </ul>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # SECTION 3: KPI COMPARISON WITH NORMAL OBSERVATIONS
    # ------------------------------------------------------------------
    st.markdown(
        "### 📊 2. KPI Comparison Against Normal Network Baseline",
        help="Compares this record's measurements against 30,765 verified normal cellular traces. Deviation is measured in units of normal standard deviation (σ).",
    )

    chart_col, table_col = st.columns([1.05, 1.15])

    with chart_col:
        dev_fig = create_record_deviation_waterfall(explanation_df.head(8))
        st.plotly_chart(dev_fig, use_container_width=True)
        st.caption(
            "💡 **How to Read This Chart:** Bars extending past **+2.0σ** or **-2.0σ** indicate KPIs that have departed significantly from normal cell behavior."
        )

    with table_col:
        st.markdown("#### 📋 Top Deviating Telemetry Measurements")
        comp_table = explanation_df[[
            "Telecom Measurement",
            "Observed Value",
            "Normal Median",
            "Normal 5th-95th Range",
            "Operational Status",
            "Z-Score (Dev.)",
        ]].copy()
        comp_table.columns = [
            "KPI Measurement",
            "Observed Value",
            "Normal Baseline (Median)",
            "Normal Range (5th–95th %)",
            "Status vs Normal",
            "Deviation (σ)",
        ]
        st.dataframe(comp_table, use_container_width=True, hide_index=True, height=340)

    st.markdown("---")

    # ------------------------------------------------------------------
    # SECTION 4: FULL 16-KPI TELECOM SUBSYSTEM INSPECTOR
    # ------------------------------------------------------------------
    st.markdown("### 📡 3. Complete 16-KPI Subsystem Measurements for This Record")
    st.caption("Inspect all 16 cellular KPIs (Average, Std Dev, Minimum, Maximum) recorded during this 12.8-second cell window:")

    norm_mean = pipeline.baseline_stats["mean"] if pipeline.baseline_stats else None
    norm_q05 = pipeline.baseline_stats["q05"] if pipeline.baseline_stats else None
    norm_q95 = pipeline.baseline_stats["q95"] if pipeline.baseline_stats else None

    kpi_rows = []
    for base_kpi in NUMERIC_KPIS:
        meta = TELECOM_KPI_METADATA.get(base_kpi, {})
        val_mean = float(record_row.get(f"{base_kpi}_mean", 0.0))
        val_std = float(record_row.get(f"{base_kpi}_std", 0.0))
        val_min = float(record_row.get(f"{base_kpi}_min", 0.0))
        val_max = float(record_row.get(f"{base_kpi}_max", 0.0))

        norm_range_str = "-"
        if norm_q05 is not None and norm_q95 is not None and f"{base_kpi}_mean" in norm_q05:
            q05_v = float(norm_q05[f"{base_kpi}_mean"])
            q95_v = float(norm_q95[f"{base_kpi}_mean"])
            norm_range_str = f"[{q05_v:.2f} to {q95_v:.2f}]"

        kpi_rows.append({
            "Subsystem": meta.get("subsystem", "Telemetry"),
            "KPI Channel": f"{meta.get('friendly', base_kpi)} ({base_kpi})",
            "Unit": meta.get("unit", "-"),
            "Observed Avg": round(val_mean, 4),
            "Normal Avg Range (5th–95th)": norm_range_str,
            "Window Min": round(val_min, 4),
            "Window Peak (Max)": round(val_max, 4),
            "Fluctuation (Std)": round(val_std, 4),
        })

    kpi_full_df = pd.DataFrame(kpi_rows)
    sub_filter = st.selectbox(
        "Filter by Telecom Subsystem:",
        options=["All 7 Subsystems (16 KPIs)"] + sorted(kpi_full_df["Subsystem"].unique().tolist()),
    )
    if sub_filter != "All 7 Subsystems (16 KPIs)":
        kpi_full_df = kpi_full_df[kpi_full_df["Subsystem"] == sub_filter]

    st.dataframe(kpi_full_df, use_container_width=True, hide_index=True, height=340)

    # Bottom Actions: Export Report for this record
    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        # Generate single-record diagnostic summary text for download
        single_report = f"""# TELECOM AI INCIDENT INVESTIGATION REPORT
Record ID: {selected_record_id}
Alert Severity: {severity} ({status})
AI Risk Score: {score_val:.4f} ({score_type})
Primary Affected Subsystem: {top_driver['Subsystem']}

## 1. TOP DEVIATING KPI MEASUREMENTS
- Primary Driver: {top_driver['Telecom Measurement']} ({top_driver['Feature']})
  Observed: {top_driver['Observed Value']} | Normal Median: {top_driver['Normal Median']} | Normal Range: {top_driver['Normal 5th-95th Range']} | Deviation: {top_driver['Z-Score (Dev.)']:+.2f} sigma
  Recommended Check: {top_driver['Recommended Action']}

- Secondary Driver: {second_driver['Telecom Measurement']} ({second_driver['Feature']})
  Observed: {second_driver['Observed Value']} | Normal Median: {second_driver['Normal Median']} | Normal Range: {second_driver['Normal 5th-95th Range']} | Deviation: {second_driver['Z-Score (Dev.)']:+.2f} sigma
  Recommended Check: {second_driver['Recommended Action']}

## 2. TOP 8 KPI DEVIATIONS TABLE
{comp_table.to_string(index=False)}
"""
        st.download_button(
            label=f"⬇️ Download Diagnostic Brief for {selected_record_id} (.txt)",
            data=single_report.encode("utf-8"),
            file_name=f"incident_report_{selected_record_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with b_col2:
        if st.button("📑 Open Full Batch Reports & Export Center →", use_container_width=True, key="btn_inv_to_reports"):
            navigate_to("📑 Reports & Export")

    render_engineering_notice()
