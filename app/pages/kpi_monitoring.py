"""
Page 05: Alert Management & Anomaly Viewing Console.
Provides real-time incident triage, severity distribution breakdowns,
temporal anomaly spike timelines, and interactive triage tables.
"""

from typing import Any, Optional, Tuple
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    navigate_to,
)
from ..components.charts import (
    create_donut_class_distribution,
    create_temporal_alert_spikes,
    create_severity_breakdown_bars,
)
from src.feature_engineering import create_synthetic_demo_features
from src.prediction import AnomalyDetectorPipeline
from .detection import _compute_record_severity_and_driver


@st.cache_data
def _generate_default_scan_results():
    pipeline = AnomalyDetectorPipeline()
    demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
    preds = pipeline.predict(demo_df, model_id="random_forest", threshold=0.30)
    preds["Record ID"] = [f"CELL-TRACE-{i+1:04d}" for i in range(len(preds))]
    return _compute_record_severity_and_driver(pipeline, preds, "Supervised", 0.30)


def render(pipeline: Optional[Any] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the Alert Management and Anomaly Viewing screen."""
    st.markdown(
        """
        <div class="noc-topbar">
            <div>
                <span style="background: rgba(239, 68, 68, 0.1); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.25);
                             font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                    05 — ALERT MANAGEMENT CONSOLE
                </span>
                <div style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em; margin-top: 4px;">
                    Real-Time Incident Triage
                </div>
            </div>
            <div style="display: flex; gap: 10px;">
                <span style="background: #0E1524; border: 1px solid #1A263D; color: #64748B; padding: 5px 12px; border-radius: 4px; font-size: 0.78rem; font-weight: 600;">
                    Telemetry Refresh: Active
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_df: Optional[pd.DataFrame] = st.session_state.get("last_results")
    if results_df is None or len(results_df) == 0:
        results_df = _generate_default_scan_results()
        st.session_state["last_results"] = results_df
        st.caption("Displaying default field scan. Execute a custom scan in Stage 04 to evaluate other telemetry.")

    total_records = len(results_df)
    anom_mask = (results_df["anomaly_prediction"] == 1).values if "anomaly_prediction" in results_df.columns else [False] * total_records
    anom_count = int(sum(anom_mask))
    normal_count = total_records - anom_count
    crit_count = sum(1 for s in results_df.get("Severity Level", []) if "Critical" in str(s))
    high_count = sum(1 for s in results_df.get("Severity Level", []) if "High" in str(s))

    # 1. Summary Ribbon
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Scanned Windows", f"{total_records}", "Active observation batch", "#38BDF8")
    with c2:
        render_metric_card("Nominal Traffic", f"{normal_count}", f"{(normal_count/total_records)*100:.1f}% within baseline", "#10B981")
    with c3:
        render_metric_card("Flagged Anomalies", f"{anom_count}", f"{(anom_count/total_records)*100:.1f}% incident rate", "#EF4444")
    with c4:
        render_metric_card("High & Critical Priority", f"{crit_count + high_count}", "Immediate engineer triage required", "#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Visual Analytics Strip
    v_col1, v_col2, v_col3 = st.columns([1, 1.6, 1])

    with v_col1:
        fig_donut = create_donut_class_distribution(normal_count, anom_count)
        st.plotly_chart(fig_donut, use_container_width=True)

    with v_col2:
        fig_spikes = create_temporal_alert_spikes(results_df)
        st.plotly_chart(fig_spikes, use_container_width=True)

    with v_col3:
        fig_sev = create_severity_breakdown_bars(results_df)
        st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Triage Table
    st.markdown("### Incident Triage & Classification")

    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
    with f_col1:
        search_query = st.text_input("Search Telemetry:", placeholder="Search by Record ID or KPI...", label_visibility="collapsed")
    with f_col2:
        filter_status = st.selectbox("Filter Severity:", ["All Severities", "Critical & High Risk Only", "Nominal Records Only"], label_visibility="collapsed")
    with f_col3:
        sort_by = st.selectbox("Sort Order:", ["Risk Score (Descending)", "Record ID", "Severity"], label_visibility="collapsed")

    display_df = results_df.copy()

    if filter_status == "Critical & High Risk Only":
        display_df = display_df[display_df["anomaly_prediction"] == 1]
    elif filter_status == "Nominal Records Only":
        display_df = display_df[display_df["anomaly_prediction"] == 0]

    if search_query:
        sq = search_query.lower()
        display_df = display_df[
            display_df["Record ID"].astype(str).str.lower().str.contains(sq)
            | display_df["Primary KPI Deviation"].astype(str).str.lower().str.contains(sq)
        ]

    if sort_by == "Risk Score (Descending)":
        display_df = display_df.sort_values(by="anomaly_score_or_prob", ascending=False)
    elif sort_by == "Record ID":
        display_df = display_df.sort_values(by="Record ID", ascending=True)

    cols_to_show = ["Record ID", "Severity Level", "Primary KPI Deviation", "anomaly_score_or_prob"]
    col_names_map = {
        "Record ID": "Cell Telemetry ID",
        "Severity Level": "NOC Severity Status",
        "Primary KPI Deviation": "Primary Root-Cause Trigger",
        "anomaly_score_or_prob": "AI Anomaly Confidence",
    }

    sub_display = display_df[cols_to_show].rename(columns=col_names_map)
    st.dataframe(sub_display, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Target Record Selection
    with st.container(border=True):
        st.markdown(
            """
            <h3 style="margin-top: 0; margin-bottom: 6px; font-size: 1rem;">Targeted Record Investigation</h3>
            <p style="color: #64748B; font-size: 0.84rem; margin-top: 0; margin-bottom: 12px;">
                Select a telemetry window to inspect multi-axis radar profiles, deviation waterfalls, and field troubleshooting guidance:
            </p>
            """,
            unsafe_allow_html=True,
        )

        c_sel, c_jump = st.columns([2.5, 1])
        available_ids = list(display_df["Record ID"]) if len(display_df) > 0 else list(results_df["Record ID"])
        
        with c_sel:
            selected_record = st.selectbox(
                "Select Record for Deep-Dive:",
                options=available_ids,
                index=0,
                key="alert_triage_record_select",
            )
            st.session_state["selected_record_id"] = selected_record

        with c_jump:
            st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
            if st.button("Investigate Record (Stage 06) →", use_container_width=True, key="btn_jump_to_investigate"):
                navigate_to("06 — Investigation & Export")
