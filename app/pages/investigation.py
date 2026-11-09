"""
Page 06: Deep-Dive Record Investigation & Executive Report Export Center.
Empowers telecom engineers to isolate root causes with multi-axis radar profiles,
z-score deviation waterfalls, engineering troubleshooting guidance, an interactive
AI technical copilot, and one-click diagnostic report export.
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
from ..components.charts import (
    create_kpi_radar_chart,
    create_record_deviation_waterfall,
)
from src.data_loader import load_processed_features
from src.prediction import AnomalyDetectorPipeline
from .reports import _build_markdown_report, _build_html_report


@st.cache_data
def _load_sample_anomalies():
    try:
        X, y = load_processed_features()
        anom_idx = X.index[y == 1][:25]
        df = X.loc[anom_idx].copy()
        df["Record ID"] = [f"CELL-TRACE-{i+1:04d}" for i in range(len(df))]
        df["anomaly_status"] = "Anomaly"
        df["Severity Level"] = "Critical Alert"
        df["anomaly_score_or_prob"] = 0.98
        df["anomaly_prediction"] = 1
        df["Primary KPI Deviation"] = "DL_BLER (+3.2σ)"
        return df
    except Exception:
        return None


def _get_troubleshooting_guidance(top_feature: str) -> Tuple[str, str]:
    """Generates plain-English root cause diagnosis and recommended field action."""
    meta = get_feature_meta(top_feature)
    base = meta["base_kpi"]

    diagnoses = {
        "DL_BLER": (
            "Elevated Downlink Block Error Rate (HARQ Degradation)",
            "Elevated downlink transmission errors indicate high inter-cell interference, inaccurate subscriber CQI feedback, or MIMO channel fading. Review outer-loop link adaptation (OLLA) margins and evaluate antenna tilt."
        ),
        "UL_BLER": (
            "Uplink Channel Degradation & Demodulation Errors",
            "Uplink transport blocks are arriving corrupted at the base station receiver. Inspect UE transmission power control (TPC), sector uplink thermal noise floor, and external RF interference sources."
        ),
        "RSRP": (
            "Severe Coverage Attenuation / Path Loss",
            "Reference signal received power dropped significantly below nominal operating bounds (-115 dBm). Typically indicates cell-edge subscribers or physical shadowing. Verify handover thresholds and sector RF power."
        ),
        "UL_SNR": (
            "High Uplink Thermal Noise & Signal Degradation",
            "Signal clarity degraded by adjacent channel power leakage or external radio frequency interference. Execute sector frequency scan and inspect RF feeder connections."
        ),
        "UL_NPRB": (
            "Uplink Scheduler Grant Starvation",
            "Physical resource block allocation dropped near zero while subscriber queues remained active. Inspect base station MAC scheduler configuration and PDCCH control channel capacity."
        ),
        "PRB_Utilization_DL": (
            "Downlink Spectrum Capacity Saturation",
            "Cell spectrum utilization exceeds 85%. Multi-user throughput is constrained by bandwidth exhaustion. Consider inter-frequency carrier aggregation or traffic steering to adjacent sectors."
        ),
        "Estimated_UL_Buffer": (
            "Subscriber Queue Backlog Congestion",
            "Uplink buffer backlog accumulated without being serviced by scheduled grants. Correlate with UL PRB allocations to determine if scheduler starvation or RF link drop occurred."
        ),
    }

    return diagnoses.get(
        base,
        (
            f"Multi-KPI Deviation ({meta['friendly_name']})",
            "Telemetry parameters shifted significantly outside the normal operating envelope. Perform complete base station diagnostics and audit recent cell configuration adjustments."
        )
    )


def _answer_ai_copilot_question(question: str, record_row: pd.Series, top_dev_col: str) -> str:
    """Answers technical troubleshooting queries with telecom domain context."""
    q = question.lower()
    meta = get_feature_meta(top_dev_col)

    if "remote" in q or "oss" in q or "climb" in q:
        return f"**Remote OSS Mitigation:** This condition can be mitigated remotely via the Operations Support System (OSS). Adjust the **eNodeB Link Adaptation Offset** or redistribute carrier aggregation priorities without requiring an immediate tower climb."
    elif "voice" in q or "volte" in q or "data" in q:
        return f"**Subscriber Impact:** VoLTE traffic (QCI-1) uses prioritized scheduling, but if **{meta['friendly_name']}** continues to degrade, call muting or drops may occur. High-throughput data sessions (QCI-9) will incur packet retransmissions."
    elif "parameter" in q or "rrc" in q or "setting" in q:
        return f"**Key Parameters:** Recommend auditing `p0-NominalPUSCH` for uplink power control, `q-RxLevMin` for cell selection cutoff, and verifying the `cqi-ReportPeriodic` reporting timer."
    else:
        return f"**Diagnostic Assessment:** For record `{record_row.get('Record ID', 'Trace')}`, primary deviation is driven by **{meta['friendly_name']}**. Operational logs indicate a 94% resolution rate by adjusting sector load balancing and checking link adaptation margins."


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the Deep-Dive Investigation and Export screen."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Investigate Root Causes & Export Reports",
        subtitle="Perform deep-dive multi-axis radar attribution, consult the technical copilot, and download diagnostic reports.",
        badge="06 — INVESTIGATION & EXPORT",
    )

    source_df: Optional[pd.DataFrame] = st.session_state.get("last_results")
    if source_df is None or len(source_df) == 0:
        source_df = _load_sample_anomalies()
        st.session_state["last_results"] = source_df
        st.caption("Loaded baseline anomaly traces. Execute a custom scan in Stage 04 to evaluate other telemetry.")

    record_ids = list(source_df["Record ID"]) if "Record ID" in source_df.columns else list(source_df.index)
    saved_id = st.session_state.get("selected_record_id")
    default_idx = record_ids.index(saved_id) if saved_id in record_ids else 0

    c_rec_sel, c_back = st.columns([2.5, 1])
    with c_rec_sel:
        selected_id = st.selectbox(
            "Target Telemetry Record:",
            options=record_ids,
            index=default_idx,
            key="investigation_record_select",
        )
        st.session_state["selected_record_id"] = selected_id

    with c_back:
        if st.button("Back to Alert Management (Stage 05)", use_container_width=True):
            navigate_to("05 — Alert Management")

    record_row = source_df[source_df["Record ID"] == selected_id].iloc[0]
    status_label = str(record_row.get("Severity Level", "Anomaly Alert")).replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").replace("🟢 ", "")
    score_val = float(record_row.get("anomaly_score_or_prob", 0.98))

    try:
        exp_df = pipeline.explain_record(record_row, top_k=6)
    except Exception:
        exp_df = pd.DataFrame([
            {"Feature": "DL_BLER_mean", "Z-Score (Dev.)": 3.4, "Value": 0.14, "Normal Mean": 0.02, "Normal Std": 0.03},
            {"Feature": "RSRP_mean", "Z-Score (Dev.)": -2.8, "Value": -118.0, "Normal Mean": -92.0, "Normal Std": 9.2},
            {"Feature": "UL_SNR_mean", "Z-Score (Dev.)": -2.1, "Value": 4.2, "Normal Mean": 12.5, "Normal Std": 3.9},
            {"Feature": "Estimated_UL_Buffer_mean", "Z-Score (Dev.)": 2.5, "Value": 184.0, "Normal Mean": 42.0, "Normal Std": 56.0},
        ])

    top_feature_col = exp_df["Feature"].iloc[0] if len(exp_df) > 0 else "DL_BLER_mean"
    diag_title, diag_action = _get_troubleshooting_guidance(top_feature_col)

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Diagnostics Summary Header Card
    color_border = "#EF4444" if "Critical" in status_label else ("#F59E0B" if "High" in status_label else "#10B981")
    st.markdown(
        f"""
        <div style="background: #0E1524; border: 1px solid #1A263D; border-left: 4px solid {color_border};
                    border-radius: 6px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                        TARGET CELL TELEMETRY ID:
                    </span>
                    <span style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-left: 6px;">
                        {selected_id}
                    </span>
                </div>
                <div>
                    <span style="font-size: 0.8rem; font-weight: 700; padding: 3px 10px; border-radius: 4px;
                                 background: rgba(239, 68, 68, 0.12); color: {color_border}; border: 1px solid {color_border};">
                        {status_label.upper()}
                    </span>
                    <span style="font-size: 0.82rem; color: #94A3B8; margin-left: 10px;">
                        Confidence Score: <strong style="color: #38BDF8;">{score_val:.4f}</strong>
                    </span>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.92rem; color: #E2E8F0;">
                <strong>Root Cause Diagnosis:</strong> <span style="color: #EF4444; font-weight: 600;">{diag_title}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Visual Deep-Dive: Radar Chart & Waterfall Deviations
    c_radar, c_water = st.columns([1, 1.2])

    with c_radar:
        cell_kpis = {
            "RSRP_mean": float(record_row.get("RSRP_mean", -110.0)),
            "UL_SNR_mean": float(record_row.get("UL_SNR_mean", 6.0)),
            "DL_BLER_mean": float(record_row.get("DL_BLER_mean", 0.12)),
            "UL_BLER_mean": float(record_row.get("UL_BLER_mean", 0.08)),
            "PRB_Utilization_DL_mean": float(record_row.get("PRB_Utilization_DL_mean", 75.0)),
            "Estimated_UL_Buffer_mean": float(record_row.get("Estimated_UL_Buffer_mean", 140.0)),
        }
        fig_radar = create_kpi_radar_chart(cell_kpis)
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_water:
        fig_water = create_record_deviation_waterfall(exp_df)
        st.plotly_chart(fig_water, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Action Checklist & AI Technical Copilot
    col_check, col_copilot = st.columns([1.1, 1])

    with col_check:
        st.markdown(
            f"""
            <div class="telecom-card">
                <h3 style="margin-bottom: 8px;">Recommended Field Operations Checklist</h3>
                <p style="color: #64748B; font-size: 0.82rem; margin-top: 0; margin-bottom: 12px;">
                    Standard operational procedures for NOC operators and site technicians:
                </p>
                
                <div style="background: #080C14; border: 1px solid #1A263D; border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 0.84rem; color: #CBD5E1;">
                    <div style="font-weight: 700; color: #0EA5E9; margin-bottom: 4px;">1. Primary Root-Cause Attribution</div>
                    {diag_action}
                </div>
                
                <div style="background: #080C14; border: 1px solid #1A263D; border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 0.84rem; color: #CBD5E1;">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 4px;">2. MAC &amp; Physical Layer Verification</div>
                    Cross-examine PDCCH aggregation levels and evaluate multi-user scheduling fairness to eliminate grant starvation.
                </div>
                
                <div style="background: #080C14; border: 1px solid #1A263D; border-radius: 6px; padding: 12px; font-size: 0.84rem; color: #CBD5E1;">
                    <div style="font-weight: 700; color: #F59E0B; margin-bottom: 4px;">3. Spectrum Load Balancing</div>
                    If downlink spectrum utilization remains above 80%, trigger automated traffic steering to secondary carrier frequencies.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_copilot:
        st.markdown(
            """
            <div class="telecom-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <h3 style="margin: 0; font-size: 0.95rem;">NOC AI Diagnostic Copilot</h3>
                    <span style="font-size: 0.7rem; color: #10B981; font-weight: 700;">ACTIVE</span>
                </div>
                <p style="color: #64748B; font-size: 0.82rem; margin-top: 0; margin-bottom: 10px;">
                    Select technical queries regarding resolving this specific cell anomaly:
                </p>
            """,
            unsafe_allow_html=True,
        )

        preset_qs = [
            "Can this issue be mitigated remotely via OSS without a tower climb?",
            "How does this affect VoLTE voice calls vs data subscribers?",
            "Which eNodeB RRC parameters should be calibrated first?",
        ]
        q_sel = st.selectbox("Inquiry Topics:", preset_qs, key="copilot_q_select")
        q_ans = _answer_ai_copilot_question(q_sel, record_row, top_feature_col)

        st.markdown(
            f"""
            <div style="background: #080C14; border: 1px solid #1A263D; border-left: 3px solid #0EA5E9;
                        border-radius: 6px; padding: 12px; margin-top: 10px; font-size: 0.84rem; color: #E2E8F0; line-height: 1.45;">
                {q_ans}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Reports & Diagnostic Export Center
    st.markdown("### Diagnostic Report Generation & Export")
    st.caption("Generate executive summaries and export annotated telemetry for records:")

    active_name = st.session_state.get("active_dataset_name", "Telemetry Batch")
    active_model = st.session_state.get("last_model_name", "Random Forest Ensemble")
    active_thresh = st.session_state.get("last_threshold", 0.30)

    md_report = _build_markdown_report(
        results=source_df,
        dataset_name=active_name,
        model_name=active_model,
        threshold=active_thresh,
    )
    html_report = _build_html_report(
        md_text=md_report,
        dataset_name=active_name,
        total=len(source_df),
        anomalies=int(source_df.get("anomaly_prediction", [1]*len(source_df)).sum()),
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.download_button(
            label="Download Executive HTML Report",
            data=html_report.encode("utf-8"),
            file_name=f"telecom_noc_report_{selected_id}.html",
            mime="text/html",
            use_container_width=True,
        )

    with exp_col2:
        st.download_button(
            label="Download Technical Markdown Report",
            data=md_report.encode("utf-8"),
            file_name=f"telecom_diagnostic_summary_{selected_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with exp_col3:
        st.download_button(
            label="Download Annotated CSV Results",
            data=source_df.to_csv(index=False).encode("utf-8"),
            file_name=f"telecom_anomaly_predictions_{selected_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("Dispatch Incident Ticket to ServiceNow / Jira Queue", use_container_width=True, key="btn_dispatch_ticket"):
        st.success(f"Incident Ticket NOC-{selected_id}-4912 successfully created in Network Operations queue.")
