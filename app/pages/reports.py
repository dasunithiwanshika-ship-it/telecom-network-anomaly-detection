"""
Page 7: Reports & Export Center.
Allows Telecom Network Engineers to download detection results (CSV) and generate/export
a professional Network Anomaly Analysis Summary Report (Markdown & Printable HTML).
"""

from datetime import datetime
from typing import Any, Optional
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_academic_disclaimer,
    navigate_to,
    get_feature_meta,
)
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS
from src.prediction import AnomalyDetectorPipeline


def _build_markdown_report(
    results: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    threshold: float,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    anomalies = int(results["anomaly_prediction"].sum())
    normals = total - anomalies
    rate = (anomalies / total) * 100 if total > 0 else 0.0

    anom_df = results[results["anomaly_prediction"] == 1].sort_values(
        by="anomaly_score_or_prob", ascending=False
    )

    crit_n = int(results["Severity Level"].str.contains("CRITICAL", regex=False).sum()) if "Severity Level" in results.columns else anomalies
    high_n = int(results["Severity Level"].str.contains("HIGH RISK", regex=False).sum()) if "Severity Level" in results.columns else 0
    mod_n = int(results["Severity Level"].str.contains("MODERATE", regex=False).sum()) if "Severity Level" in results.columns else 0

    lines = [
        "# TELECOM AI INTELLIGENCE PLATFORM — NETWORK ANOMALY ANALYSIS REPORT",
        f"**Generated Timestamp:** {now_str}",
        f"**Evaluated Telemetry Dataset:** {dataset_name}",
        f"**AI Detection Engine:** {model_name}",
        f"**Alert Sensitivity Cutoff:** {threshold:.2f}",
        "",
        "---",
        "",
        "## 1. EXECUTIVE TELEMETRY SUMMARY",
        f"- **Total Cell Windows Scanned:** {total:,}",
        f"- **Healthy Operating Windows:** {normals:,} ({(normals/total)*100:.1f}%)",
        f"- **Flagged Anomaly Alerts:** {anomalies:,} ({rate:.1f}%)",
        f"  - Critical Priority Alerts: {crit_n:,}",
        f"  - High-Risk Priority Alerts: {high_n:,}",
        f"  - Moderate-Risk Alerts: {mod_n:,}",
        "",
        "## 2. KEY TELEMETRY AVERAGES (HEALTHY VS. FLAGGED ALERTS)",
    ]

    key_metrics = [
        ("RSRP_mean", "Reference Signal Power (RSRP Avg)", "dBm"),
        ("UL_SNR_mean", "Uplink Signal-to-Noise Ratio (UL SNR Avg)", "dB"),
        ("UL_BLER_mean", "Uplink Block Error Rate (UL BLER Avg)", "Ratio"),
        ("DL_BLER_mean", "Downlink Block Error Rate (DL BLER Avg)", "Ratio"),
        ("UL_NPRB_min", "Minimum Uplink PRB Allocation (UL NPRB Min)", "PRBs"),
        ("PRB_Utilization_DL_mean", "Downlink Spectrum Utilization (DL PRB %)", "%"),
        ("Estimated_UL_Buffer_mean", "Uplink Queue Buffer Backlog (Avg)", "Bytes/KB"),
    ]

    norm_df = results[results["anomaly_prediction"] == 0]
    lines.append("| KPI Measurement | Unit | Healthy Windows Avg | Flagged Alerts Avg |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for col, friendly, unit in key_metrics:
        if col in results.columns:
            n_avg = f"{norm_df[col].mean():.3f}" if len(norm_df) > 0 else "N/A"
            a_avg = f"{anom_df[col].mean():.3f}" if len(anom_df) > 0 else "N/A"
            lines.append(f"| {friendly} (`{col}`) | {unit} | {n_avg} | {a_avg} |")

    lines.extend([
        "",
        "## 3. TOP FLAGGED ANOMALY ALERTS FOR ENGINEERING TRIAGE",
    ])

    if len(anom_df) == 0:
        lines.append("No anomaly alerts were triggered in this telemetry batch. All cell windows operated within normal parameters.")
    else:
        lines.append("| Record ID | Severity Level | Risk Score | Primary KPI Deviation | RSRP (dBm) | UL SNR (dB) | UL BLER | UL Buffer |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, r in anom_df.head(15).iterrows():
            rid = r.get("Record ID", "-")
            sev = r.get("Severity Level", "ALERT")
            sc = f"{float(r.get('anomaly_score_or_prob', 0.0)):.4f}"
            drv = r.get("Primary KPI Deviation", "Multi-KPI")
            lines.append(
                f"| {rid} | {sev} | {sc} | {drv} | {float(r['RSRP_mean']):.2f} | {float(r['UL_SNR_mean']):.2f} | {float(r['UL_BLER_mean']):.4f} | {float(r['Estimated_UL_Buffer_mean']):.1f} |"
            )

    lines.extend([
        "",
        "## 4. RECOMMENDED NOC ENGINEERING ACTIONS",
        "1. **MAC Resource Scheduling (`UL_NPRB_min`, `PRB_Utilization_DL`):** Inspect cell sectors where uplink PRB grants drop near zero or downlink PRB utilization exceeds 90%. Verify PDCCH control channel capacity and active user scheduling fairness.",
        "2. **Radio Signal Quality & Interference (`UL_SNR_mean`, `RSRP_mean`):** For alerts driven by negative UL SNR shifts, check the sector uplink interference floor and adjacent cell power leakage.",
        "3. **Block Error Rates & Buffer Congestion (`UL_BLER_mean`, `Estimated_UL_Buffer_mean`):** Correlate elevated block error rates (>10%) with UE uplink buffer buildup to isolate HARQ retransmission bottlenecks.",
        "",
        "---",
        "*Scope Note: Generated by the Telecom AI Intelligence Platform (Research Prototype based on the public AliMaatouk/TelecomTS cellular KPI benchmark). Statistical alerts require engineering verification prior to field maintenance.*",
    ])
    return "\n".join(lines)


def _build_html_report(md_text: str, dataset_name: str, total: int, anomalies: int) -> str:
    """Wrap report details into a clean, printable HTML document."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_body = md_text.replace("\n", "<br>")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Telecom AI Network Anomaly Report - {dataset_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; background: #0B1220; color: #F8FAFC; padding: 36px; max-width: 960px; margin: auto; line-height: 1.6; }}
  .header {{ background: #172235; border-left: 6px solid #14B8A6; padding: 24px; border-radius: 10px; margin-bottom: 24px; }}
  h1 {{ color: #14B8A6; margin: 0 0 8px 0; font-size: 24px; }}
  .summary-box {{ background: #111827; border: 1px solid #1E293B; padding: 18px; border-radius: 8px; margin-bottom: 20px; }}
  pre {{ background: #172235; padding: 18px; border-radius: 8px; white-space: pre-wrap; font-family: Consolas, monospace; font-size: 13px; color: #E2E8F0; border: 1px solid #1E293B; }}
  @media print {{ body {{ background: #FFFFFF; color: #000000; }} pre {{ background: #F8FAFC; color: #000000; border: 1px solid #CCC; }} }}
</style>
</head>
<body>
  <div class="header">
    <h1>Telecom AI Intelligence Platform — Diagnostic Summary Report</h1>
    <div>Generated: {now_str} | Dataset: {dataset_name} | Scanned: {total:,} Windows | Alerts: {anomalies:,}</div>
  </div>
  <pre>{md_text}</pre>
</body>
</html>"""


def render(pipeline: Optional[AnomalyDetectorPipeline] = None):
    """Render the Reports & Export Center."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Reports & Telemetry Export Center",
        subtitle="Download complete AI detection results (CSV) and generate formatted engineering diagnostic summary reports.",
        badge="REPORTING & EXPORT",
    )

    last_results: Optional[pd.DataFrame] = st.session_state.get("last_results")

    if last_results is None or len(last_results) == 0:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px dashed #334155; border-radius: 10px; padding: 30px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📑</div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem;">No Detection Results Available to Export Yet</div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 6px; max-width: 620px; margin-left: auto; margin-right: auto;">
                    Run an AI detection scan in the <strong>AI Detection Workspace</strong> first, or click the quick button below to scan our 25-window sample dataset and generate a report immediately.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_empty1, c_empty2, c_empty3 = st.columns([1, 1.5, 1])
        with c_empty2:
            if st.button("⚡ Run Sample Detection & Generate Report Now", use_container_width=True, key="btn_rep_quick"):
                with st.spinner("Scanning 25 sample cell windows and building diagnostic report..."):
                    demo_df = create_synthetic_demo_features(n_samples=25, random_state=42)
                    res = pipeline.predict(demo_df, model_id="random_forest", threshold=0.30)
                    res["Record ID"] = [f"TRACE-{i:04d}" for i in range(len(res))]
                    from .detection import _compute_record_severity_and_driver
                    res = _compute_record_severity_and_driver(pipeline, res, "Supervised", 0.30)
                    st.session_state["active_dataset_df"] = demo_df
                    st.session_state["active_dataset_name"] = "Standard Field Sample (25 Cell Windows)"
                    st.session_state["last_results"] = res
                    st.session_state["last_model_id"] = "random_forest"
                    st.session_state["last_model_name"] = "High-Precision Fault Detector (Random Forest)"
                    st.session_state["last_threshold"] = 0.30
                    st.rerun()
        return

    dataset_name = st.session_state.get("active_dataset_name", "Cellular KPI Batch")
    model_id = st.session_state.get("last_model_id", "random_forest")
    model_name = st.session_state.get("last_model_name", "High-Precision Fault Detector (Random Forest)")
    threshold = float(st.session_state.get("last_threshold", 0.30))

    total = len(last_results)
    anomalies = int(last_results["anomaly_prediction"].sum())
    normals = total - anomalies
    rate = (anomalies / total) * 100 if total > 0 else 0.0

    # Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Analyzed Dataset", f"{total:,} Windows", dataset_name, "#38BDF8")
    with m2:
        render_metric_card("Healthy Windows", f"{normals:,}", f"{(normals/total)*100:.1f}% of batch", "#10B981")
    with m3:
        render_metric_card("Flagged Alerts", f"{anomalies:,}", f"{rate:.1f}% alert rate", "#EF4444")
    with m4:
        render_metric_card("Detection Profile", model_id.split("_")[0].upper(), f"Cutoff = {threshold:.2f}", "#14B8A6")

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. CSV Exports Card
    st.markdown("### 📥 1. Export Detection Data Tables (CSV)")
    st.caption("Download the complete cell-window prediction table or export only the flagged anomaly alerts for NOC ticketing:")

    csv_col1, csv_col2 = st.columns(2)
    with csv_col1:
        st.markdown(
            f"""
            <div class="telecom-card">
                <h3>📊 Complete Batch Predictions CSV ({total:,} Rows)</h3>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Includes all {total:,} cell windows, 64 KPI features, AI risk scores, severity levels, and primary deviating KPIs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"⬇️ Download Complete Results CSV ({total:,} Rows)",
            data=last_results.to_csv(index=False).encode("utf-8"),
            file_name=f"telecom_full_predictions_{model_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with csv_col2:
        anom_only_df = last_results[last_results["anomaly_prediction"] == 1]
        st.markdown(
            f"""
            <div class="telecom-card">
                <h3>🚨 Flagged Anomaly Alerts Only CSV ({len(anom_only_df):,} Rows)</h3>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Filtered strictly to the {len(anom_only_df):,} anomalous cell windows requiring engineering review.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"⬇️ Download Flagged Alerts CSV ({len(anom_only_df):,} Rows)",
            data=anom_only_df.to_csv(index=False).encode("utf-8"),
            file_name=f"telecom_flagged_alerts_{model_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # 2. Automated Executive Analysis Summary Report
    st.markdown("### 📄 2. Automated Network Anomaly Analysis Summary Report")
    st.caption("Review the generated diagnostic summary below or download it as a formatted Markdown or Printable HTML report:")

    md_report = _build_markdown_report(last_results, dataset_name, model_name, threshold)
    html_report = _build_html_report(md_report, dataset_name, total, anomalies)

    rep_dl1, rep_dl2 = st.columns(2)
    with rep_dl1:
        st.download_button(
            label="⬇️ Download Printable HTML Engineering Report (.html)",
            data=html_report.encode("utf-8"),
            file_name=f"telecom_anomaly_diagnostic_report_{model_id}.html",
            mime="text/html",
            use_container_width=True,
        )
    with rep_dl2:
        st.download_button(
            label="⬇️ Download Markdown Analysis Summary (.md)",
            data=md_report.encode("utf-8"),
            file_name=f"telecom_anomaly_diagnostic_report_{model_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("👁️ Preview Full Engineering Analysis Summary Report", expanded=True):
        st.markdown(md_report)

    render_academic_disclaimer()
