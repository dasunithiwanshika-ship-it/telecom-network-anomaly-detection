"""
Page 4: AI Anomaly Detection Workspace & Alert Triage Table.
Provides a guided step-by-step workflow for Telecom Network Engineers:
  Step 1: Select Dataset
  Step 2: Select AI Detection Engine
  Step 3: Configure Detection Sensitivity
  Step 4: Click RUN AI DETECTION
  Step 5: Review Anomaly Alerts & Click Any Record to Investigate
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
)
from ..components.charts import create_score_distribution_histogram
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS
from src.prediction import AnomalyDetectorPipeline


def _compute_record_severity_and_driver(pipeline: AnomalyDetectorPipeline, results_df: pd.DataFrame, model_type: str, threshold: float) -> pd.DataFrame:
    """Enrich detection results with human-readable Severity Badges and Primary Deviating KPI."""
    df = results_df.copy()
    severities = []
    drivers = []

    has_baseline = pipeline.baseline_stats is not None
    if has_baseline:
        norm_mean = pipeline.baseline_stats["mean"]
        norm_std = pipeline.baseline_stats["std"].replace(0, 1.0)

    for idx, row in df.iterrows():
        is_anom = int(row["anomaly_prediction"]) == 1
        score = float(row["anomaly_score_or_prob"])

        if not is_anom:
            severities.append("🟢 NORMAL")
        else:
            if model_type == "Supervised":
                if score >= max(0.75, threshold + 0.25):
                    severities.append("🔴 CRITICAL ALERT")
                elif score >= max(0.50, threshold + 0.10):
                    severities.append("🟠 HIGH RISK")
                else:
                    severities.append("🟡 MODERATE RISK")
            else:
                if score > 0.15:
                    severities.append("🔴 CRITICAL ALERT")
                elif score > 0.05:
                    severities.append("🟠 HIGH RISK")
                else:
                    severities.append("🟡 MODERATE RISK")

        # Identify top deviating KPI for quick engineering visibility
        if has_baseline:
            feats = row[FEATURE_COLUMNS].astype(float)
            z_scores = (feats - norm_mean[FEATURE_COLUMNS]) / norm_std[FEATURE_COLUMNS]
            top_col = z_scores.abs().idxmax()
            top_z = float(z_scores[top_col])
            meta = get_feature_meta(top_col)
            drivers.append(f"{meta['base_kpi']} ({top_z:+.1f}σ)")
        else:
            drivers.append("Multi-KPI Profile")

    df["Severity Level"] = severities
    df["Primary KPI Deviation"] = drivers
    return df


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the Step-by-Step AI Anomaly Detection Workspace."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="AI Anomaly Detection Workspace",
        subtitle="Step-by-step cellular fault-risk detection: select your telemetry dataset, choose a detection profile, run AI analysis, and triage alerts.",
        badge="STEP-BY-STEP DIAGNOSTIC WORKSPACE",
    )

    # ------------------------------------------------------------------
    # STEP 1, STEP 2, STEP 3 CONFIGURATION CARDS
    # ------------------------------------------------------------------
    col_s1, col_s2, col_s3 = st.columns([1.15, 1.15, 1.1])

    # STEP 1: SELECT DATASET
    with col_s1:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">STEP 1</span>
                    <strong style="color: #F8FAFC; font-size: 1.0rem;">Select Telemetry Dataset</strong>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 6px;">
                    Choose which batch of cellular KPI windows to analyze:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_uploaded = "active_dataset_df" in st.session_state and st.session_state["active_dataset_df"] is not None
        active_name = st.session_state.get("active_dataset_name", "Custom Dataset")

        dataset_choices = []
        if has_uploaded:
            dataset_choices.append(f"✅ Active Workspace Dataset: {active_name}")
        dataset_choices.extend([
            "📡 Standard Field Sample (25 Cell Windows)",
            "🔥 High-Traffic Stress Batch (50 Cell Windows)",
            "📂 Upload a New CSV File...",
        ])

        chosen_ds_option = st.selectbox(
            "Telemetry Data Source:",
            options=dataset_choices,
            index=0,
            help="Select your previously uploaded dataset, a ready-to-use field sample, or upload a new CSV.",
        )

        working_df: Optional[pd.DataFrame] = None
        working_ds_label: str = ""

        if chosen_ds_option.startswith("✅ Active Workspace"):
            working_df = st.session_state["active_dataset_df"]
            working_ds_label = active_name
        elif "25 Cell Windows" in chosen_ds_option:
            working_df = create_synthetic_demo_features(n_samples=25, random_state=42)
            working_ds_label = "Standard Field Sample (25 Cell Windows)"
        elif "50 Cell Windows" in chosen_ds_option:
            working_df = create_synthetic_demo_features(n_samples=50, random_state=99)
            working_ds_label = "High-Traffic Stress Batch (50 Cell Windows)"
        elif "Upload a New CSV" in chosen_ds_option:
            quick_file = st.file_uploader("Upload 64-Feature KPI CSV:", type=["csv"], key="det_quick_upload")
            if quick_file is not None:
                try:
                    raw_q = pd.read_csv(quick_file)
                    working_df = raw_q
                    working_ds_label = f"Uploaded CSV ({quick_file.name})"
                except Exception as e:
                    st.error(f"Could not read CSV file: {e}")

        if working_df is not None:
            st.caption(f"📊 Ready: **{len(working_df):,} cell windows** ({working_ds_label})")

    # STEP 2: SELECT AI MODEL
    with col_s2:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">STEP 2</span>
                    <strong style="color: #F8FAFC; font-size: 1.0rem;">Select AI Detection Engine</strong>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 6px;">
                    Choose the detection engine suited for your operational goal:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_catalog = [
            (
                "random_forest",
                "High-Precision Fault Detector (Recommended)",
                "Supervised",
                "Random Forest Ensemble: 99.2%–100% precision on benchmark telemetry. Best for catching radio, BLER, and PRB scheduling faults with minimal false alarms.",
            ),
            (
                "logistic_regression",
                "Fast Linear Screening Engine",
                "Supervised",
                "Logistic Regression: Ultra-fast sub-millisecond screening for major threshold departures (95.3% precision, 74.1% recall).",
            ),
            (
                "isolation_forest",
                "Unlabelled Pattern Discovery (Isolation Forest)",
                "Unsupervised",
                "Isolation Forest: Detects unusual multi-KPI combinations without relying on historical fault labels. Useful for exploratory screening.",
            ),
            (
                "local_outlier_factor",
                "Local Density Outlier Scanner (LOF)",
                "Unsupervised",
                "Local Outlier Factor: Compares each cell window against neighboring density profiles to isolate atypical operating states.",
            ),
        ]

        current_mid = st.session_state.get("selected_model_id", "random_forest")
        idx_match = [i for i, m in enumerate(model_catalog) if m[0] == current_mid]
        default_idx = idx_match[0] if idx_match else 0

        selected_model = st.selectbox(
            "Detection Engine Profile:",
            options=model_catalog,
            index=default_idx,
            format_func=lambda x: x[1],
            help="High-Precision Fault Detector (Random Forest) is recommended for daily NOC triage.",
        )
        model_id, model_friendly_name, model_type, model_desc = selected_model
        st.session_state["selected_model_id"] = model_id
        st.caption(f"ℹ️ {model_desc}")

    # STEP 3: CONFIGURE SENSITIVITY
    with col_s3:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">STEP 3</span>
                    <strong style="color: #F8FAFC; font-size: 1.0rem;">Configure Alert Sensitivity</strong>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin-bottom: 6px;">
                    Set how aggressively the system flags potential fault risks:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        threshold = 0.30
        if model_type == "Supervised":
            preset = st.selectbox(
                "Sensitivity Preset:",
                options=[
                    "Balanced NOC Operations (Recommended — Cutoff 0.30)",
                    "Strict / Zero False Alarms (Cutoff 0.50)",
                    "High Sensitivity Screening (Cutoff 0.20)",
                    "Custom Sensitivity Cutoff...",
                ],
                index=0,
                help="Controls the anomaly probability cutoff. Lower cutoffs catch more subtle risks; higher cutoffs reduce false alarms.",
            )
            if "0.30" in preset:
                threshold = 0.30
                st.caption("🎯 **Balanced (0.30):** Captures 97.2% of benchmark faults with 99.2% precision.")
            elif "0.50" in preset:
                threshold = 0.50
                st.caption("🛡️ **Strict (0.50):** 100% precision on benchmark test data (zero false alarms).")
            elif "0.20" in preset:
                threshold = 0.20
                st.caption("🔍 **High Sensitivity (0.20):** Flags early warning signs and subtle KPI drift.")
            else:
                threshold = st.slider(
                    "Custom Alert Probability Cutoff:",
                    min_value=0.05,
                    max_value=0.95,
                    value=float(st.session_state.get("threshold", 0.30)),
                    step=0.05,
                    help="Any cell window with an anomaly probability at or above this value is flagged as an alert.",
                )
            st.session_state["threshold"] = threshold
        else:
            st.info(
                "ℹ️ **Automatic Outlier Calibration:** Unsupervised discovery engines rank cell windows automatically by statistical isolation/density."
            )

    # ------------------------------------------------------------------
    # STEP 4: VALIDATION & RUN AI DETECTION BUTTON
    # ------------------------------------------------------------------
    if working_df is None:
        st.warning("👆 Please select a dataset or upload a CSV file in **Step 1** above to continue.")
        return

    is_valid, validation_msg, cleaned_df = pipeline.validate_input_dataframe(working_df)
    if not is_valid:
        st.error(f"❌ **Telemetry Validation Error:** {validation_msg}")
        st.info("💡 Go to **📂 Data Upload & Validation** in the sidebar to inspect missing columns or download the 64-column CSV template.")
        return

    if not pipeline.is_model_available(model_id):
        st.error(f"❌ Selected detection engine (`{model_id}`) is not available on disk.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_clicked = st.button(
            f"🚀 STEP 4: RUN AI DETECTION ({len(cleaned_df):,} CELL WINDOWS)",
            use_container_width=True,
            key="btn_execute_detection",
        )

    # Execute detection when clicked
    if run_clicked:
        with st.spinner(f"Scanning {len(cleaned_df):,} cell windows using {model_friendly_name}..."):
            try:
                raw_results = pipeline.predict(cleaned_df, model_id=model_id, threshold=threshold)
                raw_results["Record ID"] = [f"TRACE-{i:04d}" for i in range(len(raw_results))]
                enriched_results = _compute_record_severity_and_driver(pipeline, raw_results, model_type, threshold)

                st.session_state["active_dataset_df"] = cleaned_df
                st.session_state["active_dataset_name"] = working_ds_label
                st.session_state["last_results"] = enriched_results
                st.session_state["last_model_id"] = model_id
                st.session_state["last_model_name"] = model_friendly_name
                st.session_state["last_threshold"] = threshold

                n_detected = int(enriched_results["anomaly_prediction"].sum())
                st.toast(f"Detection complete: {n_detected} anomaly alerts identified!", icon="✅")
            except Exception as e:
                st.error(f"❌ **Detection Execution Error:** {e}")
                return

    # ------------------------------------------------------------------
    # STEP 5: VIEW ANOMALY ALERTS & CLICK TO INVESTIGATE
    # ------------------------------------------------------------------
    if "last_results" not in st.session_state or st.session_state["last_results"] is None:
        st.markdown("---")
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px dashed #334155; border-radius: 10px; padding: 28px; text-align: center; margin-top: 10px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📡</div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.05rem;">Ready to Scan Cellular Telemetry</div>
                <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                    Click <strong>🚀 STEP 4: RUN AI DETECTION</strong> above to analyze your selected dataset and generate the Anomaly Alerts table.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results: pd.DataFrame = st.session_state["last_results"]
    # Ensure Severity Level and Primary KPI Deviation exist even if populated earlier
    if "Severity Level" not in results.columns or "Primary KPI Deviation" not in results.columns:
        results = _compute_record_severity_and_driver(pipeline, results, model_type, threshold)
        st.session_state["last_results"] = results

    total = len(results)
    anomalies = int(results["anomaly_prediction"].sum())
    normals = total - anomalies
    rate = (anomalies / total) * 100 if total > 0 else 0.0
    crit_count = int(results["Severity Level"].str.contains("CRITICAL|HIGH", regex=True).sum())

    st.markdown("---")
    st.success(
        f"✅ **AI Detection Complete:** Scanned **{total:,}** cell windows using **{st.session_state.get('last_model_name', model_friendly_name)}** "
        f"— Identified **{anomalies:,} Anomaly Alerts** ({rate:.1f}% of batch)."
    )

    st.markdown("### 🚨 Step 5: Anomaly Alerts & Diagnostic Triage")

    # Summary Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Windows Scanned", f"{total:,}", st.session_state.get("active_dataset_name", "Current Batch"), "#38BDF8")
    with m2:
        render_metric_card("Healthy Windows", f"{normals:,}", f"{(normals/total)*100:.1f}% of batch", "#10B981")
    with m3:
        render_metric_card("Flagged Anomaly Alerts", f"{anomalies:,}", f"{rate:.1f}% alert rate", "#EF4444")
    with m4:
        render_metric_card("Critical / High Risk", f"{crit_count:,}", "Immediate triage priority", "#F59E0B")

    # Interactive Alert Table & Investigation Launcher
    st.markdown("<br>", unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([1.4, 1])
    with f_col1:
        alert_filter = st.radio(
            "Filter Alert Table:",
            ["🚨 Flagged Anomaly Alerts Only", "📋 All Cell Windows", "🔴 Critical & High-Risk Only"],
            index=0 if anomalies > 0 else 1,
            horizontal=True,
        )
    with f_col2:
        sort_order = st.selectbox(
            "Sort Records By:",
            ["Highest Risk Score First", "Record ID Order"],
            index=0,
        )

    filtered_res = results.copy()
    if "Flagged Anomaly" in alert_filter:
        filtered_res = filtered_res[filtered_res["anomaly_prediction"] == 1]
    elif "Critical & High-Risk" in alert_filter:
        filtered_res = filtered_res[filtered_res["Severity Level"].str.contains("CRITICAL|HIGH", regex=True)]

    if sort_order == "Highest Risk Score First":
        filtered_res = filtered_res.sort_values(by="anomaly_score_or_prob", ascending=False)

    if len(filtered_res) == 0:
        st.info("No records match the current filter. Switch filter to **📋 All Cell Windows** to view healthy records.")
        filtered_res = results.copy()

    # Format clean engineering table for display
    table_df = pd.DataFrame({
        "Record ID": filtered_res["Record ID"],
        "Severity Level": filtered_res["Severity Level"],
        "Risk Score": filtered_res["anomaly_score_or_prob"].apply(lambda v: f"{float(v):.4f}"),
        "Primary KPI Deviation": filtered_res["Primary KPI Deviation"],
        "RSRP Avg (dBm)": filtered_res["RSRP_mean"].round(2),
        "UL SNR Avg (dB)": filtered_res["UL_SNR_mean"].round(2),
        "UL BLER Avg": filtered_res["UL_BLER_mean"].round(4),
        "UL PRB Min": filtered_res["UL_NPRB_min"].round(1),
        "UL Buffer Avg": filtered_res["Estimated_UL_Buffer_mean"].round(1),
        "DL PRB Util (%)": filtered_res["PRB_Utilization_DL_mean"].round(1),
    }).reset_index(drop=True)

    st.caption("💡 **Tip:** Click any row in the table below (or use the Record Selector) to inspect its full KPI breakdown in **Anomaly Investigation**.")

    selection_event = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=310,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Check if user clicked a row in the dataframe
    clicked_record_id = None
    if selection_event and hasattr(selection_event, "selection") and selection_event.selection.rows:
        row_idx = selection_event.selection.rows[0]
        if 0 <= row_idx < len(table_df):
            clicked_record_id = table_df.iloc[row_idx]["Record ID"]
            st.session_state["selected_record_id"] = clicked_record_id

    # Dedicated Action Bar: Select Record & Click to Investigate
    st.markdown(
        """
        <div style="background-color: #172235; border: 1px solid #14B8A6; border-radius: 10px; padding: 16px 20px; margin-top: 8px; margin-bottom: 18px;">
            <div style="font-weight: 700; color: #14B8A6; font-size: 0.95rem; margin-bottom: 4px;">
                🔬 Deep-Dive Record Investigation
            </div>
            <div style="color: #94A3B8; font-size: 0.84rem;">
                Select any flagged alert below and click <strong>Investigate Selected Alert</strong> to inspect its 16 cellular KPIs, normal baseline comparison, and engineering triage recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inv_col1, inv_col2, inv_col3 = st.columns([1.4, 1, 1])
    available_ids = table_df["Record ID"].tolist()
    default_rec_idx = 0
    if "selected_record_id" in st.session_state and st.session_state["selected_record_id"] in available_ids:
        default_rec_idx = available_ids.index(st.session_state["selected_record_id"])

    with inv_col1:
        chosen_rec_id = st.selectbox(
            "Selected Record for Investigation:",
            options=available_ids,
            index=default_rec_idx,
            format_func=lambda rid: f"{rid} — {table_df.loc[table_df['Record ID']==rid, 'Severity Level'].iloc[0]} ({table_df.loc[table_df['Record ID']==rid, 'Primary KPI Deviation'].iloc[0]})",
        )
        st.session_state["selected_record_id"] = chosen_rec_id

    with inv_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button(f"🔬 Investigate {chosen_rec_id} →", use_container_width=True, key="btn_go_investigate"):
            st.session_state["selected_record_id"] = chosen_rec_id
            navigate_to("🔬 Anomaly Investigation")

    with inv_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📑 Open Reports & Export →", use_container_width=True, key="btn_go_reports"):
            navigate_to("📑 Reports & Export")

    # Visual Score Distribution & Quick Export
    with st.expander("📊 View Risk Score Distribution Histogram & Quick CSV Download", expanded=False):
        score_fig = create_score_distribution_histogram(
            results,
            score_col="anomaly_score_or_prob",
            status_col="anomaly_status",
            title=f"Distribution of Anomaly Risk Scores ({results['score_type'].iloc[0]})",
            threshold=threshold if model_type == "Supervised" else None,
        )
        st.plotly_chart(score_fig, use_container_width=True)

        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button(
                label="⬇️ Download Full Detection Results (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"telecom_detection_results_{model_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_c2:
            anom_only = results[results["anomaly_prediction"] == 1]
            st.download_button(
                label=f"⬇️ Download Flagged Alerts Only ({len(anom_only)} Rows CSV)",
                data=anom_only.to_csv(index=False).encode("utf-8"),
                file_name=f"telecom_flagged_alerts_{model_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    render_engineering_notice()
