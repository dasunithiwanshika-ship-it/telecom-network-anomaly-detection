"""
Page 2: Run AI Anomaly Detection.
Simple, clean 2-step setup + instant Anomaly Alerts table with 1-click record inspection.
"""

from typing import Any, Optional, Tuple
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    navigate_to,
    get_feature_meta,
)
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS
from src.prediction import AnomalyDetectorPipeline


def _compute_record_severity_and_driver(
    pipeline: AnomalyDetectorPipeline,
    results_df: pd.DataFrame,
    model_type: str = "Supervised",
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Add clean Status labels and Main KPI Issue for each record."""
    df = results_df.copy()
    severities = []
    drivers = []

    has_baseline = pipeline.baseline_stats is not None
    if has_baseline:
        norm_mean = pipeline.baseline_stats["mean"]
        norm_std = pipeline.baseline_stats["std"].replace(0, 1.0)

    for _, row in df.iterrows():
        is_anom = int(row["anomaly_prediction"]) == 1
        score = float(row["anomaly_score_or_prob"])

        if not is_anom:
            severities.append("🟢 Normal")
        else:
            if model_type == "Supervised" and score >= max(0.75, threshold + 0.25):
                severities.append("🔴 Critical Alert")
            elif model_type == "Supervised" and score >= max(0.50, threshold + 0.10):
                severities.append("🟠 High Risk")
            else:
                severities.append("🔴 Anomaly Alert")

        if has_baseline:
            feats = row[FEATURE_COLUMNS].astype(float)
            z_scores = (feats - norm_mean[FEATURE_COLUMNS]) / norm_std[FEATURE_COLUMNS]
            top_col = z_scores.abs().idxmax()
            meta = get_feature_meta(top_col)
            drivers.append(f"{meta['friendly_name']}")
        else:
            drivers.append("Multi-KPI Deviation")

    df["Severity Level"] = severities
    df["Primary KPI Deviation"] = drivers
    return df


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the streamlined AI Detection page."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Run AI Anomaly Detection",
        subtitle="Select sample data or upload a CSV file, choose a detection model, and click Run AI Detection.",
        badge="DETECTION WORKSPACE",
    )

    col_left, col_right = st.columns(2)

    # STEP 1: CHOOSE DATA
    with col_left:
        st.markdown("#### 1. Choose Data")
        data_source = st.radio(
            "Select Input Data:",
            ["📡 Use Sample Network Data (25 rows)", "📂 Upload CSV File"],
            horizontal=True,
            label_visibility="collapsed",
        )

        working_df: Optional[pd.DataFrame] = None
        working_name = "Sample Network Data (25 rows)"

        if "Sample" in data_source:
            working_df = create_synthetic_demo_features(n_samples=25, random_state=42)
            st.caption("✅ Loaded 25 sample cell records (ready to scan).")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file (64 KPI columns):",
                type=["csv"],
                help="Upload a CSV containing the 64 telecom KPI columns.",
            )
            if uploaded_file is not None:
                try:
                    working_df = pd.read_csv(uploaded_file)
                    working_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Could not read CSV file: {e}")

            template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
            st.download_button(
                label="⬇️ Download CSV Template",
                data=template_df.to_csv(index=False).encode("utf-8"),
                file_name="telecom_kpi_template.csv",
                mime="text/csv",
            )

    # STEP 2: CHOOSE MODEL
    with col_right:
        st.markdown("#### 2. Choose AI Model")
        model_options = [
            ("random_forest", "Random Forest (Recommended — Best Accuracy)", "Supervised"),
            ("logistic_regression", "Logistic Regression (Fast Screening)", "Supervised"),
            ("isolation_forest", "Isolation Forest (Unlabelled Discovery)", "Unsupervised"),
            ("local_outlier_factor", "Local Outlier Factor (Density Discovery)", "Unsupervised"),
        ]

        selected_model = st.selectbox(
            "AI Model:",
            options=model_options,
            format_func=lambda x: x[1],
            label_visibility="collapsed",
        )
        model_id, model_name, model_type = selected_model

        threshold = 0.30
        if model_type == "Supervised":
            sens_choice = st.select_slider(
                "Alert Sensitivity:",
                options=["Strict (Fewer Alerts)", "Balanced (Recommended)", "High Sensitivity"],
                value="Balanced (Recommended)",
                help="Balanced (0.30) catches 97.2% of anomalies with 99.2% precision.",
            )
            if "Strict" in sens_choice:
                threshold = 0.50
            elif "High" in sens_choice:
                threshold = 0.20
            else:
                threshold = 0.30
        else:
            st.caption("ℹ️ Unsupervised model automatically scores unusual records.")

    # Optional Data Preview
    if working_df is not None:
        with st.expander(f"👁️ Preview Input Data ({len(working_df)} rows)"):
            st.dataframe(working_df.head(10), use_container_width=True, hide_index=True)

    # Validate & Run Button
    if working_df is None:
        st.info("👆 Please upload a CSV file above (or switch to Sample Network Data) to run detection.")
        return

    is_valid, validation_msg, cleaned_df = pipeline.validate_input_dataframe(working_df)
    if not is_valid:
        st.error(f"❌ **Invalid CSV Format:** {validation_msg}")
        return

    run_clicked = st.button("🚀 RUN AI DETECTION", use_container_width=True)

    if run_clicked:
        with st.spinner("Analyzing network KPIs..."):
            raw_results = pipeline.predict(cleaned_df, model_id=model_id, threshold=threshold)
            raw_results["Record ID"] = [f"TRACE-{i:04d}" for i in range(len(raw_results))]
            enriched = _compute_record_severity_and_driver(pipeline, raw_results, model_type, threshold)

            st.session_state["active_dataset_df"] = cleaned_df
            st.session_state["active_dataset_name"] = working_name
            st.session_state["last_results"] = enriched
            st.session_state["last_model_id"] = model_id
            st.session_state["last_model_name"] = model_name
            st.session_state["last_threshold"] = threshold

    # DISPLAY RESULTS IF AVAILABLE
    results: Optional[pd.DataFrame] = st.session_state.get("last_results")
    if results is None:
        return

    if "Severity Level" not in results.columns:
        results = _compute_record_severity_and_driver(pipeline, results, model_type, threshold)
        st.session_state["last_results"] = results

    total = len(results)
    anomalies = int(results["anomaly_prediction"].sum())
    normals = total - anomalies

    st.markdown("---")
    st.markdown("### 3. Detection Results")

    # 3 Simple Result Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Total Records Scanned", f"{total:,}", st.session_state.get("active_dataset_name", ""), "#38BDF8")
    with m2:
        render_metric_card("Normal Records", f"{normals:,}", "Operating normally", "#10B981")
    with m3:
        render_metric_card("Anomalies Detected", f"{anomalies:,}", "Flagged for review", "#EF4444" if anomalies > 0 else "#10B981")

    # Filter & Table
    filter_choice = st.radio(
        "Show:",
        ["🔴 Anomalies Only", "📋 All Records"],
        index=0 if anomalies > 0 else 1,
        horizontal=True,
    )

    view_df = results[results["anomaly_prediction"] == 1] if "Anomalies" in filter_choice and anomalies > 0 else results
    view_df = view_df.sort_values(by="anomaly_score_or_prob", ascending=False)

    display_table = pd.DataFrame({
        "Record ID": view_df["Record ID"],
        "Status": view_df["Severity Level"],
        "Risk Score": view_df["anomaly_score_or_prob"].round(3),
        "Main KPI Issue": view_df["Primary KPI Deviation"],
        "Signal RSRP (dBm)": view_df["RSRP_mean"].round(1),
        "Uplink SNR (dB)": view_df["UL_SNR_mean"].round(1),
        "Uplink Error (BLER)": view_df["UL_BLER_mean"].round(3),
        "Uplink Buffer": view_df["Estimated_UL_Buffer_mean"].round(1),
    }).reset_index(drop=True)

    st.caption("💡 **Click any row** below or choose a Record ID and click **Inspect Selected Record** to see full details.")

    event = st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=280,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event and hasattr(event, "selection") and event.selection.rows:
        idx = event.selection.rows[0]
        if 0 <= idx < len(display_table):
            st.session_state["selected_record_id"] = display_table.iloc[idx]["Record ID"]

    # Bottom Action Bar: Inspect or Download
    act1, act2, act3 = st.columns([1.3, 1, 1])
    rec_list = display_table["Record ID"].tolist()
    default_idx = 0
    if st.session_state.get("selected_record_id") in rec_list:
        default_idx = rec_list.index(st.session_state["selected_record_id"])

    with act1:
        chosen_id = st.selectbox("Select Record to Inspect:", options=rec_list, index=default_idx, label_visibility="collapsed")
        st.session_state["selected_record_id"] = chosen_id
    with act2:
        if st.button(f"🔍 Inspect {chosen_id} →", use_container_width=True, key="btn_det_inspect"):
            st.session_state["selected_record_id"] = chosen_id
            navigate_to("🔍 Inspect Alert")
    with act3:
        st.download_button(
            label="⬇️ Download Results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
