"""
Page 2: Anomaly Detection Workspace (Interactive Model Inference & Ingestion).
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_engineering_notice,
)
from ..components.charts import create_score_distribution_histogram
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS


def render(pipeline, dataset_tuple, benchmark_tables):
    """Render the Anomaly Detection Workspace."""
    render_hero_banner(
        title="AI Anomaly Detection Workspace",
        subtitle="Execute pre-trained supervised and unsupervised models on 64-feature cellular KPI telemetry",
    )

    col_config, col_data = st.columns([1.1, 1])

    with col_config:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>1. Select Detection Model & Parameters</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_options = [
            ("random_forest", "Random Forest (Supervised Ensemble - Recommended)", "Supervised"),
            ("logistic_regression", "Logistic Regression (Supervised Linear GLM)", "Supervised"),
            ("isolation_forest", "Isolation Forest (Unsupervised Path-Length Isolation)", "Unsupervised"),
            ("local_outlier_factor", "Local Outlier Factor (Unsupervised Density k-NN)", "Unsupervised"),
        ]

        current_mid = st.session_state.get("selected_model_id", "random_forest")
        idx_match = [i for i, m in enumerate(model_options) if m[0] == current_mid]
        default_idx = idx_match[0] if idx_match else 0

        selected_tuple = st.selectbox(
            "Select AI Anomaly Detection Algorithm:",
            options=model_options,
            index=default_idx,
            format_func=lambda x: f"[{x[2]}] {x[1]}",
        )
        model_id = selected_tuple[0]
        model_type = selected_tuple[2]
        st.session_state["selected_model_id"] = model_id

        # Availability indicator
        is_avail = pipeline.is_model_available(model_id)
        if not is_avail:
            st.error(f"❌ Model artifact `{model_id}.joblib` is not loaded in `models/`.")
        else:
            st.success(f"✅ Loaded model engine: `{model_id}.joblib` ({model_type})")

        # Threshold slider for supervised models
        threshold = 0.50
        if model_type == "Supervised":
            st.markdown("#### 🎯 Decision Threshold ($\tau$)")
            threshold = st.slider(
                "Classification Decision Cutoff:",
                min_value=0.05,
                max_value=0.95,
                value=float(st.session_state.get("threshold", 0.50)),
                step=0.05,
                help="Probability cutoff above which a trace is classified as an anomaly.",
            )
            st.session_state["threshold"] = threshold
            st.caption(
                f"Selected threshold: **{threshold:.2f}**. "
                "Lower thresholds prioritize **Recall** (catching more subtle anomalies). "
                "Higher thresholds prioritize **Precision** (eliminating false alarms)."
            )
        else:
            st.info(
                f"ℹ️ **Unsupervised Anomaly Scoring Active:** `{model_id}` generates continuous anomaly scores. "
                "In scikit-learn, negative decision scores denote anomalies; scores are calibrated and inverted "
                "such that higher scores uniformly reflect higher anomaly likelihood."
            )

    with col_data:
        st.markdown(
            """
            <div class="telecom-card">
                <h3>2. Ingest Cellular KPI Telemetry</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("Choose an input data source for inference:")
        input_mode = st.radio(
            "Data Source:",
            ["⚡ Pre-loaded Synthetic Demonstration Batch (10 rows)", "📂 Upload Custom 64-Feature CSV File"],
            label_visibility="collapsed",
        )

        uploaded_df = None

        if "Synthetic" in input_mode:
            uploaded_df = create_synthetic_demo_features(n_samples=10, random_state=42)
            st.success("Loaded 10 pre-formatted synthetic demonstration records.")
            st.caption("DISCLAIMER: These rows are simulated to match the 64-feature TelecomTS schema; they do NOT represent live operational subscriber traces.")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV containing all 64 feature columns:",
                type=["csv"],
                help="The CSV must contain the 64 engineered statistical moments.",
            )
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    st.success(f"File uploaded: {uploaded_df.shape[0]} rows, {uploaded_df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        # Download Template CSV
        sample_demo = create_synthetic_demo_features(n_samples=3, random_state=1)
        csv_template = sample_demo.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download 64-Feature CSV Template",
            data=csv_template,
            file_name="telecom_kpi_features_template.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # Step 3: Schema Validation & Execution
    st.subheader("3. Validation & Model Execution")

    if uploaded_df is None:
        st.warning("Please upload a CSV file or select the synthetic demonstration dataset above.")
        return

    # Validate Schema
    is_valid, validation_msg, cleaned_df = pipeline.validate_input_dataframe(uploaded_df)

    if not is_valid:
        st.error(f"❌ **Data Schema Validation Failed:** {validation_msg}")
        st.markdown(
            """
            **Required Structure:** Input CSV must strictly contain the 64 engineered statistical moments
            (mean, std, min, max for the 16 cellular KPIs: RSRP, DL_BLER, DL_MCS, UL_BLER, UL_MCS,
            UL_NPRB, UL_SNR, TX_Bytes, RX_Bytes, Estimated_UL_Buffer, PRBs_DL_Current, PRBs_UL_Current,
            PRB_Utilization_DL, PRB_Utilization_UL, UL_NumberOfPackets, DL_NumberOfPackets).
            """
        )
        return

    st.success(f"✅ **Validation Successful:** All 64 KPI features present and verified numeric across {len(cleaned_df)} records.")

    run_clicked = st.button("🚀 RUN AI ANOMALY DETECTION", use_container_width=True)

    if run_clicked or ("last_results" in st.session_state and st.session_state.get("last_model_id") == model_id):
        if run_clicked:
            with st.spinner(f"Running inference with {selected_tuple[1]}..."):
                try:
                    results = pipeline.predict(
                        cleaned_df, model_id=model_id, threshold=threshold
                    )
                    # Add trace index identifier
                    results["Record ID"] = [f"TRACE-{i:04d}" for i in range(len(results))]
                    st.session_state["last_results"] = results
                    st.session_state["last_model_id"] = model_id
                    st.session_state["last_threshold"] = threshold
                except Exception as e:
                    st.error(f"Inference execution error: {e}")
                    return

        results = st.session_state["last_results"]

        st.markdown("---")
        st.subheader("4. Detection Results & Diagnostic Metrics")

        total = len(results)
        anomalies = int(results["anomaly_prediction"].sum())
        normals = total - anomalies
        rate = (anomalies / total) * 100 if total > 0 else 0

        # Summary Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Processed Traces", f"{total:,}", "Batch total")
        with m2:
            render_metric_card("Healthy / Normal", f"{normals:,}", f"{(normals/total)*100:.1f}%", "#10B981")
        with m3:
            render_metric_card("Detected Anomalies", f"{anomalies:,}", f"{rate:.1f}% rate", "#EF4444")
        with m4:
            active_thresh_str = f"tau = {threshold:.2f}" if model_type == "Supervised" else "Unsupervised Score"
            render_metric_card("Decision Cutoff", active_thresh_str, model_id.split('_')[0].upper(), "#F59E0B")

        # Score distribution histogram
        st.markdown("<br>", unsafe_allow_html=True)
        score_fig = create_score_distribution_histogram(
            results,
            score_col="anomaly_score_or_prob",
            status_col="anomaly_status",
            title=f"Distribution of Anomaly Metric ({results['score_type'].iloc[0]})",
            threshold=threshold if model_type == "Supervised" else None,
        )
        st.plotly_chart(score_fig, use_container_width=True)

        # Prediction table
        st.markdown("### 📋 Detailed Trace Predictions Table")
        display_cols = [
            "Record ID",
            "anomaly_status",
            "anomaly_score_or_prob",
            "score_type",
            "RSRP_mean",
            "UL_SNR_mean",
            "UL_BLER_mean",
            "Estimated_UL_Buffer_mean",
            "PRB_Utilization_DL_mean",
        ]
        available_display = [c for c in display_cols if c in results.columns]
        st.dataframe(results[available_display], use_container_width=True, hide_index=True)

        # Export CSV Button
        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Complete Predictions CSV",
            data=csv_out,
            file_name=f"telecom_anomaly_predictions_{model_id}.csv",
            mime="text/csv",
        )

        render_engineering_notice()
