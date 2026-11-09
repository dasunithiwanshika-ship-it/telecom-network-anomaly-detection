"""
Page 04: Configure AI Detection & Model Calibration.
Supports dual operational modes:
- NOC Engineer Mode: Guided presets with calibrated decision boundaries.
- AI/ML Specialist Mode: Continuous decision threshold sliders, model selection,
  and precision-recall tradeoff controls.
"""

from typing import Any, Optional, Tuple
import pandas as pd
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_html,
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
    """Add clean Status labels and Primary KPI Deviations for each record."""
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
            severities.append("Normal")
        else:
            if model_type == "Supervised" and score >= max(0.75, threshold + 0.25):
                severities.append("Critical Alert")
            elif model_type == "Supervised" and score >= max(0.50, threshold + 0.10):
                severities.append("High Risk")
            else:
                severities.append("Moderate Risk")

        if has_baseline:
            feats = row[FEATURE_COLUMNS].astype(float)
            z_scores = (feats - norm_mean[FEATURE_COLUMNS]) / norm_std[FEATURE_COLUMNS]
            top_col = z_scores.abs().idxmax()
            top_z = z_scores[top_col]
            meta = get_feature_meta(top_col)
            drivers.append(f"{meta['friendly_name']} ({top_z:+.1f}σ)")
        else:
            drivers.append("Multi-KPI Deviation")

    df["Severity Level"] = severities
    df["Primary KPI Deviation"] = drivers
    return df


def render(pipeline: Optional[AnomalyDetectorPipeline] = None, dataset_tuple: Optional[Tuple] = None, benchmark_tables: Any = None):
    """Render the AI Detection configuration and execution screen."""
    if pipeline is None:
        pipeline = AnomalyDetectorPipeline()

    render_hero_banner(
        title="Configure AI Detection",
        subtitle="Select detection engine, calibrate alert sensitivity, and scan telemetry batches for network anomalies.",
        badge="04 — AI CONFIGURATION",
    )

    active_df = st.session_state.get("active_dataset_df")
    active_name = st.session_state.get("active_dataset_name", "Standard Field Batch (25 Cell Windows)")
    if active_df is None:
        active_df = create_synthetic_demo_features(n_samples=25, random_state=42)
        st.session_state["active_dataset_df"] = active_df
        st.session_state["active_dataset_name"] = active_name

    persona = st.session_state.get("user_persona", "NOC Engineer (Standard)")
    is_ml_mode = "AI/ML" in persona

    col_model, col_thresh = st.columns([1.2, 1])

    # 1. Model Selection
    with col_model:
        st.markdown("### 1. Detection Engine Architecture")

        model_options = [
            (
                "random_forest",
                "Random Forest Ensemble (Recommended Champion)",
                "Supervised ensemble trained on 32k cell traces. Highest precision (99.2%) and robust F1 (0.934). Minimizes false NOC alarms.",
                "Supervised",
            ),
            (
                "logistic_regression",
                "Logistic Regression Baseline",
                "Linear classifier with sub-10ms response time for low-latency edge nodes. 92.5% precision.",
                "Supervised",
            ),
            (
                "isolation_forest",
                "Isolation Forest (Zero-Day Discovery)",
                "Unsupervised tree isolation. Detects unprecedented, unlabelled signal anomalies without requiring historical incident flags.",
                "Unsupervised",
            ),
            (
                "local_outlier_factor",
                "Local Outlier Factor (Density-Based)",
                "Density-based unsupervised clustering. Identifies cell windows operating in sparse, anomalous KPI manifolds.",
                "Unsupervised",
            ),
        ]

        model_labels = [m[1] for m in model_options]
        chosen_idx = st.radio(
            "Select Detection Model:",
            range(len(model_labels)),
            format_func=lambda i: model_labels[i],
            key="detection_model_idx",
        )

        selected_model_id = model_options[chosen_idx][0]
        selected_model_title = model_options[chosen_idx][1]
        selected_model_desc = model_options[chosen_idx][2]
        selected_model_family = model_options[chosen_idx][3]

        render_html(
            f"""
            <div style="background: #0E1524; border: 1px solid #1A263D; border-left: 3px solid #0EA5E9;
                        border-radius: 6px; padding: 12px 14px; margin-top: 10px; font-size: 0.84rem; color: #94A3B8;">
                <strong style="color: #F8FAFC;">Engine Architecture:</strong> {selected_model_desc}
            </div>
            """
        )

    # 2. Alert Sensitivity & Threshold Calibration
    with col_thresh:
        st.markdown("### 2. Alert Sensitivity Calibration")

        if not is_ml_mode:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 8px;">
                    Calibrated for NOC operations to balance fault isolation against alarm fatigue:
                </div>
                """,
                unsafe_allow_html=True,
            )

            preset_choice = st.selectbox(
                "Operational Sensitivity Preset:",
                [
                    "Balanced NOC Standard (Recommended — tau = 0.30)",
                    "High Sensitivity (Early Warning — tau = 0.15)",
                    "Conservative (Zero False Alarms — tau = 0.50)",
                ],
                index=0,
                key="noc_preset_choice",
            )

            if "0.15" in preset_choice:
                threshold_val = 0.15
            elif "0.50" in preset_choice:
                threshold_val = 0.50
            else:
                threshold_val = 0.30

            st.caption(f"Decision Boundary: Probability Cutoff = {threshold_val:.2f}")

        else:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #38BDF8; margin-bottom: 8px;">
                    Advanced Mode Enabled — Continuous Decision Threshold Control:
                </div>
                """,
                unsafe_allow_html=True,
            )

            threshold_val = st.slider(
                "Decision Cutoff (tau):",
                min_value=0.05,
                max_value=0.95,
                value=0.30,
                step=0.05,
                help="Threshold probability for classifying a window as anomalous.",
                key="ml_threshold_slider",
            )

            render_html(
                f"""
                <div style="background: #080C14; border: 1px solid #162032; border-radius: 6px; padding: 10px; font-size: 0.8rem; color: #CBD5E1;">
                    <div>Cutoff: <strong style="color: #38BDF8;">tau = {threshold_val:.2f}</strong></div>
                    <div>Validation F1: <strong>0.934</strong> | Test Precision: <strong>99.2%</strong></div>
                </div>
                """
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_html(
            f"""
            <div style="background: #0B101B; border: 1px solid #1A263D; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase;">Active Telemetry Target</div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 0.92rem;">{active_name}</div>
                <div style="color: #10B981; font-size: 0.78rem; margin-top: 2px;">● {len(active_df)} Cell Windows Loaded</div>
            </div>
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Execution Action
    col_run, col_space = st.columns([1.5, 1])
    with col_run:
        if st.button("Execute AI Detection Scan", use_container_width=True, key="btn_run_detection_exec"):
            with st.spinner("Processing telemetry through AI pipeline..."):
                try:
                    preds_df = pipeline.predict(
                        active_df,
                        model_id=selected_model_id,
                        threshold=threshold_val,
                    )

                    if "Record ID" not in preds_df.columns:
                        preds_df["Record ID"] = [f"CELL-TRACE-{i+1:04d}" for i in range(len(preds_df))]

                    annotated_df = _compute_record_severity_and_driver(
                        pipeline=pipeline,
                        results_df=preds_df,
                        model_type=selected_model_family,
                        threshold=threshold_val,
                    )

                    st.session_state["last_results"] = annotated_df
                    st.session_state["last_model_id"] = selected_model_id
                    st.session_state["last_model_name"] = selected_model_title
                    st.session_state["last_threshold"] = threshold_val

                    n_anomalies = int(annotated_df["anomaly_prediction"].sum())
                    st.success(f"Detection Complete: Scanned {len(annotated_df)} windows — flagged {n_anomalies} anomalies.")
                    navigate_to("05 — Alert Management")

                except Exception as e:
                    st.error(f"Inference Pipeline Error: {e}")

    if "last_results" in st.session_state and st.session_state["last_results"] is not None:
        with col_space:
            n_curr = int(st.session_state["last_results"]["anomaly_prediction"].sum())
            if st.button(f"View Scan Results ({n_curr} Alerts)", use_container_width=True, key="btn_to_existing_alerts"):
                navigate_to("05 — Alert Management")
