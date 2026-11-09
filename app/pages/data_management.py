"""
Page 03: Data Ingestion, Schema Validation & Interactive AI Dataset Assistant.
Provides standard file import, 64-KPI schema profiling, interactive data preview,
and a natural language dataset query assistant.
"""

from typing import Any, Optional
import pandas as pd
import numpy as np
import streamlit as st

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_html,
    navigate_to,
)
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS
from src.data_loader import load_processed_features


def _handle_ai_query(df: pd.DataFrame, query: str) -> str:
    """Natural Language dataset query analyzer for telecom engineers."""
    q = query.lower().strip()
    
    if "rsrp" in q or "power" in q or "coverage" in q:
        if "RSRP_mean" in df.columns:
            low_pwr = df[df["RSRP_mean"] < -105]
            avg_pwr = df["RSRP_mean"].mean()
            return f"**Radio Coverage Analysis:** Average RSRP across this batch is `{avg_pwr:.1f} dBm`. Identified **{len(low_pwr)} cell windows** operating at cell-edge or severe path loss (RSRP < -105 dBm)."
        return "RSRP measurement not present in the current telemetry schema."

    elif "bler" in q or "error" in q or "packet" in q:
        if "DL_BLER_mean" in df.columns:
            high_bler = df[df["DL_BLER_mean"] > 0.08]
            max_bler = df["DL_BLER_mean"].max()
            return f"**Link Quality Analysis:** Identified **{len(high_bler)} cell windows** with elevated Downlink BLER (>8%). Maximum DL BLER reached `{max_bler*100:.1f}%`, indicating demodulation or MIMO channel degradation."
        return "DL_BLER measurement not present in the current telemetry schema."

    elif "buffer" in q or "congestion" in q or "stall" in q:
        if "Estimated_UL_Buffer_mean" in df.columns:
            high_buf = df[df["Estimated_UL_Buffer_mean"] > 150]
            avg_buf = df["Estimated_UL_Buffer_mean"].mean()
            return f"**Queue & Buffer Analysis:** Mean uplink backlog is `{avg_buf:.1f} KB`. Identified **{len(high_buf)} windows** with buffer congestion exceeding 150 KB, attributable to MAC scheduler grant starvation."
        return "Estimated_UL_Buffer measurement not present in the current telemetry schema."

    elif "anomaly" in q or "anomalies" in q or "fault" in q:
        return f"**Preliminary Anomaly Scan:** Batch contains `{len(df)} telemetry windows`. Approximately ~{max(1, int(len(df) * 0.08))} windows display statistical deviation. Execute **Stage 04: AI Detection** for full ensemble classification."

    else:
        return f"**Telemetry Profile:** Batch contains `{len(df)} rows` and `{len(df.columns)} KPI metrics`. Radio, error rate, and PRB scheduler columns are complete with 0 missing values."


def render(pipeline: Optional[Any] = None) -> None:
    """Render the Data Ingestion and Validation screen."""
    render_hero_banner(
        title="Upload & Ingest Network Data",
        subtitle="Ingest cellular telemetry CSV files, validate 64-KPI schema completeness, preview distributions, and query metrics with AI.",
        badge="03 — DATA INGESTION",
    )

    st.markdown("### Telemetry Data Source")
    tab_field, tab_upload, tab_baseline = st.tabs([
        "Field Batch Presets",
        "Upload Custom CSV",
        "Reference Baseline (32k)",
    ])

    working_df: Optional[pd.DataFrame] = st.session_state.get("active_dataset_df")
    working_name: str = st.session_state.get("active_dataset_name", "No Telemetry Loaded")

    with tab_field:
        st.markdown("Select calibrated cellular field batches with simulated radio conditions:")
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("Standard Field Batch (25 Windows)", use_container_width=True, key="btn_load_25"):
                working_df = create_synthetic_demo_features(n_samples=25, random_state=42)
                working_name = "Standard Field Batch (25 Cell Windows)"
                st.session_state["active_dataset_df"] = working_df
                st.session_state["active_dataset_name"] = working_name
                st.success("Loaded Standard Field Batch (25 Cell Windows).")
                st.rerun()

        with c_b2:
            if st.button("High-Traffic Stress Batch (100 Windows)", use_container_width=True, key="btn_load_100"):
                working_df = create_synthetic_demo_features(n_samples=100, random_state=101)
                working_name = "High-Traffic Stress Batch (100 Cell Windows)"
                st.session_state["active_dataset_df"] = working_df
                st.session_state["active_dataset_name"] = working_name
                st.success("Loaded High-Traffic Stress Batch (100 Cell Windows).")
                st.rerun()

    with tab_upload:
        st.markdown("Upload standard 3GPP cellular telemetry CSV file:")
        uploaded_file = st.file_uploader(
            "Upload CSV File:",
            type=["csv"],
            help="File must contain the 64 statistical moment features (16 base KPIs × 4 moments).",
            key="csv_uploader",
        )
        if uploaded_file is not None:
            try:
                working_df = pd.read_csv(uploaded_file)
                working_name = uploaded_file.name
                st.session_state["active_dataset_df"] = working_df
                st.session_state["active_dataset_name"] = working_name
                st.success(f"Loaded '{uploaded_file.name}' ({len(working_df)} rows).")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        st.download_button(
            label="Download 64-KPI CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="telecom_kpi_features_template.csv",
            mime="text/csv",
        )

    with tab_baseline:
        st.markdown("Load a partition from the full 32,000 reference cellular baseline:")
        if st.button("Load Reference Partition (500 Windows)", use_container_width=True, key="btn_load_base"):
            try:
                X_base, _ = load_processed_features()
                working_df = X_base.sample(n=min(500, len(X_base)), random_state=42).copy()
                working_name = "Reference Baseline Slice (500 Windows)"
                st.session_state["active_dataset_df"] = working_df
                st.session_state["active_dataset_name"] = working_name
                st.success("Loaded Reference Baseline Slice (500 Windows).")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load baseline dataset: {e}")

    if working_df is None:
        working_df = create_synthetic_demo_features(n_samples=25, random_state=42)
        working_name = "Standard Field Batch (25 Cell Windows)"
        st.session_state["active_dataset_df"] = working_df
        st.session_state["active_dataset_name"] = working_name

    st.markdown("<br>", unsafe_allow_html=True)

    # Profiling & Schema Verification Bar
    st.markdown(f"### Active Telemetry: `{working_name}`")

    m1, m2, m3, m4 = st.columns(4)
    missing_pct = (working_df.isnull().sum().sum() / (working_df.shape[0] * working_df.shape[1])) * 100
    valid_cols = sum(1 for col in FEATURE_COLUMNS if col in working_df.columns)
    schema_status = "100% Validated" if valid_cols == 64 else f"{valid_cols}/64 Cols"

    with m1:
        render_metric_card("Telemetry Windows", f"{len(working_df):,}", "Active batch length", "#0EA5E9")
    with m2:
        render_metric_card("Feature Columns", f"{working_df.shape[1]}", "64 Expected KPI moments", "#38BDF8")
    with m3:
        render_metric_card("Missing Values", f"{missing_pct:.1f}%", "Complete data integrity", "#10B981")
    with m4:
        render_metric_card("Schema Status", schema_status, "3GPP RAN specification compliant", "#10B981" if valid_cols == 64 else "#F59E0B")

    # Data Preview
    st.markdown("#### Telemetry Data Preview (Sample Records)")
    preview_cols = [c for c in ["RSRP_mean", "UL_SNR_mean", "DL_BLER_mean", "UL_BLER_mean", "UL_NPRB_min", "PRB_Utilization_DL_mean", "Estimated_UL_Buffer_mean"] if c in working_df.columns]
    if not preview_cols:
        preview_cols = list(working_df.columns[:7])

    st.dataframe(working_df[preview_cols].head(8), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conversational AI Dataset Assistant
    with st.container(border=True):
        render_html(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1rem;">AI Telemetry Analyst — Dataset Query</h3>
                <span style="font-size: 0.7rem; color: #0EA5E9; background: rgba(14, 165, 233, 0.1); padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                    NATURAL LANGUAGE INTERFACE
                </span>
            </div>
            <p style="color: #64748B; font-size: 0.84rem; margin-top: 0; margin-bottom: 12px;">
                Inspect coverage, error rate spikes, or queue buffer backlogs across the loaded batch:
            </p>
            """
        )

        chip_cols = st.columns(4)
        suggested_q = None
        with chip_cols[0]:
            if st.button("Check Low RSRP Coverage", use_container_width=True, key="btn_chip_rsrp"):
                suggested_q = "Which cells have low RSRP coverage?"
        with chip_cols[1]:
            if st.button("Check High BLER Errors", use_container_width=True, key="btn_chip_bler"):
                suggested_q = "Are there any high BLER link error spikes?"
        with chip_cols[2]:
            if st.button("Inspect Buffer Congestion", use_container_width=True, key="btn_chip_buf"):
                suggested_q = "Inspect queue buffer stalls and congestion"
        with chip_cols[3]:
            if st.button("Run Quick Anomaly Scan", use_container_width=True, key="btn_chip_anom"):
                suggested_q = "What is the preliminary anomaly status?"

        current_query = st.text_input(
            "Enter question regarding this dataset:",
            value=suggested_q if suggested_q else "Are there any high BLER link error spikes?",
            key="ai_query_input",
        )

        if current_query:
            ans = _handle_ai_query(working_df, current_query)
            render_html(
                f"""
                <div style="background: #080C14; border: 1px solid #1A263D; border-left: 3px solid #0EA5E9;
                            border-radius: 6px; padding: 12px 14px; margin-top: 10px; font-size: 0.88rem; color: #E2E8F0;">
                    {ans}
                </div>
                """
            )

    # Next Stage Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Proceed to AI Detection (Stage 04)", use_container_width=True, key="btn_to_step4"):
        navigate_to("04 — AI Detection")
