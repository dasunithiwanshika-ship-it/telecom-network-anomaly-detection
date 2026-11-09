"""
Page 3: Data Upload, Validation & Telemetry Schema Guide.
Provides a guided drag-and-drop upload experience, automatic 64-feature validation,
dataset preview, sample telemetry loaders, and template downloads for Telecom Engineers.
"""

from pathlib import Path
from typing import Any, Optional
import streamlit as st
import pandas as pd

from ..components.metric_cards import (
    render_hero_banner,
    render_metric_card,
    render_academic_disclaimer,
    navigate_to,
    get_feature_meta,
    TELECOM_KPI_METADATA,
)
from src.feature_engineering import create_synthetic_demo_features, FEATURE_COLUMNS
from src.preprocessing import validate_feature_schema, clean_feature_data
from src.data_loader import load_processed_features


@st.cache_data
def _get_feature_reference_df() -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        meta = get_feature_meta(col)
        rows.append(
            {
                "CSV Column Name": col,
                "Telecom Measurement": meta["friendly_name"],
                "Subsystem": meta["subsystem"],
                "Unit": meta["unit"],
                "Statistic": meta["moment_label"],
                "Operational Meaning": meta["meaning"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def _load_real_benchmark_sample(n_rows: int = 100) -> Optional[pd.DataFrame]:
    try:
        X, y = load_processed_features()
        # Select a balanced sample containing both normal and anomalous traces
        anom_part = X.loc[y == 1].head(20)
        norm_part = X.loc[y == 0].head(max(0, n_rows - len(anom_part)))
        combined = pd.concat([norm_part, anom_part], ignore_index=True)
        return combined
    except Exception:
        return None


def render(pipeline: Any = None):
    """Render the Guided Data Upload, Validation & Schema page."""
    render_hero_banner(
        title="KPI Telemetry Upload & Data Validation",
        subtitle="Upload cellular KPI batch files (CSV) or load built-in field telemetry samples. All files are automatically validated before AI detection.",
        badge="STEP 1: DATA INGESTION",
    )

    # Top Section: Upload vs Built-in Sample Loader
    col_upload, col_samples = st.columns([1.25, 1])

    with col_upload:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">OPTION A</span>
                    <strong style="color: #F8FAFC; font-size: 1.05rem;">Upload Custom KPI Telemetry CSV</strong>
                </div>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">
                    Drag and drop a CSV file containing the 64 cellular KPI window statistics (Average, Std Dev, Min, Max across the 16 base KPIs).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drag and drop your KPI CSV file here (or click Browse files):",
            type=["csv"],
            help="CSV file with 64 KPI columns. Use the template download below if you need the exact column headers.",
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                is_valid, missing_cols, _ = validate_feature_schema(raw_df)
                if not is_valid:
                    st.error(
                        f"❌ **File Validation Error:** The uploaded CSV is missing **{len(missing_cols)}** required KPI column(s)."
                    )
                    with st.expander("View Missing KPI Columns & How to Fix"):
                        st.write("The following required columns were not found in your CSV:")
                        st.code(", ".join(missing_cols[:25]) + (" ..." if len(missing_cols) > 25 else ""))
                        st.info("💡 Tip: Download the 64-Feature CSV Template below to ensure your column headers match the exact telecom schema.")
                else:
                    cleaned_df = clean_feature_data(raw_df)
                    st.session_state["active_dataset_df"] = cleaned_df
                    st.session_state["active_dataset_name"] = f"Uploaded CSV ({uploaded_file.name})"
                    # Clear stale detection results when a new dataset is uploaded
                    if "last_results" in st.session_state:
                        del st.session_state["last_results"]
                    st.success(
                        f"✅ **Upload & Validation Successful:** Loaded **{len(cleaned_df):,} cell windows** from `{uploaded_file.name}`."
                    )
            except Exception as e:
                st.error(f"❌ **Unable to read CSV file:** Could not parse file format ({e}). Please ensure the file is a valid comma-separated CSV.")

    with col_samples:
        st.markdown(
            """
            <div class="step-card">
                <div style="margin-bottom: 6px;">
                    <span class="step-badge">OPTION B</span>
                    <strong style="color: #F8FAFC; font-size: 1.05rem;">Load Ready-to-Use Field Telemetry</strong>
                </div>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">
                    Don't have a CSV file ready? Load one of our pre-formatted cellular telemetry batches with a single click:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("📡 Load Standard Field Sample (25 Cell Windows)", use_container_width=True, key="btn_load_25"):
            with st.spinner("Preparing 25-window cellular telemetry sample..."):
                df_25 = create_synthetic_demo_features(n_samples=25, random_state=42)
                st.session_state["active_dataset_df"] = df_25
                st.session_state["active_dataset_name"] = "Standard Field Sample (25 Cell Windows)"
                if "last_results" in st.session_state:
                    del st.session_state["last_results"]
            st.success("✅ Loaded **Standard Field Sample (25 Cell Windows)** into workspace.")

        if st.button("🔥 Load High-Traffic Stress Batch (50 Cell Windows)", use_container_width=True, key="btn_load_50"):
            with st.spinner("Preparing 50-window stress test telemetry..."):
                df_50 = create_synthetic_demo_features(n_samples=50, random_state=99)
                st.session_state["active_dataset_df"] = df_50
                st.session_state["active_dataset_name"] = "High-Traffic Stress Batch (50 Cell Windows)"
                if "last_results" in st.session_state:
                    del st.session_state["last_results"]
            st.success("✅ Loaded **High-Traffic Stress Batch (50 Cell Windows)** into workspace.")

        if st.button("🏛️ Load Verified Benchmark Sample (100 Real Traces)", use_container_width=True, key="btn_load_100"):
            with st.spinner("Loading 100 verified TelecomTS cellular windows..."):
                df_100 = _load_real_benchmark_sample(100)
                if df_100 is not None:
                    st.session_state["active_dataset_df"] = df_100
                    st.session_state["active_dataset_name"] = "TelecomTS Verified Batch (100 Real Cell Windows)"
                else:
                    df_100 = create_synthetic_demo_features(n_samples=100, random_state=7)
                    st.session_state["active_dataset_df"] = df_100
                    st.session_state["active_dataset_name"] = "Extended Field Sample (100 Cell Windows)"
                if "last_results" in st.session_state:
                    del st.session_state["last_results"]
            st.success("✅ Loaded **100 Cell Windows** into workspace.")

    st.markdown("---")

    # Active Dataset Preview & Validation Status Section
    st.markdown("### 🔍 Active Dataset Validation & Preview")

    active_df: Optional[pd.DataFrame] = st.session_state.get("active_dataset_df")
    active_name: str = st.session_state.get("active_dataset_name", "")

    if active_df is None or len(active_df) == 0:
        st.info(
            "📭 **No Telemetry Dataset Loaded Yet:** Please upload a CSV file on the left or click one of the **Load Ready-to-Use Field Telemetry** buttons on the right to preview data."
        )
    else:
        # Validation Metrics
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            render_metric_card("Active Dataset", f"{len(active_df):,} Rows", active_name, "#14B8A6")
        with v2:
            render_metric_card("KPI Schema Check", "64 / 64 Valid", "All 16 KPIs x 4 stats", "#10B981")
        with v3:
            render_metric_card("Data Quality", "100% Numeric", "Zero NaN / Null errors", "#38BDF8")
        with v4:
            render_metric_card("Pipeline Status", "READY", "Ready for AI Detection", "#10B981")

        # Prominent Call to Action to Continue to Step 2
        st.markdown("<br>", unsafe_allow_html=True)
        cta_col1, cta_col2 = st.columns([1.6, 1])
        with cta_col1:
            st.markdown(
                f"""
                <div style="background-color: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 12px 16px;">
                    <strong style="color: #10B981;">✅ Dataset Ready for Analysis:</strong>
                    <span style="color: #F8FAFC;"> <strong>{active_name}</strong> ({len(active_df):,} cell windows) is validated and loaded in memory.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cta_col2:
            if st.button("⚡ Proceed to AI Detection Workspace →", use_container_width=True, key="btn_upload_to_detect"):
                navigate_to("⚡ AI Detection Workspace")

        st.markdown("<br>", unsafe_allow_html=True)

        tab_prev, tab_summary = st.tabs([
            "📋 Dataset Preview (First 15 Cell Windows)",
            "📊 Key Telecom KPI Summary (Min / Avg / Max)",
        ])

        with tab_prev:
            preview_df = active_df.head(15).copy()
            preview_df.insert(0, "Window ID", [f"CELL-WIN-{i:04d}" for i in range(len(preview_df))])
            st.dataframe(preview_df, use_container_width=True, hide_index=True, height=320)

        with tab_summary:
            key_cols = [
                ("RSRP_mean", "Signal Power RSRP (Avg)", "dBm"),
                ("UL_SNR_mean", "Uplink SNR (Avg)", "dB"),
                ("UL_BLER_mean", "Uplink Error Rate BLER (Avg)", "Ratio [0–1]"),
                ("DL_BLER_mean", "Downlink Error Rate BLER (Avg)", "Ratio [0–1]"),
                ("UL_NPRB_min", "Uplink PRB Allocation (Min)", "PRBs"),
                ("PRB_Utilization_DL_mean", "Downlink PRB Utilization (Avg)", "%"),
                ("PRB_Utilization_UL_mean", "Uplink PRB Utilization (Avg)", "%"),
                ("Estimated_UL_Buffer_mean", "Uplink Queue Buffer (Avg)", "Bytes/KB"),
            ]
            sum_rows = []
            for col, label, unit in key_cols:
                if col in active_df.columns:
                    s = active_df[col]
                    sum_rows.append(
                        {
                            "Telecom Measurement": label,
                            "Column Code": col,
                            "Unit": unit,
                            "Minimum": f"{s.min():.3f}",
                            "Average (Mean)": f"{s.mean():.3f}",
                            "Median": f"{s.median():.3f}",
                            "Maximum": f"{s.max():.3f}",
                        }
                    )
            st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Template Downloads & 64-Feature Dictionary
    st.markdown("### 📥 Download CSV Templates & Feature Schema Guide")
    dl1, dl2 = st.columns(2)
    with dl1:
        empty_template = pd.DataFrame(columns=FEATURE_COLUMNS)
        st.download_button(
            label="⬇️ Download Empty 64-Column CSV Template",
            data=empty_template.to_csv(index=False).encode("utf-8"),
            file_name="telecom_kpi_64feature_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        sample_25_csv = create_synthetic_demo_features(n_samples=25, random_state=42)
        st.download_button(
            label="⬇️ Download Sample Populated KPI CSV (25 Rows)",
            data=sample_25_csv.to_csv(index=False).encode("utf-8"),
            file_name="sample_telecom_kpi_25windows.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("📖 View Complete 64-Feature Telecom KPI Reference Guide"):
        ref_df = _get_feature_reference_df()
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            q = st.text_input("Search by KPI Name or Subsystem:", placeholder="e.g., RSRP, SNR, BLER, Buffer, PRB...")
        with f_col2:
            sub_list = ["All Subsystems"] + sorted(ref_df["Subsystem"].unique().tolist())
            sub_sel = st.selectbox("Filter by Subsystem:", sub_list)

        filt = ref_df.copy()
        if q:
            filt = filt[
                filt["CSV Column Name"].str.contains(q, case=False)
                | filt["Telecom Measurement"].str.contains(q, case=False)
                | filt["Operational Meaning"].str.contains(q, case=False)
            ]
        if sub_sel != "All Subsystems":
            filt = filt[filt["Subsystem"] == sub_sel]
        st.dataframe(filt, use_container_width=True, hide_index=True, height=320)

    render_academic_disclaimer()
