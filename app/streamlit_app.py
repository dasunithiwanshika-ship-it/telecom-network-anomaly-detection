"""
Telecom AI Anomaly Intelligence Dashboard
Interactive web interface for comparing supervised and unsupervised anomaly detection models
on public cellular network KPI data (AliMaatouk/TelecomTS).
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to sys.path so src imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import NUMERIC_KPIS, FEATURE_COLUMNS, load_processed_features
from src.prediction import AnomalyDetectorPipeline
from src.feature_engineering import create_synthetic_demo_features

# -------------------------------------------------------------
# Streamlit App Configuration & Custom Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="Telecom AI Anomaly Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Telecom Dark/Navy Styling
st.markdown(
    """
    <style>
    /* Dark Navy/Slate Header Theme */
    .main-header {
        background: linear-gradient(135deg, #0a192f 0%, #172a45 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .metric-card {
        background: #112240;
        border: 1px solid #233554;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #00e5ff;
        margin: 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #a8b2d1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .badge-normal {
        background-color: #065f46;
        color: #a7f3d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-anomaly {
        background-color: #991b1b;
        color: #fecaca;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .disclaimer-box {
        background-color: rgba(254, 243, 199, 0.08);
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #fbbf24;
        margin: 15px 0 25px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# Cached Resource Loaders
# -------------------------------------------------------------
@st.cache_resource
def get_prediction_pipeline() -> AnomalyDetectorPipeline:
    return AnomalyDetectorPipeline(models_dir=os.path.join(project_root, "models"))


@st.cache_data
def get_dataset_sample():
    pkl_path = os.path.join(project_root, "data", "processed", "telecom_features_64.pkl")
    if os.path.exists(pkl_path):
        X, y = load_processed_features(pkl_path)
        return X, y
    return None, None


@st.cache_data
def get_benchmark_tables():
    res_dir = os.path.join(project_root, "results")
    comp_path = os.path.join(res_dir, "unified_4model_comparison_table.csv")
    perm_path = os.path.join(res_dir, "rf_permutation_importance.csv")
    gini_path = os.path.join(res_dir, "rf_gini_feature_importance.csv")
    thresh_path = os.path.join(res_dir, "threshold_sensitivity_analysis.csv")
    iso_sens_path = os.path.join(res_dir, "isolation_forest_contamination_sensitivity.csv")
    lof_sens_path = os.path.join(res_dir, "lof_k_sensitivity.csv")

    tables = {}
    if os.path.exists(comp_path):
        tables["comparison"] = pd.read_csv(comp_path)
    if os.path.exists(perm_path):
        tables["permutation"] = pd.read_csv(perm_path)
    if os.path.exists(gini_path):
        tables["gini"] = pd.read_csv(gini_path)
    if os.path.exists(thresh_path):
        tables["threshold"] = pd.read_csv(thresh_path)
    if os.path.exists(iso_sens_path):
        tables["iso_sens"] = pd.read_csv(iso_sens_path)
    if os.path.exists(lof_sens_path):
        tables["lof_sens"] = pd.read_csv(lof_sens_path)

    return tables


# -------------------------------------------------------------
# Sidebar Navigation & Context
# -------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?auto=format&fit=crop&w=400&q=80",
        caption="Cellular KPI Intelligence",
        use_container_width=True,
    )
    st.markdown("## 📡 Navigation")
    page = st.radio(
        "Select Module:",
        [
            "🏠 1. Overview",
            "📊 2. Dataset Explorer",
            "🔍 3. Anomaly Detection Demo",
            "⚖️ 4. Model Comparison",
            "💡 5. Feature Importance & Explainability",
            "📖 6. Research & About",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    pipeline = get_prediction_pipeline()
    available_models = [m for m in pipeline.AVAILABLE_MODELS if pipeline.is_model_available(m)]
    st.success(f"**{len(available_models)}/4 Models Loaded**")
    for m in pipeline.AVAILABLE_MODELS:
        status_icon = "🟢" if pipeline.is_model_available(m) else "🔴"
        st.markdown(f"{status_icon} `{m}`")

    st.markdown("---")
    st.caption("AI/ML Research Prototype | AliMaatouk/TelecomTS")


# -------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------
if page == "🏠 1. Overview":
    st.markdown(
        """
        <div class="main-header">
            <h1>Telecom AI Anomaly Intelligence Dashboard</h1>
            <p>Comparative Evaluation of Supervised and Unsupervised Machine Learning Models for Cellular Network Anomaly Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Academic Research & Prototype Limitation Notice:</strong><br>
            This system is an independent research prototype evaluated on the public <code>AliMaatouk/TelecomTS</code> benchmark dataset.
            The data is <strong>NOT</strong> SLTMobitel operational network telemetry, does not reflect Sri Lankan telecommunication infrastructure,
            and does not predict physical field dispatch or hardware breakdowns. Results demonstrate comparative algorithmic detection performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">32,000</div><div class="metric-lbl">Total Traces</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">30,765</div><div class="metric-lbl">Normal Traces (96.14%)</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-val" style="color:#ef4444">1,235</div><div class="metric-lbl">Anomalies (3.86%)</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">64</div><div class="metric-lbl">KPI Feature Moments</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">4</div><div class="metric-lbl">Evaluated Models</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.subheader("🎯 Research Problem & Core Question")
        st.markdown(
            """
            In modern cellular networks (4G LTE / 5G), operators continuously record multi-dimensional Key Performance Indicators (KPIs)
            including radio signal levels, packet error rates, modulation schemes, and buffer occupancies. Detecting abnormal network states
            (such as uplink jamming, buffer overflows, and handover anomalies) is critical for service reliability.

            **Core Research Question:**
            > *How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?*
            """
        )

        st.markdown("### 🔄 End-to-End Methodological Architecture")
        st.markdown(
            r"""
            1. **Raw Telemetry Ingestion**: 32,000 multi-channel traces (128 time steps per sample across 16 numerical KPIs).
            2. **Feature Engineering**: Extraction of 4 summary moments ($\mu, \sigma, \min, \max$) yielding **64 engineered features**.
            3. **Supervised Modeling**:
               - *Logistic Regression (GLM Baseline)* with L2 regularization and feature standardization.
               - *Random Forest Classifier (Bagged Ensemble)* tuned via 5-Fold Stratified Cross-Validation.
            4. **Unsupervised Anomaly Detection**:
               - *Isolation Forest* isolating sparse points via recursive random hyperplanes.
               - *Local Outlier Factor (LOF)* evaluating local reachability densities under novelty detection.
            5. **Evaluation & Explainability**: Stratified test evaluation ($N=6,400$), error decomposition, Gini MDI, and Permutation Importance.
            """
        )

    with c_right:
        st.subheader("📊 Dataset Class Distribution")
        donut_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Normal Cellular Traces", "Anomalous Traces"],
                    values=[30765, 1235],
                    hole=0.55,
                    marker=dict(colors=["#10b981", "#ef4444"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        donut_fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )
        st.plotly_chart(donut_fig, use_container_width=True)

        st.info(
            "💡 **Class Imbalance:** Anomalies represent only 3.86% of the dataset. "
            "Precision-Recall AUC and F1-score are the primary evaluation metrics rather than accuracy or ROC-AUC."
        )


# -------------------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------------------
elif page == "📊 2. Dataset Explorer":
    st.markdown(
        """
        <div class="main-header">
            <h1>Dataset Explorer & KPI Analytics</h1>
            <p>Inspection of the 16 Numerical Cellular KPIs & 64 Statistical Moments</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    X, y = get_dataset_sample()
    if X is None:
        st.error("Processed features file not found. Please ensure data preprocessing has completed.")
    else:
        st.markdown("### 🗂️ Cellular KPI Categories")
        kpi_groups = {
            "Radio Signal Quality": ["RSRP", "UL_SNR"],
            "Packet Error Rates": ["UL_BLER", "DL_BLER"],
            "Modulation & Coding": ["UL_MCS", "DL_MCS"],
            "Resource Allocation (PRB)": ["UL_NPRB", "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL", "PRB_Utilization_UL"],
            "Throughput & Traffic Volume": ["TX_Bytes", "RX_Bytes", "UL_NumberOfPackets", "DL_NumberOfPackets"],
            "Buffer Health": ["Estimated_UL_Buffer"],
        }

        col1, col2, col3 = st.columns(3)
        for i, (grp_name, kpis) in enumerate(kpi_groups.items()):
            col = [col1, col2, col3][i % 3]
            with col:
                st.markdown(f"**{grp_name}:**")
                st.markdown(" ".join([f"`{k}`" for k in kpis]))

        st.markdown("---")
        st.subheader("📈 Statistical Distribution: Normal vs. Anomaly")

        selected_kpi = st.selectbox(
            "Select KPI to examine distribution:",
            NUMERIC_KPIS,
            index=NUMERIC_KPIS.index("RSRP"),
        )
        selected_moment = st.selectbox("Select Statistical Moment:", ["mean", "std", "min", "max"], index=0)
        feature_col = f"{selected_kpi}_{selected_moment}"

        plot_df = pd.DataFrame({
            feature_col: X[feature_col].values,
            "Class": np.where(y.values == 1, "Anomaly (1)", "Normal (0)"),
        })

        c1, c2 = st.columns(2)
        with c1:
            box_fig = px.box(
                plot_df,
                x="Class",
                y=feature_col,
                color="Class",
                color_discrete_map={"Normal (0)": "#10b981", "Anomaly (1)": "#ef4444"},
                title=f"Boxplot Comparison: {feature_col}",
            )
            box_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(box_fig, use_container_width=True)

        with c2:
            hist_fig = px.histogram(
                plot_df,
                x=feature_col,
                color="Class",
                barmode="overlay",
                marginal="box",
                color_discrete_map={"Normal (0)": "#10b981", "Anomaly (1)": "#ef4444"},
                title=f"Histogram Overlay: {feature_col}",
            )
            hist_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Feature Matrix Sample Preview")
        st.dataframe(X.head(10), use_container_width=True)


# -------------------------------------------------------------
# PAGE 3: ANOMALY DETECTION DEMO
# -------------------------------------------------------------
elif page == "🔍 3. Anomaly Detection Demo":
    st.markdown(
        """
        <div class="main-header">
            <h1>Interactive Anomaly Detection Demo</h1>
            <p>Upload a CSV with the 64 engineered KPI features and run live batch predictions across trained models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_prediction_pipeline()

    col_ctrl, col_info = st.columns([1.2, 1])

    with col_ctrl:
        model_choice = st.selectbox(
            "Select Detection Model:",
            [
                ("random_forest", "Random Forest (Supervised Ensemble - Recommended)"),
                ("logistic_regression", "Logistic Regression (Supervised GLM Baseline)"),
                ("isolation_forest", "Isolation Forest (Unsupervised Tree Isolation)"),
                ("local_outlier_factor", "Local Outlier Factor (Unsupervised Density k-NN)"),
            ],
            format_func=lambda x: x[1],
        )[0]

        threshold = 0.50
        if model_choice in ["random_forest", "logistic_regression"]:
            threshold = st.slider(
                "Classification Decision Threshold (tau):",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="Probability cutoff above which a trace is classified as an anomaly.",
            )

    with col_info:
        st.markdown("### 📥 Need Sample Data?")
        st.markdown(
            "Download a pre-formatted synthetic test CSV containing the 64 engineered KPI feature headers. "
            "*(Synthetic demonstration data; not live telecom observations)*."
        )
        sample_csv_path = os.path.join(project_root, "app", "sample_telecom_kpi_features.csv")
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Synthetic Sample CSV (10 rows)",
                    data=f.read(),
                    file_name="sample_telecom_kpi_features.csv",
                    mime="text/csv",
                )

    st.markdown("---")
    st.subheader("📂 Upload KPI Feature CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV containing all 64 feature columns:",
        type=["csv"],
        help="The CSV must contain the 64 engineered statistical moments.",
    )

    use_demo = st.button("⚡ Or Load Synthetic Demo Data (10 Samples)")

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file loaded: {input_df.shape[0]} rows, {input_df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif use_demo:
        input_df = create_synthetic_demo_features(n_samples=10, random_state=42)
        st.info("Loaded 10 synthetic demonstration samples.")

    if input_df is not None:
        # Validate schema
        is_valid, msg, cleaned_df = pipeline.validate_input_dataframe(input_df)
        if not is_valid:
            st.error(f"❌ **Validation Failed:** {msg}")
        else:
            st.success("✅ **Input Schema Validated:** All 64 KPI features present and formatted correctly.")

            if st.button("🚀 Run Anomaly Detection"):
                with st.spinner(f"Running inference with {model_choice}..."):
                    try:
                        results = pipeline.predict(
                            cleaned_df, model_id=model_choice, threshold=threshold
                        )

                        total = len(results)
                        anomalies = int(results["anomaly_prediction"].sum())
                        normals = total - anomalies
                        rate = (anomalies / total) * 100

                        st.markdown("<br>", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Traces Processed", f"{total}")
                        with m2:
                            st.metric("Normal Traces", f"{normals}")
                        with m3:
                            st.metric("Detected Anomalies", f"{anomalies}", delta=f"{rate:.1f}% rate", delta_color="inverse")
                        with m4:
                            st.metric("Decision Threshold", f"{threshold:.2f}")

                        # Score distribution chart
                        st.subheader("📊 Anomaly Score / Probability Distribution")
                        score_fig = px.histogram(
                            results,
                            x="anomaly_score_or_prob",
                            color="anomaly_status",
                            color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
                            title=f"Distribution of Predicted Scores ({results['score_type'].iloc[0]})",
                            nbins=20,
                        )
                        score_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(score_fig, use_container_width=True)

                        st.subheader("📋 Prediction Results Table")
                        display_cols = [
                            "anomaly_status",
                            "anomaly_score_or_prob",
                            "score_type",
                            "RSRP_mean",
                            "UL_SNR_mean",
                            "UL_BLER_mean",
                            "Estimated_UL_Buffer_mean",
                        ]
                        available_display = [c for c in display_cols if c in results.columns]
                        st.dataframe(results[available_display], use_container_width=True)

                        # Download results
                        csv_output = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Full Predictions CSV",
                            data=csv_output,
                            file_name=f"telecom_anomaly_predictions_{model_choice}.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Inference error: {e}")


# -------------------------------------------------------------
# PAGE 4: MODEL COMPARISON
# -------------------------------------------------------------
elif page == "⚖️ 4. Model Comparison":
    st.markdown(
        """
        <div class="main-header">
            <h1>Comparative Evaluation: Supervised vs. Unsupervised</h1>
            <p>Empirical Performance Benchmark on Reserved Test Partition (N = 6,400 traces, 247 anomalies)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tables = get_benchmark_tables()

    if "comparison" in tables:
        st.subheader("🏆 Unified 4-Model Comparative Benchmark Table")
        st.dataframe(tables["comparison"], use_container_width=True)

        st.markdown(
            """
            > **Key Takeaway:** Supervised models (Random Forest F1: **96.44%**, Logistic Regression F1: **83.37%**)
            > significantly outperform unsupervised models (Local Outlier Factor F1: **29.90%**, Isolation Forest F1: **26.77%**).
            > Cellular KPI anomalies are contextual multi-feature interactions rather than extreme geometric outliers in all 64 dimensions.
            """
        )

    st.markdown("---")
    st.subheader("📈 ROC and Precision-Recall Curves")

    fig_path = os.path.join(project_root, "results", "unified_4model_roc_pr_comparison.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="ROC and Precision-Recall Comparison across all 4 Models", use_container_width=True)
    else:
        st.warning("ROC & PR comparison plot not found.")

    st.markdown("---")
    st.subheader("🔲 Confusion Matrices Comparison")
    cm_path = os.path.join(project_root, "results", "unified_4model_confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="2x2 Confusion Matrices Grid on Test Set (N=6,400)", use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Sensitivity & Robustness Sweeps")
    t1, t2 = st.tabs(["Isolation Forest Contamination Sweep", "LOF Neighborhood Size (k) Sweep"])
    with t1:
        if "iso_sens" in tables:
            st.dataframe(tables["iso_sens"], use_container_width=True)
            st.caption("Contamination determines the proportion of outliers and sets the internal decision threshold.")
    with t2:
        if "lof_sens" in tables:
            st.dataframe(tables["lof_sens"], use_container_width=True)
            st.caption("Neighborhood size k dictates the local density reference pool for reachability computations.")


# -------------------------------------------------------------
# PAGE 5: FEATURE IMPORTANCE & EXPLAINABILITY
# -------------------------------------------------------------
elif page == "💡 5. Feature Importance & Explainability":
    st.markdown(
        """
        <div class="main-header">
            <h1>Feature Importance & Model Explainability</h1>
            <p>Attribution Analysis of KPI Features Contributing to Cellular Anomaly Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tables = get_benchmark_tables()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Tree Gini Importance (MDI)")
        st.markdown(
            "Measures the total decrease in Gini impurity brought by splits on a given feature across all 100 decision trees."
        )
        if "gini" in tables:
            top_gini = tables["gini"].head(12)
            fig_gini = px.bar(
                top_gini[::-1],
                x="MDI_Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#10b981"],
                title="Top 12 Gini Feature Importances",
            )
            fig_gini.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gini, use_container_width=True)

    with col2:
        st.subheader("🎲 Permutation Feature Importance")
        st.markdown(
            "Model-agnostic evaluation measuring the degradation in test accuracy when values of a specific feature are randomly shuffled."
        )
        if "permutation" in tables:
            top_perm = tables["permutation"].head(12)
            fig_perm = px.bar(
                top_perm[::-1],
                x="Permutation_Mean",
                y="Feature",
                error_x=top_perm["Permutation_Std"][::-1],
                orientation="h",
                color_discrete_sequence=["#00e5ff"],
                title="Top 12 Permutation Feature Importances",
            )
            fig_perm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Methodological & Interpretability Disclaimers")
    st.markdown(
        r"""
        - **Correlation vs. Physical Causation**: High importance (e.g., in `UL_BLER`, `UL_SNR`, and `Estimated_UL_Buffer`) indicates strong statistical correlation with injected anomaly states in TelecomTS. It does **not** prove that buffer size or error rate physically caused the fault.
        - **Collinearity**: Correlated features share split importance in tree models. Standardized linear coefficients in Logistic Regression show directional log-odds but can suffer from variance inflation under collinearity.
        - **Static Moments vs. Time-Series**: Compression of 128 timesteps into 4 static moments ($\mu, \sigma, \min, \max$) loses temporal ordering, explaining why temporal anomalies (like frequent handovers) are harder to separate.
        """
    )


# -------------------------------------------------------------
# PAGE 6: RESEARCH & ABOUT
# -------------------------------------------------------------
elif page == "📖 6. Research & About":
    st.markdown(
        """
        <div class="main-header">
            <h1>Research Context & System Specifications</h1>
            <p>Academic Background, Dataset Attribution, and Viva Defense Reference</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 📚 Project Metadata
        - **Project Title**: AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks using Public Cellular Network KPI Data
        - **Research Title**: Comparative Evaluation of Supervised and Unsupervised Machine Learning Models for Network Anomaly Detection Using Public Telecom KPI Data
        - **Research Question**: *How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?*
        - **Dataset Source**: Hugging Face `AliMaatouk/TelecomTS` ([Hugging Face Link](https://huggingface.co/datasets/AliMaatouk/TelecomTS) | [GitHub Link](https://github.com/Ali-maatouk/TelecomTS))
        - **Benchmark Split**: 80/20 Stratified Partition ($N_{\text{train}}=25,600$, $N_{\text{test}}=6,400$, 247 anomalies, 3.86% prevalence).

        ---

        ### 🔬 Core Scientific Conclusions
        1. **Supervised vs. Unsupervised Disparity**: Supervised Random Forest achieves **96.44% F1-score** and **99.54% PR-AUC** with zero false alarms on the test set. In contrast, unsupervised Isolation Forest achieves **26.77% F1-score** and Local Outlier Factor achieves **29.90% F1-score**.
        2. **Why Unsupervised Models Struggle on Cellular KPIs**: Cellular network anomalies often manifest as subtle, localized perturbations (e.g., buffer buildup during high throughput, or handover frequency oscillations) rather than extreme isolated outliers in Euclidean or orthogonal projection space. Without labels to distinguish anomalous multi-variable configurations, unsupervised density estimators incur significant false positives (174-193 false alarms).
        3. **Key Predictive Features**: Signal-to-noise ratio (`UL_SNR_mean`), block error rate (`UL_BLER_mean`), buffer occupancy (`Estimated_UL_Buffer_mean`), and resource block allocations (`UL_NPRB_min`) exhibit the highest discriminative power.

        ---

        ### 🛡️ Explicit Research Boundaries & Scope
        - **Independent Prototype**: Built exclusively with open-source scientific tools (`scikit-learn`, `pandas`, `streamlit`, `plotly`).
        - **No SLTMobitel Operational Data**: This research does not use, expose, or replicate proprietary SLTMobitel telecommunication records, cell site locations, or customer data.
        - **No Direct Physical Fault Prediction**: Predictions indicate statistical KPI deviation matching synthetic TelecomTS profiles, not physical copper/fiber cable cuts or equipment burnouts.
        """
    )
