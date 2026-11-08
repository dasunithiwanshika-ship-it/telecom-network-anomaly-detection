"""
Page 7: Research Centre & Notebooks
Presents the formal research methodology, research question, 12 research notebooks,
experimental artifacts, and academic publication references.
"""

from pathlib import Path
import streamlit as st
from app.components.metric_cards import render_hero_banner, render_academic_disclaimer

NOTEBOOKS_META = [
    {"num": "01", "name": "01_project_understanding.ipynb", "phase": "Foundations", "desc": "Domain formulation, cellular KPI hierarchy, research objectives, and Hugging Face dataset ingestion."},
    {"num": "02", "name": "02_dataset_exploration.ipynb", "phase": "Foundations", "desc": "Multivariate tensor inspection across 32,000 time series, verifying (128, 16) dimensions and missing value integrity."},
    {"num": "03", "name": "03_eda.ipynb", "phase": "Exploration", "desc": "Distribution analysis, skewness assessment, cross-KPI correlation heatmaps, and physical radio behavior profiling."},
    {"num": "04", "name": "04_feature_engineering.ipynb", "phase": "Feature Eng", "desc": "Temporal statistical windowing generating 64 moment features (mean, std, min, max) across 16 base cell KPIs."},
    {"num": "05", "name": "05_baseline_logistic_regression.ipynb", "phase": "Supervised", "desc": "Linear classification baseline establishing initial benchmark with L2 regularization and StandardScaler scaling."},
    {"num": "06", "name": "06_threshold_imbalance_analysis.ipynb", "phase": "Supervised", "desc": "Class imbalance investigation (3.86% prevalence), precision-recall trade-offs, and probability calibration."},
    {"num": "07", "name": "07_model_selection_and_experiment_plan.ipynb", "phase": "Methodology", "desc": "Formal experiment protocol, stratified 80/20 train/test split design, and metric evaluation criteria."},
    {"num": "08", "name": "08_random_forest_baseline.ipynb", "phase": "Supervised", "desc": "Non-linear bagged ensemble baseline with 100 decision trees and default scikit-learn hyperparameters."},
    {"num": "09", "name": "09_random_forest_tuning.ipynb", "phase": "Supervised", "desc": "Grid search cross-validation optimizing tree depth, minimum leaf samples, and split criteria for generalization."},
    {"num": "10", "name": "10_model_comparison.ipynb", "phase": "Supervised", "desc": "Supervised model showdown comparing LR and RF with ROC-AUC, PR-AUC, confusion matrices, and error analysis."},
    {"num": "11", "name": "11_unsupervised_experiments.ipynb", "phase": "Unsupervised", "desc": "Unsupervised anomaly detection with Isolation Forest and Local Outlier Factor; contamination parameter sensitivity."},
    {"num": "12", "name": "12_threshold_tuning_and_xai.ipynb", "phase": "Optimization & XAI", "desc": "5-Fold CV probability threshold optimization (tau=0.30) and permutation feature importance on held-out test data."}
]

def render():
    render_hero_banner(
        title="Research Centre & Notebook Index",
        subtitle="Formal academic research methodology, 12-stage experimental notebook progression, and artifact repository.",
        badge="ACADEMIC RESEARCH"
    )
    
    render_academic_disclaimer()
    
    st.markdown("### 1. Research Framework & Core Question")
    
    col_rq1, col_rq2 = st.columns([3, 2])
    with col_rq1:
        st.markdown(
            """
            <div style="background-color: #172235; border-left: 4px solid #14B8A6; padding: 18px; border-radius: 6px;">
                <div style="color: #14B8A6; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Primary Research Question</div>
                <div style="color: #F8FAFC; font-size: 1.05rem; font-weight: 600; line-height: 1.5; font-style: italic;">
                    "How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?"
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_rq2:
        st.markdown(
            """
            <div style="background-color: #172235; border: 1px solid #1E293B; padding: 18px; border-radius: 6px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Key Dataset Citation</div>
                <div style="color: #38BDF8; font-size: 0.95rem; font-weight: 600;">Hugging Face: AliMaatouk/TelecomTS</div>
                <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">
                    32,000 cellular traces &bull; 128 timesteps &bull; 16 KPIs &bull; 25,600 train / 6,400 test split
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 2. Experimental Progression: The 12 Research Notebooks")
    st.markdown(
        """
        The project follows a 12-stage research and implementation lifecycle. 
        Each notebook encapsulates an independent, reproducible experimental milestone saved in the `notebooks/` directory.
        """
    )
    
    # Filter by phase
    phases = ["All Phases", "Foundations", "Exploration", "Feature Eng", "Supervised", "Methodology", "Unsupervised", "Optimization & XAI"]
    selected_phase = st.selectbox("Filter Notebooks by Research Phase", phases)
    
    filtered_nb = NOTEBOOKS_META if selected_phase == "All Phases" else [nb for nb in NOTEBOOKS_META if nb["phase"] == selected_phase]
    
    for nb in filtered_nb:
        st.markdown(
            f"""
            <div style="background-color: #172235; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background-color: #1E293B; color: #14B8A6; font-weight: 700; font-size: 0.85rem; padding: 3px 8px; border-radius: 4px;">Day {nb['num']}</span>
                        <span style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">{nb['name']}</span>
                    </div>
                    <span style="background-color: rgba(56, 189, 248, 0.15); color: #38BDF8; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">{nb['phase']}</span>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.4;">{nb['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 3. Summary of Major Research Findings")
    
    st.markdown(
        """
        1. **Supervised Superiority on Labeled Telecom Modes**:
           Random Forest achieved a **96.44% F1-score** at default threshold $\\tau=0.50$ (100% precision, 93.12% recall), reaching **98.16% F1-score** at validation-tuned threshold $\\tau=0.30$ (99.17% precision, 97.17% recall).
        
        2. **Unsupervised Precision Ceiling**:
           Unsupervised anomaly detection (Isolation Forest and Local Outlier Factor) achieved high ranking ability (**ROC-AUC ~74–76%**), but sustained a precision ceiling of **26–30%** when evaluated against binary labels. In an unlabelled environment, unsupervised models effectively identify rare statistical behavior, but normal variations in mobile traffic volume trigger high false alarms.
           
        3. **Feature Cardinality vs Functional Sensitivity**:
           Tree-based impurity measures (Gini MDI) ranked `RSRP_mean` as the most important feature (11.26%) due to continuous splitting opportunities. However, true permutation feature importance on held-out test data identified **`UL_NPRB_min`** as the single most critical functional bottleneck ($+1.88\\%$ test error degradation upon shuffling).
           
        4. **Production Latency Viability**:
           Logistic Regression offers near-instantaneous inference (0.17 ms/1k records) and Random Forest scales comfortably at 8.17 ms/1k records. Local Outlier Factor is computationally prohibitive for live high-throughput stream processing (129.32 ms/1k records) due to memory-bound $k$-NN distance scans.
        """
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 4. Technical Documentation & Viva Resources")
    st.markdown(
        """
        The repository includes complete engineering and research documentation:
        - **Comprehensive Research Report**: `reports/final_research_report.md`
        - **Viva Preparation & Defense Guide**: `docs/viva_preparation_guide.md`
        - **Platform Demonstration Guide**: `docs/demonstration_guide.md`
        - **Data Quality & Leakage Audit**: `results/unified_4model_comparison_table.csv`
        """
    )
