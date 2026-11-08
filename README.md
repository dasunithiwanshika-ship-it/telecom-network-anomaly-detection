# Telecom AI Intelligence Platform
## AI-Powered Network Anomaly Detection & KPI Intelligence
### Comparative Evaluation of Supervised and Unsupervised Machine Learning Models on Public Cellular Network KPI Data

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20v2.0-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)

---

### ⚠️ Academic Research Scope & Disclaimer
> **Important Notice:** This repository is an independent academic research prototype using the open-access **TelecomTS** benchmark dataset (`AliMaatouk/TelecomTS`).  
> The data does **NOT** represent SLTMobitel operational network telemetry, does **NOT** reflect Sri Lankan telecommunication infrastructure or customer records, and does **NOT** predict physical field dispatch or hardware breakdowns. Results demonstrate comparative algorithmic detection performance on public cellular KPI benchmarks.

---

## 📌 Project Overview & Research Question

In modern cellular networks (4G LTE / 5G), operators continuously record multi-dimensional Key Performance Indicators (KPIs) reflecting radio channel quality, packet transmission error rates, modulation schemes, and buffer queues. Detecting anomalous network states (such as uplink jamming, buffer overflows, and handover anomalies) is critical for service reliability.

**Core Research Question:**
> *How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?*

### 🔬 Verified Key Findings
1. **Supervised Superiority**: On the reserved 80/20 test partition ($N=6,400$), **Random Forest** achieved an outstanding **96.44% F1-score**, **100.00% precision**, and **99.54% PR-AUC** with **zero false alarms** at the canonical threshold $\tau = 0.50$.
2. **Operational Threshold Flexibility**: When the classification threshold is tuned strictly via 5-Fold Stratified Cross-Validation on the training set ($\tau = 0.30$), Random Forest captures **97.17% of all test anomalies** (240/247) with **99.17% precision** (only 2 false positives).
3. **Unsupervised Limitations**: Unsupervised models struggle significantly on cellular KPIs (**Local Outlier Factor F1: 29.90%**, **Isolation Forest F1: 26.77%**). Benign operational extremes (such as heavy bandwidth surges) are misclassified as anomalies, generating 174–193 false alarms.
4. **Core Predictive KPIs**: Feature attribution via Gini impurity reduction and permutation feature importance identifies uplink resource block allocation (`UL_NPRB_min`), signal-to-noise ratio (`UL_SNR_mean`), block error rate (`UL_BLER_mean`), and buffer occupancy (`Estimated_UL_Buffer_mean`) as the primary discriminative indicators.
5. **Inference Latency Profile**: Random Forest delivers sub-10ms batch throughput (8.17 ms / 1k records) and Logistic Regression executes in 0.17 ms / 1k records, whereas Local Outlier Factor requires 129.32 ms / 1k records due to memory-bound $k$-NN distance scans.

---

## 📊 Comprehensive 4-Model Comparative Benchmark

Evaluated on the exact same deterministic 80/20 stratified test partition ($N_{\text{test}} = 6,400$ traces: 6,153 Normal, 247 Anomalies, 3.86% prevalence):

| Model Name | Paradigm | Configuration Details | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | FN | TN | Fit Time | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Supervised (GLM) | $C=1.0, \tau=0.50$, StandardScaler | 95.31% | 74.09% | 83.37% | 97.85% | 88.55% | 183 | 9 | 64 | 6,144 | **0.45 s** | **0.17 ms** / 1k |
| **Random Forest (Canonical)**| Supervised (Ensemble) | $B=100, \tau=0.50, \text{leaf}=1, \text{sqrt}$ | **100.00%** | **93.12%** | **96.44%** | **99.98%** | **99.54%** | **230** | **0** | **17** | **6,153** | 9.04 s | 8.17 ms / 1k |
| *Random Forest (CV-Tuned)* | Supervised (Ensemble) | $B=100, \tau=0.30\text{ (from 5-Fold CV)}$ | 99.17% | 97.17% | 98.16% | 99.98% | 99.54% | 240 | 2 | 7 | 6,151 | 9.04 s | 8.17 ms / 1k |
| **Isolation Forest (Prior)**| Unsupervised (Trees) | $B=100, c=0.0386\text{ (training prior)}$| 26.05% | 27.53% | 26.77% | 76.12% | 21.21% | 68 | 193 | 179 | 5,960 | 0.53 s | 7.55 ms / 1k |
| *Isolation Forest (Auto)* | Unsupervised (Trees) | $B=100, c=\text{'auto' (pure unlabelled)}$| 18.07% | 34.82% | 23.79% | 76.12% | 21.21% | 86 | 390 | 161 | 5,763 | 0.53 s | 7.55 ms / 1k |
| **Local Outlier Factor**| Unsupervised (k-NN) | $k=20, \text{novelty}=\text{True}, c=0.0386$ | 29.84% | 29.96% | 29.90% | 74.38% | 23.04% | 74 | 174 | 173 | 5,979 | 3.33 s | 129.32 ms / 1k |

*Methodological Notes:*
1. **Unbiased Baseline:** Primary evaluation reports canonical $\tau = 0.50$ for Random Forest. When $\tau$ is tuned strictly via 5-Fold Stratified CV on $X_{\text{train}}$ ($\tau = 0.30$), it recovers 10 additional faults with only 2 false alarms.
2. **Unsupervised Ranking Invariance:** Isolation Forest continuous ranking metrics (ROC-AUC = 76.12%, PR-AUC = 21.21%) are completely invariant to the contamination parameter cutoff.

---

## 🏗️ Repository Architecture

```text
telecom-network-anomaly-detection/
│
├── notebooks/                                # Jupyter Research Notebooks (Days 1–12)
│   ├── 01_project_understanding.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_logistic_regression.ipynb
│   ├── 06_threshold_imbalance_analysis.ipynb
│   ├── 07_model_selection_and_experiment_plan.ipynb
│   ├── 08_random_forest_baseline.ipynb
│   ├── 09_random_forest_tuning.ipynb
│   ├── 10_model_comparison.ipynb
│   ├── 11_unsupervised_experiments.ipynb             # Phase 2: Isolation Forest & LOF
│   └── 12_threshold_tuning_and_xai.ipynb             # Phase 3: Benchmark & Explainability
│
├── src/                                      # Production & ML Engineering Pipeline
│   ├── __init__.py
│   ├── data_loader.py                        # Dataset loading & 80/20 stratified split
│   ├── feature_engineering.py                # 64 summary moments feature extraction
│   ├── preprocessing.py                      # Robust schema validation & TelecomPreprocessor
│   ├── evaluation.py                         # Standardized classification metrics & ROC/PR
│   ├── prediction.py                         # Unified inference & explain_record triage engine
│   └── train_and_evaluate_all_models.py      # End-to-end benchmark execution script
│
├── app/                                      # Enterprise Streamlit Platform (v2.0)
│   ├── streamlit_app.py                      # Main entrypoint & multi-page router
│   ├── sample_telecom_kpi_features.csv       # Pre-formatted synthetic demonstration data
│   ├── components/                           # Enterprise UI Components
│   │   ├── __init__.py
│   │   ├── styles.py                         # Premium dark theme CSS (#0B1220 / #14B8A6)
│   │   ├── sidebar.py                        # NOC sidebar with live model engine monitors
│   │   ├── metric_cards.py                   # Responsive metric cards and banner headers
│   │   └── charts.py                         # Dark theme Plotly interactive visualizers
│   └── pages/                                # Specialized Platform Modules
│       ├── __init__.py
│       ├── overview.py                       # 1. Network Overview & Telemetry Summary
│       ├── detection.py                      # 2. Anomaly Detection Workspace (CSV batch)
│       ├── kpi_monitoring.py                 # 3. KPI Monitoring (16 KPIs in 7 domains)
│       ├── investigation.py                  # 4. Anomaly Investigation (Z-Score Triage)
│       ├── ai_insights.py                    # 5. AI Explainability (Gini MDI vs Permutation)
│       ├── model_intelligence.py             # 6. Model Intelligence & Benchmarking
│       ├── research_centre.py                # 7. Research Centre & 12 Notebook Directory
│       ├── data_management.py                # 8. Data Management & 64-Feature Dictionary
│       └── about.py                          # 9. About, Architecture & Academic Boundaries
│
├── models/                                   # Serialized Model Artifacts
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── isolation_forest.joblib
│   ├── local_outlier_factor.joblib
│   ├── normal_baseline_stats.joblib          # Precomputed stats (N=30,765) for triage
│   └── feature_names.json                    # Exact 64-feature column schema
│
├── results/                                  # Verified Metric Tables & Plots
│   ├── unified_4model_comparison_table.csv
│   ├── unified_4model_roc_pr_comparison.png
│   ├── unified_4model_confusion_matrices.png
│   ├── rf_gini_feature_importance.csv
│   ├── rf_permutation_importance.csv
│   ├── rf_feature_importance_and_permutation.png
│   ├── threshold_sensitivity_analysis.csv
│   ├── threshold_sensitivity_tradeoff.png
│   ├── isolation_forest_contamination_sensitivity.csv
│   └── lof_k_sensitivity.csv
│
├── docs/                                     # Documentation, Guides & Reports
│   ├── day-8-summary-report.md
│   ├── day-9-tuning-report.md
│   ├── day-10-comparison-report.md
│   ├── demonstration_guide.md                # 9-page presentation & demonstration guide
│   └── viva_preparation_guide.md             # Technical Viva Voce Q&A defense
│
├── reports/
│   └── final_research_report.md              # 23-section comprehensive academic research report
│
├── tests/                                    # Automated Unit Test Suite (pytest)
│   ├── test_dashboard.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_feature_engineering.py
│   ├── test_prediction.py
│   └── test_preprocessing.py
│
├── requirements.txt                          # Pinned project dependencies
├── conftest.py                               # Pytest configuration
└── README.md                                 # Master repository documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/dasunithiwanshika-ship-it/telecom-network-anomaly-detection.git
cd telecom-network-anomaly-detection

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Automated Unit Tests
```bash
python -m pytest -v tests/
```
*(All 22 unit tests cover data loading, 64-feature extraction, schema validation, pipeline inference, triage deviation analysis, and component compiling).*

### 3. Launch the Enterprise Platform
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📡 The 9 Enterprise Platform Modules

1. **🌐 Network Overview**: Executive summary with KPI telemetry counts, 32k trace metrics, class distribution donut chart, and operational boundaries.
2. **⚡ Anomaly Detection Workspace**: Interactive inference supporting 64-feature CSV uploads, missing column validation, model selection (RF, LR, IF, LOF), decision threshold sliders, and prediction CSV export.
3. **📊 KPI Monitoring & Health**: Multi-metric health analysis categorized into 7 telecom domains (Radio, Signal, Error Rates, Modulation, Scheduling, Traffic, Buffer) with interactive boxplots and distributions.
4. **🔬 Anomaly Investigation Workspace**: Forensic root-cause triage evaluating individual anomalous records against 30,765 normal baseline traces using exact z-score deviations and percentile confidence bands.
5. **💡 AI Explainability & Feature Intelligence**: Comparison of Gini Impurity (MDI) vs Permutation Feature Importance evaluated on held-out test data, explaining feature cardinality bias and correlation vs causation.
6. **🧠 Model Intelligence & Benchmarking**: Empirical 4-model comparison table, multi-metric grouped bar charts, 2x2 confusion matrices, decision threshold sensitivity curves, and computational latency benchmarks.
7. **📚 Research Centre & Notebook Index**: Academic research question, methodology, index of all 12 experimental notebooks, and publication links.
8. **🗄️ Data Management & Feature Schema**: Interactive 64-feature schema dictionary with moment definitions, CSV template download, and synthetic telemetry sample generation.
9. **ℹ️ About & Platform Architecture**: Complete end-to-end data pipeline flow, technology stack specifications, author credentials, and strict SLTMobitel industry boundary statements.

---

## 📖 Key Documentation Links
- [Final Research Report (23 Sections)](reports/final_research_report.md)
- [Enterprise Platform Demonstration Guide](docs/demonstration_guide.md)
- [Technical Viva Voce Defense Guide](docs/viva_preparation_guide.md)
- [Day 10 Comparative Model Evaluation Report](docs/day-10-comparison-report.md)
- [Day 9 Random Forest Tuning Report](docs/day-9-tuning-report.md)
- [Day 8 Random Forest Baseline Report](docs/day-8-summary-report.md)

---

## 📚 References & Dataset Citation
- **TelecomTS Dataset**: Maatouk, A., et al. "TelecomTS: A Multivariate Time Series Dataset for Cellular Network Anomaly Detection." *arXiv preprint*, 2024. [Hugging Face Repository](https://huggingface.co/datasets/AliMaatouk/TelecomTS).
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. "Isolation Forest." *IEEE ICDM*, 2008.
- **Local Outlier Factor**: Breunig, M. M., et al. "LOF: Identifying Density-Based Local Outliers." *ACM SIGMOD*, 2000.
- **Random Forest**: Breiman, L. "Random Forests." *Machine Learning*, 2001.
