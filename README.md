# Telecom AI Anomaly Intelligence & Fault-Risk Detection
## Comparative Evaluation of Supervised and Unsupervised Machine Learning Models on Public Cellular Network KPI Data

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)](tests/)

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
1. **Supervised Superiority**: On the reserved 80/20 test partition ($N=6,400$), **Random Forest** achieved an outstanding **96.44% F1-score**, **100.00% precision**, and **99.54% PR-AUC** with **zero false alarms**.
2. **Unsupervised Limitations**: Unsupervised models struggle significantly on cellular KPIs (**Local Outlier Factor F1: 29.90%**, **Isolation Forest F1: 26.77%**). Benign operational extremes (such as heavy bandwidth surges) are misclassified as anomalies, generating 174–193 false alarms.
3. **Core Predictive KPIs**: Feature attribution via Gini impurity reduction and permutation feature importance identifies uplink signal-to-noise ratio (`UL_SNR_mean`), block error rate (`UL_BLER_mean`), buffer occupancy (`Estimated_UL_Buffer_mean`), and resource block allocation (`UL_NPRB_min`) as the primary discriminative indicators.

---

## 📊 Comprehensive 4-Model Comparative Benchmark

Evaluated on the exact same deterministic 80/20 stratified test partition ($N_{\text{test}} = 6,400$ traces: 6,153 Normal, 247 Anomalies, 3.86% prevalence):

| Model Name | Paradigm | Configuration Details | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | FN | TN | Fit Time | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Supervised (GLM) | $C=1.0, \tau=0.50$, StandardScaler | 95.31% | 74.09% | 83.37% | 97.85% | 88.55% | 183 | 9 | 64 | 6,144 | **0.45 s** | **0.17 ms** / 1k |
| **Random Forest** | Supervised (Ensemble) | $B=100, \text{leaf}=1, \text{feat}=\text{sqrt}$ | **100.00%** | **93.12%** | **96.44%** | **99.98%** | **99.54%** | **230** | **0** | **17** | **6,153** | 9.04 s | 8.17 ms / 1k |
| **Isolation Forest** | Unsupervised (Trees) | $B=100, c=0.0386$, random_state=42 | 26.05% | 27.53% | 26.77% | 76.12% | 21.21% | 68 | 193 | 179 | 5,960 | 0.53 s | 7.55 ms / 1k |
| **Local Outlier Factor**| Unsupervised (k-NN) | $k=20, \text{novelty}=\text{True}, c=0.0386$ | 29.84% | 29.96% | 29.90% | 74.38% | 23.04% | 74 | 174 | 173 | 5,979 | 3.33 s | 129.32 ms / 1k |

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
│   ├── 11_unsupervised_anomaly_detection.ipynb        # Phase 2: Isolation Forest & LOF
│   └── 12_final_model_evaluation_and_explainability.ipynb # Phase 3: Benchmark & Explainability
│
├── src/                                      # Reusable Production & ML Modules
│   ├── __init__.py
│   ├── data_loader.py                        # Dataset loading & 80/20 stratified split
│   ├── feature_engineering.py                # 64 summary moments feature extraction
│   ├── preprocessing.py                      # Schema validation & TelecomPreprocessor
│   ├── evaluation.py                         # Standardized classification metrics & ROC/PR
│   ├── prediction.py                         # Unified inference pipeline for all 4 models
│   └── train_and_evaluate_all_models.py      # End-to-end benchmark execution script
│
├── app/                                      # Interactive Web Application
│   ├── streamlit_app.py                      # Modern 6-page Telecom Intelligence Dashboard
│   └── sample_telecom_kpi_features.csv       # Pre-formatted synthetic demonstration data
│
├── models/                                   # Serialized Model Artifacts
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── isolation_forest.joblib
│   └── local_outlier_factor.joblib
│
├── results/                                  # Verified Metric Tables & Plots
│   ├── unified_4model_comparison_table.csv
│   ├── unified_4model_roc_pr_comparison.png
│   ├── unified_4model_confusion_matrices.png
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
│   ├── demonstration_guide.md                # Presentation & demonstration guide
│   └── viva_preparation_guide.md             # Technical Viva Voce Q&A defense
│
├── reports/
│   └── final_research_report.md              # 23-section comprehensive academic research report
│
├── tests/                                    # Automated Unit Test Suite (pytest)
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   ├── test_evaluation.py
│   └── test_dashboard.py
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
*(All 18 unit tests cover data loading, 64-feature extraction, schema validation, pipeline inference, and error handling).*

### 3. Re-execute the End-to-End Experiment Benchmark
```bash
python src/train_and_evaluate_all_models.py
```
*(Trains all 4 models, computes permutation importance, runs sensitivity sweeps, and exports publication figures).*

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📡 Interactive Dashboard Features
The **Telecom AI Anomaly Intelligence Dashboard** provides 6 specialized modules:
1. **Overview**: Project metadata, 32k trace metrics, class distribution donut chart, and scope limitations.
2. **Dataset Explorer**: Functional categorization of 16 cellular KPIs, interactive normal vs. anomaly boxplots, and histograms.
3. **Anomaly Detection Demo**: Interactive CSV uploader supporting the 64 features, input validation, threshold slider, real-time predictions, score distribution plots, and downloadable CSV results.
4. **Model Comparison**: Unified comparative metrics table, interactive Plotly ROC/PR curves, and sensitivity sweeps.
5. **Feature Importance & Explainability**: Top 12 Gini MDI and Permutation Importance bar charts with standard deviation bars.
6. **Research & About**: Full problem definition, citations, and viva preparation guide.

---

## 📖 Key Documentation Links
- [Final Research Report (23 Sections)](reports/final_research_report.md)
- [Project Demonstration & Presentation Guide](docs/demonstration_guide.md)
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
