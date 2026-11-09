# Project Demonstration & Walkthrough Guide
## Telecom AI Intelligence Platform (v2.0 Enterprise Hybrid Edition)

**Project Title:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks using Public Cellular Network KPI Data  
**Student / Intern:** Dasuni Thiwanshika  
**Target Date:** November 2026  

This guide provides a structured, step-by-step walkthrough for presenting the project to supervisors, evaluators, and interviewers.

---

### 1. What Problem Does This Project Address?
- Cellular telecommunication networks (4G LTE / 5G) produce massive volumes of multivariate performance telemetry across cell sites.
- Traditional Network Operations Centers (NOC) rely on static, univariate threshold alarms (e.g., "raise alert if BLER > 10%"). These generate either excessive false alarms or miss complex multi-channel faults.
- Operators need automated, multivariate machine learning systems to detect network anomalies before end-users experience dropped calls, packet loss, or service degradation.

### 2. What Dataset is Used (TelecomTS)?
- We use the open-access **TelecomTS** benchmark published by Ali Maatouk et al. on Hugging Face (`AliMaatouk/TelecomTS`).
- **32,000 trace windows** sampled at 10 Hz over 128 discrete time steps (~12.7 seconds).
- **16 continuous numerical KPIs** covering radio signal quality, block error rates, modulation, PRB utilization, throughput, and packet queues.
- **3.86% anomaly prevalence** (30,765 normal traces, 1,235 anomalous traces).
- *Crucial boundary:* This is public cellular research data; it is **NOT** SLTMobitel operational network telemetry or Sri Lankan customer records.

### 3. How Are the 64 Features Generated?
- Each of the 16 numerical KPIs across the 128 timesteps is summarized into 4 statistical distribution moments:
  1. Mean ($\mu$)
  2. Standard deviation ($\sigma$)
  3. Minimum ($\min$)
  4. Maximum ($\max$)
- $16 \times 4 = 64$ engineered features.
- All target metadata (anomaly type, affected KPIs, descriptions) are strictly excluded from model feature sets.

### 4. How Do the Supervised Models Work?
- **Logistic Regression (GLM Baseline)**: Standardizes features with `StandardScaler` and fits an L2-regularized linear decision boundary. Fast and interpretable ($0.17\text{ ms}$ inference per 1k samples), but limited by linearity.
- **Random Forest (Ensemble)**: 100 deep decision trees bagging random feature subsets (`max_features='sqrt'`). Captures non-linear feature interactions without requiring feature normalization.

### 5. How Do the Unsupervised Models Work?
- **Isolation Forest**: Fits random orthogonal trees **without anomaly labels**. Samples requiring fewer random splits to isolate are assigned high anomaly scores.
- **Local Outlier Factor (LOF)**: Compares the local reachability density of an observation to its $k$-nearest neighbors in standardized feature space under novelty detection (`novelty=True`).

### 6. How Are the Models Compared?
All four models are evaluated on the exact same deterministic 80/20 stratified test partition ($N=6,400$, 247 anomalies):
- **Random Forest (Canonical $\tau=0.50$)**: Precision: **100.00%**, Recall: **93.12%**, F1: **96.44%**, ROC-AUC: **99.98%**, PR-AUC: **99.54%** (0 false alarms).
- **Random Forest (CV-Tuned $\tau=0.30$)**: Precision: **99.17%**, Recall: **97.17%**, F1: **98.16%** (Captures 240/247 faults with only 2 false alarms).
- **Logistic Regression**: Precision: **95.31%**, Recall: **74.09%**, F1: **83.37%**, ROC-AUC: **97.85%**, PR-AUC: **88.55%** (9 false alarms).
- **Local Outlier Factor**: Precision: **29.84%**, Recall: **29.96%**, F1: **29.90%**, ROC-AUC: **74.38%**, PR-AUC: **23.04%** (174 false alarms).
- **Isolation Forest**: Precision: **26.05%**, Recall: **27.53%**, F1: **26.77%**, ROC-AUC: **76.12%**, PR-AUC: **21.21%** (193 false alarms).

### 7. How Does the Enterprise Platform Work?
Run `streamlit run app/streamlit_app.py`:
- **Module 1 (Network Overview)**: Executive summary, 32k dataset telemetry metrics, class distribution donut chart, and operational boundaries.
- **Module 2 (Anomaly Detection Workspace)**: 64-feature CSV uploader, schema validation, synthetic generator, model selector, probability threshold controls, score histograms, and downloadable results.
- **Module 3 (KPI Monitoring & Health)**: Multi-metric health analysis categorized into 7 telecom domains (Radio, Signal, Error Rates, Modulation, Scheduling, Traffic, Buffer) with interactive boxplots and distributions.
- **Module 4 (Anomaly Investigation Workspace)**: Forensic root-cause triage evaluating individual anomalous records against 30,765 normal baseline traces using exact z-score deviations and percentile confidence bands.
- **Module 5 (AI Explainability & Feature Intelligence)**: Comparison of Gini Impurity (MDI) vs Permutation Feature Importance evaluated on held-out test data, explaining feature cardinality bias and correlation vs causation.
- **Module 6 (Model Intelligence & Benchmarking)**: Empirical 4-model comparison table, multi-metric grouped bar charts, 2x2 confusion matrices, decision threshold sensitivity curves, and computational latency benchmarks.
- **Module 7 (Research Centre & Notebook Index)**: Academic research question, methodology, index of all 12 experimental notebooks, and publication links.
- **Module 8 (Data Management & Feature Schema)**: Interactive 64-feature schema dictionary with moment definitions, CSV template download, and synthetic telemetry sample generation.
- **Module 9 (About & Platform Architecture)**: Complete end-to-end data pipeline flow, technology stack specifications, author credentials, and strict SLTMobitel industry boundary statements.

### 8. What Do the Results Mean?
- **Supervised models heavily outperform unsupervised methods** on cellular KPI telemetry (+66% F1 gain). Cellular anomalies are subtle contextual interactions rather than extreme geometric outliers. Unsupervised models mistake legitimate peak traffic surges for anomalies.
- **Uplink health dominates detection**: Physical uplink signal quality (`UL_SNR`), packet loss (`UL_BLER`), and resource block starvation (`UL_NPRB_min`) are the most informative features.

### 9. What Can the Project NOT Claim?
- Cannot claim prediction of actual physical SLTMobitel DP/FDP cable cuts or field dispatches.
- Cannot claim deployment in live commercial network production.
- Cannot claim that feature correlations prove physical fault causation.

### 10. Possible Future Improvements
- Sequential neural networks (LSTM, GRU, Transformers) directly on the 128 timesteps to catch the 7 temporal handover anomalies missed by static summary moments.
- Two-stage hybrid deployment where Random Forest provides primary automated triage and Isolation Forest monitors background zero-day drifts.
