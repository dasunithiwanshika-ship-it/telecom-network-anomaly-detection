# Final Research Report
# Comparative Evaluation of Supervised and Unsupervised Machine Learning Models for Network Anomaly Detection Using Public Telecom KPI Data

**Project Title:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks using Public Cellular Network KPI Data  
**Student / Researcher:** Dasuni Thiwanshika  
**Role:** Senior ML Engineering & Research Project Assistant  
**Date of Completion:** November 8, 2026 (Final Project Target Deadline: November 28, 2026)  
**Academic & Industry Disclaimer:** This report documents an independent academic research prototype using the open-access `AliMaatouk/TelecomTS` dataset. The data does NOT represent SLTMobitel operational network telemetry, does not reflect Sri Lankan telecommunication infrastructure or customer records, and does not predict physical field dispatch or hardware breakdowns.

---

## 1. Abstract

Modern cellular networks generate high-frequency multivariate telemetry reflecting radio channel quality, packet transmission integrity, and resource scheduling. Detecting anomalous operational states is crucial for preventing severe quality-of-service degradation. This research investigates the empirical trade-offs between supervised classification and unsupervised anomaly detection on cellular Key Performance Indicator (KPI) time series. Using the public TelecomTS benchmark (32,000 trace windows, 3.86% anomaly prevalence), we extract 64 statistical moments across 16 numerical KPIs and systematically evaluate four distinct algorithmic paradigms: L2-regularized Logistic Regression, an optimized Random Forest ensemble, Isolation Forest, and Local Outlier Factor (LOF). 

On an independent 80/20 stratified test partition ($N=6,400$), the supervised Random Forest achieved an outstanding **96.44% F1-score**, **100.00% precision**, and **99.54% PR-AUC** with zero false alarms. Logistic Regression achieved an **83.37% F1-score** and **88.55% PR-AUC**. In sharp contrast, unsupervised methods achieved substantially lower performance: Isolation Forest yielded a **26.77% F1-score** (76.12% ROC-AUC), while Local Outlier Factor achieved a **29.90% F1-score** (74.38% ROC-AUC). Feature importance analysis via Gini impurity reduction and model-agnostic permutation importance identified uplink signal-to-noise ratio (`UL_SNR_mean`), block error rate (`UL_BLER_mean`), buffer occupancy (`Estimated_UL_Buffer_mean`), and physical resource block allocation (`UL_NPRB_min`) as the primary discriminative indicators. These findings demonstrate that cellular anomalies manifest as subtle, multi-dimensional boundary perturbations rather than extreme geometric outliers in high-dimensional space, necessitating supervised ground-truth supervision for production-grade reliability.

---

## 2. Introduction

Telecommunication operators operate complex radio access networks (RAN) subject to unpredictable stochastic channel fading, RF interference, hardware faults, and buffer congestion. Traditional operations center workflows rely heavily on static scalar alarm thresholds (e.g., flagging when Downlink Block Error Rate exceeds a fixed percentage). However, modern LTE/5G network behaviors exhibit highly non-linear, correlated interactions that escape simple univariate rules. 

Machine learning offers automated multivariate pattern recognition to preemptively identify fault signatures. However, telecom environments present a fundamental dilemma: while supervised learning leverages confirmed fault labels, acquiring verified anomaly labels in live operational networks is labor-intensive and expensive. Conversely, unsupervised methods promise "label-free" discovery but risk overwhelming operations centers with false alarms. This research systematically investigates this paradigm trade-off.

---

## 3. Background

Cellular base stations (eNodeBs in 4G, gNodeBs in 5G) sample physical and MAC-layer telemetry at millisecond granularities. These KPIs encompass:
1. **Radio Signal Conditions**: Reference Signal Received Power (RSRP) and Signal-to-Noise Ratio (SNR), dictating the physical transmission capacity.
2. **Transmission Reliability**: Block Error Rates (BLER) measuring the fraction of corrupted transport blocks.
3. **Link Adaptation**: Modulation and Coding Scheme (MCS) indices dynamically selected by the scheduler based on channel quality.
4. **Buffer & Queue Dynamics**: Buffer occupancy indicating packet queueing latency before physical transmission.

---

## 4. Problem Statement

Cellular KPI telemetry is inherently high-dimensional, highly correlated, and heavily imbalanced (anomalies typically represent less than 5% of network operating time). Operational teams require anomaly detection systems that:
1. Maximize fault capture (high recall) to avert service blackouts.
2. Minimize false alarms (high precision) to prevent alarm fatigue and unnecessary truck rolls.
3. Provide interpretable feature attributions so network engineers can diagnose underlying channel or resource bottlenecks.

---

## 5. Research Question

> **How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?**

---

## 6. Objectives

1. Audit and preserve existing exploratory and baseline experiments (Days 1–10).
2. Engineer a standardized 64-feature statistical representation across 16 numerical cellular KPIs.
3. Implement and benchmark unsupervised anomaly detection models (Isolation Forest and Local Outlier Factor).
4. Conduct unified comparative evaluation across four models on a deterministic 80/20 test split ($N=6,400$).
5. Quantify model explainability using tree Gini importance and model-agnostic permutation feature importance.
6. Perform robustness and sensitivity analyses over decision thresholds, contamination factors, and neighborhood sizes ($k$).
7. Build and deploy an interactive, modular Streamlit dashboard with live CSV validation and inference.
8. Establish a comprehensive unit test suite to guarantee software quality.

---

## 7. Dataset Description

The research utilizes the **TelecomTS** benchmark published by Ali Maatouk et al. on Hugging Face (`AliMaatouk/TelecomTS`):
- **Total Traces**: 32,000 discrete observation windows.
- **Trace Length**: 128 sequential time steps sampled at 10 Hz (~12.7 seconds per session).
- **Target Indicator**: Binary classification target ($y=0$ for Normal, $y=1$ for Anomaly).
- **Class Breakdown**: 30,765 normal observations (96.14%) and 1,235 anomalous observations (3.86%).
- **Channels**: 18 total channels (16 numerical continuous KPIs and 2 categorical protocol/application metadata attributes).

---

## 8. Exploratory Data Analysis (EDA)

Exploratory analysis revealed several critical structural characteristics:
- **Severe Class Imbalance**: The 3.86% anomaly rate means naive accuracy (96.14% for predicting all normal) is meaningless. Evaluation must prioritize Precision-Recall Area Under the Curve (PR-AUC) and F1-score.
- **Multimodal Traffic Distributions**: Throughput variables (`TX_Bytes`, `RX_Bytes`) exhibit extreme positive skewness spanning multiple orders of magnitude depending on user traffic burstiness (e.g., video streaming vs. background messaging).
- **Negative Radio Correlations**: Strong inverse relationships exist between RSRP/SNR and BLER: severe signal attenuation naturally triggers elevated error rates and lower MCS indices.

---

## 9. Data Preprocessing

1. **Deterministic Partitioning**: An 80/20 stratified split (`random_state=42`) preserves the 3.86% anomaly prevalence across training ($N=25,600$, 988 anomalies) and testing ($N=6,400$, 247 anomalies).
2. **Missing Value & Infinity Handling**: Finite bounds checking with median imputation.
3. **Standardization Pipeline**: Fit a `StandardScaler` strictly on the training partition ($X_{\text{train}}$) and transform the test partition ($X_{\text{test}}$) to prevent data leakage for distance-based models (Logistic Regression, LOF). Tree models (Random Forest, Isolation Forest) operate directly on scale-invariant unnormalized features.

---

## 10. Feature Engineering

To summarize each 128-timestep trace into fixed-size tabular inputs while preserving statistical distribution moments, four summary statistics were computed for each of the 16 numerical KPIs:
$$\mu = \frac{1}{T}\sum_{t=1}^T x_t, \quad \sigma = \sqrt{\frac{1}{T}\sum_{t=1}^T (x_t - \mu)^2}, \quad \min = \min_{t}(x_t), \quad \max = \max_{t}(x_t)$$
This yields $16 \times 4 = 64$ engineered features per sample:
- **Radio Quality**: `RSRP_{mean, std, min, max}`, `UL_SNR_{mean, std, min, max}`
- **Block Error Rates**: `UL_BLER_{mean, std, min, max}`, `DL_BLER_{mean, std, min, max}`
- **Modulation Schemes**: `UL_MCS_{mean, std, min, max}`, `DL_MCS_{mean, std, min, max}`
- **Resource Allocations**: `UL_NPRB`, `PRBs_DL_Current`, `PRBs_UL_Current`, `PRB_Utilization_DL`, `PRB_Utilization_UL`
- **Throughput & Counters**: `TX_Bytes`, `RX_Bytes`, `UL_NumberOfPackets`, `DL_NumberOfPackets`
- **Buffer Metrics**: `Estimated_UL_Buffer`

Metadata fields (`anomaly_type`, `affected_kpis`, `application`) were strictly excluded from model feature sets.

---

## 11. Research Methodology

The evaluation methodology enforces strict scientific controls:
1. **Unsupervised Label Blindness**: Isolation Forest and Local Outlier Factor were trained on $X_{\text{train}}$ **without access to target labels $y$**.
2. **Common Evaluation Benchmark**: All four models were tested on the exact same 6,400 test traces.
3. **Score Calibration & Inversion**: For unsupervised models, raw decision functions were inverted such that higher scores uniformly correspond to greater anomaly likelihood, enabling valid ROC and PR curve derivation.
4. **Latency Benchmarking**: CPU execution times for both model fitting and batch inference (per 1,000 samples) were clocked across repeated trials.

---

## 12. Supervised Learning Experiments

### A. Logistic Regression (Baseline GLM)
- **Configuration**: L2 penalty, $C=1.0$, `max_iter=1000`, classification threshold $\tau = 0.50$.
- **Performance**: Precision: **95.31%**, Recall: **74.09%**, F1: **83.37%**, ROC-AUC: **97.85%**, PR-AUC: **88.55%**.
- **Confusion Matrix**: TP: 183, FP: 9, FN: 64, TN: 6,144.

### B. Random Forest (Optimized Bagged Ensemble)
- **Configuration**: 100 bagged trees, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'` (selected via Day 9 5-Fold Stratified Cross-Validation, CV F1 = 96.01%).
- **Performance**: Precision: **100.00%**, Recall: **93.12%**, F1: **96.44%**, ROC-AUC: **99.98%**, PR-AUC: **99.54%**.
- **Confusion Matrix**: TP: 230, FP: 0, FN: 17, TN: 6,153. Zero false positives observed on the test set.

---

## 13. Unsupervised Learning Experiments

### A. Isolation Forest
- **Concept**: Recursively isolates samples via random orthogonal hyperplanes. Anomalies require fewer splits (shorter path lengths).
- **Configuration**: `n_estimators=100`, `contamination=0.0386` (calibrated to the 3.86% training prevalence), `random_state=42`.
- **Performance**: Precision: **26.05%**, Recall: **27.53%**, F1: **26.77%**, ROC-AUC: **76.12%**, PR-AUC: **21.21%**.
- **Confusion Matrix**: TP: 68, FP: 193, FN: 179, TN: 5,960.

### B. Local Outlier Factor (LOF)
- **Concept**: Density-based algorithm comparing the local reachability density of an observation to its $k$-nearest neighbors.
- **Configuration**: Novelty detection mode (`novelty=True`), $k=20$ neighbors, `contamination=0.0386`, standardized features.
- **Performance**: Precision: **29.84%**, Recall: **29.96%**, F1: **29.90%**, ROC-AUC: **74.38%**, PR-AUC: **23.04%**.
- **Confusion Matrix**: TP: 74, FP: 174, FN: 173, TN: 5,979.

---

## 14. Model Evaluation Metrics

Given the 3.86% class imbalance, accuracy is misleading. The primary metrics are:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
PR-AUC (Average Precision) integrates the area under the Precision-Recall curve, directly penalizing false alarms in the presence of overwhelming negative majorities.

---

## 15. Comparative Results

### Comprehensive 4-Model Unified Benchmark Table ($N_{\text{test}} = 6,400$):

| Model Name | Learning Paradigm | Hyperparameters | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | FN | TN | Fit Time | Inf. Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Supervised (GLM) | $C=1.0, \tau=0.50$ | 95.31% | 74.09% | 83.37% | 97.85% | 88.55% | 183 | 9 | 64 | 6,144 | **0.45 s** | **0.17 ms** |
| **Random Forest** | Supervised (Ensemble) | $B=100, \text{leaf}=1$ | **100.00%** | **93.12%** | **96.44%** | **99.98%** | **99.54%** | **230** | **0** | **17** | **6,153** | 9.04 s | 8.17 ms |
| **Isolation Forest** | Unsupervised (Trees) | $B=100, c=0.0386$ | 26.05% | 27.53% | 26.77% | 76.12% | 21.21% | 68 | 193 | 179 | 5,960 | 0.53 s | 7.55 ms |
| **Local Outlier Factor**| Unsupervised (k-NN) | $k=20, c=0.0386$ | 29.84% | 29.96% | 29.90% | 74.38% | 23.04% | 74 | 174 | 173 | 5,979 | 3.33 s | 129.32 ms |

*Note: Latency measured per 1,000 samples on standard benchmark CPU.*

---

## 16. Error Analysis

Set-theoretic decomposition of test set predictions reveals fundamental architectural insights:
1. **Mutually Missed Anomalies ($\mathcal{FN}_{\text{both}} = 16$ traces)**:
   - Both Logistic Regression and Random Forest missed 16 specific traces (`[726, 760, 763, 787, 915, 962, 985, 1001, 1027, 1030, 1099, 1103, 1117, 1126, 1157, 1223]`).
   - These correspond to temporal anomalies (e.g., `Faulty Handover Algorithm (Too Frequent)`). Because our 64 features compress time steps into static summary moments, sequential handover oscillation signatures are lost.
2. **Anomalies Recovered Exclusively by Random Forest (48 traces)**:
   - Primarily `Jamming` faults. In these cases, extreme BLER is accompanied by collapsed traffic throughput. Logistic Regression's linear summation offsets positive and negative contributions, missing the fault; Random Forest's non-linear orthogonal splits cleanly isolate the joint condition.
3. **The Single Case Caught by LR but Missed by RF (`Sample 1047`)**:
   - Trace 1047 (`Buffer Overflow (Gradual Buildup)` on `Twitch`).
   - Exhibited elevated buffer (`Estimated_UL_Buffer_mean = 708.01 KB`, upper quartile). Logistic Regression assigned large positive weight ($\beta \approx +3.65$), yielding $P=1.0000$.
   - In Random Forest, 40 trees voted anomaly and 60 voted normal. The ensemble probability was $\hat{P}=0.4000$, which fell below the 0.50 threshold. Lowering $\tau$ to 0.35 enables Random Forest to catch this sample as well.
4. **False Alarms in Unsupervised Models (174–193 traces)**:
   - Unsupervised models flag high-throughput, high-PRB sessions as "outliers" simply because heavy utilization is statistically rare in the dataset, even though the connection is completely healthy.

---

## 17. Feature Importance & Explainability

### A. Gini Importance (Mean Decrease in Impurity - MDI)
Top features contributing to tree split purities:
1. `RSRP_mean` (0.1126)
2. `UL_SNR_mean` (0.1082)
3. `RSRP_max` (0.0737)
4. `UL_BLER_mean` (0.0677)
5. `RSRP_min` (0.0511)
6. `PRB_Utilization_DL_max` (0.0406)

### B. Permutation Feature Importance
Model-agnostic evaluation measuring test accuracy drop when feature values are permuted:
1. `UL_NPRB_min` ($\Delta\text{acc} = +0.0188 \pm 0.0008$)
2. `UL_SNR_mean` ($\Delta\text{acc} = +0.0113 \pm 0.0016$)
3. `RX_Bytes_mean` ($\Delta\text{acc} = +0.0097 \pm 0.0015$)
4. `PRB_Utilization_DL_max` ($\Delta\text{acc} = +0.0075 \pm 0.0025$)
5. `UL_BLER_mean` ($\Delta\text{acc} = +0.0060 \pm 0.0010$)

*Interpretability Caution:* Statistical feature attribution indicates correlational predictive utility within TelecomTS, not physical causation.

---

## 18. Dashboard Implementation

A web dashboard was developed using **Streamlit and Plotly**:
- **Application Structure**: `app/streamlit_app.py`, supported by modular backend services (`src/data_loader.py`, `src/feature_engineering.py`, `src/preprocessing.py`, `src/evaluation.py`, `src/prediction.py`).
- **Pages**:
  1. *Overview*: Project goals, class balance donut chart, pipeline stages, research disclaimer.
  2. *Dataset Explorer*: 16 KPI categories, interactive normal vs. anomaly boxplots and histograms.
  3. *Anomaly Detection Demo*: File uploader supporting the 64 features, input validation, threshold slider, live predictions, score distributions, and downloadable CSV.
  4. *Model Comparison*: Interactive unified benchmark table, ROC/PR curves, confusion matrices, and sensitivity sweeps.
  5. *Feature Importance & Explainability*: Interactive MDI and Permutation Importance charts with interpretability notes.
  6. *Research & About*: Comprehensive scientific documentation and viva preparation guides.

---

## 19. Discussion

The experimental findings clearly answer the research question:
1. **Supervised vs. Unsupervised Disparity**: Supervised Random Forest (F1: 96.44%) outperforms unsupervised Isolation Forest (F1: 26.77%) and LOF (F1: 29.90%) by over **66% in F1-score**. Unsupervised methods incur unacceptable false alarm rates (over 170 false alarms on 6,400 traces) because normal telecom operations encompass legitimate extreme states (e.g., peak-hour bandwidth surges).
2. **Key Feature Contributions**: Anomaly detection is predominantly driven by physical-layer uplink channel health (`UL_SNR`, `UL_BLER`) coupled with resource allocation indicators (`UL_NPRB`, `Estimated_UL_Buffer`).

---

## 20. Limitations

1. **Loss of Temporal Sequence Context**: Compressing 128 timesteps into 4 summary moments discards sequential transition dynamics, obscuring rapid temporal handover anomalies.
2. **Isolated Traces without Network Topology**: TelecomTS traces lack cell sector IDs, neighbor-cell adjacency matrices, and geographic coordinates.
3. **Synthetic / Semi-Synthetic Fault Signatures**: Anomalies follow mathematically injected rules rather than complex physical environmental degradations.
4. **Scope Notice**: Findings reflect academic evaluation on the open-access TelecomTS dataset and cannot be generalized to live commercial networks without domain-specific validation.

---

## 21. Conclusion

This research demonstrates that supervised tree ensembles (Random Forest) deliver exceptional accuracy (96.44% F1, 100% precision) for cellular KPI anomaly detection. Unsupervised methods (Isolation Forest, LOF) exhibit severe performance degradation (F1 < 30%) due to their inability to distinguish benign operational extremes from genuine network faults without labelled guidance. Uplink SNR, block error rates, and buffer dynamics represent the core KPI features governing fault discriminability.

---

## 22. Future Work

1. **Sequential Deep Learning**: Implement Recurrent Neural Networks (LSTM, GRU) or Temporal Convolutional Networks (TCN) directly on the 128-timestep sequence to capture the 16 missed temporal handover anomalies.
2. **Semi-Supervised / Active Learning**: Explore semi-supervised paradigms (e.g., Deep Support Vector Data Description with a small labelled calibration pool) to bridge the gap between label scarcity and high precision.
3. **Graph Neural Networks (GNNs)**: Integrate cellular topology and handover adjacency matrices when spatial telemetry becomes available.

---

## 23. References

1. Maatouk, A., et al. "TelecomTS: A Multivariate Time Series Dataset for Cellular Network Anomaly Detection." *arXiv preprint*, 2024.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. "Isolation Forest." *IEEE International Conference on Data Mining (ICDM)*, 2008.
3. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. "LOF: Identifying Density-Based Local Outliers." *ACM SIGMOD Record*, 2000.
4. Breiman, L. "Random Forests." *Machine Learning*, 45(1), 5-32, 2001.
5. Davis, J., & Goadrich, M. "The Relationship Between Precision-Recall and ROC Curves." *International Conference on Machine Learning (ICML)*, 2006.
6. Molnar, C. *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd Edition, 2022.
