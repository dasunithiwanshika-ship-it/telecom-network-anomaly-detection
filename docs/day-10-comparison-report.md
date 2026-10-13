# Internship Daily Summary Report — Day 10
**Project:** Telecom Network Anomaly Detection Prototype  
**Date:** October 13, 2026  
**Intern:** Dasuni Thiwanshika  
**Branch:** `feat/day-10-model-evaluation`  
**Dataset:** Public TelecomTS Dataset (Hugging Face `AliMaatouk/TelecomTS`)  
**Scope Notice:** Academic research prototype using open-access cellular KPI data; does not represent measured performance or faults on SLTMobitel's live network infrastructure.

---

## 1. Executive Summary & Objective

The objective of Day 10 was to conduct a rigorous, empirical comparative evaluation across the machine learning models developed during the internship:
1. **Day 5 Baseline**: L2-regularized Logistic Regression ($C=1.0$, standardized features, classification threshold $\tau = 0.50$).
2. **Day 8 Baseline**: Unconstrained Random Forest ($B=100$ bagged decision trees, `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`).
3. **Day 9 Selected Configuration**: 5-Fold Stratified Cross-Validation selected Random Forest (`max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`), identified as the **best among the 18 hyperparameter configurations tested**.

All models were evaluated on the exact same deterministic 80/20 stratified test partition ($N_{\text{test}}=6,400$ traces: 6,153 Normal, 247 Anomalies, $3.86\%$ prevalence). Hyperparameter search findings from Day 9 were reused without rerunning the 90-fit cross-validation grid.

### Key Measured Results on the TelecomTS Test Split ($N=6,400$):
- **Performance Comparison**: Random Forest achieved higher test metrics across all dimensions compared to Logistic Regression:
  - **Precision**: **100.00%** vs. **95.31%** (+4.69%)
  - **Recall**: **93.12%** vs. **74.09%** (+19.03%)
  - **F1-Score**: **96.44%** vs. **83.37%** (+13.07%)
  - **ROC-AUC**: **99.98%** vs. **97.85%** (+2.13%)
  - **PR-AUC (Average Precision)**: **99.54%** vs. **88.55%** (+10.99%)
- **Zero False Positives in Random Forest**: On the 6,400 test traces, Random Forest produced **0 false positives**, whereas Logistic Regression produced **9 false positives**.
- **Set-Theoretic Error Decomposition**:
  - Mutually missed by both models ($\mathcal{FN}_{\text{both}} = \mathcal{FN}_{\text{LR}} \cap \mathcal{FN}_{\text{RF}}$): **16 samples**.
  - Detected by Random Forest but missed by Logistic Regression ($\mathcal{FN}_{\text{LR}} \setminus \mathcal{FN}_{\text{RF}}$): **48 samples** (net fault recovery).
  - Detected by Logistic Regression but missed by Random Forest ($\mathcal{FN}_{\text{RF}} \setminus \mathcal{FN}_{\text{LR}}$): Exactly **1 sample** (`Index 1047`: Uplink Buffer Overflow trace).
- **Benchmark Latency**: On local hardware, single-batch inference latency was measured at **0.15 ms per 1,000 samples** for Logistic Regression and **8.58 ms per 1,000 samples** for Random Forest.

---

## 2. Comprehensive 3-Model Comparison Table

The table below summarizes the architectural properties, classification metrics, computational latencies, and complexity footprints measured on the reserved test split:

| Metric / Property | Day 5 Logistic Regression | Day 8 Baseline Random Forest | Day 9 Selected Random Forest | Empirical Comparison on Test Split |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | Linear Generalized Linear Model (GLM) | Non-linear Bagged Ensemble (100 Decision Trees) | 5-Fold CV Selected Ensemble (100 Decision Trees) | Linear hyperplane vs. recursive orthogonal partitioning |
| **Hyperparameters** | `penalty='l2', C=1.0, threshold=0.50` | `n_estimators=100, depth=None, leaf=1, feat='sqrt'` | `n_estimators=100, depth=None, leaf=1, feat='sqrt'` | Best among the 18 configurations tested in Day 9 CV |
| **Feature Preprocessing** | `StandardScaler` (Z-score normalized) | Raw statistical features (Scale invariant) | Raw statistical features (Scale invariant) | Trees require no feature scaling; LR requires normalization |
| **Anomaly Precision (Test Set)** | **95.31%** | **100.00%** | **100.00%** | +4.69% (0 false alarms on test set) |
| **Anomaly Recall (Test Set)** | **74.09%** | **93.12%** | **93.12%** | +19.03% (230 vs. 183 true positives) |
| **Anomaly F1-Score (Test Set)** | **83.37%** | **96.44%** | **96.44%** | +13.07% F1 gain on test split |
| **ROC-AUC (Test Set)** | **97.85%** | **99.98%** | **99.98%** | +2.13% ranking discrimination |
| **PR-AUC (Average Precision)** | **88.55%** | **99.54%** | **99.54%** | +10.99% under 3.86% class prevalence |
| **True Positives (TP)** | 183 | 230 | 230 | +47 detected anomaly traces |
| **False Positives (FP)** | 9 | **0** | **0** | -9 false alarms on test set |
| **False Negatives (FN)**| 64 | 17 | 17 | -47 missed anomaly traces |
| **True Negatives (TN)** | 6,144 | 6,153 | 6,153 | +9 correctly classified normal traces |
| **Training Fit Latency** | **0.32 s** | 9.78 s | 8.34 s *(22.87 min CV tuning budget)* | LR trains ~25x faster for a single fit |
| **Measured Inference Latency** | **0.15 ms** / 1k samples | **8.58 ms** / 1k samples | **8.58 ms** / 1k samples | Measured on benchmark system CPU |
| **Model Structure** | 64 linear weights + 1 intercept bias | Max Depth 44, Mean Depth 30.80 | Max Depth 44, Mean Depth 30.80 | Linear formula vs. 100 deep decision trees |
| **Structural Units Count** | **65 scalar linear parameters** | **29,136 decision and leaf nodes** | **29,136 decision and leaf nodes** | Structurally distinct representations; not equivalent units |

---

## 3. Metric Analysis & Trade-offs on the Test Split

### A. Precision vs. Recall Considerations
- **False Positives**: On this 6,400-sample test partition, Logistic Regression generated **9 false positives** ($95.31\%$ precision). In an operational monitoring scenario, false alarms could lead to unnecessary investigation workflows. Random Forest produced **0 false positives** ($100.00\%$ precision) on the test split.
- **False Negatives**: Logistic Regression missed **64 out of 247 anomalies** (recall of $74.09\%$), leaving roughly a quarter of anomalies unflagged. Random Forest missed **17 out of 247 anomalies** (recall of $93.12\%$), identifying **48 additional anomalies** that Logistic Regression failed to detect.

### B. ROC-AUC vs. PR-AUC under 3.86% Prevalence
- **ROC-AUC Sensitivity**: Because the negative class comprises $96.14\%$ of the test set ($N=6,153$), 9 false positives yield a False Positive Rate of only $9 / 6,153 = 0.146\%$. Consequently, Logistic Regression achieves a high ROC-AUC of **97.85%**, which masks its precision degradation.
- **PR-AUC as the Evaluator**: Precision-Recall focuses on performance on the minority anomalous class. Logistic Regression achieved **88.55% PR-AUC**, whereas Random Forest achieved **99.54% PR-AUC**, demonstrating sustained precision across operational thresholds on this test partition.

---

## 4. Set-Theoretic Per-Sample Error Analysis

We conducted an exact set decomposition of test sample predictions across the 6,400 test traces:

```
                          TOTAL TEST ANOMALIES (N = 247)
        ┌─────────────────────────────────────────────────────────────┐
        │  LOGISTIC REGRESSION DETECTED (TP = 183)                    │
        │  ┌──────────────────────────────┬─────────────────────────┐ │
        │  │ Caught by Both Models        │ Caught by LR ONLY       │ │
        │  │ (TP_both = 182)              │ (Sample 1047)           │ │
        │  └──────────────────────────────┴─────────────────────────┘ │
        ├─────────────────────────────────────────────────────────────┤
        │  LOGISTIC REGRESSION MISSED (FN = 64)                       │
        │  ┌──────────────────────────────┬─────────────────────────┐ │
        │  │ Caught by RF ONLY            │ Missed by BOTH Models   │ │
        │  │ (FN_LR \ FN_RF = 48)         │ (FN_both = 16)          │ │
        │  └──────────────────────────────┴─────────────────────────┘ │
        └─────────────────────────────────────────────────────────────┘
```

### Exact Error Set Quantities & Sample Indices:
- **Total Test Anomalies**: 247
- **Mutually Missed Anomalies ($\mathcal{FN}_{\text{both}} = \mathcal{FN}_{\text{LR}} \cap \mathcal{FN}_{\text{RF}}$)**: **16 samples**  
  Exact indices: `[726, 760, 763, 787, 915, 962, 985, 1001, 1027, 1030, 1099, 1103, 1117, 1126, 1157, 1223]`
- **Anomalies Detected Exclusively by Random Forest ($\mathcal{FN}_{\text{LR}} \setminus \mathcal{FN}_{\text{RF}}$)**: **48 samples**  
  (e.g., Samples `8, 36, 93`)
- **Anomalies Detected Exclusively by Logistic Regression ($\mathcal{FN}_{\text{RF}} \setminus \mathcal{FN}_{\text{LR}}$)**: Exactly **1 sample**  
  Exact index: `[1047]`
- **False Positives in Logistic Regression ($\mathcal{FP}_{\text{LR}}$)**: **9 samples**  
  Exact indices: `[3452, 5222, 6448, 6957, 7113, 8108, 8625, 11686, 12355]`
- **False Positives in Random Forest ($\mathcal{FP}_{\text{RF}}$)**: **0 samples**

---

### Empirical Review of Specific Failure Cases

#### 1. The Single Anomaly Caught by LR but Missed by RF (`Sample 1047`)
- **Dataset Labels**: Application: `Twitch`, Anomaly Type: `Buffer Overflow (Gradual Buildup)`.
- **Model Predictions**: Logistic Regression $P = 1.000000$ (Predicted Anomaly), Random Forest $P = 0.400000$ (Predicted Normal).
- **Observed Feature Values**:
  - `Estimated_UL_Buffer_mean`: **708.01 KB**. Across normal training traces, the empirical median of `Estimated_UL_Buffer_mean` is 171.62 KB, and the 75th percentile is 339.16 KB. Sample 1047 is located in the upper quartile (~86th percentile) of buffer values.
  - `TX_Bytes_mean`: **79,826.5 Bytes**.
  - `RSRP_mean`: **-115.94 dBm**, `UL_SNR_mean`: **19.39 dB**, `UL_BLER_mean`: **0.1071**, `DL_BLER_mean`: **0.0013**.
- **Model Analysis**:
  - In Logistic Regression, `Estimated_UL_Buffer_mean` carries a large positive standardized coefficient ($\beta \approx +3.65$). The upper-quartile buffer value multiplied by this coefficient elevates the linear dot product $\mathbf{w}^T \mathbf{x} + b$, resulting in a high predicted probability ($P = 1.0000$).
  - In Random Forest, we distinguish individual tree predictions from the ensemble probability:
    - **Individual Tree Predictions**: Exactly 40 out of the 100 individual decision trees produced a discrete `predict()` output of 1 (Anomaly), while 60 trees produced 0 (Normal).
    - **Ensemble Anomaly Probability**: `RandomForestClassifier.predict_proba()` returned $\hat{P}(y = 1 \mid \mathbf{x}) = 0.4000$. In scikit-learn, ensemble predictions are based on the averaged class probabilities across all trees rather than a hard-vote fraction. Because the averaged anomaly probability of $0.4000$ was below the default decision threshold ($\tau = 0.50$), the ensemble predicted Normal (0), resulting in a false negative.
    - We avoid making unsupported claims about why individual trees made their predictions, noting only the measured feature values and model mechanics.

#### 2. False Positives in Logistic Regression (`Sample 3452`)
- **Dataset Labels**: Ground truth is Normal (`anomaly_present = 'No'`).
- **Model Predictions**: LR $P = 0.9281$ (False Positive), RF $P = 0.0400$ (Correctly Normal).
- **Observed Feature Values**:
  - `PRB_Utilization_DL_mean = 56.66%`, `RX_Bytes_mean = 7,593.7 Bytes`.
  - Radio indicators: `RSRP_mean = -100.69 dBm`, `UL_SNR_mean = 19.08 dB`, `UL_BLER_mean = 0.0814`.
- **Model Analysis**:
  - Logistic Regression assigns a large positive coefficient to `PRB_Utilization_DL_mean` ($\beta \approx +4.91$) and positive weights to byte throughput. In this trace, high PRB utilization and byte volume contribute additive positive terms to the linear sum, pushing the probability past 0.50.
  - Random Forest assigns probability $P = 0.0400$, correctly classifying the trace as Normal.

#### 3. Mutual False Negatives (`FN_both = 16 Samples`)
- **Dataset Labels**: The metadata for these traces includes anomaly types such as `Faulty Handover Algorithm (Too Frequent)` and `Faulty RF Filters (Temporal)` (e.g., Samples 760 and 763).
- **Observed Feature Values (Sample 760)**:
  - `UL_BLER_mean = 0.1096`, `UL_SNR_mean = 20.61 dB`, `RSRP_mean = -112.68 dBm`, `Estimated_UL_Buffer_mean = 294.34 KB`.
- **Model Analysis**:
  - Each TelecomTS sample trace consists of 128 discrete time steps (~12.7 seconds at 10 Hz). When compressed into 4 static moments (mean, std, min, max), temporal sequence ordering is lost.
  - For Sample 760, average error rates and signal levels fall within ranges commonly observed in normal operational traces. Neither model detected the fault signature from these summary statistics alone.

#### 4. The 48 Anomalies Recovered by Random Forest (`FN_LR only`)
- **Dataset Labels**: Primarily `Jamming` anomalies (e.g., Samples 8, 36, 93).
- **Observed Predictions (Sample 8)**: LR $P = 0.3798$ (Missed), RF $P = 1.0000$ (Detected).
- **Model Analysis**:
  - In these traces, degraded signal quality is accompanied by altered traffic volumes. In Logistic Regression, negative contributions from reduced traffic volume offset positive contributions from elevated error rates.
  - Random Forest partitions features non-linearly, enabling it to detect anomalous combinations even when individual feature values might pull in opposing directions in a linear model.

---

## 5. Feature Attribution Analysis: Linear Coefficients vs. Tree Gini Importance

```
TOP LOGISTIC REGRESSION COEFFICIENTS (Standardized Log-Odds Beta)
PRB_Utilization_DL_mean : +4.91  [Positive linear weight]
DL_MCS_mean             : -3.82  [Negative linear weight]
Estimated_UL_Buffer_mean: +3.65  [Positive linear weight]
RSRP_mean               : -3.12  [Negative linear weight]

TOP RANDOM FOREST GINI IMPORTANCES (Mean Decrease in Impurity, MDI)
UL_BLER_mean            : 0.142  [Top split criterion across forest]
UL_SNR_mean             : 0.118  [Uplink signal quality discriminator]
RSRP_mean               : 0.096  [Downlink signal coverage indicator]
Estimated_UL_Buffer_mean: 0.081  [Buffer utilization discriminator]
```

### Methodological Distinctions:
1. **Logistic Regression Standardized Coefficients ($\beta$)**:
   - Quantifies the estimated change in log-odds of the positive class per standard deviation increase in the feature, *holding other features fixed*.
   - *Limitation*: Assumes monotonic, linear relationships. Collinear features may experience coefficient inflation under L2 regularization.
2. **Random Forest Gini Importance (MDI)**:
   - Measures the total decrease in Gini impurity brought by splits on a given feature across all 100 trees, normalized to sum to 1.0.
   - *Limitation*: Non-directional (does not indicate whether higher values increase or decrease anomaly probability) and can be biased toward continuous features with high cardinality.

---

## 6. Computational Complexity & Measured Latency

| Property | Day 5 Logistic Regression | Day 8 / Day 9 Random Forest |
| :--- | :--- | :--- |
| **Training Fit Latency** | **0.32 s** (single fit) | **8.34 s** (single fit) / 22.87 min (18-config CV grid) |
| **Inference Latency (per 1,000 samples)** | **0.15 ms** | **8.58 ms** |
| **Structural Representation** | **65 scalar parameters** (64 weights + 1 intercept) | **29,136 decision and leaf nodes** across 100 trees |
| **Memory Footprint** | < 1 KB (coefficients vector) | ~ 4.5 MB (tree structures) |

*Clarification on SLAs*: No specific operational SLA threshold was defined for this benchmark project. Inference latency was measured at 0.15 ms (LR) and 8.58 ms (RF) per 1,000 samples on the local benchmark CPU. Whether 8.58 ms per 1,000 samples is suitable for production depends on the target ingestion rate, streaming window size, and hardware environment.

*Clarification on Complexity Units*: Linear parameters and decision tree nodes represent fundamentally different concepts. A linear parameter is a continuous scalar multiplier, whereas a tree node is a conditional branching split or terminal leaf in a piecewise constant model. They are presented as distinct architectural properties, not interchangeable metrics.

---

## 7. TelecomTS Limitations & Scope Boundaries

1. **Window-Level Summary Statistics vs. Sequential Dynamics**:
   - Each TelecomTS sample trace consists of 128 time steps (~12.7 seconds at 10 Hz sampling rate). Aggregating each KPI into 4 summary statistics (mean, std, min, max) loses sequential ordering, which limits detection of temporal patterns (such as frequent handovers) whose summary statistics overlap with normal traces.
2. **Absence of Cellular Topology & Neighbor-Cell Context**:
   - TelecomTS samples are isolated session traces. The dataset does not include cell sector identifiers, neighbor-cell lists, handover adjacency matrices, or geographic coordinates.
3. **Synthetic / Semi-Synthetic Fault Signatures**:
   - Anomalies in TelecomTS follow synthetic injection patterns. Real cellular environments exhibit unpredictable stochastic variations and hardware degradations.
4. **Scope Notice & Operator Independence**:
   - TelecomTS is an open-access research benchmark developed by Ali Maatouk et al. Results reflect machine learning evaluations on this benchmark and must not be interpreted as measured performance or infrastructure telemetry from SLTMobitel's commercial cellular network.

---

## 8. Validation Performed

1. **Notebook Execution Scope**:
   - The notebook code cells were executed sequentially in a local environment where the precomputed 64-feature dataset (`data/processed/telecom_features_64.pkl`) was loaded.
   - Model fitting, prediction generation, latency benchmarking, metric computation, comparison table generation, and plotting were executed live and verified.
   - Full raw-data-to-features extraction from `AliMaatouk/TelecomTS` (all 32,000 JSONL records) was verified on Day 4 and Day 8, but was not re-run from scratch during Day 10 validation.
   - The cell containing `!pip install ...` was validated for syntax but skipped in the active local environment.
2. **Re-used Results**:
   - The Day 9 cross-validation search (18 configurations, 90 fits, 22.87 minutes) was **not** re-run; its finding that `max_depth=None, min_samples_leaf=1, max_features='sqrt'` was the best configuration tested was reused.

---

## 9. Deliverables & Git Status

### Deliverables:
- [10_model_comparison.ipynb](file:///d:/Telecome/telecom-network-anomaly-detection/notebooks/10_model_comparison.ipynb): Validated comparative notebook with complete markdown explanations, live outputs, and visualizations.
- [day-10-comparison-report.md](file:///d:/Telecome/telecom-network-anomaly-detection/docs/day-10-comparison-report.md): Formal internship summary report for Day 10.
- `results/day10_model_comparison_table.csv`: Machine-readable comparison table.
- `results/day10_error_analysis_summary.csv`: Machine-readable error overlap table.
- `results/day10_roc_pr_comparison.png`: ROC and PR comparison plot.
- `results/day10_confusion_matrices.png`: Confusion matrices plot.
- `results/day10_feature_coefficients_vs_importance.png`: Feature comparison plot.

### Git Status:
- **Branch**: `feat/day-10-model-evaluation`
- **Untracked Files**: Only the two approved files:
  - `docs/day-10-comparison-report.md`
  - `notebooks/10_model_comparison.ipynb`
- **Result Artifacts**: Preserved in `results/` and ignored by `.gitignore`.
- **Day 1–9 Files & History**: Preserved and completely unchanged.
- **Main Branch**: Untouched.
- **Commit/Push/PR**: None initiated. Ready for review and approval.
