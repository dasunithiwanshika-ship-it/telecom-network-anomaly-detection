# Internship Daily Summary Report — Day 8
**Project:** Telecom Network Anomaly Detection Prototype  
**Date:** October 9, 2026  
**Intern:** Dasuni Thiwanshika  
**Branch:** `feat/day-8-random-forest`  
**Dataset:** Public TelecomTS Dataset (Hugging Face `AliMaatouk/TelecomTS`)  
**Scope Notice:** Academic research prototype using open-access cellular KPI data; does not represent measured performance or faults on SLTMobitel's live network infrastructure.

---

## 1. Today's Objective
The objective of Day 8 was to train and evaluate our second supervised machine learning model—a **Random Forest Classifier**—for telecom network KPI anomaly detection, and determine whether a non-linear ensemble method outperforms our Day 5 Logistic Regression baseline across precision, recall, F1-score, ROC-AUC, PR-AUC, and false-alarm mitigation.

---

## 2. Tasks Completed
- [x] Initialized dedicated feature branch `feat/day-8-random-forest` from `feat/day-7-model-selection`.
- [x] Designed and authored standalone Google Colab-ready notebook: `notebooks/08_random_forest_baseline.ipynb`.
- [x] Built executable Python script: `08_random_forest_baseline.py` supporting standalone headless runs.
- [x] Ingested and validated all 32,000 time-series samples directly from Hugging Face `AliMaatouk/TelecomTS`.
- [x] Engineered 64 tabular statistical features across 16 numerical KPIs (Mean, Std, Min, Max).
- [x] Implemented stratified 80/20 train-test split (`random_state=42`) preserving 3.86% anomaly prevalence.
- [x] Trained baseline Random Forest Classifier ($B=100$ trees, `random_state=42`, `n_jobs=-1`).
- [x] Evaluated test set performance: Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix.
- [x] Conducted comprehensive side-by-side empirical comparison against Day 5 Logistic Regression.
- [x] Analyzed Gini Feature Importance (Mean Decrease in Impurity) and exported top 15 KPI rankings and visualization.
- [x] Formulated Git commit plan and documented research conclusions for internship diary and university viva.

---

## 3. Dataset & Feature Engineering Summary
- **Source**: `AliMaatouk/TelecomTS` (full chunked dataset, 32,000 samples).
- **Target Distribution**:
  - Normal (`0`): 30,765 samples (96.14%)
  - Anomaly (`1`): 1,235 samples (3.86%)
  - *Class Imbalance Ratio*: ~25:1.
- **16 Numerical Radio & Traffic KPIs**:
  - Signal Quality: `RSRP`, `UL_SNR`
  - Block Error & Modulation: `DL_BLER`, `UL_BLER`, `DL_MCS`, `UL_MCS`
  - Physical Resource Blocks (PRBs): `UL_NPRB`, `PRBs_DL_Current`, `PRBs_UL_Current`, `PRB_Utilization_DL`, `PRB_Utilization_UL`
  - Traffic Volumes & Buffers: `TX_Bytes`, `RX_Bytes`, `Estimated_UL_Buffer`, `UL_NumberOfPackets`, `DL_NumberOfPackets`
- **Statistical Aggregation**: 4 statistics per KPI (Mean, Standard Deviation, Min, Max) yielding a feature matrix $\mathbf{X} \in \mathbb{R}^{32000 \times 64}$.
- **Stratified Split (80/20, `random_state=42`)**:
  - Training Set: 25,600 samples (24,612 Normal, 988 Anomaly, rate = 3.86%)
  - Testing Set: 6,400 samples (6,153 Normal, 247 Anomaly, rate = 3.86%)
  - *Feature Scaling*: Tree-based models are scale-invariant; features were used directly without `StandardScaler`.

---

## 4. Model Training & Evaluation Results
The Random Forest model ($B=100$ estimators) trained in **11.84 seconds** using all CPU cores.

### Classification Report (Test Set: $N=6,400$):
```
              precision    recall  f1-score   support

      Normal     0.9972    1.0000    0.9986      6153
     Anomaly     1.0000    0.9312    0.9644       247

    accuracy                         0.9973      6400
   macro avg     0.9986    0.9656    0.9815      6400
weighted avg     0.9974    0.9973    0.9973      6400
```

### Confusion Matrix:
$$\begin{pmatrix} 6153 & 0 \\ 17 & 230 \end{pmatrix}$$
- **True Negatives (TN)**: 6,153
- **False Positives (FP)**: 0 *(Zero false alarms!)*
- **False Negatives (FN)**: 17 *(Missed anomalies)*
- **True Positives (TP)**: 230 *(Correctly detected anomalies)*
- **ROC-AUC Score**: **0.9998**
- **PR-AUC (Average Precision)**: **0.9954**

---

## 5. Comparison: Random Forest vs. Logistic Regression Baseline

| Metric | Day 5 Logistic Regression Baseline | Day 8 Random Forest Baseline | Empirical Delta | Operational Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **Precision (Anomaly)** | 95.31% | **100.00%** | **+4.69%** | Completely eliminated false alerts on test set. |
| **Recall (Anomaly)** | 74.09% | **93.12%** | **+19.03%** | Detected 47 additional real network anomalies. |
| **F1-Score (Anomaly)** | 83.37% | **96.44%** | **+13.07%** | Outstanding harmonic balance under heavy imbalance. |
| **ROC-AUC** | 97.85% | **99.98%** | **+2.13%** | Near-perfect ranking capability across all thresholds. |
| **PR-AUC** | ~88.42% | **99.54%** | **+11.12%** | High precision maintained across virtually all recall levels. |
| **False Positives (FP)** | 9 | **0** | **-9 (-100%)** | No wasted network engineer dispatch hours. |
| **False Negatives (FN)** | 64 | **17** | **-47 (-73.4%)** | Drastic 73.4% reduction in unflagged anomalous periods. |

---

## 6. Feature Importance Findings (Top 15 KPIs)
Computed via Mean Decrease in Impurity (MDI / Gini Importance):

| Rank | Feature | Gini Importance | Domain Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `RSRP_mean` | 0.1126 (11.26%) | Radio signal coverage attenuation is the leading indicator. |
| 2 | `UL_SNR_mean` | 0.1082 (10.82%) | Uplink channel noise/interference degradation. |
| 3 | `RSRP_max` | 0.0737 (7.37%) | Peak coverage drops during handovers or fading. |
| 4 | `UL_BLER_mean` | 0.0677 (6.77%) | Uplink block error rate indicates severe packet corruption. |
| 5 | `RSRP_min` | 0.0511 (5.11%) | Lowest observed signal level within the window. |
| 6 | `PRB_Utilization_DL_max` | 0.0406 (4.06%) | Cell tower downlink bandwidth saturation. |
| 7 | `RX_Bytes_mean` | 0.0376 (3.76%) | Throughput drops during abnormal connection states. |
| 8 | `PRB_Utilization_DL_mean`| 0.0328 (3.28%) | Sustained congestion on downlink radio channels. |
| 9 | `UL_NPRB_min` | 0.0272 (2.72%) | Minimum physical resource blocks allocated in uplink. |
| 10 | `UL_SNR_max` | 0.0268 (2.68%) | Uplink signal-to-noise ratio dynamics. |
| 11 | `UL_BLER_max` | 0.0220 (2.20%) | Peak packet error bursts. |
| 12 | `DL_NumberOfPackets_mean`| 0.0217 (2.17%) | Packet arrival rate perturbations. |
| 13 | `DL_BLER_max` | 0.0216 (2.16%) | Downlink block error rate spikes. |
| 14 | `RX_Bytes_min` | 0.0198 (1.98%) | Downlink traffic starvation periods. |
| 15 | `UL_SNR_min` | 0.0184 (1.84%) | Severe signal-to-noise degradation. |

---

## 7. Challenges & Solutions Encountered
1. **Missing Package Requirement (`ModuleNotFoundError: No module named 'datasets'`)**:
   - *Problem*: The local Python environment initially lacked Hugging Face `datasets`.
   - *Solution*: Identified error, requested intern/mentor approval per protocol, and installed `datasets` (version 5.1.0).
2. **Drive C Disk Space Constraint during Hugging Face Hub Download (`OS Error 112`)**:
   - *Problem*: Hugging Face default cache (`C:\Users\...\.cache\huggingface`) ran out of disk space on Drive C (only 50 MB free).
   - *Solution*: Redirected `HF_HOME`, `HF_DATASETS_CACHE`, and temporary extraction paths to Drive D (`D:\hf_cache`, 70+ GB free), cleared incomplete cache files on C:, and successfully downloaded and processed all 33 shards.
3. **Execution Runtime & Memory Stability**:
   - *Problem*: Computing 512,000 pandas Series across 32,000 samples.
   - *Solution*: Optimized feature calculation loop and utilized parallel tree construction (`n_jobs=-1`), training 100 trees in 11.84 seconds.

---

## 8. Key Learning Outcomes for Viva & Thesis
1. **Linear vs. Non-linear Classifiers in Telecom**: Logistic Regression's linear hyperplane struggles when anomalies are characterized by non-linear conjunctions (e.g., poor SNR is only anomalous when paired with high PRB demand). Random Forest's orthogonal recursive partitioning naturally captures these multi-variable relationships.
2. **Overcoming the Precision-Recall Trade-off**: On Day 6, lowering thresholds or weighting classes in Logistic Regression gained recall but caused severe false alarms (260 FPs). Random Forest improved recall to 93.12% while maintaining 100% precision (0 FPs).
3. **Feature Scale Invariance**: Decision tree splits are invariant to monotonic transformations, rendering z-score standardization unnecessary.
4. **MDI Interpretability**: Gini importance allows engineers to identify which radio layers (Layer 1 RF coverage vs. Layer 2 MAC PRBs) are most informative for anomaly monitoring.

---

## 9. Files Created or Modified
- `notebooks/08_random_forest_baseline.ipynb`: Complete Day 8 notebook with independent loading, markdown, outputs, and conclusions.
- `08_random_forest_baseline.py`: Standalone Python script executing the full pipeline.
- `results/day8_rf_feature_importance.csv`: Exported feature importance table.
- `results/day8_rf_feature_importance.png`: High-resolution horizontal bar plot of top 15 features.
- `docs/day-8-summary-report.md`: Formal daily internship report.

---

## 10. Git Commits & Branch Status
- **Current Branch**: `feat/day-8-random-forest`
- **Branch Tracking**: Local branch created from `feat/day-7-model-selection`.
- **Git Commit Plan**:
  1. `feat: add day 8 random forest baseline notebook and execution script`
  2. `feat: add day 8 evaluation results and feature importance visualization`
  3. `docs: add day 8 internship summary report`

---

## 11. Next Steps for Day 9
1. **Hyperparameter Tuning & Tree Depth Optimization**: Experiment with `max_depth`, `min_samples_split`, and `max_features` to evaluate whether tree pruning preserves high accuracy while reducing inference latency.
2. **Cost-Sensitive Learning / Class-Weighted Random Forest**: Test `class_weight='balanced'` to see if the remaining 17 false negatives can be caught without introducing false positives.
3. **Gradient Boosting Exploration (XGBoost / LightGBM)**: Benchmark sequential boosting against our Random Forest bagging baseline.
