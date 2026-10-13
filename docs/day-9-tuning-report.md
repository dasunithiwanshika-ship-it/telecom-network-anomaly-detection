# Internship Daily Summary Report — Day 9
**Project:** Telecom Network Anomaly Detection Prototype  
**Date:** October 13, 2026  
**Intern:** Dasuni Thiwanshika  
**Branch:** `feat/day-9-rf-tuning`  
**Dataset:** Public TelecomTS Dataset (Hugging Face `AliMaatouk/TelecomTS`)  
**Scope Notice:** Academic research prototype using open-access cellular KPI data; does not represent measured performance or faults on SLTMobitel's live network infrastructure.

---

## 1. Today's Objective
The objective of Day 9 was to investigate how Random Forest hyperparameters affect anomaly detection sensitivity, model generalization, and tree complexity. Specifically, we sought to determine whether shallower trees (`max_depth=10, 20`), increased leaf size (`min_samples_leaf=2, 5`), or wider feature sampling (`max_features=0.5`) could reduce model complexity and inference cost without sacrificing anomaly recall and precision.

---

## 2. Research Questions & Empirical Answers

### Question 1: Can shallower trees maintain strong anomaly-detection performance?
**Empirical Finding: No.**  
Restricting tree depth to `max_depth=10` severely degraded anomaly recall from **92.51% down to 80.67%–85.53%** in 5-fold cross-validation, reducing mean F1-score from **96.01% down to 89.23%–92.08%**. Moderately constrained trees (`max_depth=20`) performed better (mean F1 ~94.73%), but still missed approximately **2.3% more real anomalies**. Because network anomalies represent only 3.86% of samples, decision trees require deeper paths (~30–44 splits) to carve out the precise non-linear boundaries separating subtle network faults from normal traffic fluctuations.

### Question 2: How do minimum leaf size and feature subsampling affect the model?
**Empirical Finding:**
1. **Minimum Leaf Size (`min_samples_leaf`)**: Increasing leaf size from 1 to 2 or 5 acts as strong regularization. While precision remained near-perfect (>99.6%), recall dropped consistently across all depths because small clusters of rare anomalous patterns were pruned into majority normal leaves.
2. **Feature Subsampling (`max_features`)**: Evaluating 50% of features (32 KPIs) per split instead of `'sqrt'` (~8 KPIs) provided **no performance advantage** (mean CV F1 was 95.42% vs. 96.01%) while inflating training fit time by **nearly 400%** (from 8.06 s up to 31.01 s per fold). Feature decorrelation via `sqrt` subspace sampling is both computationally lighter and more accurate.

### Question 3: Can model complexity be reduced without substantially sacrificing anomaly recall and F1-score?
**Empirical Finding: No.**  
Any structural constraint on tree depth or leaf capacity resulted in a direct penalty to anomaly recall (ranging from -2.3% to -11.8%). The unconstrained forest (`max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`) remains the empirical global optimum on this dataset.

---

## 3. Tasks Completed
- [x] Initialized dedicated feature branch `feat/day-9-rf-tuning` from `feat/day-8-random-forest`.
- [x] Verified Day 8 feature extraction parity (16 numerical KPIs, 4 statistical moments, sample standard deviation with $ddof=1$, 64 features total).
- [x] Verified 80/20 stratified split determinism ($N_{\text{train}}=25,600$, $N_{\text{test}}=6,400$) and confirmed baseline ground truth reproduction ($TN=6,153, FP=0, FN=17, TP=230$).
- [x] Executed 5-Fold Stratified Cross-Validation strictly on the training set across an 18-combination hyperparameter grid (90 fits total).
- [x] Tracked cross-validation duration (22.87 min), final refit duration (8.34 s), and baseline fit duration (11.26 s) as distinct metrics.
- [x] Evaluated optimal tuned model on the untouched reserved test set.
- [x] Measured tree complexity metrics (maximum depth, mean depth, total node count, mean nodes per tree).
- [x] Exported cross-validation results table (`results/day9_rf_grid_search_results.csv`) and model comparison table (`results/day9_model_comparison.csv`).
- [x] Generated hyperparameter ranking visualization and complexity vs. training cost trade-off plot.
- [x] Authored self-contained, Google Colab-ready notebook: `notebooks/09_random_forest_tuning.ipynb`.
- [x] Migrated executable script to `src/09_random_forest_tuning.py` and cleaned temporary files.

---

## 4. Hyperparameter Search Configuration & Results

### Grid Search Setup
- **Training Population**: 25,600 samples (988 Anomalies, 3.86% prevalence)
- **Validation Strategy**: 5-Fold Stratified Cross-Validation (`StratifiedKFold`, `shuffle=True`, `random_state=42`)
- **Primary Optimization Metric**: Anomaly-class F1-score (`scoring='f1'`, refit on `f1`)
- **Parameter Grid**:
  - `max_depth`: `[10, 20, None]`
  - `min_samples_leaf`: `[1, 2, 5]`
  - `max_features`: `["sqrt", 0.5]`
  - Fixed: `n_estimators=100`, `random_state=42`
- **Total Search Fits**: 18 candidates $\times$ 5 folds = 90 fits

### 5-Fold Cross-Validation Rankings (Top 10 Combinations)

| Rank | `max_depth` | `min_samples_leaf` | `max_features` | Mean CV F1 | Std CV F1 | Mean CV Precision | Mean CV Recall | Mean CV ROC-AUC | Mean Fit Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **None** | **1** | **sqrt** | **0.9601** | **0.0055** | **0.9978** | **0.9251** | **0.9983** | **8.06 s** |
| 2 | None | 2 | sqrt | 0.9557 | 0.0068 | 0.9967 | 0.9180 | 0.9973 | 10.26 s |
| 3 | None | 1 | 0.5 | 0.9542 | 0.0049 | 0.9946 | 0.9170 | 0.9961 | 31.01 s |
| 4 | None | 2 | 0.5 | 0.9542 | 0.0057 | 0.9946 | 0.9170 | 0.9962 | 33.03 s |
| 5 | 20 | 1 | sqrt | 0.9473 | 0.0084 | 0.9978 | 0.9018 | 0.9976 | 5.78 s |
| 6 | 20 | 1 | 0.5 | 0.9458 | 0.0066 | 0.9956 | 0.9008 | 0.9922 | 25.63 s |
| 7 | None | 5 | 0.5 | 0.9452 | 0.0074 | 0.9956 | 0.8998 | 0.9948 | 29.66 s |
| 8 | 20 | 2 | 0.5 | 0.9452 | 0.0050 | 0.9956 | 0.8998 | 0.9926 | 29.23 s |
| 9 | 20 | 2 | sqrt | 0.9451 | 0.0091 | 0.9967 | 0.8988 | 0.9962 | 5.63 s |
| 10 | None | 5 | sqrt | 0.9439 | 0.0112 | 0.9966 | 0.8968 | 0.9974 | 8.57 s |

---

## 5. Final Evaluation on Reserved Test Set ($N=6,400$)

The selected model (`max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`) was evaluated on the unseen test set:

### Classification Report:
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
- **False Positives (FP)**: 0 *(Zero false alarms preserved)*
- **False Negatives (FN)**: 17 *(Missed anomalies)*
- **True Positives (TP)**: 230
- **ROC-AUC**: **0.9998**
- **PR-AUC (Average Precision)**: **0.9954**

---

## 6. Head-to-Head Comparison: Day 8 Baseline vs. Day 9 Tuned Model

| Metric / Property | Day 8 Baseline RF | Day 9 Tuned RF | Empirical Comparison / Takeaway |
| :--- | :--- | :--- | :--- |
| **Hyperparameters** | `max_depth=None`, `leaf=1`, `feat='sqrt'` | `max_depth=None`, `leaf=1`, `feat='sqrt'` | Selected by 5-fold CV out of 18 candidates. |
| **5-Fold CV F1-Score** | N/A *(Trained directly on train set)* | **96.01%** | Confirms robust generalization across splits. |
| **Test Precision (Anomaly)** | 100.00% | 100.00% | Zero false alarms maintained. |
| **Test Recall (Anomaly)** | 93.12% | 93.12% | 230 of 247 faults caught. |
| **Test F1-Score (Anomaly)** | 96.44% | 96.44% | Identical test metrics on verified split. |
| **Test ROC-AUC** | 0.9998 | 0.9998 | Flawless class ranking across all thresholds. |
| **Test PR-AUC** | 0.9954 | 0.9954 | Consistent precision across full recall range. |
| **True Positives (TP)** | 230 | 230 | Identical fault coverage. |
| **False Positives (FP)** | 0 | 0 | Identical zero-false-alarm guarantee. |
| **False Negatives (FN)** | 17 | 17 | Identical 17 missed anomaly windows. |
| **Max Tree Depth** | 44 | 44 | Deepest tree path across 100 trees. |
| **Mean Tree Depth** | 30.80 | 30.80 | Average tree depth. |
| **Total Tree Nodes** | 29,136 | 29,136 | Total decision nodes across forest. |
| **Mean Nodes per Tree** | 291.36 | 291.36 | Structural footprint per tree. |
| **Full Fit Duration** | 11.26 s | 8.34 s | Single fit execution latency on full train set. |
| **CV Search Duration** | N/A | 1,371.95 s (22.87 min) | Total computational budget for 90 CV fits. |

---

## 7. Key Learning Outcomes for Viva & Thesis

1. **Honest Model Selection in Academic Research**:
   Hyperparameter tuning does not guarantee a metric increase over an already strong, well-chosen baseline. Demonstrating that 5-fold cross-validation rigorously confirmed the baseline's parameter set validates the scientific integrity of the Day 8 model rather than inventing cosmetic gains.
2. **The Danger of Premature Tree Pruning under Imbalance**:
   In balanced classification, shallow trees often prevent overfitting. In telecom anomaly detection under a 25:1 imbalance, shallow trees prune out rare anomaly leaves, severely depressing recall (from 93.12% down to 80.67%).
3. **Computational Efficiency of the Random Subspace Method**:
   Using `max_features='sqrt'` (8 features) achieved higher F1 (96.01% vs 95.42%) in one-quarter of the training time compared to `max_features=0.5` (32 features), proving the theoretical power of Breiman's tree decorrelation principle.

---

## 8. Files Created, Moved, or Modified
- `src/09_random_forest_tuning.py`: Standalone execution script organized under `src/`.
- `notebooks/09_random_forest_tuning.ipynb`: Complete, Google Colab-runnable Day 9 notebook with Markdown explanations, outputs, and conclusions.
- `docs/day-9-tuning-report.md`: Formal daily internship report.
- `results/day9_rf_grid_search_results.csv`: Table of all 18 hyperparameter combinations ranked by 5-fold CV F1.
- `results/day9_model_comparison.csv`: Head-to-head empirical comparison table.
- `results/day9_rf_hyperparameter_search.png`: Visualization of hyperparameter rankings.
- `results/day9_complexity_tradeoff.png`: Scatter plot of fit duration vs. anomaly F1-score across tree depths.

---

## 9. Git Status & Next Steps for Day 10
- **Branch**: `feat/day-9-rf-tuning`
- **Main Branch**: Untouched.
- **Next Step**: Await user review and confirmation before proceeding to Day 10 (Final Model Evaluation and Error Analysis across Logistic Regression, Day 8 Baseline Random Forest, and Day 9 Tuned Random Forest).
