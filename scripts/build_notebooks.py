"""
Script to generate notebooks 11 and 12 using standard library json.
"""

import os
import json

os.makedirs("notebooks", exist_ok=True)

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(text):
    lines = [l + "\n" for l in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(code):
    lines = [l + "\n" for l in code.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# -------------------------------------------------------------
# Notebook 11: Unsupervised Anomaly Detection
# -------------------------------------------------------------
cells11 = [
    markdown_cell("""# Experiment 11: Unsupervised Anomaly Detection
## Isolation Forest & Local Outlier Factor on Public Cellular Network KPI Data

**Project:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks  
**Research Question:** How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?  
**Dataset Reference:** Hugging Face `AliMaatouk/TelecomTS` (32,000 public cellular traces, 64 statistical moments)  
**Research Scope Notice:** Independent research prototype using public cellular KPI data. The dataset does NOT represent SLTMobitel operational records, Sri Lankan telecom infrastructure, or confirmed physical fault causes.

---

### Objectives
1. Implement **Isolation Forest (IF)**: an unsupervised tree ensemble based on path length isolation.
2. Implement **Local Outlier Factor (LOF)**: an unsupervised density-based method comparing local reachability density.
3. Fit both models strictly **without anomaly labels**.
4. Generate anomaly scores, decision functions, and discrete predictions.
5. Evaluate predictions against reference test labels on the exact same 80/20 test split ($N=6,400$).
6. Investigate the effect of the contamination hyperparameter and neighborhood size ($k$)."""),

    code_cell("""import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Step 1: Load precomputed 64 features
pickle_path = "../data/processed/telecom_features_64.pkl"
if not os.path.exists(pickle_path):
    pickle_path = "data/processed/telecom_features_64.pkl"

with open(pickle_path, "rb") as f:
    X, y = pickle.load(f)

# Step 2: Deterministic 80/20 stratified partition
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
print(f"Test Class Balance: Normal={sum(y_test==0)}, Anomaly={sum(y_test==1)} ({y_test.mean()*100:.2f}%)")"""),

    markdown_cell("""## Part 1: Isolation Forest (iForest)
### Theoretical Background
Isolation Forest isolates anomalies by recursively partitioning randomly selected features at random cut-points.
Because anomalies are few and structurally divergent, they require fewer random orthogonal splits to isolate from the majority population, resulting in shorter average path lengths in the isolation trees:
$$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$
where $E(h(x))$ is the expected path length and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree.

Crucially, **no target labels ($y$) are provided during model fitting**."""),

    code_cell("""# Fit Isolation Forest (Unsupervised)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.0386, # matches training anomaly rate for balanced cutoff
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

# Invert decision function: lower scikit-learn scores mean abnormal; inversion ensures higher = more anomalous
iso_scores = -iso_forest.decision_function(X_test)
iso_preds = np.where(iso_forest.predict(X_test) == -1, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_test, iso_preds).ravel()
print("Isolation Forest Test Metrics:")
print(f"  Precision: {precision_score(y_test, iso_preds)*100:.2f}%")
print(f"  Recall:    {recall_score(y_test, iso_preds)*100:.2f}%")
print(f"  F1-Score:  {f1_score(y_test, iso_preds)*100:.2f}%")
print(f"  ROC-AUC:   {roc_auc_score(y_test, iso_scores)*100:.2f}%")
print(f"  PR-AUC:    {average_precision_score(y_test, iso_scores)*100:.2f}%")
print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")"""),

    markdown_cell("""## Part 2: Local Outlier Factor (LOF)
### Theoretical Background
Local Outlier Factor measures the local deviation of a given data point with respect to its surrounding neighbors.
It calculates:
1. $k$-distance and reachability distance to neighbors.
2. Local reachability density (lrd).
3. The ratio of local reachability density of neighbors to that of the point itself (LOF score).

To evaluate on unseen test observations ($X_{test}$), we employ scikit-learn's **novelty detection mode (`novelty=True`)**, fitting on standardized training data."""),

    code_cell("""# Standardize features for distance-based k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lof_novelty = LocalOutlierFactor(
    n_neighbors=20,
    novelty=True,
    contamination=0.0386,
    n_jobs=-1
)
lof_novelty.fit(X_train_scaled)

lof_scores = -lof_novelty.decision_function(X_test_scaled)
lof_preds = np.where(lof_novelty.predict(X_test_scaled) == -1, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_test, lof_preds).ravel()
print("Local Outlier Factor (k=20) Test Metrics:")
print(f"  Precision: {precision_score(y_test, lof_preds)*100:.2f}%")
print(f"  Recall:    {recall_score(y_test, lof_preds)*100:.2f}%")
print(f"  F1-Score:  {f1_score(y_test, lof_preds)*100:.2f}%")
print(f"  ROC-AUC:   {roc_auc_score(y_test, lof_scores)*100:.2f}%")
print(f"  PR-AUC:    {average_precision_score(y_test, lof_scores)*100:.2f}%")
print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")"""),

    markdown_cell("""## Part 3: Contamination & Sensitivity Sweeps
We analyze how the contamination hyperparameter in Isolation Forest and the neighborhood size ($k$) in LOF influence detection rates."""),

    code_cell("""# Load and display saved sensitivity sweeps
results_dir = "../results" if os.path.exists("../results") else "results"
iso_sens = pd.read_csv(os.path.join(results_dir, "isolation_forest_contamination_sensitivity.csv"))
lof_sens = pd.read_csv(os.path.join(results_dir, "lof_k_sensitivity.csv"))

print("Isolation Forest Contamination Sweep:")
print(iso_sens)

print("\\nLocal Outlier Factor Neighborhood Size (k) Sweep:")
print(lof_sens)""")
]

nb11 = create_notebook(cells11)
with open("notebooks/11_unsupervised_anomaly_detection.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb11, f, indent=1)
print("Saved notebooks/11_unsupervised_anomaly_detection.ipynb")


# -------------------------------------------------------------
# Notebook 12: Unified Comparative Evaluation & Explainability
# -------------------------------------------------------------
cells12 = [
    markdown_cell("""# Experiment 12: Unified Model Comparison, Feature Importance & Explainability

**Project:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks  
**Research Question:** How does unsupervised anomaly detection compare with supervised classification for detecting network anomalies in public telecom KPI data, and which KPI features contribute most to detection?  
**Models Evaluated:**
1. Logistic Regression (Supervised GLM baseline)
2. Random Forest (Supervised Ensemble)
3. Isolation Forest (Unsupervised Isolation Tree)
4. Local Outlier Factor (Unsupervised Density-based k-NN)

---

### Objectives
1. Unified 4-model performance comparison on the reserved test set ($N=6,400$).
2. Receiver Operating Characteristic (ROC) and Precision-Recall (PR) comparative analysis.
3. Feature importance analysis via Mean Decrease in Impurity (Gini) and Permutation Feature Importance.
4. Threshold sensitivity and classification trade-off analysis.
5. Multi-model error analysis."""),

    code_cell("""import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "../results" if os.path.exists("../results") else "results"

# Load unified comparison table
comp_df = pd.read_csv(os.path.join(results_dir, "unified_4model_comparison_table.csv"))
print("Unified 4-Model Comparative Benchmark Table:")
print(comp_df[["Model Name", "Learning Paradigm", "Test Precision", "Test Recall", "Test F1-Score", "ROC-AUC", "PR-AUC (Avg Precision)"]])"""),

    markdown_cell("""## Feature Importance & Explainability
We evaluate two complementary feature importance methodologies:
1. **Gini Importance (MDI)**: Measures total impurity reduction across tree splits. Fast to compute, but non-directional and can favor high-cardinality continuous features.
2. **Permutation Feature Importance**: Model-agnostic technique that measures degradation in test accuracy when values of a specific feature are randomly shuffled, breaking its relationship with the target."""),

    code_cell("""# Display top 10 permutation importance features
perm_df = pd.read_csv(os.path.join(results_dir, "rf_permutation_importance.csv"))
print("Top 10 Features by Permutation Importance (Accuracy Degradation):")
print(perm_df.head(10))"""),

    markdown_cell("""## Decision Threshold Sensitivity
We evaluate how shifting the classification cutoff $\\tau$ from 0.05 to 0.95 impacts the balance between false alarms and missed faults."""),

    code_cell("""thresh_df = pd.read_csv(os.path.join(results_dir, "threshold_sensitivity_analysis.csv"))
print("Threshold Sensitivity Grid (Sample):")
print(thresh_df[thresh_df["Threshold"].isin([0.1, 0.3, 0.5, 0.7, 0.9])])""")
]

nb12 = create_notebook(cells12)
with open("notebooks/12_final_model_evaluation_and_explainability.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb12, f, indent=1)
print("Saved notebooks/12_final_model_evaluation_and_explainability.ipynb")
