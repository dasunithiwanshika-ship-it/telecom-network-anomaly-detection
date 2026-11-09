"""
Complete experiment execution script:
1. Trains and serializes Logistic Regression, Random Forest, Isolation Forest, and Local Outlier Factor.
2. Evaluates all 4 models on the deterministic reserved test set (N=6,400).
3. Computes permutation feature importance for model explainability.
4. Performs threshold and sensitivity sweeps.
5. Saves metric CSVs, model artifacts, and professional publication-quality figures.
"""

import os
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Set clean visualization aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

print("=" * 70)
print("TELECOM ANOMALY DETECTION: 4-MODEL COMPREHENSIVE EXPERIMENT BENCHMARK")
print("=" * 70)

# Step 1: Load precomputed 64 features
print("\n[Step 1] Loading precomputed 64 KPI features...")
with open("data/processed/telecom_features_64.pkl", "rb") as f:
    X, y = pickle.load(f)

feature_names = list(X.columns)
print(f"  Dataset shape: {X.shape}, Target distribution: Normal={sum(y==0)}, Anomaly={sum(y==1)} ({y.mean()*100:.2f}%)")

# Step 2: Stratified 80/20 train-test split
print("\n[Step 2] Creating deterministic 80/20 stratified partition (random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: N={len(X_train)} (Anomalies={sum(y_train==1)}), Test: N={len(X_test)} (Anomalies={sum(y_test==1)})")

# Step 3: Feature Scaling
print("\n[Step 3] Fitting StandardScaler for distance/linear models...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.joblib")
print("  Saved 'models/scaler.joblib'")

# -------------------------------------------------------------
# Model 1: Logistic Regression (Supervised Baseline)
# -------------------------------------------------------------
print("\n[Step 4] Training Model 1: Logistic Regression (L2, C=1.0)...")
t0 = time.time()
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_fit_time = time.time() - t0

t0 = time.time()
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_inf_time = (time.time() - t0) / len(X_test) * 1000 * 1000 # ms / 1k samples
lr_preds = (lr_probs >= 0.50).astype(int)

joblib.dump(lr_model, "models/logistic_regression.joblib")
print(f"  LR Fit time: {lr_fit_time:.2f}s, Inference: {lr_inf_time:.2f} ms/1k samples. Saved 'models/logistic_regression.joblib'")

# -------------------------------------------------------------
# Model 2: Random Forest (Supervised Ensemble)
# -------------------------------------------------------------
print("\n[Step 5] Training Model 2: Random Forest (n=100, max_depth=None, leaf=1, sqrt)...")
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_fit_time = time.time() - t0

t0 = time.time()
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_inf_time = (time.time() - t0) / len(X_test) * 1000 * 1000 # ms / 1k samples
rf_preds = rf_model.predict(X_test)

joblib.dump(rf_model, "models/random_forest.joblib")
print(f"  RF Fit time: {rf_fit_time:.2f}s, Inference: {rf_inf_time:.2f} ms/1k samples. Saved 'models/random_forest.joblib'")

# -------------------------------------------------------------
# Model 3: Isolation Forest (Unsupervised Isolation Tree Ensemble)
# -------------------------------------------------------------
print("\n[Step 6] Training Model 3: Isolation Forest (Unsupervised, fitted on X_train WITHOUT labels)...")
# Using contamination=0.0386 matching data anomaly rate for balanced decision cutoff
t0 = time.time()
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.0386,
    random_state=42,
    n_jobs=-1,
)
iso_model.fit(X_train) # Unsupervised fitting!
iso_fit_time = time.time() - t0

t0 = time.time()
# Raw decision function: negative = anomaly. We invert so higher = more abnormal.
iso_raw_decision = iso_model.decision_function(X_test)
iso_scores = -iso_raw_decision
iso_inf_time = (time.time() - t0) / len(X_test) * 1000 * 1000
iso_preds = np.where(iso_model.predict(X_test) == -1, 1, 0)

joblib.dump(iso_model, "models/isolation_forest.joblib")
print(f"  IF Fit time: {iso_fit_time:.2f}s, Inference: {iso_inf_time:.2f} ms/1k samples. Saved 'models/isolation_forest.joblib'")

# -------------------------------------------------------------
# Model 4: Local Outlier Factor (Unsupervised Density-based Novelty Detection)
# -------------------------------------------------------------
print("\n[Step 7] Training Model 4: Local Outlier Factor (novelty=True, k=20, contamination=0.0386)...")
t0 = time.time()
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    novelty=True,
    contamination=0.0386,
    n_jobs=-1,
)
lof_model.fit(X_train_scaled) # Unsupervised fitting on standardized features!
lof_fit_time = time.time() - t0

t0 = time.time()
# Raw decision function: negative = anomaly. Inverted: higher = more abnormal.
lof_raw_decision = lof_model.decision_function(X_test_scaled)
lof_scores = -lof_raw_decision
lof_inf_time = (time.time() - t0) / len(X_test) * 1000 * 1000
lof_preds = np.where(lof_model.predict(X_test_scaled) == -1, 1, 0)

joblib.dump(lof_model, "models/local_outlier_factor.joblib")
print(f"  LOF Fit time: {lof_fit_time:.2f}s, Inference: {lof_inf_time:.2f} ms/1k samples. Saved 'models/local_outlier_factor.joblib'")

# -------------------------------------------------------------
# Step 8: Metric Computation & Unified Benchmark Table
# -------------------------------------------------------------
print("\n[Step 8] Computing test set classification metrics...")

models_eval = [
    ("Logistic Regression", "Supervised (Linear GLM)", "L2, C=1.0, tau=0.50, StandardScaler", lr_preds, lr_probs, lr_fit_time, lr_inf_time),
    ("Random Forest", "Supervised (Bagged Ensemble)", "n=100, depth=None, leaf=1, sqrt", rf_preds, rf_probs, rf_fit_time, rf_inf_time),
    ("Isolation Forest", "Unsupervised (Isolation Trees)", "n=100, contamination=0.0386, random_state=42", iso_preds, iso_scores, iso_fit_time, iso_inf_time),
    ("Local Outlier Factor", "Unsupervised (Local Density k-NN)", "k=20, novelty=True, contamination=0.0386, StandardScaler", lof_preds, lof_scores, lof_fit_time, lof_inf_time),
]

summary_rows = []
for name, mtype, settings, preds, scores, f_time, i_time in models_eval:
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, scores)
    pr = average_precision_score(y_test, scores)

    summary_rows.append({
        "Model Name": name,
        "Learning Paradigm": mtype,
        "Configuration Details": settings,
        "Test Precision": f"{prec * 100:.2f}%",
        "Test Recall": f"{rec * 100:.2f}%",
        "Test F1-Score": f"{f1 * 100:.2f}%",
        "ROC-AUC": f"{roc * 100:.2f}%",
        "PR-AUC (Avg Precision)": f"{pr * 100:.2f}%",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Training Fit Latency (s)": round(f_time, 2),
        "Inference Latency (ms/1k)": round(i_time, 2),
    })

comparison_df = pd.DataFrame(summary_rows)
comparison_df.to_csv("results/unified_4model_comparison_table.csv", index=False)
print("  Saved 'results/unified_4model_comparison_table.csv'")
print(comparison_df[["Model Name", "Learning Paradigm", "Test Precision", "Test Recall", "Test F1-Score", "ROC-AUC", "PR-AUC (Avg Precision)"]])

# -------------------------------------------------------------
# Step 9: Publication Plots
# -------------------------------------------------------------
print("\n[Step 9] Generating unified benchmark plots...")

# Plot 1: Unified ROC and PR Curves (4 models)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Isolation Forest": "#d62728", "Local Outlier Factor": "#9467bd"}

for name, _, _, _, scores, _, _ in models_eval:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_val = roc_auc_score(y_test, scores)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val*100:.2f}%)", color=colors[name], lw=2)

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUC = 50.0%)")
axes[0].set_xlabel("False Positive Rate (FPR)")
axes[0].set_ylabel("True Positive Rate (TPR / Recall)")
axes[0].set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

for name, _, _, _, scores, _, _ in models_eval:
    p_curve, r_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc_val = average_precision_score(y_test, scores)
    axes[1].plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc_val*100:.2f}%)", color=colors[name], lw=2)

axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.6, label=f"Baseline Prevalence ({y_test.mean()*100:.2f}%)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall (PR) Curve Comparison", fontsize=12, fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("results/unified_4model_roc_pr_comparison.png", dpi=300)
plt.close(fig)
print("  Saved 'results/unified_4model_roc_pr_comparison.png'")

# Plot 2: 2x2 Confusion Matrices Grid
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
cm_models = [
    ("Logistic Regression (Supervised)", lr_preds, axes[0, 0]),
    ("Random Forest (Supervised)", rf_preds, axes[0, 1]),
    ("Isolation Forest (Unsupervised)", iso_preds, axes[1, 0]),
    ("Local Outlier Factor (Unsupervised)", lof_preds, axes[1, 1]),
]

for title, preds, ax in cm_models:
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred Normal (0)", "Pred Anomaly (1)"],
        yticklabels=["True Normal (0)", "True Anomaly (1)"],
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
fig.savefig("results/unified_4model_confusion_matrices.png", dpi=300)
plt.close(fig)
print("  Saved 'results/unified_4model_confusion_matrices.png'")

# -------------------------------------------------------------
# Step 10: Model Explainability: Random Forest Feature Importance & Permutation Importance
# -------------------------------------------------------------
print("\n[Step 10] Computing Feature Importance & Permutation Importance on Random Forest...")
# 1. Gini MDI
rf_gini_imp = pd.DataFrame({
    "Feature": feature_names,
    "MDI_Importance": rf_model.feature_importances_
}).sort_values(by="MDI_Importance", ascending=False)
rf_gini_imp.to_csv("results/rf_gini_feature_importance.csv", index=False)

# 2. Permutation Importance on a representative stratified test sample of 1,200 samples
print("  Computing permutation importance over 1,200 test samples (5 repeats)...")
sample_test_idx = X_test.sample(n=1200, random_state=42).index
perm_result = permutation_importance(
    rf_model,
    X_test.loc[sample_test_idx],
    y_test.loc[sample_test_idx],
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "Feature": feature_names,
    "Permutation_Mean": perm_result.importances_mean,
    "Permutation_Std": perm_result.importances_std,
}).sort_values(by="Permutation_Mean", ascending=False)
perm_df.to_csv("results/rf_permutation_importance.csv", index=False)
print("  Saved 'results/rf_permutation_importance.csv'")

# Plot Feature Importance Comparison: Gini MDI vs Permutation Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

top12_gini = rf_gini_imp.head(12)
axes[0].barh(top12_gini["Feature"][::-1], top12_gini["MDI_Importance"][::-1], color="#2ca02c", alpha=0.85)
axes[0].set_xlabel("Mean Decrease in Impurity (Gini Importance)")
axes[0].set_title("Random Forest: Top 12 Gini Feature Importances", fontsize=11, fontweight="bold")
axes[0].grid(True, alpha=0.3)

top12_perm = perm_df.head(12)
axes[1].barh(
    top12_perm["Feature"][::-1],
    top12_perm["Permutation_Mean"][::-1],
    xerr=top12_perm["Permutation_Std"][::-1],
    color="#1f77b4",
    alpha=0.85,
    capsize=4,
)
axes[1].set_xlabel("Mean Decrease in Accuracy (Permutation Importance)")
axes[1].set_title("Random Forest: Top 12 Permutation Feature Importances", fontsize=11, fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("results/rf_feature_importance_and_permutation.png", dpi=300)
plt.close(fig)
print("  Saved 'results/rf_feature_importance_and_permutation.png'")

# -------------------------------------------------------------
# Step 11: Robustness & Sensitivity Sweeps
# -------------------------------------------------------------
print("\n[Step 11] Performing Sensitivity Analyses...")

# A. Threshold Sensitivity on Supervised Models
tau_grid = np.linspace(0.05, 0.95, 19)
thresh_records = []
for tau in tau_grid:
    # Logistic Regression
    lr_t_preds = (lr_probs >= tau).astype(int)
    lr_p = precision_score(y_test, lr_t_preds, zero_division=0)
    lr_r = recall_score(y_test, lr_t_preds, zero_division=0)
    lr_f = f1_score(y_test, lr_t_preds, zero_division=0)

    # Random Forest
    rf_t_preds = (rf_probs >= tau).astype(int)
    rf_p = precision_score(y_test, rf_t_preds, zero_division=0)
    rf_r = recall_score(y_test, rf_t_preds, zero_division=0)
    rf_f = f1_score(y_test, rf_t_preds, zero_division=0)

    thresh_records.append({
        "Threshold": round(float(tau), 2),
        "LR_Precision": round(lr_p, 4),
        "LR_Recall": round(lr_r, 4),
        "LR_F1": round(lr_f, 4),
        "RF_Precision": round(rf_p, 4),
        "RF_Recall": round(rf_r, 4),
        "RF_F1": round(rf_f, 4),
    })

thresh_df = pd.DataFrame(thresh_records)
thresh_df.to_csv("results/threshold_sensitivity_analysis.csv", index=False)
print("  Saved 'results/threshold_sensitivity_analysis.csv'")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(thresh_df["Threshold"], thresh_df["LR_Precision"], "b-o", label="Precision", markersize=4)
axes[0].plot(thresh_df["Threshold"], thresh_df["LR_Recall"], "r--s", label="Recall", markersize=4)
axes[0].plot(thresh_df["Threshold"], thresh_df["LR_F1"], "g-^", label="F1-Score", markersize=4)
axes[0].set_title("Logistic Regression: Threshold Trade-off Curve", fontweight="bold")
axes[0].set_xlabel("Classification Decision Threshold (tau)")
axes[0].set_ylabel("Metric Value")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresh_df["Threshold"], thresh_df["RF_Precision"], "b-o", label="Precision", markersize=4)
axes[1].plot(thresh_df["Threshold"], thresh_df["RF_Recall"], "r--s", label="Recall", markersize=4)
axes[1].plot(thresh_df["Threshold"], thresh_df["RF_F1"], "g-^", label="F1-Score", markersize=4)
axes[1].set_title("Random Forest: Threshold Robustness Curve", fontweight="bold")
axes[1].set_xlabel("Classification Decision Threshold (tau)")
axes[1].set_ylabel("Metric Value")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("results/threshold_sensitivity_tradeoff.png", dpi=300)
plt.close(fig)
print("  Saved 'results/threshold_sensitivity_tradeoff.png'")

# B. Contamination Sensitivity for Isolation Forest
contam_grid = [0.01, 0.02, 0.0386, 0.05, 0.08, 0.10, 0.15]
iso_records = []
for c in contam_grid:
    iso = IsolationForest(n_estimators=100, contamination=c, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    c_preds = np.where(iso.predict(X_test) == -1, 1, 0)
    p = precision_score(y_test, c_preds, zero_division=0)
    r = recall_score(y_test, c_preds, zero_division=0)
    f = f1_score(y_test, c_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, c_preds).ravel()
    iso_records.append({
        "Contamination": c,
        "Precision": round(p * 100, 2),
        "Recall": round(r * 100, 2),
        "F1_Score": round(f * 100, 2),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    })

iso_sens_df = pd.DataFrame(iso_records)
iso_sens_df.to_csv("results/isolation_forest_contamination_sensitivity.csv", index=False)
print("  Saved 'results/isolation_forest_contamination_sensitivity.csv'")

# C. Neighborhood Size (k) Sensitivity for LOF
k_grid = [10, 20, 30, 50]
lof_records = []
for k in k_grid:
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.0386, n_jobs=-1)
    lof.fit(X_train_scaled)
    k_preds = np.where(lof.predict(X_test_scaled) == -1, 1, 0)
    p = precision_score(y_test, k_preds, zero_division=0)
    r = recall_score(y_test, k_preds, zero_division=0)
    f = f1_score(y_test, k_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, k_preds).ravel()
    lof_records.append({
        "k_neighbors": k,
        "Precision": round(p * 100, 2),
        "Recall": round(r * 100, 2),
        "F1_Score": round(f * 100, 2),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    })

lof_sens_df = pd.DataFrame(lof_records)
lof_sens_df.to_csv("results/lof_k_sensitivity.csv", index=False)
print("  Saved 'results/lof_k_sensitivity.csv'")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS, BENCHMARKS, MODELS, AND FIGURES COMPLETED SUCCESSFULLY!")
print("=" * 70)
