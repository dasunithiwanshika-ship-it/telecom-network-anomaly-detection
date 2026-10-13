"""
Day 9 — Random Forest Hyperparameter Tuning and Complexity Analysis
Telecom Network Anomaly Detection Project
Dataset: AliMaatouk/TelecomTS (Hugging Face)
Author: Dasuni Thiwanshika
Branch: feat/day-9-rf-tuning
"""

import os
# Configure cache directories on D drive if needed for local execution
if os.path.exists("D:/"):
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "D:/hf_cache/datasets")
    os.environ.setdefault("TEMP", "D:/temp")
    os.environ.setdefault("TMP", "D:/temp")

import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

def main():
    print("=" * 70, flush=True)
    print("Day 9: Random Forest Hyperparameter Tuning & Model Complexity", flush=True)
    print("Telecom Network Anomaly Detection (Public TelecomTS)", flush=True)
    print("=" * 70, flush=True)

    cache_file = "data/processed/telecom_features_64.pkl"

    if os.path.exists(cache_file):
        print(f"\n--- Loading precomputed 64 features from local cache ---", flush=True)
        t_load = time.time()
        with open(cache_file, "rb") as f:
            X, y = pickle.load(f)
        print(f"Loaded features in {time.time() - t_load:.2f} seconds. Shape: {X.shape}", flush=True)
    else:
        # Step 2: Load TelecomTS
        print("\n--- Step 2: Loading TelecomTS dataset ---", flush=True)
        start_time = time.time()
        cache_dir = "D:/hf_cache/datasets" if os.path.exists("D:/") else None
        dataset = load_dataset(
            "AliMaatouk/TelecomTS",
            data_files={"full": "**/chunked.jsonl"},
            cache_dir=cache_dir
        )
        data = dataset["full"]
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds. Total samples: {len(data)}", flush=True)

        # Step 3: Recreate 64 tabular statistical features exactly as Day 8
        print("\n--- Step 3: Recreating 64 engineered features ---", flush=True)
        t_feat = time.time()
        numerical_kpis = [
            "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS",
            "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer",
            "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL",
            "PRB_Utilization_UL", "UL_NumberOfPackets", "DL_NumberOfPackets"
        ]
        
        feature_rows = []
        for sample in data:
            row = {}
            for kpi in numerical_kpis:
                values = sample["KPIs"][kpi]
                row[f"{kpi}_mean"] = sum(values) / len(values)
                row[f"{kpi}_std"] = pd.Series(values).std()  # ddof=1, exact Day 8 logic
                row[f"{kpi}_min"] = min(values)
                row[f"{kpi}_max"] = max(values)
            feature_rows.append(row)
            
        X = pd.DataFrame(feature_rows)
        y = pd.Series([
            1 if sample["labels"]["anomaly_present"] == "Yes" else 0
            for sample in data
        ])
        print(f"64 features extracted in {time.time() - t_feat:.2f} seconds. Feature shape: {X.shape}", flush=True)
        
        # Save to cache for instant future reuse
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((X, y), f)
        print(f"Cached features to {cache_file}", flush=True)

    # Step 4: Rigorous Validation of Setup and Split
    print("\n--- Step 4: Validating feature definitions and stratified train-test split ---", flush=True)
    # Check 1: Sample counts
    assert len(X) == 32000, f"Expected 32,000 samples, got {len(X)}"
    assert len(y) == 32000, f"Expected 32,000 target labels, got {len(y)}"
    # Check 2: Feature count & column names
    assert X.shape[1] == 64, f"Expected 64 features, got {X.shape[1]}"
    print("Feature count verified: Exactly 64 statistical KPI features.", flush=True)

    # Check 3: Target distribution
    normal_count = (y == 0).sum()
    anomaly_count = (y == 1).sum()
    assert normal_count == 30765, f"Expected 30,765 normal samples, got {normal_count}"
    assert anomaly_count == 1235, f"Expected 1,235 anomalous samples, got {anomaly_count}"
    print(f"Target distribution verified: Normal={normal_count} (96.14%), Anomaly={anomaly_count} (3.86%)", flush=True)

    # Check 4: Deterministic 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    assert len(X_train) == 25600, f"Expected 25,600 training samples, got {len(X_train)}"
    assert len(X_test) == 6400, f"Expected 6,400 test samples, got {len(X_test)}"
    assert (y_train == 1).sum() == 988, f"Expected 988 training anomalies, got {(y_train == 1).sum()}"
    assert (y_test == 1).sum() == 247, f"Expected 247 test anomalies, got {(y_test == 1).sum()}"
    print(f"Training split: 25,600 samples (988 anomalies, rate = {y_train.mean():.4f})", flush=True)
    print(f"Testing split: 6,400 samples (247 anomalies, rate = {y_test.mean():.4f})", flush=True)

    # Check 5: Ground truth reproduction of Day 8 Baseline
    print("\n--- Verifying Day 8 Baseline Ground Truth Reproduction on Test Split ---", flush=True)
    base_t0 = time.time()
    day8_baseline = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    day8_baseline.fit(X_train, y_train)
    day8_baseline_fit_time = time.time() - base_t0
    print(f"Day 8 baseline fit time: {day8_baseline_fit_time:.2f} seconds", flush=True)

    y_pred_base = day8_baseline.predict(X_test)
    y_prob_base = day8_baseline.predict_proba(X_test)[:, 1]

    cm_base = confusion_matrix(y_test, y_pred_base)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    prec_base = precision_score(y_test, y_pred_base)
    rec_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    roc_auc_base = roc_auc_score(y_test, y_prob_base)
    pr_auc_base = average_precision_score(y_test, y_prob_base)

    print(f"Baseline Confusion Matrix: TN={tn_b}, FP={fp_b}, FN={fn_b}, TP={tp_b}", flush=True)
    print(f"Baseline Metrics: Precision={prec_base*100:.2f}%, Recall={rec_base*100:.2f}%, F1={f1_base*100:.2f}%", flush=True)
    print(f"Baseline ROC-AUC: {roc_auc_base:.4f}, PR-AUC: {pr_auc_base:.4f}", flush=True)

    # Strict assertion verifying split identity against Day 8 notebook ground truth
    assert tn_b == 6153 and fp_b == 0 and fn_b == 17 and tp_b == 230, \
        f"Ground truth verification failed! Got TN={tn_b}, FP={fp_b}, FN={fn_b}, TP={tp_b}"
    print(">> GROUND TRUTH REPRODUCTION CONFIRMED: Test split is 100% identical to Day 8 <<", flush=True)

    # Baseline complexity metrics
    base_depths = [e.get_depth() for e in day8_baseline.estimators_]
    base_nodes = [e.tree_.node_count for e in day8_baseline.estimators_]
    base_max_depth = max(base_depths)
    base_mean_depth = np.mean(base_depths)
    base_total_nodes = sum(base_nodes)
    base_mean_nodes = np.mean(base_nodes)
    print(f"Baseline Complexity: Max Depth={base_max_depth}, Mean Depth={base_mean_depth:.2f}, Total Nodes={base_total_nodes:,}", flush=True)

    # Step 5: Hyperparameter Grid Search on Training Set Only
    print("\n" + "=" * 70, flush=True)
    print("STEP 5: HYPERPARAMETER GRID SEARCH (TRAINING SET ONLY, 5-FOLD STRATIFIED CV)", flush=True)
    print("=" * 70, flush=True)
    param_grid = {
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5]
    }
    print(f"Candidate Grid: {param_grid}", flush=True)
    print("18 parameter combinations x 5 folds = 90 fits total.", flush=True)
    print("Optimization Objective: Maximize Anomaly F1-Score on CV validation folds.", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    # Base RF uses n_jobs=-1 to parallelize tree construction across all 4 CPU cores
    rf_for_grid = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=rf_for_grid,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit="f1",
        n_jobs=1,
        verbose=2
    )

    print("\nStarting 5-fold cross-validation grid search...", flush=True)
    cv_start_time = time.time()
    grid_search.fit(X_train, y_train)
    cv_duration = time.time() - cv_start_time
    refit_duration = grid_search.refit_time_

    print("\n" + "=" * 70, flush=True)
    print("GRID SEARCH COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Total Cross-Validation Duration: {cv_duration:.2f} seconds ({cv_duration/60:.2f} minutes)", flush=True)
    print(f"Selected-Model Refit Time on Full Training Set: {refit_duration:.2f} seconds", flush=True)
    print(f"Day 8 Baseline Fit Time: {day8_baseline_fit_time:.2f} seconds", flush=True)
    print("=" * 70, flush=True)

    # Step 6: Process and Record Cross-Validation Results
    print("\n--- Step 6: Analyzing Cross-Validation Search Results ---", flush=True)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    summary_cols = [
        "param_max_depth", "param_min_samples_leaf", "param_max_features",
        "mean_test_f1", "std_test_f1",
        "mean_test_precision", "std_test_precision",
        "mean_test_recall", "std_test_recall",
        "mean_test_roc_auc", "mean_fit_time"
    ]
    cv_summary = cv_results[summary_cols].copy()
    cv_summary["param_max_depth"] = cv_summary["param_max_depth"].fillna("None")
    cv_summary = cv_summary.sort_values(by="mean_test_f1", ascending=False).reset_index(drop=True)
    cv_summary.index = cv_summary.index + 1  # 1-indexed rank

    print("\nFull Hyperparameter Search Rankings (Mean 5-Fold Anomaly F1):", flush=True)
    print(cv_summary.to_string(), flush=True)

    os.makedirs("results", exist_ok=True)
    cv_summary_path = "results/day9_rf_grid_search_results.csv"
    cv_summary.to_csv(cv_summary_path, index_label="rank")
    print(f"\nSaved cross-validation results table to {cv_summary_path}", flush=True)

    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_
    print(f"\nBest Hyperparameters by CV F1-score: {best_params}", flush=True)
    print(f"Best 5-Fold CV Anomaly F1-score: {best_cv_f1:.4f} ({best_cv_f1*100:.2f}%)", flush=True)

    # Step 7: Evaluate Selected Model Once on Untouched Test Set
    print("\n--- Step 7: Evaluating Selected Model on Reserved Test Set (N=6,400) ---", flush=True)
    best_model = grid_search.best_estimator_

    eval_t0 = time.time()
    y_pred_tuned = best_model.predict(X_test)
    y_prob_tuned = best_model.predict_proba(X_test)[:, 1]
    test_eval_duration = time.time() - eval_t0

    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()
    prec_tuned = precision_score(y_test, y_pred_tuned)
    rec_tuned = recall_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)
    pr_auc_tuned = average_precision_score(y_test, y_prob_tuned)

    print("\nTuned Model Test Classification Report:", flush=True)
    print(classification_report(y_test, y_pred_tuned, target_names=["Normal", "Anomaly"], digits=4), flush=True)
    print(f"Confusion Matrix: TN={tn_t}, FP={fp_t}, FN={fn_t}, TP={tp_t}", flush=True)
    print(f"Precision: {prec_tuned*100:.2f}%", flush=True)
    print(f"Recall: {rec_tuned*100:.2f}%", flush=True)
    print(f"F1-Score: {f1_tuned*100:.2f}%", flush=True)
    print(f"ROC-AUC: {roc_auc_tuned:.4f}", flush=True)
    print(f"PR-AUC (Average Precision): {pr_auc_tuned:.4f}", flush=True)

    # Step 8: Model Complexity Measurement
    tuned_depths = [e.get_depth() for e in best_model.estimators_]
    tuned_nodes = [e.tree_.node_count for e in best_model.estimators_]
    tuned_max_depth = max(tuned_depths)
    tuned_mean_depth = np.mean(tuned_depths)
    tuned_total_nodes = sum(tuned_nodes)
    tuned_mean_nodes = np.mean(tuned_nodes)

    print("\n--- Step 8: Model Complexity Comparison ---", flush=True)
    print(f"Tuned Model Max Depth: {tuned_max_depth} (vs Baseline: {base_max_depth})", flush=True)
    print(f"Tuned Model Mean Depth: {tuned_mean_depth:.2f} (vs Baseline: {base_mean_depth:.2f})", flush=True)
    print(f"Tuned Model Total Nodes: {tuned_total_nodes:,} (vs Baseline: {base_total_nodes:,})", flush=True)
    print(f"Tuned Model Mean Nodes/Tree: {tuned_mean_nodes:.2f} (vs Baseline: {base_mean_nodes:.2f})", flush=True)

    # Step 9: Head-to-Head Comparison Table
    print("\n" + "=" * 75, flush=True)
    print("HEAD-TO-HEAD COMPARISON: DAY 8 BASELINE VS DAY 9 TUNED RANDOM FOREST", flush=True)
    print("=" * 75, flush=True)
    comparison_df = pd.DataFrame({
        "Metric / Property": [
            "Hyperparameters",
            "Cross-Validation F1-score",
            "Test Precision (Anomaly)",
            "Test Recall (Anomaly)",
            "Test F1-score (Anomaly)",
            "Test ROC-AUC",
            "Test PR-AUC",
            "True Positives (TP)",
            "False Positives (FP)",
            "False Negatives (FN)",
            "True Negatives (TN)",
            "Max Tree Depth",
            "Mean Tree Depth",
            "Total Tree Nodes",
            "Mean Nodes per Tree",
            "Full Fit Duration",
            "CV Search Duration (90 fits)"
        ],
        "Day 8 Baseline RF": [
            "max_depth=None, min_samples_leaf=1, max_features='sqrt'",
            "N/A (Evaluated directly on test)",
            f"{prec_base*100:.2f}%",
            f"{rec_base*100:.2f}%",
            f"{f1_base*100:.2f}%",
            f"{roc_auc_base:.4f}",
            f"{pr_auc_base:.4f}",
            f"{tp_b}",
            f"{fp_b}",
            f"{fn_b}",
            f"{tn_b}",
            f"{base_max_depth}",
            f"{base_mean_depth:.2f}",
            f"{base_total_nodes:,}",
            f"{base_mean_nodes:.2f}",
            f"{day8_baseline_fit_time:.2f} s",
            "N/A"
        ],
        "Day 9 Tuned RF": [
            str(best_params),
            f"{best_cv_f1*100:.2f}%",
            f"{prec_tuned*100:.2f}%",
            f"{rec_tuned*100:.2f}%",
            f"{f1_tuned*100:.2f}%",
            f"{roc_auc_tuned:.4f}",
            f"{pr_auc_tuned:.4f}",
            f"{tp_t}",
            f"{fp_t}",
            f"{fn_t}",
            f"{tn_t}",
            f"{tuned_max_depth}",
            f"{tuned_mean_depth:.2f}",
            f"{tuned_total_nodes:,}",
            f"{tuned_mean_nodes:.2f}",
            f"{refit_duration:.2f} s",
            f"{cv_duration:.2f} s ({cv_duration/60:.2f} min)"
        ]
    })
    print(comparison_df.to_string(index=False), flush=True)

    comparison_path = "results/day9_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved head-to-head comparison table to {comparison_path}", flush=True)

    # Step 10: Visualizations
    print("\n--- Step 10: Generating Visualizations ---", flush=True)
    # Plot 1: Hyperparameter ranking
    plt.figure(figsize=(12, 7))
    labels = [
        f"d={row['param_max_depth']}, l={row['param_min_samples_leaf']}, f={row['param_max_features']}"
        for _, row in cv_summary.iterrows()
    ]
    colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(cv_summary))]
    plt.barh(labels[::-1], cv_summary["mean_test_f1"][::-1], color=colors[::-1], xerr=cv_summary["std_test_f1"][::-1], capsize=3)
    plt.xlabel("Mean 5-Fold CV Anomaly F1-Score")
    plt.ylabel("Hyperparameter Candidates (depth, leaf, features)")
    plt.title("Random Forest Hyperparameter Tuning: 5-Fold Stratified CV F1-Score")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot1_path = "results/day9_rf_hyperparameter_search.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved hyperparameter search plot to {plot1_path}", flush=True)

    # Plot 2: Complexity vs Training Cost
    plt.figure(figsize=(10, 6))
    depth_palette = {"10": "#ff7f0e", "20": "#1f77b4", "None": "#2ca02c"}
    for depth in ["10", "20", "None"]:
        sub = cv_summary[cv_summary["param_max_depth"].astype(str) == depth]
        plt.scatter(
            sub["mean_fit_time"],
            sub["mean_test_f1"],
            s=120,
            label=f"max_depth = {depth}",
            color=depth_palette[depth],
            edgecolor="black",
            alpha=0.85
        )
    plt.xlabel("Mean Fit Time per Fold (seconds)")
    plt.ylabel("Mean Anomaly F1-Score (5-Fold CV)")
    plt.title("Computational Cost vs. Anomaly Detection Performance")
    plt.legend(title="Tree Depth Limit")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot2_path = "results/day9_complexity_tradeoff.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved complexity tradeoff plot to {plot2_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Day 9 Experiment Complete! All deliverables saved under results/", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
