"""
Day 8 — Random Forest Baseline Training and Evaluation
Telecom Network Anomaly Detection Project
Dataset: AliMaatouk/TelecomTS (Hugging Face)
Author: Dasuni Thiwanshika
"""

import os
# Configure cache directories on D drive if needed for local execution
if os.path.exists("D:/"):
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "D:/hf_cache/datasets")
    os.environ.setdefault("TEMP", "D:/temp")
    os.environ.setdefault("TMP", "D:/temp")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

def main():
    print("=" * 65)
    print("Day 8: Random Forest Baseline Training & Evaluation")
    print("Telecom Network Anomaly Detection Prototype (Public TelecomTS)")
    print("=" * 65)

    # Step 2: Load TelecomTS
    print("\n--- Step 2: Loading TelecomTS dataset ---")
    start_time = time.time()
    cache_dir = "D:/hf_cache/datasets" if os.path.exists("D:/") else None
    dataset = load_dataset(
        "AliMaatouk/TelecomTS",
        data_files={"full": "**/chunked.jsonl"},
        cache_dir=cache_dir
    )
    data = dataset["full"]
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    print("Number of samples:", len(data))
    print("Dataset columns:", data.column_names)

    # Step 3: Check the KPI structure
    print("\n--- Step 3: Checking KPI structure ---")
    print("Available KPI names:")
    print(list(data[0]["KPIs"].keys()))
    print("\nExample anomaly label:")
    print(data[0]["labels"]["anomaly_present"])

    # Step 4: Recreate the 64 features
    print("\n--- Step 4: Recreating 64 features ---")
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
            row[f"{kpi}_std"] = pd.Series(values).std()
            row[f"{kpi}_min"] = min(values)
            row[f"{kpi}_max"] = max(values)
        feature_rows.append(row)
        
    X = pd.DataFrame(feature_rows)
    y = pd.Series([
        1 if sample["labels"]["anomaly_present"] == "Yes" else 0
        for sample in data
    ])

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())
    print("\nTarget percentage:")
    print((y.value_counts(normalize=True) * 100).sort_index())

    # Step 5: Prepare stratified train-test split
    print("\n--- Step 5: Preparing stratified train-test split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training anomaly rate:", round(y_train.mean(), 4))
    print("Testing anomaly rate:", round(y_test.mean(), 4))
    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())
    print("\nTesting class counts:")
    print(y_test.value_counts().sort_index())

    # Step 6: Train Random Forest
    print("\n--- Step 6: Training Random Forest (100 trees, random_state=42) ---")
    rf_start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print(f"Random Forest training completed in {time.time() - rf_start:.2f} seconds.")

    # Step 7: Evaluate the model
    print("\n--- Step 7: Evaluating Random Forest model ---")
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    pr_auc = round(average_precision_score(y_test, y_prob), 4)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC (Average Precision):", pr_auc)

    # Step 8: Feature importance
    print("\n--- Step 8: Feature Importance ---")
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))

    # Save outputs to results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/day8_rf_feature_importance.csv"
    feature_importance.to_csv(csv_path, index=False)
    print(f"\nSaved feature importance table to {csv_path}")

    # Generate and save bar plot
    top_features = feature_importance.head(15).sort_values(by="Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Importance"], color="#1f77b4")
    plt.xlabel("Feature Importance (Mean Decrease in Impurity)")
    plt.ylabel("KPI Feature")
    plt.title("Top 15 Random Forest Features for Telecom Anomaly Detection")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_path = "results/day8_rf_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")

if __name__ == "__main__":
    main()
