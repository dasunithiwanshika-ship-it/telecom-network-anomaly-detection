"""
Inference and batch prediction pipeline for telecom network anomaly detection.
Provides robust CSV validation, trained model inference, and output formatting.
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from .data_loader import FEATURE_COLUMNS
from .preprocessing import validate_feature_schema, clean_feature_data


class AnomalyDetectorPipeline:
    """
    Unified inference pipeline supporting Logistic Regression, Random Forest,
    Isolation Forest, and Local Outlier Factor.
    """

    AVAILABLE_MODELS = [
        "random_forest",
        "logistic_regression",
        "isolation_forest",
        "local_outlier_factor",
    ]

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[Any] = None
        self.baseline_stats: Optional[Dict[str, Any]] = None
        self._load_all_available_models()

    def _load_all_available_models(self) -> None:
        """Attempt to load saved model artifacts from disk."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return

        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Warning: Failed to load scaler from {scaler_path}: {e}")

        stats_path = os.path.join(self.models_dir, "normal_baseline_stats.joblib")
        if os.path.exists(stats_path):
            try:
                self.baseline_stats = joblib.load(stats_path)
            except Exception as e:
                print(f"Warning: Failed to load baseline stats from {stats_path}: {e}")

        for model_id in self.AVAILABLE_MODELS:
            path = os.path.join(self.models_dir, f"{model_id}.joblib")
            if os.path.exists(path):
                try:
                    self.models[model_id] = joblib.load(path)
                except Exception as e:
                    print(f"Warning: Failed to load model '{model_id}' from {path}: {e}")

    def is_model_available(self, model_id: str) -> bool:
        return model_id in self.models

    def validate_input_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """
        Validate incoming dataframe columns and data integrity.
        """
        if not isinstance(df, pd.DataFrame):
            return False, "Input data is not a pandas DataFrame.", df

        if df.empty:
            return False, "Input DataFrame is completely empty.", df

        is_valid, missing = validate_feature_schema(df)
        if not is_valid:
            sample_missing = missing[:5]
            suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            msg = (
                f"Missing {len(missing)} required KPI feature columns: "
                f"{', '.join(sample_missing)}{suffix}. "
                "Ensure your input contains the required 64 engineered statistical moments."
            )
            return False, msg, df

        try:
            cleaned = clean_feature_data(df[FEATURE_COLUMNS])
        except Exception as e:
            return False, f"Error cleaning input feature data: {str(e)}", df

        return True, "Validation successful.", cleaned

    def predict(
        self,
        df: pd.DataFrame,
        model_id: str = "random_forest",
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Execute anomaly prediction on an input DataFrame of 64 features.

        Args:
            df: DataFrame containing the 64 engineered KPI features.
            model_id: Identifier of model ('random_forest', 'logistic_regression',
                      'isolation_forest', 'local_outlier_factor').
            threshold: Classification threshold for supervised class probabilities (default: 0.50).

        Returns:
            DataFrame containing original columns plus 'anomaly_prediction',
            'anomaly_status', and 'anomaly_score_or_prob'.
        """
        is_valid, msg, cleaned_df = self.validate_input_dataframe(df)
        if not is_valid:
            raise ValueError(msg)

        if model_id not in self.models:
            raise KeyError(
                f"Model '{model_id}' is not loaded. "
                f"Available models: {list(self.models.keys())}"
            )

        model = self.models[model_id]
        X_df = cleaned_df[FEATURE_COLUMNS]

        result_df = df.copy()

        if model_id == "logistic_regression":
            if self.scaler is None:
                raise RuntimeError("StandardScaler artifact is required for Logistic Regression inference.")
            X_scaled = self.scaler.transform(X_df)
            probs = model.predict_proba(X_scaled)[:, 1]
            preds = (probs >= threshold).astype(int)
            result_df["anomaly_score_or_prob"] = np.round(probs, 4)
            result_df["score_type"] = "Supervised Anomaly Probability (P)"

        elif model_id == "random_forest":
            probs = model.predict_proba(X_df)[:, 1]
            preds = (probs >= threshold).astype(int)
            result_df["anomaly_score_or_prob"] = np.round(probs, 4)
            result_df["score_type"] = "Supervised Anomaly Probability (P)"

        elif model_id == "isolation_forest":
            raw_scores = model.decision_function(X_df)
            # In scikit-learn, decision_function < 0 indicates anomaly.
            # Invert for ranking: higher value = more anomalous
            inv_scores = -raw_scores
            # Normalize scores to [0, 1] range for intuitive dashboard display
            s_min, s_max = inv_scores.min(), inv_scores.max()
            norm_scores = (inv_scores - s_min) / (s_max - s_min + 1e-9)
            raw_preds = model.predict(X_df)
            preds = np.where(raw_preds == -1, 1, 0)
            result_df["anomaly_score_or_prob"] = np.round(norm_scores, 4)
            result_df["score_type"] = "Normalized Isolation Anomaly Score"

        elif model_id == "local_outlier_factor":
            if self.scaler is None:
                raise RuntimeError("StandardScaler artifact is required for Local Outlier Factor inference.")
            X_scaled = self.scaler.transform(X_df)
            raw_scores = model.decision_function(X_scaled)
            inv_scores = -raw_scores
            s_min, s_max = inv_scores.min(), inv_scores.max()
            norm_scores = (inv_scores - s_min) / (s_max - s_min + 1e-9)
            raw_preds = model.predict(X_scaled)
            preds = np.where(raw_preds == -1, 1, 0)
            result_df["anomaly_score_or_prob"] = np.round(norm_scores, 4)
            result_df["score_type"] = "Normalized LOF Outlier Score"

        else:
            raise NotImplementedError(f"Model ID '{model_id}' is not implemented.")

        result_df["anomaly_prediction"] = preds
        result_df["anomaly_status"] = np.where(preds == 1, "Anomaly", "Normal")

        return result_df

    def explain_record(
        self, record: Union[pd.Series, Dict[str, Any]], top_k: int = 6
    ) -> pd.DataFrame:
        """
        Explain why a specific observation deviates from normal network behavior.
        Computes empirical z-score deviations relative to the normal cellular population.

        Args:
            record: Series or dict containing values for the 64 KPI features.
            top_k: Number of highest deviating features to return.

        Returns:
            Ranked DataFrame of top deviating features with statistical comparisons.
        """
        if self.baseline_stats is None:
            raise RuntimeError("Baseline normal statistics artifact not loaded.")

        if isinstance(record, dict):
            s = pd.Series(record)
        else:
            s = record.copy()

        norm_mean = self.baseline_stats["mean"]
        norm_std = self.baseline_stats["std"]
        norm_median = self.baseline_stats["median"]
        norm_q05 = self.baseline_stats["q05"]
        norm_q95 = self.baseline_stats["q95"]

        rows = []
        for col in FEATURE_COLUMNS:
            if col in s.index:
                val = float(s[col])
                m = float(norm_mean[col])
                sd = float(norm_std[col]) if float(norm_std[col]) > 1e-9 else 1.0
                z = (val - m) / sd
                rows.append({
                    "Feature": col,
                    "Observed Value": round(val, 4),
                    "Normal Mean": round(m, 4),
                    "Normal Median": round(float(norm_median[col]), 4),
                    "Normal 5th-95th Range": f"[{float(norm_q05[col]):.2f}, {float(norm_q95[col]):.2f}]",
                    "Z-Score (Dev.)": round(z, 2),
                    "Abs Z-Score": round(abs(z), 2),
                    "Status": "Severely Elevated" if z > 3.0 else ("Elevated" if z > 1.5 else ("Severely Depressed" if z < -3.0 else ("Depressed" if z < -1.5 else "Normal Range"))),
                })

        df_exp = pd.DataFrame(rows).sort_values(by="Abs Z-Score", ascending=False)
        return df_exp.head(top_k)
