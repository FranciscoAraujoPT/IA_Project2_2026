"""
model.py — Logistic regression trained on forward-chained features.

Keeps the model, scaler, encodings, and metrics together.
Call train() whenever data or chain steps change.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

try:
    from .chain import ChainStep, apply_chain
except ImportError:
    from chain import ChainStep, apply_chain


class PurchaseModel:
    def __init__(self) -> None:
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.encodings: dict[str, dict[str, int]] = {}
        self.feature_cols: list[str] = []
        self.is_trained = False
        self.metrics: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, outcome_col: str, steps: list[ChainStep]) -> dict:
        """
        Fit the model on df using the provided chain steps.
        outcome_col must be a binary 0/1 column in df.
        """
        if df.empty or outcome_col not in df.columns:
            raise ValueError(f"DataFrame is empty or missing outcome column '{outcome_col}'")

        # Build encodings for categorical encode steps
        self.encodings = {}
        for step in steps:
            if step.enabled and step.type == "encode":
                col = step.config["col"]
                if col in df.columns:
                    unique_vals = df[col].astype(str).unique().tolist()
                    self.encodings[col] = {v: i for i, v in enumerate(unique_vals)}

        # Apply forward chain
        df_feat = apply_chain(df, steps, self.encodings)

        # Feature columns = all numeric except outcome
        outcome_vals = pd.to_numeric(df[outcome_col], errors="coerce")
        valid_mask = outcome_vals.notna()

        # Collect feature cols from the chain output (exclude outcome and non-numeric)
        candidate_cols = [c for c in df_feat.columns if c != outcome_col]
        numeric_candidates = []
        for col in candidate_cols:
            series = pd.to_numeric(df_feat[col], errors="coerce")
            if series.notna().sum() > len(df_feat) * 0.5:
                numeric_candidates.append(col)
                df_feat[col] = series

        self.feature_cols = numeric_candidates
        if not self.feature_cols:
            raise ValueError("No usable numeric features found after chain application")

        X = df_feat.loc[valid_mask, self.feature_cols].fillna(0).values
        y = (outcome_vals[valid_mask] > 0.5).astype(int).values

        if len(X) < 10:
            raise ValueError(f"Not enough valid training samples ({len(X)} < 10)")

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_trained = True

        # Metrics
        y_pred = self.clf.predict(X_scaled)
        y_prob = self.clf.predict_proba(X_scaled)[:, 1]
        self.metrics = {
            "samples": int(len(y)),
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "positive_rate": round(float(y.mean()), 4),
            "features": len(self.feature_cols),
        }
        print(f"[model] Trained | samples={self.metrics['samples']} | "
              f"AUC={self.metrics['roc_auc']} | acc={self.metrics['accuracy']} | "
              f"features={self.metrics['features']}")
        return self.metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, row: dict, steps: list[ChainStep]) -> float:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet")

        df = pd.DataFrame([row])
        df_feat = apply_chain(df, steps, self.encodings)

        X = np.zeros((1, len(self.feature_cols)))
        for j, col in enumerate(self.feature_cols):
            if col in df_feat.columns:
                val = pd.to_numeric(df_feat[col].iloc[0], errors="coerce")
                X[0, j] = float(val) if not np.isnan(val) else 0.0

        X_scaled = self.scaler.transform(X)
        return float(self.clf.predict_proba(X_scaled)[0][1])

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> list[dict]:
        if not self.is_trained:
            return []
        coefs = np.abs(self.clf.coef_[0])
        pairs = sorted(zip(self.feature_cols, coefs), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(w), 5)} for f, w in pairs]

    def get_metrics(self) -> dict:
        return {"trained": self.is_trained, **self.metrics}
