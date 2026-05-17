"""
model.py — Logistic regression trained on forward-chained features.

Keeps the model, scaler, encodings, and metrics together.
Call train() whenever data or chain steps change.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
        self.models = {
            "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
        self.scaler = StandardScaler()
        self.encodings: dict[str, dict[str, int]] = {}
        self.feature_cols: list[str] = []
        self.is_trained = False
        self.metrics_by_model: dict[str, dict] = {}

    def _resolve_model_id(self, model_id: str | None) -> str:
        model_id = model_id or "logistic_regression"
        if model_id not in self.models:
            raise ValueError(f"Unknown model '{model_id}'")
        return model_id

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
        self.models["logistic_regression"].fit(X_scaled, y)
        self.models["random_forest"].fit(X, y)
        self.models["gradient_boosting"].fit(X, y)
        self.is_trained = True

        self.metrics_by_model = {}
        for model_id, clf in self.models.items():
            eval_X = X_scaled if model_id == "logistic_regression" else X
            y_pred = clf.predict(eval_X)
            y_prob = clf.predict_proba(eval_X)[:, 1]
            self.metrics_by_model[model_id] = {
                "model_id": model_id,
                "trained": True,
                "samples": int(len(y)),
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
                "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
                "positive_rate": round(float(y.mean()), 4),
                "features": len(self.feature_cols),
            }

        lr_metrics = self.metrics_by_model["logistic_regression"]
        print(f"[model] Trained 3 models | samples={lr_metrics['samples']} | "
              f"LR AUC={lr_metrics['roc_auc']} | features={lr_metrics['features']}")
        return lr_metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, row: dict, steps: list[ChainStep], model_id: str | None = None) -> float:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet")
        model_id = self._resolve_model_id(model_id)

        df = pd.DataFrame([row])
        df_feat = apply_chain(df, steps, self.encodings)

        X = np.zeros((1, len(self.feature_cols)))
        for j, col in enumerate(self.feature_cols):
            if col in df_feat.columns:
                val = pd.to_numeric(df_feat[col].iloc[0], errors="coerce")
                X[0, j] = float(val) if not np.isnan(val) else 0.0

        eval_X = self.scaler.transform(X) if model_id == "logistic_regression" else X
        return float(self.models[model_id].predict_proba(eval_X)[0][1])

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, model_id: str | None = None) -> list[dict]:
        if not self.is_trained:
            return []
        model_id = self._resolve_model_id(model_id)
        clf = self.models[model_id]
        if hasattr(clf, "coef_"):
            weights = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
        else:
            weights = np.zeros(len(self.feature_cols))
        pairs = sorted(zip(self.feature_cols, weights), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(w), 5)} for f, w in pairs]

    def get_metrics(self, model_id: str | None = None) -> dict:
        if not self.is_trained:
            return {"trained": False}
        model_id = self._resolve_model_id(model_id)
        return {"trained": True, **self.metrics_by_model.get(model_id, {})}

    def list_models(self) -> list[dict]:
        labels = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
        }
        return [
            {
                "id": model_id,
                "name": labels[model_id],
                "status": "online" if self.is_trained else "waiting_for_data",
                "supports_prediction": self.is_trained,
                "metrics": self.metrics_by_model.get(model_id, {"trained": False}),
            }
            for model_id in self.models
        ]
