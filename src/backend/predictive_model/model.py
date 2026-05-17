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
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

try:
    from .chain import ChainStep, apply_chain
except ImportError:
    from chain import ChainStep, apply_chain


def _to_py(val):
    """Convert numpy scalars to plain Python for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if (np.isnan(val) or np.isinf(val)) else float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


class PurchaseModel:
    def __init__(self) -> None:
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.encodings: dict[str, dict[str, int]] = {}
        self.feature_cols: list[str] = []
        self.is_trained = False
        self.metrics: dict = {}
        self.cv_metrics: dict = {}
        self.cv_test_rows: list[dict] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, outcome_col: str, steps: list[ChainStep], k: int = 5) -> dict:
        """
        Fit the model on df using the provided chain steps.
        outcome_col must be a binary 0/1 column in df.
        Also computes k-fold cross-validation metrics and stores row-level
        test predictions (one prediction per row from its held-out fold).
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

        # Training metrics (on full data — for display only)
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

        # Source columns used as chain inputs (for test row storage)
        source_cols_set: set[str] = set()
        for step in steps:
            if not step.enabled:
                continue
            cfg = step.config
            if step.type in ("log", "binary_threshold", "passthrough", "encode"):
                c = cfg.get("col", "")
                if c and c in df.columns and c != outcome_col:
                    source_cols_set.add(c)
            elif step.type in ("interaction", "ratio"):
                for key in ("colA", "colB"):
                    c = cfg.get(key, "")
                    if c and c in df.columns and c != outcome_col:
                        source_cols_set.add(c)
        source_cols_list = sorted(source_cols_set)
        orig_df = (
            df.loc[valid_mask][source_cols_list].reset_index(drop=True)
            if source_cols_list else pd.DataFrame()
        )

        # K-Fold: manual loop so we get both aggregate metrics AND row-level predictions
        min_class_count = int(np.bincount(y).min())
        actual_k = min(k, min_class_count)
        self.cv_metrics = {}
        self.cv_test_rows = []

        if actual_k >= 2:
            kf = StratifiedKFold(n_splits=actual_k, shuffle=True, random_state=42)
            fold_accs: list[float] = []
            fold_aucs: list[float] = []
            fold_f1s:  list[float] = []
            fold_precs: list[float] = []
            fold_recs:  list[float] = []
            cv_test_rows: list[dict] = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
                fold_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                fold_clf.fit(X_scaled[train_idx], y[train_idx])

                y_test = y[test_idx]
                probs = fold_clf.predict_proba(X_scaled[test_idx])[:, 1]
                preds = (probs >= 0.5).astype(int)

                fold_accs.append(float(accuracy_score(y_test, preds)))
                fold_aucs.append(float(roc_auc_score(y_test, probs)))
                fold_f1s.append(float(f1_score(y_test, preds, zero_division=0)))
                fold_precs.append(float(precision_score(y_test, preds, zero_division=0)))
                fold_recs.append(float(recall_score(y_test, preds, zero_division=0)))

                for i, orig_idx in enumerate(test_idx):
                    prob = float(probs[i])
                    real = int(y[orig_idx])
                    row: dict = {}
                    if not orig_df.empty:
                        for col in source_cols_list:
                            row[col] = _to_py(orig_df.iloc[orig_idx][col])
                    row["_fold"] = fold_idx + 1
                    row["_real"] = real
                    row["_prob"] = round(prob, 4)
                    row["_predicted"] = int(prob >= 0.5)
                    row["_correct"] = real == int(prob >= 0.5)
                    cv_test_rows.append(row)

            def _ms(vals: list[float]) -> dict:
                a = np.array(vals)
                return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}

            self.cv_metrics = {
                "k": actual_k,
                "accuracy":  _ms(fold_accs),
                "roc_auc":   _ms(fold_aucs),
                "f1":        _ms(fold_f1s),
                "precision": _ms(fold_precs),
                "recall":    _ms(fold_recs),
                "fold_aucs": [round(v, 4) for v in fold_aucs],
            }
            self.cv_test_rows = cv_test_rows

        print(f"[model] Trained | samples={self.metrics['samples']} | "
              f"AUC={self.metrics['roc_auc']} | acc={self.metrics['accuracy']} | "
              f"features={self.metrics['features']}")
        if self.cv_metrics:
            print(f"[model] CV({actual_k}-fold) AUC={self.cv_metrics['roc_auc']['mean']:.4f} "
                  f"± {self.cv_metrics['roc_auc']['std']:.4f} | "
                  f"test rows stored: {len(self.cv_test_rows)}")
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

    def get_cv_metrics(self) -> dict:
        return self.cv_metrics
