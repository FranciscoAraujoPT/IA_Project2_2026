"""
model.py — Multi-model training with forward-chained features and K-Fold CV.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
        self.cv_metrics: dict = {}
        self.cv_test_rows: list[dict] = []

    def _resolve_model_id(self, model_id: str | None) -> str:
        model_id = model_id or "logistic_regression"
        if model_id not in self.models:
            raise ValueError(f"Unknown model '{model_id}'")
        return model_id

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, outcome_col: str, steps: list[ChainStep], k: int = 5) -> dict:
        """
        Fit all models on df using the provided chain steps.
        outcome_col must be a binary 0/1 column in df.
        Also computes k-fold cross-validation metrics (LR only) and stores
        row-level test predictions (one prediction per row from its held-out fold).
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

        # Fit all three models
        self.models["logistic_regression"].fit(X_scaled, y)
        self.models["random_forest"].fit(X, y)
        self.models["gradient_boosting"].fit(X, y)
        self.is_trained = True

        # Per-model training metrics
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

        # Source columns used as chain inputs (for K-Fold row storage)
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

        # K-Fold CV (Logistic Regression only — consistent, interpretable baseline)
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

        lr_metrics = self.metrics_by_model["logistic_regression"]
        print(f"[model] Trained 3 models | samples={lr_metrics['samples']} | "
              f"LR AUC={lr_metrics['roc_auc']} | features={lr_metrics['features']}")
        if self.cv_metrics:
            print(f"[model] CV({actual_k}-fold) AUC={self.cv_metrics['roc_auc']['mean']:.4f} "
                  f"± {self.cv_metrics['roc_auc']['std']:.4f} | "
                  f"test rows stored: {len(self.cv_test_rows)}")
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

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self, model_id: str | None = None) -> dict:
        if not self.is_trained:
            return {"trained": False}
        model_id = self._resolve_model_id(model_id)
        return {"trained": True, **self.metrics_by_model.get(model_id, {})}

    def get_cv_metrics(self) -> dict:
        return self.cv_metrics

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
