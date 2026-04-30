"""
PurchaseModel — Logistic Regression predictive model for customer purchase probability.

Uses scikit-learn with feature engineering for:
  - response_time_min  : how long the customer waited for a reply
  - price              : product price in USD
  - patience_level     : customer patience score 0–1
  - segment            : customer spending tier (low / medium / high)
  - time_of_day        : morning / afternoon / evening
  - complexity         : product complexity 1–10

Adds interaction features (response_time * patience, price * complexity) to
capture non-linear relationships that a plain linear model would miss.
"""

import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class PurchaseModel:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.is_trained = False
        self.metrics: dict = {}
        self.feature_names = [
            'response_time_min', 'price', 'patience_level',
            'segment_enc', 'time_of_day_enc', 'complexity',
            # Interaction / engineered features
            'rt_x_impatience',   # response_time * (1 - patience)
            'price_x_complexity', # price * complexity / 10
            'log_rt',            # log(response_time) — diminishing returns
        ]

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and log features to the dataframe."""
        df = df.copy()
        df['rt_x_impatience'] = df['response_time_min'] * (1.0 - df['patience_level'])
        df['price_x_complexity'] = df['price'] * df['complexity'] / 10.0
        df['log_rt'] = np.log1p(df['response_time_min'])
        return df

    def train(self, db_path: str) -> dict:
        """Load data from SQLite, engineer features, and fit the model."""
        conn = sqlite3.connect(db_path)
        query = """
        SELECT
            i.response_time_min,
            p.price,
            c.patience_level,
            c.segment,
            p.complexity,
            i.time_of_day,
            i.bought
        FROM interactions i
        JOIN customers c ON i.customer_id = c.customer_id
        JOIN products p  ON i.product_id  = p.product_id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            raise ValueError("No training data found in database")

        # Encode categoricals
        for col in ['segment', 'time_of_day']:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        df = self._engineer(df)

        X = df[self.feature_names].values
        y = df['bought'].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Basic training metrics
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        self.metrics = {
            'samples': int(len(df)),
            'accuracy': float(round(accuracy_score(y, y_pred), 4)),
            'roc_auc': float(round(roc_auc_score(y, y_prob), 4)),
            'positive_rate': float(round(y.mean(), 4)),
        }
        print(f"[model] Trained on {self.metrics['samples']} samples | "
              f"AUC={self.metrics['roc_auc']} | Acc={self.metrics['accuracy']}")
        return self.metrics

    def predict(self, input_data: dict) -> float:
        """Return the probability of purchase (0–1) for one data point."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        df = pd.DataFrame([input_data])

        # Encode categoricals — handle unseen labels gracefully
        for col in ['segment', 'time_of_day']:
            le = self.label_encoders[col]
            val = df[col].iloc[0]
            if val not in le.classes_:
                # Default to the middle class if unknown
                df[f'{col}_enc'] = 1
            else:
                df[f'{col}_enc'] = le.transform(df[col])

        df = self._engineer(df)
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]
        return float(prob)

    def get_metrics(self) -> dict:
        return self.metrics if self.is_trained else {}
