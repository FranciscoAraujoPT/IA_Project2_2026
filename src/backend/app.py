"""
app.py — Flask API server.

Startup:
  cd src/backend
  python app.py
"""

from __future__ import annotations
import os
import sys
import pandas as pd

# Allow running as   python app.py   (no -m needed)
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictive_model import (
    PurchaseModel, ChainStore, DatasetManager,
    auto_generate_steps,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "db", "data.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── Singletons ────────────────────────────────────────────────────────────────

datasets = DatasetManager(DB_PATH)
chain_store = ChainStore(DB_PATH)
model = PurchaseModel()

app = Flask(__name__)
CORS(app)


# ── Training helper ───────────────────────────────────────────────────────────

def retrain() -> None:
    config = datasets.get_config()
    outcome_col = config.get("outcome_col", "")
    if not outcome_col:
        return
    df = datasets.get_merged_df()
    if df.empty:
        return
    steps = chain_store.list_steps()
    k_folds = int(config.get("k_folds", 5))
    try:
        model.train(df, outcome_col, steps, k=k_folds)
    except Exception as exc:
        print(f"[app] Training failed: {exc}")


# ── Dataset routes ────────────────────────────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    return jsonify(datasets.list_datasets())


@app.route("/api/datasets/upload", methods=["POST"])
def upload_dataset():
    csv_text = request.get_data(as_text=True)
    if not csv_text.strip():
        return jsonify({"error": "Empty CSV body"}), 400

    label = request.headers.get("X-Dataset-Label", "Unnamed Dataset")
    outcome_hint = request.headers.get("X-Outcome-Col", "")
    response_hint = request.headers.get("X-Response-Col", "")
    is_primary = request.headers.get("X-Is-Primary", "false").lower() == "true"

    try:
        result = datasets.upload_dataset(label, csv_text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_id = result["dataset_id"]
    schema = result["schema"]

    # If this is the first/primary dataset, auto-configure and generate chain
    existing_datasets = datasets.list_datasets()
    if is_primary or len(existing_datasets) == 1:
        # Resolve outcome col
        col_names = [c["name"] for c in schema]
        resolved_outcome = outcome_hint or next(
            (c["name"] for c in schema if any(kw in c["name"].lower() for kw in ["buy", "bought", "purchase", "convert", "outcome"])),
            col_names[-1] if col_names else ""
        )
        resolved_response = response_hint or next(
            (c["name"] for c in schema if c["type"] == "numeric" and any(kw in c["name"].lower() for kw in ["time", "wait", "response", "delay"]) and c["name"] != resolved_outcome),
            next((c["name"] for c in schema if c["type"] == "numeric" and c["name"] != resolved_outcome), "")
        )
        existing_k = datasets.get_config().get("k_folds", 5)
        datasets.set_config(resolved_outcome, resolved_response, dataset_id, existing_k)

        # Auto-generate chain steps
        auto_steps = auto_generate_steps(schema, resolved_outcome)
        chain_store.replace_all(auto_steps)

        retrain()

    return jsonify(result)


@app.route("/api/datasets/<int:dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id: int):
    try:
        datasets.delete_dataset(dataset_id)
        retrain()
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<int:dataset_id>/cols", methods=["PUT"])
def update_cols(dataset_id: int):
    body = request.get_json(force=True)
    cols = body.get("cols", [])
    datasets.update_active_cols(dataset_id, cols)
    retrain()
    return jsonify({"success": True})


# ── Join rules ────────────────────────────────────────────────────────────────

@app.route("/api/joins", methods=["GET"])
def list_joins():
    return jsonify(datasets.list_join_rules())


@app.route("/api/joins", methods=["POST"])
def add_join():
    body = request.get_json(force=True)
    try:
        rule_id = datasets.add_join_rule(
            body["left_id"], body["right_id"],
            body["left_col"], body["right_col"],
            body.get("join_type", "left")
        )
        retrain()
        return jsonify({"rule_id": rule_id, "success": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/joins/<int:rule_id>", methods=["DELETE"])
def delete_join(rule_id: int):
    datasets.remove_join_rule(rule_id)
    retrain()
    return jsonify({"success": True})


# ── Config ────────────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(datasets.get_config())


@app.route("/api/config", methods=["POST"])
def set_config():
    body = request.get_json(force=True)
    datasets.set_config(
        body.get("outcome_col", ""),
        body.get("response_col", ""),
        int(body.get("primary_id", 0)),
        int(body.get("k_folds", 5)),
    )
    retrain()
    return jsonify({"success": True})


# ── Chain steps ───────────────────────────────────────────────────────────────

@app.route("/api/chain", methods=["GET"])
def list_chain():
    steps = chain_store.list_steps()
    return jsonify([{
        "step_id": s.step_id, "step_order": s.step_order,
        "name": s.name, "type": s.type, "config": s.config, "enabled": s.enabled,
    } for s in steps])


@app.route("/api/chain", methods=["POST"])
def add_chain_step():
    body = request.get_json(force=True)
    step_id = chain_store.add_step(
        body.get("step_order", 99),
        body["name"], body["type"],
        body.get("config", {}),
        body.get("enabled", True)
    )
    retrain()
    return jsonify({"step_id": step_id, "success": True})


@app.route("/api/chain/<int:step_id>", methods=["PUT"])
def update_chain_step(step_id: int):
    body = request.get_json(force=True)
    chain_store.update_step(
        step_id,
        enabled=body.get("enabled"),
        config=body.get("config"),
        name=body.get("name"),
    )
    retrain()
    return jsonify({"success": True})


@app.route("/api/chain/<int:step_id>", methods=["DELETE"])
def delete_chain_step(step_id: int):
    chain_store.delete_step(step_id)
    retrain()
    return jsonify({"success": True})


# ── Merged columns (after join) ───────────────────────────────────────────────

@app.route("/api/merged-columns", methods=["GET"])
def merged_columns():
    """Return the actual column names of the merged dataframe so the frontend
    can show them in the Chain editor (especially joined columns like purchases_purchase_value)."""
    import json, math
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    try:
        df = datasets.get_merged_df()
        if df.empty:
            return jsonify([])
        from predictive_model.datasets import infer_schema
        cols = sanitize(infer_schema(df))
        return app.response_class(
            json.dumps(cols, allow_nan=False),
            mimetype="application/json"
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Model ─────────────────────────────────────────────────────────────────────

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(model.get_metrics())


@app.route("/api/importance", methods=["GET"])
def get_importance():
    return jsonify(model.feature_importance())


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(datasets.get_stats())


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    if not model.is_trained:
        return jsonify({"error": "Model not trained yet"}), 503
    steps = chain_store.list_steps()
    try:
        prob = model.predict(body, steps)
        return jsonify({"probability": prob, "percentage": round(prob * 100, 1)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict-inputs", methods=["GET"])
def predict_inputs():
    """Return the raw source columns (with stats) that the chain reads as inputs."""
    import json, math

    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    try:
        steps = chain_store.list_steps()
        df = datasets.get_merged_df()
        if df.empty:
            return jsonify([])

        config = datasets.get_config()
        outcome_col = config.get("outcome_col", "")

        # Collect all raw source columns referenced by enabled chain steps
        source_cols = {}
        for step in steps:
            if not step.enabled:
                continue
            cfg = step.config
            cols_to_check = []
            if step.type in ("log", "binary_threshold", "passthrough", "encode"):
                cols_to_check = [cfg.get("col", "")]
            elif step.type in ("interaction", "ratio"):
                cols_to_check = [cfg.get("colA", ""), cfg.get("colB", "")]

            for col in cols_to_check:
                if col and col in df.columns and col != outcome_col and col not in source_cols:
                    series = df[col]
                    if pd.api.types.is_numeric_dtype(series):
                        source_cols[col] = {
                            "name": col,
                            "type": "numeric",
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "mean": float(series.mean()),
                            "uniqueValues": None,
                        }
                    else:
                        unique_vals = series.dropna().astype(str).unique().tolist()
                        source_cols[col] = {
                            "name": col,
                            "type": "categorical",
                            "min": None,
                            "max": None,
                            "mean": None,
                            "uniqueValues": unique_vals,
                        }

        result = sanitize(list(source_cols.values()))
        return app.response_class(
            json.dumps(result, allow_nan=False),
            mimetype="application/json"
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── K-Fold & Testing ──────────────────────────────────────────────────────────

@app.route("/api/kfold-metrics", methods=["GET"])
def get_kfold_metrics():
    return jsonify(model.get_cv_metrics())


@app.route("/api/sample-rows", methods=["GET"])
def sample_rows():
    import json, math

    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    try:
        n = int(request.args.get("n", 20))
        df = datasets.get_merged_df()
        config = datasets.get_config()
        outcome_col = config.get("outcome_col", "")

        if df.empty or not outcome_col or outcome_col not in df.columns:
            return jsonify({"outcome_col": outcome_col, "rows": []})

        valid = df[df[outcome_col].notna()].copy()
        sample = valid.sample(min(n, len(valid)), random_state=None)

        steps = chain_store.list_steps()
        source_cols: set[str] = set()
        for step in steps:
            if not step.enabled:
                continue
            cfg = step.config
            if step.type in ("log", "binary_threshold", "passthrough", "encode"):
                c = cfg.get("col", "")
                if c:
                    source_cols.add(c)
            elif step.type in ("interaction", "ratio"):
                for key in ("colA", "colB"):
                    c = cfg.get(key, "")
                    if c:
                        source_cols.add(c)

        cols_to_return = [c for c in source_cols if c in sample.columns and c != outcome_col]
        cols_to_return.append(outcome_col)

        rows = [sanitize(row.to_dict()) for _, row in sample[cols_to_return].iterrows()]
        return app.response_class(
            json.dumps({"outcome_col": outcome_col, "rows": rows}, allow_nan=False),
            mimetype="application/json",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/kfold-sample", methods=["GET"])
def kfold_sample():
    """Return a percentage sample of K-Fold test rows (pre-computed during training)."""
    import json, math, random as rand_mod

    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    if not model.is_trained:
        return jsonify({"error": "Model not trained yet"}), 503

    rows = model.cv_test_rows
    if not rows:
        return jsonify({"outcome_col": "", "rows": [], "total": 0, "k": 0})

    pct = max(0.01, min(1.0, float(request.args.get("pct", 0.30))))
    threshold = float(request.args.get("threshold", 0.5))

    n = max(1, round(len(rows) * pct))
    sample = rand_mod.sample(rows, min(n, len(rows)))

    # Apply current threshold to recompute predicted/correct
    result_rows = []
    for row in sample:
        r = dict(row)
        r["_predicted"] = int(r["_prob"] >= threshold)
        r["_correct"] = r["_real"] == r["_predicted"]
        result_rows.append(sanitize(r))

    result_rows.sort(key=lambda r: r.get("_fold", 0))

    config = datasets.get_config()
    outcome_col = config.get("outcome_col", "")
    k_val = model.cv_metrics.get("k", 0)

    return app.response_class(
        json.dumps({
            "outcome_col": outcome_col,
            "rows": result_rows,
            "total": len(rows),
            "k": k_val,
        }, allow_nan=False),
        mimetype="application/json",
    )


@app.route("/api/predict-compare", methods=["POST"])
def predict_compare():
    body = request.get_json(force=True)
    if not model.is_trained:
        return jsonify({"error": "Model not trained yet"}), 503

    row = body.get("row", {})
    real_outcome = body.get("real_outcome")
    threshold = float(body.get("threshold", 0.5))

    steps = chain_store.list_steps()
    try:
        prob = model.predict(row, steps)
        predicted = int(prob >= threshold)
        real = int(real_outcome) if real_outcome is not None else None
        correct = (predicted == real) if real is not None else None
        return jsonify({
            "probability": prob,
            "percentage": round(prob * 100, 1),
            "predicted": predicted,
            "real": real,
            "correct": correct,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[app] DB: {os.path.abspath(DB_PATH)}")
    retrain()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)