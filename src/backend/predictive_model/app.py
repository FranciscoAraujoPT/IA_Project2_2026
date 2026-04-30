"""
Flask backend for Customer Response Time Analyzer.

Routes:
  GET  /stats               — purchase rate bucketed by response time
  POST /predict             — purchase probability prediction
  GET  /metrics             — model training metrics
  GET  /dataset             — list all interactions (joined)
  POST /dataset             — insert a new interaction
  POST /dataset/bulk        — bulk insert rows (for CSV import)
  DELETE /dataset/<id>      — delete an interaction

Run with:
  python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
from backend.predictive_model.model import PurchaseModel
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin if frontend served separately

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'interactions.db')

# ── Model singleton ────────────────────────────────────────────────────────────
purchase_model = PurchaseModel()


def get_conn():
    return sqlite3.connect(DB_PATH)


def train_model():
    """Train (or retrain) the model from the current database."""
    if os.path.exists(DB_PATH):
        try:
            purchase_model.train(DB_PATH)
        except Exception as e:
            print(f"[model] Training failed: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_conn()
        query = """
        SELECT
            CAST(i.response_time_min / 10 AS INT) * 10 AS bucket_start,
            AVG(i.bought)  AS purchase_rate,
            COUNT(*)       AS total
        FROM interactions i
        GROUP BY bucket_start
        ORDER BY bucket_start
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        stats = []
        for _, row in df.iterrows():
            stats.append({
                "range": f"{int(row['bucket_start'])}-{int(row['bucket_start']) + 10}",
                "purchaseRate": float(row['purchase_rate']),
                "total": int(row['total']),
            })
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        prob = purchase_model.predict(data)
        return jsonify({"purchase_probability": prob})
    except RuntimeError as e:
        return jsonify({"error": str(e), "hint": "Model not trained — add data first"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(purchase_model.get_metrics())


@app.route('/dataset', methods=['GET'])
def get_dataset():
    try:
        conn = get_conn()
        query = """
        SELECT
            i.interaction_id,
            i.response_time_min,
            p.price,
            c.patience_level,
            c.segment,
            i.time_of_day,
            p.complexity,
            i.bought,
            i.customer_id,
            i.product_id
        FROM interactions i
        JOIN customers c ON i.customer_id = c.customer_id
        JOIN products p  ON i.product_id  = p.product_id
        ORDER BY i.interaction_id DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/dataset', methods=['POST'])
def add_row():
    data = request.get_json(force=True)
    required = ['response_time_min', 'price', 'patience_level', 'segment', 'time_of_day', 'complexity', 'bought']
    missing = [f for f in required if data.get(f) is None and f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        conn = get_conn()
        cur = conn.cursor()

        cust_id = int(data.get('customer_id', 1))
        prod_id = int(data.get('product_id', 1))

        # Ensure customer exists
        cur.execute('INSERT OR IGNORE INTO customers (customer_id, segment, avg_spend, patience_level) VALUES (?, ?, ?, ?)',
                    (cust_id, data['segment'], 100, float(data['patience_level'])))

        # Ensure product exists
        cur.execute('INSERT OR IGNORE INTO products (product_id, category, price, complexity) VALUES (?, ?, ?, ?)',
                    (prod_id, 'General', float(data['price']), int(data['complexity'])))

        cur.execute("""
            INSERT INTO interactions (customer_id, product_id, question_time, response_time_min, question_length, time_of_day, bought)
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
        """, (cust_id, prod_id, float(data['response_time_min']), 50, data['time_of_day'], int(bool(data['bought']))))

        conn.commit()
        new_id = cur.lastrowid
        conn.close()

        train_model()  # Retrain with new data
        return jsonify({"interaction_id": new_id, "success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/dataset/bulk', methods=['POST'])
def bulk_insert():
    body = request.get_json(force=True)
    rows = body.get('rows', [])
    if not rows:
        return jsonify({"error": "No rows provided"}), 400

    try:
        conn = get_conn()
        cur = conn.cursor()
        inserted = 0

        for row in rows:
            cust_id = int(row.get('customer_id', 1))
            prod_id = int(row.get('product_id', 1))
            cur.execute('INSERT OR IGNORE INTO customers (customer_id, segment, avg_spend, patience_level) VALUES (?, ?, ?, ?)',
                        (cust_id, row.get('segment', 'medium'), 100, float(row.get('patience_level', 0.5))))
            cur.execute('INSERT OR IGNORE INTO products (product_id, category, price, complexity) VALUES (?, ?, ?, ?)',
                        (prod_id, 'Imported', float(row.get('price', 100)), int(row.get('complexity', 5))))
            cur.execute("""
                INSERT INTO interactions (customer_id, product_id, question_time, response_time_min, question_length, time_of_day, bought)
                VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
            """, (cust_id, prod_id, float(row.get('response_time_min', 15)), 50,
                  row.get('time_of_day', 'afternoon'), int(bool(row.get('bought', 0)))))
            inserted += 1

        conn.commit()
        conn.close()

        train_model()  # Retrain after bulk import
        return jsonify({"inserted": inserted, "success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/dataset/<int:interaction_id>', methods=['DELETE'])
def delete_row(interaction_id):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('DELETE FROM interactions WHERE interaction_id = ?', (interaction_id,))
        conn.commit()
        deleted = cur.rowcount
        conn.close()

        if deleted == 0:
            return jsonify({"error": "Row not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"[app] Using database: {os.path.abspath(DB_PATH)}")
    train_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
