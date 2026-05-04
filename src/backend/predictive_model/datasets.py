"""
datasets.py — Manages an arbitrary number of uploaded CSV datasets.

Supports:
  - Union (UNION ALL): datasets with the same columns are stacked
  - Join: datasets linked by user-defined key column pairs
  - Mixed: user controls which columns to keep from each dataset

The merged result is what the model trains on.
"""

from __future__ import annotations
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional


# ── Schema inference ──────────────────────────────────────────────────────────

def infer_schema(df: pd.DataFrame) -> list[dict]:
    schema = []
    for col in df.columns:
        series = df[col]
        # Use pandas dtype as primary signal — handles NaN-heavy join columns correctly
        is_numeric_dtype = pd.api.types.is_numeric_dtype(series)
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = numeric.notna().sum()
        non_empty = (series.astype(str).str.strip() != "").sum()
        frac_numeric = non_null / max(non_empty, 1)
        if is_numeric_dtype or frac_numeric > 0.8:
            def _safe(v):
                try:
                    f = float(v)
                    return round(f, 4) if (f == f) else None  # f == f is False for NaN
                except Exception:
                    return None
            schema.append({
                "name": col, "type": "numeric",
                "min": _safe(numeric.min()) if non_null > 0 else None,
                "max": _safe(numeric.max()) if non_null > 0 else None,
                "uniqueValues": None,
            })
        else:
            unique_vals = series.astype(str).unique().tolist()
            schema.append({
                "name": col, "type": "categorical",
                "min": None, "max": None,
                "uniqueValues": unique_vals[:50],
            })
    return schema


def parse_csv(text: str) -> pd.DataFrame:
    from io import StringIO
    lines = text.strip().splitlines()
    if len(lines) < 2:
        raise ValueError("CSV must have at least a header and one data row")
    return pd.read_csv(StringIO(text))


# ── SQLite-backed dataset registry ───────────────────────────────────────────

class DatasetManager:
    """
    Stores each uploaded CSV in its own SQLite table (dataset_rows_<id>).
    Tracks metadata in dataset_registry.
    Applies union/join rules to produce a single merged DataFrame for training.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS dataset_registry (
                    dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    label        TEXT    NOT NULL,
                    schema_json  TEXT    NOT NULL DEFAULT '[]',
                    row_count    INTEGER NOT NULL DEFAULT 0,
                    active_cols  TEXT    NOT NULL DEFAULT '[]',
                    uploaded_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS join_rules (
                    rule_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    left_id      INTEGER NOT NULL,
                    right_id     INTEGER NOT NULL,
                    left_col     TEXT    NOT NULL,
                    right_col    TEXT    NOT NULL,
                    join_type    TEXT    NOT NULL DEFAULT 'left'
                );

                CREATE TABLE IF NOT EXISTS project_config (
                    id           INTEGER PRIMARY KEY CHECK (id = 1),
                    outcome_col  TEXT    NOT NULL DEFAULT '',
                    response_col TEXT    NOT NULL DEFAULT '',
                    primary_id   INTEGER NOT NULL DEFAULT 0
                );
            """)

    # ── Dataset CRUD ──────────────────────────────────────────────────────────

    def upload_dataset(self, label: str, csv_text: str) -> dict:
        df = parse_csv(csv_text)
        schema = infer_schema(df)
        now = pd.Timestamp.utcnow().isoformat()

        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO dataset_registry (label, schema_json, row_count, active_cols, uploaded_at) VALUES (?,?,?,?,?)",
                (label, json.dumps(schema), len(df), json.dumps(df.columns.tolist()), now)
            )
            dataset_id = cur.lastrowid

            # Store rows in a per-dataset table
            table = f"dataset_rows_{dataset_id}"
            df.to_sql(table, conn, if_exists="replace", index=False)

        return {"dataset_id": dataset_id, "label": label, "row_count": len(df), "schema": schema}

    def delete_dataset(self, dataset_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM dataset_registry WHERE dataset_id=?", (dataset_id,))
            conn.execute("DELETE FROM join_rules WHERE left_id=? OR right_id=?", (dataset_id, dataset_id))
            conn.execute(f"DROP TABLE IF EXISTS dataset_rows_{dataset_id}")

    def list_datasets(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT dataset_id, label, schema_json, row_count, active_cols, uploaded_at FROM dataset_registry ORDER BY dataset_id"
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "dataset_id": r[0], "label": r[1],
                "schema": json.loads(r[2]),
                "row_count": r[3],
                "active_cols": json.loads(r[4]),
                "uploaded_at": r[5],
            })
        return result

    def update_active_cols(self, dataset_id: int, cols: list[str]) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE dataset_registry SET active_cols=? WHERE dataset_id=?",
                         (json.dumps(cols), dataset_id))

    # ── Join rules ────────────────────────────────────────────────────────────

    def add_join_rule(self, left_id: int, right_id: int,
                      left_col: str, right_col: str, join_type: str = "left") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO join_rules (left_id, right_id, left_col, right_col, join_type) VALUES (?,?,?,?,?)",
                (left_id, right_id, left_col, right_col, join_type)
            )
            return cur.lastrowid

    def remove_join_rule(self, rule_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM join_rules WHERE rule_id=?", (rule_id,))

    def list_join_rules(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT rule_id, left_id, right_id, left_col, right_col, join_type FROM join_rules ORDER BY rule_id"
            ).fetchall()
        return [{"rule_id": r[0], "left_id": r[1], "right_id": r[2],
                 "left_col": r[3], "right_col": r[4], "join_type": r[5]} for r in rows]

    # ── Project config ────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM project_config WHERE id=1").fetchone()
        if not row:
            return {"outcome_col": "", "response_col": "", "primary_id": 0}
        return {"outcome_col": row[1], "response_col": row[2], "primary_id": row[3]}

    def set_config(self, outcome_col: str, response_col: str, primary_id: int) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO project_config (id, outcome_col, response_col, primary_id)
                VALUES (1, ?, ?, ?)
            """, (outcome_col, response_col, primary_id))

    # ── Merge datasets into one DataFrame ────────────────────────────────────

    def get_merged_df(self) -> pd.DataFrame:
        datasets = self.list_datasets()
        if not datasets:
            return pd.DataFrame()

        config = self.get_config()
        rules = self.list_join_rules()

        with self._conn() as conn:
            # Load each dataset respecting active_cols
            frames: dict[int, pd.DataFrame] = {}
            for ds in datasets:
                table = f"dataset_rows_{ds['dataset_id']}"
                df = pd.read_sql(f"SELECT * FROM \"{table}\"", conn)
                active = ds["active_cols"]
                # Keep only active cols that exist
                keep = [c for c in active if c in df.columns]
                frames[ds["dataset_id"]] = df[keep] if keep else df

        if not rules:
            # No join rules — try union if schemas are compatible, else just use primary
            primary_id = config.get("primary_id") or datasets[0]["dataset_id"]
            primary = frames.get(primary_id, pd.DataFrame())

            # Union datasets that share the same columns as primary
            primary_cols = set(primary.columns)
            to_union = [primary]
            for ds_id, df in frames.items():
                if ds_id == primary_id:
                    continue
                if set(df.columns) == primary_cols:
                    to_union.append(df)

            return pd.concat(to_union, ignore_index=True) if to_union else primary

        # Apply join rules in order (graph traversal)
        # Build adjacency: which datasets have been merged
        primary_id = config.get("primary_id") or datasets[0]["dataset_id"]
        result = frames.get(primary_id, pd.DataFrame()).copy()
        merged_ids = {primary_id}

        # Build label lookup for readable prefixes
        label_map = {ds["dataset_id"]: ds["label"] for ds in datasets}

        def _safe_prefix(label: str) -> str:
            """Use the last meaningful word of the label as prefix (e.g. 'purchases' from any filename)."""
            import re
            # Split on non-alphanumeric, take the last non-empty token
            parts = [p for p in re.split(r"[^a-zA-Z0-9]+", label) if p]
            # Skip generic suffixes like 'csv', 'train', 'test', 'data'
            skip = {"csv", "train", "test", "data", "file", "set", "dataset"}
            meaningful = [p.lower() for p in reversed(parts) if p.lower() not in skip]
            return meaningful[0] if meaningful else "ds"

        # Keep applying rules until no more can be applied
        remaining_rules = list(rules)
        max_passes = len(remaining_rules) + 1
        for _ in range(max_passes):
            if not remaining_rules:
                break
            applied_any = False
            still_remaining = []
            for rule in remaining_rules:
                l_id, r_id = rule["left_id"], rule["right_id"]
                l_col, r_col = rule["left_col"], rule["right_col"]
                join_type = rule["join_type"]

                if l_id in merged_ids and r_id in frames:
                    right_df = frames[r_id].copy()
                    prefix = _safe_prefix(label_map.get(r_id, f"ds{r_id}"))
                    # Prefix ALL right columns (including the join key after merge)
                    rename = {c: f"{prefix}_{c}" for c in right_df.columns if c != r_col}
                    right_df = right_df.rename(columns=rename)
                    result = result.merge(
                        right_df, left_on=l_col, right_on=r_col,
                        how=join_type, suffixes=("", f"_{prefix}")
                    )
                    # Drop the (now redundant) right join-key column
                    if r_col in result.columns and r_col != l_col:
                        result = result.drop(columns=[r_col], errors="ignore")
                    merged_ids.add(r_id)
                    applied_any = True
                elif r_id in merged_ids and l_id in frames:
                    left_df = frames[l_id].copy()
                    prefix = _safe_prefix(label_map.get(l_id, f"ds{l_id}"))
                    rename = {c: f"{prefix}_{c}" for c in left_df.columns if c != l_col}
                    left_df = left_df.rename(columns=rename)
                    result = result.merge(
                        left_df, left_on=r_col, right_on=l_col,
                        how=join_type, suffixes=("", f"_{prefix}")
                    )
                    if l_col in result.columns and l_col != r_col:
                        result = result.drop(columns=[l_col], errors="ignore")
                    merged_ids.add(l_id)
                    applied_any = True
                else:
                    still_remaining.append(rule)

            remaining_rules = still_remaining
            if not applied_any:
                break

        return result

    def get_stats(self) -> list[dict]:
        """Bucket the response_col by outcome for charting."""
        config = self.get_config()
        x_col = config.get("response_col", "")
        y_col = config.get("outcome_col", "")
        if not x_col or not y_col:
            return []

        df = self.get_merged_df()
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return []

        x_num = pd.to_numeric(df[x_col], errors="coerce")
        y_num = pd.to_numeric(df[y_col], errors="coerce")
        valid = x_num.notna() & y_num.notna()
        x_num, y_num = x_num[valid], y_num[valid]

        if x_num.dtype.kind in "fiu":
            buckets = 10
            x_min, x_max = x_num.min(), x_num.max()
            step = (x_max - x_min) / buckets if x_max > x_min else 1
            groups = []
            for i in range(buckets):
                lo = x_min + i * step
                hi = lo + step
                mask = (x_num >= lo) & (x_num < hi)
                total = int(mask.sum())
                pos = int((y_num[mask] > 0.5).sum())
                groups.append({
                    "range": f"{lo:.1f}–{hi:.1f}",
                    "purchaseRate": pos / total if total else 0,
                    "total": total,
                })
            return groups

        # Categorical
        groups: dict[str, dict] = {}
        for x_val, y_val in zip(df[x_col].astype(str), y_num):
            if x_val not in groups:
                groups[x_val] = {"total": 0, "pos": 0}
            groups[x_val]["total"] += 1
            if y_val > 0.5:
                groups[x_val]["pos"] += 1

        return [{"range": k, "purchaseRate": v["pos"] / v["total"] if v["total"] else 0,
                 "total": v["total"]} for k, v in groups.items()]