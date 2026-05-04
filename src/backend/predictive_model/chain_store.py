"""
chain_store.py — Persists forward chain steps in SQLite.
"""

from __future__ import annotations
import json
import sqlite3
from .chain import ChainStep


class ChainStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chain_steps (
                    step_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    step_order  INTEGER NOT NULL,
                    name        TEXT    NOT NULL,
                    type        TEXT    NOT NULL,
                    config_json TEXT    NOT NULL DEFAULT '{}',
                    enabled     INTEGER NOT NULL DEFAULT 1
                )
            """)

    def list_steps(self) -> list[ChainStep]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT step_id, step_order, name, type, config_json, enabled FROM chain_steps ORDER BY step_order"
            ).fetchall()
        return [ChainStep(
            step_id=r[0], step_order=r[1], name=r[2],
            type=r[3], config=json.loads(r[4]), enabled=bool(r[5])
        ) for r in rows]

    def replace_all(self, steps: list[ChainStep]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chain_steps")
            for s in steps:
                conn.execute(
                    "INSERT INTO chain_steps (step_order, name, type, config_json, enabled) VALUES (?,?,?,?,?)",
                    (s.step_order, s.name, s.type, json.dumps(s.config), 1 if s.enabled else 0)
                )

    def add_step(self, step_order: int, name: str, stype: str, config: dict, enabled: bool = True) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO chain_steps (step_order, name, type, config_json, enabled) VALUES (?,?,?,?,?)",
                (step_order, name, stype, json.dumps(config), 1 if enabled else 0)
            )
            return cur.lastrowid

    def update_step(self, step_id: int, enabled: bool | None = None,
                    config: dict | None = None, name: str | None = None) -> None:
        with self._conn() as conn:
            if enabled is not None:
                conn.execute("UPDATE chain_steps SET enabled=? WHERE step_id=?", (1 if enabled else 0, step_id))
            if config is not None:
                conn.execute("UPDATE chain_steps SET config_json=? WHERE step_id=?", (json.dumps(config), step_id))
            if name is not None:
                conn.execute("UPDATE chain_steps SET name=? WHERE step_id=?", (name, step_id))

    def delete_step(self, step_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chain_steps WHERE step_id=?", (step_id,))

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chain_steps")
