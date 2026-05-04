"""
chain.py — Forward chaining feature engine.

Applies a sequence of named steps to a dataframe, each step
able to reference columns produced by earlier steps.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChainStep:
    step_id: int
    step_order: int
    name: str
    type: str          # log | interaction | ratio | binary_threshold | encode | passthrough
    config: dict[str, Any]
    enabled: bool = True


def apply_chain(df: pd.DataFrame, steps: list[ChainStep],
                encodings: dict[str, dict[str, int]]) -> pd.DataFrame:
    """
    Apply enabled steps in order to df, adding new columns.
    Each step can reference columns created by previous steps (forward chaining).
    Returns a copy with extra columns appended.
    """
    df = df.copy()

    for step in sorted(steps, key=lambda s: s.step_order):
        if not step.enabled:
            continue
        try:
            _apply_one(df, step, encodings)
        except Exception as exc:
            # Never crash on a bad step — fill with zeros
            df[step.name] = 0.0
            print(f"[chain] Step '{step.name}' failed: {exc}")

    return df


def _apply_one(df: pd.DataFrame, step: ChainStep,
               encodings: dict[str, dict[str, int]]) -> None:
    cfg = step.config
    name = step.name

    if step.type == "log":
        col = cfg["col"]
        df[name] = np.log1p(df[col].clip(lower=0).fillna(0))

    elif step.type == "interaction":
        a, b = cfg["colA"], cfg["colB"]
        df[name] = df[a].fillna(0) * df[b].fillna(0)

    elif step.type == "ratio":
        a, b = cfg["colA"], cfg["colB"]
        df[name] = df[a].fillna(0) / (df[b].fillna(0) + 1e-9)

    elif step.type == "binary_threshold":
        col = cfg["col"]
        threshold = float(cfg.get("threshold", 0))
        df[name] = (df[col].fillna(0) > threshold).astype(float)

    elif step.type == "encode":
        col = cfg["col"]
        enc = encodings.get(col, {})
        df[name] = df[col].astype(str).map(enc).fillna(0).astype(float)

    elif step.type == "passthrough":
        col = cfg["col"]
        df[name] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def auto_generate_steps(schema: list[dict], outcome_col: str) -> list[ChainStep]:
    """
    Heuristically create an initial set of chain steps from schema metadata.
    Called once when a primary dataset is uploaded.
    """
    numeric = [c for c in schema if c["type"] == "numeric" and c["name"] != outcome_col]
    categorical = [c for c in schema if c["type"] == "categorical" and c["name"] != outcome_col]

    steps: list[ChainStep] = []
    order = 1

    # Passthrough all numeric columns (makes them explicit in the feature set)
    for col_meta in numeric:
        steps.append(ChainStep(
            step_id=order, step_order=order,
            name=col_meta["name"],
            type="passthrough",
            config={"col": col_meta["name"], "label": col_meta["name"]},
            enabled=True,
        ))
        order += 1

    # Log transform columns with high range (skewed distributions)
    for col_meta in numeric:
        mn = col_meta.get("min") or 0
        mx = col_meta.get("max") or 1
        if mx > 0 and (mx / (mn + 1)) > 10:
            steps.append(ChainStep(
                step_id=order, step_order=order,
                name=f"log_{col_meta['name']}",
                type="log",
                config={"col": col_meta["name"], "label": f"log({col_meta['name']})"},
                enabled=True,
            ))
            order += 1

    # Interactions between top-3 numerics by range
    top = sorted(numeric, key=lambda c: (c.get("max") or 0) - (c.get("min") or 0), reverse=True)[:3]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i]["name"], top[j]["name"]
            steps.append(ChainStep(
                step_id=order, step_order=order,
                name=f"{a}_x_{b}",
                type="interaction",
                config={"colA": a, "colB": b, "label": f"{a} × {b}"},
                enabled=True,
            ))
            order += 1

    # Ratio of first two numerics
    if len(numeric) >= 2:
        a, b = numeric[0]["name"], numeric[1]["name"]
        steps.append(ChainStep(
            step_id=order, step_order=order,
            name=f"ratio_{a}_{b}",
            type="ratio",
            config={"colA": a, "colB": b, "label": f"{a} / {b}"},
            enabled=True,
        ))
        order += 1

    # Binary threshold at midpoint for first 3 numerics
    for col_meta in numeric[:3]:
        mn = col_meta.get("min") or 0
        mx = col_meta.get("max") or 1
        threshold = (mn + mx) / 2
        steps.append(ChainStep(
            step_id=order, step_order=order,
            name=f"{col_meta['name']}_high",
            type="binary_threshold",
            config={"col": col_meta["name"], "threshold": threshold,
                    "label": f"{col_meta['name']} > {threshold:.1f}"},
            enabled=True,
        ))
        order += 1

    # Categorical encodings (off by default — user opts in)
    for col_meta in categorical[:6]:
        steps.append(ChainStep(
            step_id=order, step_order=order,
            name=f"{col_meta['name']}_enc",
            type="encode",
            config={"col": col_meta["name"], "label": f"encode({col_meta['name']})"},
            enabled=False,
        ))
        order += 1

    return steps
