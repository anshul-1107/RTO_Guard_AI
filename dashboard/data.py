"""Read-side helpers for the dashboard (Postgres or in-memory store)."""

from __future__ import annotations

import json

import pandas as pd
import psycopg

from agent.store import MemoryStore
from rto_guard.config import settings

RUN_COLS = ["order_id", "thread_id", "rto_probability", "band", "action", "offer_pct",
            "status", "needs_approval", "violations", "message", "cost_usd", "latency_ms",
            "updated_at"]


def db_ok() -> bool:
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3):
            return True
    except Exception:
        return False


def _query(sql: str) -> pd.DataFrame:
    with psycopg.connect(settings.pg_dsn) as c:
        cur = c.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def runs(store) -> pd.DataFrame:
    if isinstance(store, MemoryStore):
        df = pd.DataFrame(list(store.runs.values()))
    else:
        df = _query(f"SELECT {', '.join(RUN_COLS)} FROM agent_runs ORDER BY updated_at DESC")
    if df.empty:
        return pd.DataFrame(columns=RUN_COLS)
    df["violations"] = df["violations"].map(
        lambda v: json.loads(v) if isinstance(v, str) else (v or []))
    return df


def pending(store) -> pd.DataFrame:
    df = runs(store)
    return df[df["status"] == "PENDING_APPROVAL"]


def outbox(store) -> pd.DataFrame:
    if isinstance(store, MemoryStore):
        df = pd.DataFrame(store.outbox)
        return df.iloc[::-1] if not df.empty else df
    return _query("SELECT order_id, to_masked, body, send_at FROM agent_outbox ORDER BY id DESC")


def estimated_savings(df: pd.DataFrame, econ) -> float:
    """Expected net ₹ from contacting customers, using model probabilities."""
    contacted = df[df["status"] == "CUSTOMER_CONTACTED"]
    p = contacted["rto_probability"].astype(float)
    per = (p * econ.intervention_success * econ.rto_cost
           - econ.intervention_cost
           - (1 - p) * econ.friction_loss_rate * econ.order_margin)
    return float(per.sum())
