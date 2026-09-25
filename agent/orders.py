"""Fetch orders from Postgres/Supabase (falls back to the local parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from rto_guard.config import settings

PARQUET = Path("data/raw/orders.parquet")
DROP = {"is_rto"}  # the label must never reach the agent


def _clean(row: dict) -> dict:
    """Plain JSON-safe types (numpy/pandas/Decimal) so state can be checkpointed."""
    out = {}
    for k, v in row.items():
        if k in DROP:
            continue
        if hasattr(v, "isoformat"):
            v = v.isoformat()
        elif hasattr(v, "item"):
            v = v.item()
        out[k] = v
    for k in ("order_value", "discount_pct"):
        out[k] = float(out[k])
    return out


def get_order(order_id: str) -> dict:
    try:
        with psycopg.connect(settings.pg_dsn, row_factory=dict_row, connect_timeout=3) as c:
            row = c.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,)).fetchone()
            if row:
                return _clean(row)
    except psycopg.OperationalError:
        pass
    df = pd.read_parquet(PARQUET)
    hit = df[df.order_id == order_id]
    if hit.empty:
        raise KeyError(order_id)
    return _clean(hit.iloc[0].to_dict())


def sample_orders(n: int = 10, seed: int = 0, cod_only: bool = True) -> list[str]:
    """Recent orders (test period, unseen by the model) for demos. DB first, parquet fallback."""
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3) as c:
            where = "WHERE payment_mode = 'COD'" if cod_only else ""
            rows = c.execute(
                f"SELECT order_id FROM (SELECT order_id, payment_mode FROM orders "
                f"ORDER BY order_ts DESC LIMIT 20000) t {where} "
                f"ORDER BY md5(order_id || %s) LIMIT %s",
                (str(seed), n),
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except psycopg.OperationalError:
        pass
    df = pd.read_parquet(PARQUET).sort_values("order_ts").tail(20_000)
    if cod_only:
        df = df[df.payment_mode == "COD"]
    return df.sample(n, random_state=seed)["order_id"].tolist()
