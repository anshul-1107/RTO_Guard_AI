"""Fetch orders from Postgres/Supabase (falls back to the local parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from rto_guard.config import settings

PARQUET = Path("data/raw/orders.parquet")
DROP = {"is_rto"}  # the label must never reach the agent


def _ensure_parquet() -> Path:
    """Generate sample orders on the fly if orders.parquet is missing."""
    if not PARQUET.exists() or PARQUET.stat().st_size == 0:
        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        from data.generate import generate

        df = generate(n_customers=5_000, n_orders=20_000, seed=42)
        df.to_parquet(PARQUET, index=False)
    return PARQUET


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
    except Exception:
        pass
    _ensure_parquet()
    df = pd.read_parquet(PARQUET)
    hit = df[df.order_id == order_id]
    if hit.empty:
        if not df.empty:
            return _clean(df.iloc[0].to_dict())
        raise KeyError(order_id)
    return _clean(hit.iloc[0].to_dict())


def sample_orders(n: int = 10, seed: int = 0, cod_only: bool = True) -> list[str]:
    """Recent orders (test period, unseen by the model) for demos. DB first, parquet fallback."""
    seed_int = abs(int(seed)) % (2**31 - 1)
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3) as c:
            where = "WHERE payment_mode = 'COD'" if cod_only else ""
            rows = c.execute(
                f"SELECT order_id FROM (SELECT order_id, payment_mode FROM orders "
                f"ORDER BY order_ts DESC LIMIT 20000) t {where} "
                f"ORDER BY md5(order_id || %s) LIMIT %s",
                (str(seed_int), n),
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except Exception:
        pass
    _ensure_parquet()
    df = pd.read_parquet(PARQUET).sort_values("order_ts").tail(20_000)
    if cod_only:
        df = df[df.payment_mode == "COD"]
    sample_size = min(n, len(df))
    return df.sample(sample_size, random_state=seed_int)["order_id"].tolist()

