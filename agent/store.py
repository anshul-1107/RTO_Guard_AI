"""
Where the agent writes its side effects.

- outbox: mock WhatsApp messages (shown in the dashboard instead of being sent)
- runs:   one row per processed order (decision, approval, cost, latency, trace)

MemoryStore for tests / offline; PostgresStore for Supabase / local Postgres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_outbox (
    id          BIGSERIAL PRIMARY KEY,
    order_id    TEXT NOT NULL,
    channel     TEXT NOT NULL,
    to_masked   TEXT NOT NULL,
    body        TEXT NOT NULL,
    send_at     TIMESTAMPTZ NOT NULL,
    status      TEXT NOT NULL DEFAULT 'QUEUED',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_runs (
    order_id        TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    rto_probability REAL,
    band            TEXT,
    action          TEXT,
    offer_pct       REAL,
    status          TEXT NOT NULL,
    needs_approval  BOOLEAN,
    approval        JSONB,
    violations      JSONB,
    citations       JSONB,
    message         TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,
    events          JSONB,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Store(Protocol):
    def send(self, msg: dict) -> None: ...
    def save_run(self, run: dict) -> None: ...
    def get_run(self, order_id: str) -> dict | None: ...


@dataclass
class MemoryStore:
    outbox: list[dict] = field(default_factory=list)
    runs: dict[str, dict] = field(default_factory=dict)

    def send(self, msg: dict) -> None:
        self.outbox.append(msg)

    def save_run(self, run: dict) -> None:
        self.runs[run["order_id"]] = run

    def get_run(self, order_id: str) -> dict | None:
        return self.runs.get(order_id)


class PostgresStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        with psycopg.connect(dsn) as c:
            c.execute(SCHEMA)
            c.commit()

    def send(self, msg: dict) -> None:
        with psycopg.connect(self.dsn) as c:
            c.execute(
                "INSERT INTO agent_outbox (order_id, channel, to_masked, body, send_at)"
                " VALUES (%(order_id)s, %(channel)s, %(to_masked)s, %(body)s, %(send_at)s)",
                msg,
            )

    def get_run(self, order_id: str) -> dict | None:
        from psycopg.rows import dict_row

        with psycopg.connect(self.dsn, row_factory=dict_row) as c:
            return c.execute(
                "SELECT * FROM agent_runs WHERE order_id = %s", (order_id,)
            ).fetchone()

    def save_run(self, run: dict) -> None:
        cols = [
            "order_id", "thread_id", "rto_probability", "band", "action", "offer_pct",
            "status", "needs_approval", "approval", "violations", "citations", "message",
            "input_tokens", "output_tokens", "cost_usd", "latency_ms", "events",
        ]
        vals = {
            k: json.dumps(run.get(k), default=str)
            if k in {"approval", "violations", "citations", "events"} else run.get(k)
            for k in cols
        }
        placeholders = ", ".join(f"%({k})s" for k in cols)
        updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in cols[1:])
        with psycopg.connect(self.dsn) as c:
            c.execute(
                f"INSERT INTO agent_runs ({', '.join(cols)}) VALUES ({placeholders})"
                f" ON CONFLICT (order_id) DO UPDATE SET {updates}, updated_at = now()",
                vals,
            )
