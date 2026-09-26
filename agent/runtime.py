"""Wire real components: Gemini, ML predictor, RAG retriever, Postgres store + checkpointer."""

from __future__ import annotations

from functools import lru_cache

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.rows import dict_row

from agent.graph import Deps, build_graph
from agent.llm import get_llm
from agent.store import MemoryStore, PostgresStore
from ml.predict import get_predictor
from rto_guard.config import settings


def _db_ok() -> bool:
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3):
            return True
    except Exception:
        return False


def _retrieve(query: str, courier: str | None) -> list[dict]:
    if not _db_ok():
        return [
            {
                "citation": "policy/general.md",
                "content": "Verify all high risk COD orders with WhatsApp confirmation. Offer 10% prepaid discount to convert risky COD orders to prepaid.",
            }
        ]
    try:
        from rag.retriever import search_policy

        return [
            {"citation": r.citation, "content": r.content}
            for r in search_policy(query, k=3, courier=courier)
        ]
    except Exception:
        return [
            {
                "citation": "policy/general.md",
                "content": "Verify all high risk COD orders with WhatsApp confirmation.",
            }
        ]



def make_checkpointer():
    """Postgres checkpointer = approvals survive restarts. prepare_threshold=0 for poolers."""
    if not _db_ok():
        return InMemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver

    conn = psycopg.connect(
        settings.pg_dsn, autocommit=True, row_factory=dict_row, prepare_threshold=0
    )
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_agent():
    store = PostgresStore(settings.pg_dsn) if _db_ok() else MemoryStore()
    deps = Deps(
        llm=get_llm(),
        predictor=get_predictor(),
        retrieve=_retrieve,
        store=store,
        sale_active=settings.sale_active,
        brand=settings.brand_name,
    )
    return build_graph(deps, make_checkpointer()), deps
