"""
Public retrieval API used by the agent: search_policy() returns chunks with
citations, and format_context() turns them into an LLM-ready context block.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg

from rag.embeddings import Embedder, get_embedder
from rag.store import SearchResult, hybrid_search
from rto_guard.config import settings


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return get_embedder()


def search_policy(
    query: str,
    k: int = 5,
    doc_type: str | None = None,
    courier: str | None = None,
    mode: str = "hybrid",
    conn: psycopg.Connection | None = None,
) -> list[SearchResult]:
    qv = _embedder().embed_query(query)
    if conn is not None:
        return hybrid_search(conn, query, qv, k, mode, doc_type, courier)
    with psycopg.connect(settings.pg_dsn) as c:
        return hybrid_search(c, query, qv, k, mode, doc_type, courier)


def format_context(results: list[SearchResult]) -> str:
    """Numbered sources so the LLM can cite [1], [2]..."""
    blocks = [f"[{i}] ({r.citation})\n{r.content}" for i, r in enumerate(results, 1)]
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How much discount can we give for prepaid conversion?"
    for r in search_policy(q):
        print(f"{r.score:.4f}  {r.citation}")
