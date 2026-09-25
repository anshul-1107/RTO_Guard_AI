from pathlib import Path

import psycopg
import pytest

from rag.chunking import chunk_directory, chunk_markdown
from rag.embeddings import FakeEmbedder
from rag.store import _or_tsquery, delete_stale, ensure_schema, hybrid_search, upsert_chunks
from rto_guard.config import settings

DOC = """---
title: Test Policy
doc_type: brand_sop
applies_to: all
---
# Test Policy

## First rule
Always verify COD orders.

## Second rule
Token advance is ₹99.
"""


def test_chunking_sections_and_context_header():
    chunks = chunk_markdown(DOC, "test-policy")
    assert [c.section for c in chunks] == ["First rule", "Second rule"]
    assert chunks[1].content.startswith("Test Policy > Second rule")
    assert chunks[0].metadata == {"doc_type": "brand_sop", "applies_to": "all"}


def test_chunk_ids_stable_and_unique():
    a = chunk_markdown(DOC, "x")
    b = chunk_markdown(DOC, "x")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_long_section_is_split():
    long_body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_markdown(f"---\ntitle: T\n---\n## Big\n{long_body}", "t")
    assert len(chunks) > 1
    assert all(len(c.content) < 1500 for c in chunks)


def test_all_policy_docs_parse():
    chunks = chunk_directory(Path("rag/docs"))
    assert len({c.doc_id for c in chunks}) >= 15
    assert all(c.metadata["doc_type"] != "unknown" for c in chunks)


def test_or_tsquery_is_safe():
    # stopwords like "the" are kept here; Postgres drops them inside to_tsquery
    assert _or_tsquery("What's the ₹99 token?") == "what | the | 99 | token"
    assert _or_tsquery("!!!") == "''"


# ---------- integration (needs Postgres + pgvector) ----------


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=2):
            return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")


@pytest.fixture
def conn():
    """Isolated schema so tests never touch real ingested data."""
    with psycopg.connect(settings.pg_dsn) as c:
        c.execute("DROP SCHEMA IF EXISTS test_rag CASCADE")
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute("CREATE SCHEMA test_rag")
        c.execute("SET search_path TO test_rag, public")
        c.commit()
        yield c
        c.execute("DROP SCHEMA test_rag CASCADE")
        c.commit()


@needs_db
def test_ingest_and_hybrid_search(conn):
    emb = FakeEmbedder(dim=64)
    ensure_schema(conn, 64)
    chunks = chunk_directory(Path("rag/docs"))
    upsert_chunks(conn, chunks, emb.embed_documents([c.content for c in chunks]))

    q = "Shadowfax NDR response window"
    res = hybrid_search(conn, q, emb.embed_query(q), k=3)
    assert res[0].doc_id == "courier-shadowfax"
    assert res[0].citation == "Shadowfax Courier Terms > NDR response window"


@needs_db
def test_courier_filter_excludes_other_couriers(conn):
    emb = FakeEmbedder(dim=64)
    ensure_schema(conn, 64)
    chunks = chunk_directory(Path("rag/docs"))
    upsert_chunks(conn, chunks, emb.embed_documents([c.content for c in chunks]))

    q = "RTO charges"
    res = hybrid_search(conn, q, emb.embed_query(q), k=10, applies_to="Bluedart")
    couriers = {r.applies_to for r in res}
    assert couriers <= {"all", "Bluedart"}


@needs_db
def test_upsert_idempotent_and_stale_cleanup(conn):
    emb = FakeEmbedder(dim=64)
    ensure_schema(conn, 64)
    chunks = chunk_markdown(DOC, "t")
    vecs = emb.embed_documents([c.content for c in chunks])
    upsert_chunks(conn, chunks, vecs)
    upsert_chunks(conn, chunks, vecs)
    assert conn.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == 2

    removed = delete_stale(conn, [chunks[0].chunk_id])
    assert removed == 1
