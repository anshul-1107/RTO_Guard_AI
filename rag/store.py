"""
pgvector store (works on local Docker Postgres and Supabase).

Hybrid search = vector similarity (HNSW, cosine) + Postgres full-text search,
fused with Reciprocal Rank Fusion (RRF). Keyword search catches exact terms
like "₹99", "Partial COD" or "NDR"; vectors catch paraphrases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import psycopg

from rag.chunking import Chunk

RRF_K = 60

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    doc_title   TEXT NOT NULL,
    section     TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    applies_to  TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{{}}',
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON policy_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON policy_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON policy_chunks (doc_id);
"""


@dataclass
class SearchResult:
    id: str
    doc_id: str
    doc_title: str
    section: str
    doc_type: str
    applies_to: str
    content: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.doc_title} > {self.section}"


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _or_tsquery(text: str) -> str:
    """Natural-language question -> OR tsquery ('refund | cod | ...')."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    words = [w for w in dict.fromkeys(words) if len(w) > 1]
    return " | ".join(words) if words else "''"


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    conn.execute(SCHEMA.format(dim=dim))
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], vectors: list[list[float]]
) -> int:
    """Idempotent insert: chunk id = content hash, so unchanged chunks are skipped."""
    with conn.cursor() as cur:
        for ch, vec in zip(chunks, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO policy_chunks
                    (id, doc_id, doc_title, section, doc_type, applies_to, content, metadata,
                     embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,
                (
                    ch.chunk_id, ch.doc_id, ch.doc_title, ch.section,
                    ch.metadata.get("doc_type", "unknown"), ch.metadata.get("applies_to", "all"),
                    ch.content, json.dumps(ch.metadata), _vec_literal(vec),
                ),
            )
    conn.commit()
    return len(chunks)


def delete_stale(conn: psycopg.Connection, keep_ids: list[str]) -> int:
    """Drop chunks whose content no longer exists in the docs folder."""
    cur = conn.execute("DELETE FROM policy_chunks WHERE NOT (id = ANY(%s))", (keep_ids,))
    conn.commit()
    return cur.rowcount


def existing_ids(conn: psycopg.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM policy_chunks").fetchall()}


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_vec: list[float],
    k: int = 5,
    mode: str = "hybrid",  # hybrid | vector | keyword
    doc_type: str | None = None,
    applies_to: str | None = None,
    candidates: int = 20,
) -> list[SearchResult]:
    w_vec = 0.0 if mode == "keyword" else 1.0
    w_fts = 0.0 if mode == "vector" else 1.0

    filters, params = ["TRUE"], {}
    if doc_type:
        filters.append("doc_type = %(doc_type)s")
        params["doc_type"] = doc_type
    if applies_to:
        filters.append("(applies_to = 'all' OR applies_to = %(applies_to)s)")
        params["applies_to"] = applies_to
    where = " AND ".join(filters)

    sql = f"""
    WITH vec AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> %(qv)s::vector) AS r
        FROM policy_chunks WHERE {where}
        ORDER BY embedding <=> %(qv)s::vector LIMIT %(n)s
    ),
    fts AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
        FROM policy_chunks, to_tsquery('english', %(tq)s) q
        WHERE tsv @@ q AND {where}
        ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(n)s
    ),
    fused AS (
        SELECT id,
               %(wv)s * COALESCE(1.0 / ({RRF_K} + vec.r), 0)
             + %(wf)s * COALESCE(1.0 / ({RRF_K} + fts.r), 0) AS score
        FROM vec FULL OUTER JOIN fts USING (id)
    )
    SELECT c.id, c.doc_id, c.doc_title, c.section, c.doc_type, c.applies_to, c.content,
           f.score
    FROM fused f JOIN policy_chunks c USING (id)
    WHERE f.score > 0
    ORDER BY f.score DESC
    LIMIT %(k)s
    """
    params.update(qv=_vec_literal(query_vec), tq=_or_tsquery(query), n=candidates, k=k,
                  wv=w_vec, wf=w_fts)
    rows = conn.execute(sql, params).fetchall()
    return [SearchResult(*r[:7], float(r[7])) for r in rows]
