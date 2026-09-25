"""
Ingest policy docs: chunk -> embed only new/changed chunks -> upsert -> delete stale chunks.
Re-running is cheap: unchanged docs cost zero embedding calls.

Run:  python -m rag.ingest
"""

from pathlib import Path

import psycopg

from rag.chunking import chunk_directory
from rag.embeddings import get_embedder
from rag.store import delete_stale, ensure_schema, existing_ids, upsert_chunks
from rto_guard.config import settings

DOCS = Path("rag/docs")


def main() -> None:
    chunks = chunk_directory(DOCS)
    embedder = get_embedder()
    with psycopg.connect(settings.pg_dsn) as conn:
        ensure_schema(conn, embedder.dim)
        have = existing_ids(conn)
        new = [c for c in chunks if c.chunk_id not in have]
        vectors = embedder.embed_documents([c.content for c in new]) if new else []
        upsert_chunks(conn, new, vectors)
        removed = delete_stale(conn, [c.chunk_id for c in chunks])
    docs = len({c.doc_id for c in chunks})
    print(f"{docs} docs, {len(chunks)} chunks | embedded {len(new)} new | removed {removed} stale")


if __name__ == "__main__":
    main()
