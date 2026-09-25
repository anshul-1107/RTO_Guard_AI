"""
Retrieval eval: vector vs keyword vs hybrid on a golden question set.

Metrics (doc-level and section-level): Hit@1, Hit@3, MRR@5.
Questions are paraphrased on purpose (e.g. "part payment" for "Partial COD"),
so pure keyword search is not enough.

Run:  python -m evals.eval_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg

from rag.retriever import search_policy
from rto_guard.config import settings

GOLDEN = Path("evals/retrieval_golden.jsonl")
MODES = ["keyword", "vector", "hybrid"]


def score(rows: list[dict], conn: psycopg.Connection, mode: str) -> dict:
    hit1 = hit3 = mrr = s_hit3 = 0.0
    for row in rows:
        res = search_policy(row["q"], k=5, mode=mode, conn=conn)
        docs = [r.doc_id for r in res]
        secs = [(r.doc_id, r.section) for r in res]
        if docs[:1] == [row["doc"]]:
            hit1 += 1
        if row["doc"] in docs[:3]:
            hit3 += 1
        if (row["doc"], row["section"]) in secs[:3]:
            s_hit3 += 1
        if row["doc"] in docs:
            mrr += 1 / (docs.index(row["doc"]) + 1)
    n = len(rows)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "section_hit@3": s_hit3 / n, "mrr@5": mrr / n}


def main() -> dict:
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    results = {}
    with psycopg.connect(settings.pg_dsn) as conn:
        for mode in MODES:
            results[mode] = score(rows, conn, mode)

    print(f"retrieval eval | {len(rows)} questions | embeddings={settings.embeddings_provider}")
    print(f"{'mode':<9}" + "".join(f"{m:>15}" for m in results["hybrid"]))
    for mode, m in results.items():
        print(f"{mode:<9}" + "".join(f"{v:>15.2f}" for v in m.values()))
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    Path("evals/results/retrieval.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
