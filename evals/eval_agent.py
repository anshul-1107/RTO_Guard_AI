"""
Agent eval: runs every golden scenario through the full LangGraph agent.

Measures the LLM on its own (before guardrails) AND the full system (after):
  raw_action_accuracy     LLM picked an allowed action
  raw_offer_compliance    LLM stayed within the discount limit
  raw_address_ask         LLM asked for address details when needed
  guardrail_interventions how often the policy engine had to correct the LLM
  final_compliance        final action allowed, or escalated to a human   (must be 100%)
  message_first_pass      LLM message passed all checks on the first try
  message_fallback_rate   template used after 2 failed attempts
  llm_skip_correct        prepaid / low-risk orders never called the LLM
  judge_*                 LLM-as-judge scores (Gemini only)
  cost / latency          per LLM-handled order

Exit code 1 if quality gates fail, so CI blocks regressions.

Run:  python -m evals.eval_agent            (LLM_PROVIDER=fake for an offline smoke run)
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import IST, Deps, build_graph
from agent.llm import FakeLLM, get_llm
from agent.store import MemoryStore
from evals.agent_cases import BASE, CASES
from rto_guard.config import settings

GATES = {"final_compliance": 1.0, "raw_action_accuracy": 0.80, "llm_skip_correct": 1.0}


class FixedPredictor:
    def __init__(self, band: str, prob: float):
        self.band, self.prob = band, prob

    def predict(self, order):
        band, prob = self.band, self.prob

        class P:
            def to_dict(self):
                return {"order_id": order["order_id"], "rto_probability": prob,
                        "risk_band": band, "threshold": 0.24,
                        "reasons": [{"feature": "is_cod", "text": "Cash-on-delivery order",
                                     "impact": 0.4}]}
        return P()


def _retrieve(query: str, courier: str | None) -> list[dict]:
    from rag.retriever import search_policy

    return [{"citation": r.citation, "content": r.content}
            for r in search_policy(query, k=3, courier=courier)]


def run_case(case: dict, llm, i: int) -> dict:
    store = MemoryStore()
    deps = Deps(llm=llm, predictor=FixedPredictor(case["band"], case["prob"]),
                retrieve=_retrieve, store=store, sale_active=case.get("sale", False),
                brand=settings.brand_name,
                now=lambda: datetime(2026, 9, 25, 12, 0, tzinfo=IST))
    graph = build_graph(deps, InMemorySaver())
    order = {**BASE, **case["order"], "order_id": f"EVAL{i:04d}"}
    cfg = {"configurable": {"thread_id": f"eval-{i}"}}
    state = graph.invoke({"order": order, "thread_id": f"eval-{i}"}, cfg)
    escalated = False
    while state.get("__interrupt__"):
        escalated = True
        state = graph.invoke(Command(resume={"decision": "approve", "by": "eval"}), cfg)

    events = {e["node"]: e for e in state["events"]}
    raw = events.get("decide_action", {})
    usage = state.get("usage", [])
    msg_calls = [u for u in usage if u["step"] == "message"]
    return {
        "name": case["name"],
        "llm_called": bool(usage) or "decide_action" in events,
        "raw_action": raw.get("action"),
        "raw_offer": raw.get("offer"),
        "final_action": state["decision"]["action"],
        "final_ask_address": state["decision"].get("ask_address_details"),
        "violations": state.get("violations", []),
        "escalated": escalated,
        "message": state.get("message_template"),
        "message_source": state.get("message_source"),
        "message_first_pass": state.get("message_source") == "llm" and len(msg_calls) == 1,
        "language": state.get("facts", {}).get("preferred_language", "English"),
        "cost_usd": sum(u["cost_usd"] for u in usage),
        "latency_ms": sum(e.get("ms", 0) for e in state["events"]),
        "status": state.get("status"),
        "raw_decision_ask": _raw_ask(state),
    }


def _raw_ask(state: dict):
    # enforce() may force ask_address_details=True; record what the LLM itself said
    for e in state["events"]:
        if e["node"] == "decide_action":
            return e.get("ask_address")
    return None


def main() -> int:
    llm = get_llm()
    judge_llm = None if isinstance(llm, FakeLLM) else llm
    rows = []
    for i, case in enumerate(CASES):
        r = run_case(case, llm, i)
        r["case"] = case
        rows.append(r)
        mark = "✓" if r["final_action"] in case["allowed"] or r["escalated"] else "✗"
        print(f"{mark} {case['name']:<28} raw={r['raw_action'] or '-':<17} "
              f"final={r['final_action']:<17} esc={r['escalated']!s:<5} "
              f"fix={len(r['violations'])}")

    llm_rows = [r for r in rows if not r["case"].get("no_llm")]
    skip_rows = [r for r in rows if r["case"].get("no_llm")]
    offer_rows = [r for r in llm_rows if "max_offer" in r["case"] and r["raw_offer"] is not None]
    addr_rows = [r for r in llm_rows if r["case"].get("ask_address")]
    msg_rows = [r for r in llm_rows if r["message"]]

    def rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    m = {
        "cases": len(rows),
        "raw_action_accuracy": rate([r["raw_action"] in r["case"]["allowed"] for r in llm_rows]),
        "raw_offer_compliance": rate(
            [r["raw_offer"] <= r["case"]["max_offer"] for r in offer_rows]),
        "raw_address_ask": rate([bool(r["raw_decision_ask"]) for r in addr_rows]),
        "final_address_ask": rate([bool(r["final_ask_address"]) for r in addr_rows]),
        "guardrail_interventions": rate([bool(r["violations"]) for r in llm_rows]),
        "final_compliance": rate(
            [r["final_action"] in r["case"]["allowed"] or r["escalated"] for r in rows]),
        "escalation_rate": rate([r["escalated"] for r in llm_rows]),
        "message_first_pass": rate([r["message_first_pass"] for r in msg_rows]),
        "message_fallback_rate": rate([r["message_source"] == "fallback" for r in msg_rows]),
        "llm_skip_correct": rate([not r["llm_called"] for r in skip_rows]),
        "avg_cost_usd_per_llm_order": round(
            statistics.mean(r["cost_usd"] for r in llm_rows), 6),
        "avg_latency_ms": int(statistics.mean(r["latency_ms"] for r in llm_rows)),
        "p95_latency_ms": int(sorted(r["latency_ms"] for r in llm_rows)[
            max(0, int(len(llm_rows) * 0.95) - 1)]),
    }

    if judge_llm and msg_rows:
        from evals.judge import judge

        scores = []
        for r in msg_rows:
            s, _ = judge(judge_llm, r["message"], r["final_action"], r["language"])
            r["judge"] = s.model_dump()
            scores.append(s)
        m["judge_warmth"] = round(statistics.mean(s.warmth for s in scores), 2)
        m["judge_clarity"] = round(statistics.mean(s.clarity for s in scores), 2)
        m["judge_policy"] = round(statistics.mean(s.policy for s in scores), 2)
        m["judge_language_ok"] = rate([s.language_ok for s in scores])

    print(f"\nagent eval | llm={settings.llm_provider} | {len(rows)} cases")
    for k, v in m.items():
        print(f"  {k:<28} {v}")

    Path("evals/results").mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.pop("case")
    Path("evals/results/agent.json").write_text(
        json.dumps({"metrics": m, "rows": rows}, indent=2, default=str))

    failed = [k for k, t in GATES.items() if m.get(k) is not None and m[k] < t]
    if failed:
        print(f"\nGATE FAILED: {failed}")
        return 1
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
