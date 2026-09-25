"""
RTO Guard agent (LangGraph).

mask_pii -> score_risk -> [LOW / prepaid] -> auto_ship ---------------------> finalize
                       -> check_rules -> retrieve_policy -> decide_action -> enforce
                             -> [customer-facing] draft_message -> [approval?] human_approval
                             -> execute -> finalize

- ML model scores the order; LLM (Gemini) chooses the intervention using RAG policy context.
- Policy engine enforces hard limits on whatever the LLM proposes.
- Costly / corrected actions pause at human_approval (LangGraph interrupt) and resume later,
  even after a restart, thanks to the checkpointer (Postgres in prod, memory in tests).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.llm import LLM
from agent.pii import fill_name, mask_order
from agent.policy import (
    check_message,
    compute_facts,
    enforce_decision,
    fallback_message,
    next_send_time,
    offer_amount,
)
from agent.prompts import (
    DECISION_SYSTEM,
    MESSAGE_PROMPT,
    MESSAGE_SYSTEM,
    decision_prompt,
)
from agent.schemas import CUSTOMER_FACING, Action, Decision, MessageDraft
from agent.store import Store
from agent.tracing import wrap_node

IST = ZoneInfo("Asia/Kolkata")
MAX_SOURCES = 8


class AgentState(TypedDict, total=False):
    thread_id: str
    order: dict
    masked: dict
    prediction: dict
    facts: dict
    sources: list[dict]
    decision: dict
    violations: list[str]
    needs_approval: bool
    approval_reasons: list[str]
    message_template: str
    message_source: str
    message_issues: list[str]
    approval: dict
    status: str
    events: Annotated[list[dict], operator.add]
    usage: Annotated[list[dict], operator.add]


@dataclass
class Deps:
    llm: LLM
    predictor: Any                                   # ml.predict.RTOPredictor
    retrieve: Callable[[str, str | None], list[dict]]  # (query, courier) -> [{citation, content}]
    store: Store
    sale_active: bool = False
    brand: str = "Kavya Threads"
    now: Callable[[], datetime] = field(default=lambda: datetime.now(IST))


def _ev(node: str, t0: float, **data) -> list[dict]:
    return [{"node": node, "ms": int((time.perf_counter() - t0) * 1000), **data}]


def build_graph(deps: Deps, checkpointer=None):
    # ---------------- nodes ----------------
    def mask_pii(s: AgentState) -> dict:
        t0 = time.perf_counter()
        return {"masked": mask_order(s["order"]), "events": _ev("mask_pii", t0)}

    def score_risk(s: AgentState) -> dict:
        t0 = time.perf_counter()
        pred = deps.predictor.predict(s["order"]).to_dict()
        return {
            "prediction": pred,
            "events": _ev("score_risk", t0, prob=pred["rto_probability"], band=pred["risk_band"]),
        }

    def auto_ship(s: AgentState) -> dict:
        t0 = time.perf_counter()
        why = "prepaid" if s["order"]["payment_mode"] != "COD" else "low risk"
        return {
            "decision": Decision(action=Action.SHIP, reasoning=f"Auto-ship: {why}").model_dump(
                mode="json"
            ),
            "status": "SHIPPED",
            "needs_approval": False,
            "events": _ev("auto_ship", t0, reason=why),
        }

    def check_rules(s: AgentState) -> dict:
        t0 = time.perf_counter()
        f = compute_facts(s["masked"], s["prediction"]["risk_band"], deps.sale_active)
        return {"facts": f.to_dict(), "events": _ev("check_rules", t0, band=f.band, notes=f.notes)}

    def retrieve_policy(s: AgentState) -> dict:
        t0 = time.perf_counter()
        f, courier = s["facts"], s["order"]["courier"]
        reasons = ", ".join(r["text"] for r in s["prediction"]["reasons"])
        queries: list[tuple[str, str | None]] = [
            (f"{f['band']} risk COD order verification. Reasons: {reasons}", None),
            ("prepaid conversion offer discount limits and approval", None),
            (f"{courier} NDR response window and RTO charges", courier),
            ("customer message tone, language and forbidden words", None),
        ]
        if f["is_repeat_rto"]:
            queries.append(("repeat RTO customer partial COD token advance", None))
        if f["is_high_value"]:
            queries.append(("high value COD order agent call-back courier selection", None))
        if f["incomplete_address"]:
            queries.append(("incomplete address house number landmark correction", None))
        if f["sale_active"]:
            queries.append(("festive sale rules and offer limits", None))

        seen, sources = set(), []
        for q, c in queries:
            for hit in deps.retrieve(q, c)[:3]:
                if hit["citation"] not in seen and len(sources) < MAX_SOURCES:
                    seen.add(hit["citation"])
                    sources.append(hit)
        return {
            "sources": sources,
            "events": _ev("retrieve_policy", t0, queries=len(queries), sources=len(sources)),
        }

    def decide_action(s: AgentState) -> dict:
        t0 = time.perf_counter()
        context = "\n\n".join(
            f"[{i}] ({src['citation']})\n{src['content']}" for i, src in enumerate(s["sources"], 1)
        )
        prompt = decision_prompt(s["masked"], s["prediction"], s["facts"], context)
        try:
            d, usage = deps.llm.structured(
                DECISION_SYSTEM.format(brand=deps.brand), prompt, Decision
            )
            return {
                "decision": d.model_dump(mode="json"),
                "usage": [{"step": "decide", **usage.to_dict()}],
                "events": _ev("decide_action", t0, action=d.action.value, offer=d.offer_pct,
                              ask_address=d.ask_address_details),
            }
        except Exception as e:  # LLM down: safe default + human review
            d = Decision(action=Action.WHATSAPP_CONFIRM, reasoning=f"LLM unavailable: {e}")
            return {
                "decision": d.model_dump(mode="json"),
                "approval_reasons": ["LLM unavailable, safe default used"],
                "events": _ev("decide_action", t0, error=str(e)[:200]),
            }

    def enforce(s: AgentState) -> dict:
        t0 = time.perf_counter()
        from agent.policy import Facts

        facts = Facts(**s["facts"])
        res = enforce_decision(Decision.model_validate(s["decision"]), facts)
        reasons = list(s.get("approval_reasons", [])) + res.approval_reasons
        return {
            "decision": res.decision.model_dump(mode="json"),
            "violations": res.violations,
            "needs_approval": bool(reasons),
            "approval_reasons": reasons,
            "events": _ev(
                "enforce", t0, action=res.decision.action.value, violations=res.violations
            ),
        }

    def draft_message(s: AgentState) -> dict:
        t0 = time.perf_counter()
        d = Decision.model_validate(s["decision"])
        o = s["masked"]
        template = fallback_message(o, d.action, d.offer_pct, d.ask_address_details)
        details = []
        if d.action == Action.PREPAID_OFFER:
            details.append(
                f"{d.offer_pct:g}% off (₹{offer_amount(o['order_value'], d.offer_pct):.0f}) "
                "if paid online within 12 hours"
            )
        if d.action == Action.PARTIAL_COD:
            details.append("pay ₹99 advance online, rest on delivery, within 12 hours")
        if d.action == Action.AGENT_CALLBACK:
            details.append("our team will call to confirm order and address")
        if d.action == Action.WHATSAPP_CONFIRM:
            details.append("tap Confirm or Cancel")
        if d.ask_address_details:
            details.append("ask for house number and nearby landmark")

        usage, feedback, issues = [], "", []
        for _ in range(2):
            prompt = MESSAGE_PROMPT.format(
                action=d.action.value, order_id=o["order_id"], value=f"{o['order_value']:.0f}",
                language=s["facts"]["preferred_language"], details="; ".join(details),
                feedback=feedback, template=template,
            )
            try:
                m, u = deps.llm.structured(
                    MESSAGE_SYSTEM.format(brand=deps.brand), prompt, MessageDraft
                )
            except Exception as e:
                issues = [f"LLM error: {e}"]
                break
            usage.append({"step": "message", **u.to_dict()})
            issues = check_message(m.text, o, d.action, d.offer_pct)
            if not issues:
                return {
                    "message_template": m.text, "message_source": "llm", "message_issues": [],
                    "usage": usage, "events": _ev("draft_message", t0, source="llm"),
                }
            feedback = "PREVIOUS ATTEMPT FAILED CHECKS, FIX: " + "; ".join(issues)

        return {
            "message_template": template, "message_source": "fallback",
            "message_issues": issues, "usage": usage,
            "events": _ev("draft_message", t0, source="fallback", issues=issues),
        }

    def human_approval(s: AgentState) -> dict:
        t0 = time.perf_counter()
        # visible in the dashboard queue; idempotent, so safe when the node re-runs on resume
        deps.store.save_run(_record(s, "PENDING_APPROVAL"))
        answer = interrupt({
            "order_id": s["order"]["order_id"],
            "rto_probability": s["prediction"]["rto_probability"],
            "band": s["facts"]["band"],
            "decision": s["decision"],
            "message": s.get("message_template"),
            "reasons": s.get("approval_reasons", []),
            "violations": s.get("violations", []),
            "sources": [src["citation"] for src in s["sources"]],
        })
        # answer: {"decision": approve|reject|edit, "message": str?, "by": str?, "note": str?}
        out: dict = {"approval": answer, "events": _ev("human_approval", t0, **answer)}
        if answer.get("decision") == "edit" and answer.get("message"):
            out["message_template"] = answer["message"]
            out["message_source"] = "human"
            out["message_issues"] = check_message(
                answer["message"], s["masked"], Action(s["decision"]["action"]),
                s["decision"]["offer_pct"],
            )
        if answer.get("decision") == "reject":
            out["status"] = "REJECTED"
        return out

    def execute(s: AgentState) -> dict:
        t0 = time.perf_counter()
        action = Action(s["decision"]["action"])
        if action == Action.SHIP:
            return {"status": "SHIPPED", "events": _ev("execute", t0, action="SHIP")}
        if action == Action.HOLD:
            return {"status": "ON_HOLD", "events": _ev("execute", t0, action="HOLD")}
        if action == Action.CANCEL_REQUEST:
            return {"status": "CANCELLED_BY_OPS", "events": _ev("execute", t0, action="CANCEL")}

        first = str(s["order"].get("customer_name", "there")).split()[0]
        body = fill_name(s["message_template"], first)
        send_at = next_send_time(deps.now())
        deps.store.send({
            "order_id": s["order"]["order_id"],
            "channel": "whatsapp_mock",
            "to_masked": s["masked"]["customer_phone_masked"],
            "body": body,
            "send_at": send_at,
        })
        return {
            "status": "CUSTOMER_CONTACTED",
            "events": _ev("execute", t0, action=action.value, send_at=send_at.isoformat()),
        }

    def _record(s: AgentState, status: str) -> dict:
        usage = s.get("usage", [])
        pred = s.get("prediction", {})
        d = s.get("decision", {})
        return {
            "order_id": s["order"]["order_id"],
            "thread_id": s.get("thread_id", ""),
            "rto_probability": pred.get("rto_probability"),
            "band": s.get("facts", {}).get("band", pred.get("risk_band")),
            "action": d.get("action"),
            "offer_pct": d.get("offer_pct"),
            "status": status,
            "needs_approval": s.get("needs_approval", False),
            "approval": s.get("approval"),
            "violations": s.get("violations", []),
            "citations": [src["citation"] for src in s.get("sources", [])],
            "message": s.get("message_template"),
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
            "cost_usd": round(sum(u["cost_usd"] for u in usage), 6),
            "latency_ms": sum(e.get("ms", 0) for e in s.get("events", [])),
            "events": s.get("events", []),
        }

    def finalize(s: AgentState) -> dict:
        deps.store.save_run(_record(s, s.get("status", "UNKNOWN")))
        return {}

    # ---------------- routing ----------------
    def after_score(s: AgentState) -> str:
        o = s["order"]
        if o["payment_mode"] != "COD":
            return "auto_ship"
        repeat = int(o["customer_prior_rto"]) >= 2
        if s["prediction"]["risk_band"] == "LOW" and not repeat and not deps.sale_active:
            return "auto_ship"
        return "check_rules"

    def after_rules(s: AgentState) -> str:
        f = s["facts"]
        if f["band"] == "LOW" and not f["partial_cod_mandatory"]:
            return "auto_ship"
        return "retrieve_policy"

    def after_enforce(s: AgentState) -> str:
        if Action(s["decision"]["action"]) in CUSTOMER_FACING:
            return "draft_message"
        return "human_approval" if s.get("needs_approval") else "execute"

    def after_message(s: AgentState) -> str:
        if s.get("needs_approval") or s.get("message_source") == "fallback":
            return "human_approval"
        return "execute"

    def after_approval(s: AgentState) -> str:
        return "finalize" if s.get("status") == "REJECTED" else "execute"

    # ---------------- graph ----------------
    g = StateGraph(AgentState)
    for name, fn in [
        ("mask_pii", mask_pii), ("score_risk", score_risk), ("auto_ship", auto_ship),
        ("check_rules", check_rules), ("retrieve_policy", retrieve_policy),
        ("decide_action", decide_action), ("enforce", enforce),
        ("draft_message", draft_message), ("human_approval", human_approval),
        ("execute", execute), ("finalize", finalize),
    ]:
        g.add_node(name, wrap_node(name, fn))

    g.add_edge(START, "mask_pii")
    g.add_edge("mask_pii", "score_risk")
    g.add_conditional_edges("score_risk", after_score, ["auto_ship", "check_rules"])
    g.add_edge("auto_ship", "finalize")
    g.add_conditional_edges("check_rules", after_rules, ["auto_ship", "retrieve_policy"])
    g.add_edge("retrieve_policy", "decide_action")
    g.add_edge("decide_action", "enforce")
    g.add_conditional_edges(
        "enforce", after_enforce, ["draft_message", "human_approval", "execute"]
    )
    g.add_conditional_edges("draft_message", after_message, ["human_approval", "execute"])
    g.add_conditional_edges("human_approval", after_approval, ["execute", "finalize"])
    g.add_edge("execute", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
