from dataclasses import dataclass
from datetime import datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import IST, Deps, build_graph
from agent.llm import FakeLLM, Usage
from agent.policy import check_message, compute_facts, next_send_time
from agent.schemas import Action, Decision
from agent.store import MemoryStore

ORDER = {
    "order_id": "ORD0000123",
    "customer_id": "C000001",
    "order_ts": "2026-09-01T14:00:00",
    "order_hour": 14,
    "pincode": "800001",
    "state": "Karnataka",
    "city_tier": 2,
    "courier": "Delhivery",
    "category": "fashion",
    "payment_mode": "COD",
    "order_value": 1500.0,
    "num_items": 1,
    "discount_pct": 10.0,
    "is_first_order": True,
    "customer_prior_orders": 0,
    "customer_prior_rto": 0,
    "address_text": "H.No 12, MG Road, near SBI ATM, 800001",
    "address_len": 38,
    "address_has_landmark": True,
    "customer_name": "Priya",
    "customer_phone": "9876543210",
}


@dataclass
class _Pred:
    band: str
    prob: float

    def to_dict(self):
        return {"order_id": "x", "rto_probability": self.prob, "risk_band": self.band,
                "threshold": 0.24, "reasons": [{"feature": "is_cod", "text": "COD", "impact": 1}]}


class StubPredictor:
    def __init__(self, band="MEDIUM", prob=0.35):
        self.band, self.prob = band, prob

    def predict(self, order):
        return _Pred(self.band, self.prob)


def stub_retrieve(query, courier):
    return [{"citation": f"Doc > {query[:20]}", "content": "policy text"}]


def make(llm=None, band="MEDIUM", prob=0.35, hour=12, sale=False):
    store = MemoryStore()
    llm = llm or FakeLLM()
    deps = Deps(llm=llm, predictor=StubPredictor(band, prob), retrieve=stub_retrieve,
                store=store, sale_active=sale,
                now=lambda: datetime(2026, 9, 25, hour, 0, tzinfo=IST))
    return build_graph(deps, InMemorySaver()), store, llm


def run(graph, order, tid="t1"):
    cfg = {"configurable": {"thread_id": tid}}
    return graph.invoke({"order": order, "thread_id": tid}, cfg), cfg


# ---------------- routing ----------------

def test_prepaid_ships_without_llm():
    g, store, llm = make(band="HIGH", prob=0.7)
    state, _ = run(g, {**ORDER, "payment_mode": "PREPAID"})
    assert state["status"] == "SHIPPED" and llm.prompts == [] and store.outbox == []


def test_low_risk_cod_ships_without_llm():
    g, store, llm = make(band="LOW", prob=0.1)
    state, _ = run(g, ORDER)
    assert state["status"] == "SHIPPED" and llm.prompts == []
    assert store.runs["ORD0000123"]["action"] == "SHIP"


def test_medium_risk_gets_auto_prepaid_offer():
    g, store, _ = make()
    state, _ = run(g, ORDER)
    assert state["status"] == "CUSTOMER_CONTACTED"
    assert state["decision"]["action"] == "PREPAID_OFFER"
    assert state["decision"]["offer_pct"] == 5
    assert not state["needs_approval"]
    msg = store.outbox[0]["body"]
    assert msg.startswith("Hi Priya") and "ORD0000123" in msg and "5%" in msg
    assert store.outbox[0]["to_masked"] == "98XXXXXX10"


# ---------------- guardrails + human in the loop ----------------

def test_excessive_offer_is_clamped_and_needs_approval():
    llm = FakeLLM(decision_override=Decision(
        action=Action.PREPAID_OFFER, offer_pct=15, reasoning="x"))
    g, store, _ = make(llm=llm, band="HIGH", prob=0.6)
    state, cfg = run(g, ORDER)
    assert "__interrupt__" in state
    payload = state["__interrupt__"][0].value
    assert payload["decision"]["offer_pct"] == 10
    assert store.outbox == []
    assert store.runs["ORD0000123"]["status"] == "PENDING_APPROVAL"

    state = g.invoke(Command(resume={"decision": "approve", "by": "ops"}), cfg)
    assert state["status"] == "CUSTOMER_CONTACTED"
    assert "10%" in store.outbox[0]["body"]


def test_reject_sends_nothing():
    llm = FakeLLM(decision_override=Decision(
        action=Action.PREPAID_OFFER, offer_pct=10, reasoning="x"))
    g, store, _ = make(llm=llm, band="HIGH", prob=0.6)
    state, cfg = run(g, ORDER)
    state = g.invoke(Command(resume={"decision": "reject", "by": "ops"}), cfg)
    assert state["status"] == "REJECTED" and store.outbox == []


def test_human_edit_is_used():
    llm = FakeLLM(decision_override=Decision(
        action=Action.PREPAID_OFFER, offer_pct=10, reasoning="x"))
    g, store, _ = make(llm=llm, band="HIGH", prob=0.6)
    _, cfg = run(g, ORDER)
    edited = "Hi {first_name}, order ORD0000123 of ₹1500: pay online for 10% off!"
    g.invoke(Command(resume={"decision": "edit", "message": edited, "by": "ops"}), cfg)
    assert store.outbox[0]["body"].startswith("Hi Priya, order ORD0000123")


def test_repeat_rto_forced_to_partial_cod():
    llm = FakeLLM(decision_override=Decision(action=Action.WHATSAPP_CONFIRM, reasoning="x"))
    g, _, _ = make(llm=llm)
    state, _ = run(g, {**ORDER, "customer_prior_orders": 4, "customer_prior_rto": 2})
    assert state["decision"]["action"] == "PARTIAL_COD"
    assert state["violations"]
    assert "__interrupt__" in state


def test_high_value_whatsapp_upgraded_to_callback():
    llm = FakeLLM(decision_override=Decision(action=Action.WHATSAPP_CONFIRM, reasoning="x"))
    g, _, _ = make(llm=llm)
    state, _ = run(g, {**ORDER, "order_value": 4500.0})
    assert state["decision"]["action"] == "AGENT_CALLBACK"


def test_cancel_needs_approval_and_never_messages_customer():
    llm = FakeLLM(decision_override=Decision(action=Action.CANCEL_REQUEST, reasoning="x"))
    g, store, _ = make(llm=llm, band="HIGH", prob=0.8)
    state, cfg = run(g, ORDER)
    assert "__interrupt__" in state
    state = g.invoke(Command(resume={"decision": "approve", "by": "ops"}), cfg)
    assert state["status"] == "CANCELLED_BY_OPS" and store.outbox == []


def test_bad_message_falls_back_to_template_and_review():
    bad = "Hi {first_name}, your order ORD0000123 of ₹1500 looks suspicious. Pay 5% less!"
    g, store, _ = make(llm=FakeLLM(messages=[bad, bad]))
    state, cfg = run(g, ORDER)
    assert state["message_source"] == "fallback"
    assert "__interrupt__" in state
    g.invoke(Command(resume={"decision": "approve"}), cfg)
    assert "suspicious" not in store.outbox[0]["body"]


def test_llm_failure_uses_safe_default():
    class Broken:
        prompts: list = []

        def structured(self, system, prompt, schema):
            raise TimeoutError("gemini down")

    g, _, _ = make(llm=Broken())
    state, _ = run(g, ORDER)
    assert state["decision"]["action"] == "WHATSAPP_CONFIRM"
    assert "__interrupt__" in state


def test_no_pii_reaches_llm():
    g, _, llm = make(band="HIGH", prob=0.6)
    run(g, ORDER)
    joined = "\n".join(llm.prompts)
    assert llm.prompts
    for secret in ["9876543210", "Priya", "H.No 12, MG Road"]:
        assert secret not in joined


def test_message_scheduled_for_9am_at_night():
    g, store, _ = make(hour=23)
    run(g, ORDER)
    assert store.outbox[0]["send_at"].hour == 9


def test_usage_and_cost_logged():
    g, store, _ = make()
    run(g, ORDER)
    run_row = store.runs["ORD0000123"]
    assert run_row["input_tokens"] > 0 and run_row["citations"]


# ---------------- policy engine ----------------

def _masked(**kw):
    from agent.pii import mask_order
    return mask_order({**ORDER, **kw})


def test_festive_rule_raises_band():
    f = compute_facts(_masked(order_hour=2, discount_pct=40.0), "MEDIUM", sale_active=True)
    assert f.band == "HIGH" and f.max_offer_pct == 5  # sale caps offers at 5%


def test_total_discount_cap():
    f = compute_facts(_masked(discount_pct=44.0), "HIGH", sale_active=False)
    assert f.max_offer_pct == 6  # 50 - 44


def test_no_offer_on_cheap_orders():
    assert not compute_facts(_masked(order_value=399.0), "HIGH", False).offer_eligible


def test_incomplete_address_detected():
    f = compute_facts(_masked(address_text="Station Road, 800001", address_has_landmark=False),
                      "MEDIUM", False)
    assert f.incomplete_address


@pytest.mark.parametrize("text,issue", [
    ("Hi {first_name}, order ORD0000123 ₹1500 flagged as risk", "Banned word"),
    ("Hello, order ORD0000123 ₹1500", "Missing {first_name}"),
    ("Hi {first_name}, order ₹1500", "Missing order ID"),
])
def test_message_checks(text, issue):
    issues = check_message(text, ORDER, Action.WHATSAPP_CONFIRM, 0)
    assert any(i.startswith(issue) for i in issues)


def test_send_window():
    assert next_send_time(datetime(2026, 9, 25, 7, tzinfo=IST)).hour == 9
    assert next_send_time(datetime(2026, 9, 25, 15, tzinfo=IST)).hour == 15
    assert next_send_time(datetime(2026, 9, 25, 22, tzinfo=IST)).day == 26


def test_fake_llm_usage_type():
    _, u = FakeLLM().structured("s", "FACTS:\n{\"partial_cod_mandatory\": true}\n", Decision)
    assert isinstance(u, Usage)
