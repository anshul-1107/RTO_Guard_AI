"""
Deterministic policy engine: "the LLM proposes, the rules enforce".

compute_facts()      -> hard facts from the order (mirrors rag/docs policies)
enforce_decision()   -> validates / corrects the LLM decision, decides if approval is needed
check_message()      -> validates the customer message

Numbers here must match rag/docs. The LLM reads the docs for reasoning; this
module guarantees nothing unsafe ships even if the LLM gets it wrong.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta

from agent.schemas import Action, Decision

# --- constants mirrored from rag/docs ---
MIN_OFFER_ORDER_VALUE = 499
STANDARD_OFFER_PCT, STANDARD_OFFER_CAP = 5.0, 150.0
ENHANCED_OFFER_PCT, ENHANCED_OFFER_CAP = 10.0, 250.0
MAX_TOTAL_DISCOUNT = 50.0
NO_OFFER_ABOVE_SALE_DISCOUNT = 45.0
HIGH_VALUE_COD = 3000
PARTIAL_COD_MIN_VALUE_HIGH_RISK = 1999
PARTIAL_COD_TOKEN = 99
REPEAT_RTO_MIN = 2
REPEAT_RTO_WAIVER_DELIVERIES = 10
SEND_START, SEND_END = time(9, 0), time(21, 0)
MAX_MESSAGE_WORDS = 60
HINGLISH_STATES = {"Uttar Pradesh", "Bihar", "Rajasthan", "Delhi"}
BANNED_WORDS = ["suspicious", "fraud", "risk", "blacklist", "rto", "score", "return to origin"]
BAND_ORDER = ["LOW", "MEDIUM", "HIGH"]


@dataclass
class Facts:
    band: str                       # after festive adjustment
    model_band: str                 # raw model band
    is_cod: bool
    order_value: float
    is_repeat_rto: bool
    repeat_rto_waiver_possible: bool
    is_high_value: bool
    incomplete_address: bool
    offer_eligible: bool
    max_auto_offer_pct: float       # allowed without approval
    max_offer_pct: float            # allowed with approval
    partial_cod_eligible: bool
    partial_cod_mandatory: bool
    sale_active: bool
    preferred_language: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_facts(order: dict, model_band: str, sale_active: bool) -> Facts:
    notes: list[str] = []
    is_cod = order["payment_mode"] == "COD"
    value = float(order["order_value"])
    discount = float(order["discount_pct"])

    band = model_band
    if sale_active and discount > 30 and 0 <= int(order["order_hour"]) <= 4:
        band = BAND_ORDER[min(BAND_ORDER.index(band) + 1, 2)]
        if band != model_band:
            notes.append(f"Festive rule: band raised {model_band} -> {band}")

    prior_orders = int(order["customer_prior_orders"])
    prior_rto = int(order["customer_prior_rto"])
    is_repeat = prior_rto >= REPEAT_RTO_MIN
    waiver = is_repeat and (prior_orders - prior_rto) > REPEAT_RTO_WAIVER_DELIVERIES

    addr = order.get("address", {})
    incomplete = (
        not addr.get("has_house_no", True)
        or not addr.get("has_landmark", True)
        or addr.get("address_len", 99) < 25
    )

    offer_ok = (
        is_cod and value >= MIN_OFFER_ORDER_VALUE and discount <= NO_OFFER_ABOVE_SALE_DISCOUNT
    )
    if band == "HIGH" and not sale_active:
        max_offer = ENHANCED_OFFER_PCT
    else:
        max_offer = STANDARD_OFFER_PCT
    max_offer = min(max_offer, MAX_TOTAL_DISCOUNT - discount) if offer_ok else 0.0

    return Facts(
        band=band,
        model_band=model_band,
        is_cod=is_cod,
        order_value=value,
        is_repeat_rto=is_repeat,
        repeat_rto_waiver_possible=waiver,
        is_high_value=is_cod and value > HIGH_VALUE_COD,
        incomplete_address=incomplete,
        offer_eligible=offer_ok,
        max_auto_offer_pct=min(STANDARD_OFFER_PCT, max_offer),
        max_offer_pct=max_offer,
        partial_cod_eligible=is_cod and (
            is_repeat or (band == "HIGH" and value > PARTIAL_COD_MIN_VALUE_HIGH_RISK)
        ),
        partial_cod_mandatory=is_cod and is_repeat,
        sale_active=sale_active,
        preferred_language="Hinglish" if order.get("state") in HINGLISH_STATES else "English",
        notes=notes,
    )


def offer_amount(order_value: float, pct: float) -> float:
    cap = STANDARD_OFFER_CAP if pct <= STANDARD_OFFER_PCT else ENHANCED_OFFER_CAP
    return round(min(order_value * pct / 100, cap), 2)


@dataclass
class Enforcement:
    decision: Decision
    violations: list[str]
    needs_approval: bool
    approval_reasons: list[str]


def enforce_decision(decision: Decision, f: Facts) -> Enforcement:
    d = decision.model_copy(deep=True)
    v: list[str] = []
    approval: list[str] = []

    # repeat RTO customers must get partial COD (unless a human waives it)
    if f.partial_cod_mandatory and d.action != Action.PARTIAL_COD:
        if f.repeat_rto_waiver_possible:
            approval.append("Repeat-RTO rule not applied: waiver needs Ops Lead")
        else:
            v.append(f"Repeat RTO customer: {d.action.value} -> PARTIAL_COD")
            d.action, d.offer_pct = Action.PARTIAL_COD, 0

    if d.action == Action.PARTIAL_COD and not f.partial_cod_eligible:
        v.append("Partial COD not eligible -> WHATSAPP_CONFIRM")
        d.action = Action.WHATSAPP_CONFIRM

    if d.action == Action.PREPAID_OFFER:
        if not f.offer_eligible:
            v.append("Order not eligible for prepaid offer -> WHATSAPP_CONFIRM")
            d.action, d.offer_pct = Action.WHATSAPP_CONFIRM, 0
        elif d.offer_pct <= 0:
            v.append("Prepaid offer with 0% -> set to standard 5%")
            d.offer_pct = min(STANDARD_OFFER_PCT, f.max_offer_pct)
        elif d.offer_pct > f.max_offer_pct:
            v.append(f"Offer {d.offer_pct}% above limit -> clamped to {f.max_offer_pct}%")
            d.offer_pct = f.max_offer_pct
        if d.action == Action.PREPAID_OFFER and d.offer_pct > f.max_auto_offer_pct:
            approval.append(f"Offer {d.offer_pct}% is above standard 5%: Ops Lead approval")
    elif d.offer_pct:
        d.offer_pct = 0

    # high-value risky COD needs a human call, not just WhatsApp
    if f.is_high_value and f.band in {"MEDIUM", "HIGH"} and d.action == Action.WHATSAPP_CONFIRM:
        v.append("High-value COD: WHATSAPP_CONFIRM -> AGENT_CALLBACK")
        d.action = Action.AGENT_CALLBACK

    if d.action == Action.SHIP and f.band in {"MEDIUM", "HIGH"} and f.is_cod:
        v.append("MEDIUM/HIGH COD orders must be verified: SHIP -> WHATSAPP_CONFIRM")
        d.action = Action.WHATSAPP_CONFIRM

    if d.action == Action.CANCEL_REQUEST:
        approval.append("Cancellation always needs Ops Lead approval")

    if f.incomplete_address and d.action in {
        Action.WHATSAPP_CONFIRM, Action.PREPAID_OFFER, Action.PARTIAL_COD
    }:
        d.ask_address_details = True

    if v:
        approval.append("Guardrails corrected the AI decision: human review")
    return Enforcement(d, v, bool(approval), approval)


def check_message(text: str, order: dict, action: Action, offer_pct: float) -> list[str]:
    issues: list[str] = []
    low = text.lower()
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            issues.append(f"Banned word: '{w}'")
    if len(text.split()) > MAX_MESSAGE_WORDS:
        issues.append(f"Too long: {len(text.split())} words (max {MAX_MESSAGE_WORDS})")
    if "{first_name}" not in text:
        issues.append("Missing {first_name} placeholder")
    if order["order_id"] not in text:
        issues.append("Missing order ID")
    value = float(order["order_value"])
    value_forms = {f"{value:,.0f}", f"{value:.0f}", f"{value:,.2f}", f"{value:.2f}"}
    if not any(vf in text for vf in value_forms):
        issues.append("Missing order value")
    if action == Action.PARTIAL_COD and str(PARTIAL_COD_TOKEN) not in text:
        issues.append("Partial COD message must mention ₹99")
    if action == Action.PREPAID_OFFER and f"{offer_pct:g}%" not in text:
        issues.append(f"Prepaid offer message must mention {offer_pct:g}%")
    return issues


def fallback_message(order: dict, action: Action, offer_pct: float, ask_address: bool) -> str:
    """Safe template used when the LLM message fails validation twice."""
    oid, val = order["order_id"], f"{float(order['order_value']):.0f}"
    base = f"Hi {{first_name}}, thanks for your order {oid} of ₹{val}."
    if action == Action.PREPAID_OFFER:
        amt = offer_amount(float(order["order_value"]), offer_pct)
        body = f" Pay online now and get {offer_pct:g}% off (₹{amt:.0f}). Link valid 12 hours."
    elif action == Action.PARTIAL_COD:
        body = " Please pay a ₹99 advance online to confirm; the rest is payable on delivery."
    elif action == Action.AGENT_CALLBACK:
        body = " Our team will call you shortly to confirm your order and delivery details."
    else:
        body = " Please tap Confirm to ship it, or Cancel if you no longer need it."
    addr = " Please also share your house number and a nearby landmark." if ask_address else ""
    return base + body + addr


def next_send_time(now: datetime) -> datetime:
    """Messages only 9 AM - 9 PM IST (now must be IST)."""
    if SEND_START <= now.time() < SEND_END:
        return now
    day = now.date() if now.time() < SEND_START else now.date() + timedelta(days=1)
    return datetime.combine(day, SEND_START, tzinfo=now.tzinfo)
