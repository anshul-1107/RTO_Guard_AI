"""Structured outputs the LLM must return (enforced via Gemini response_schema)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    SHIP = "SHIP"                          # no intervention
    WHATSAPP_CONFIRM = "WHATSAPP_CONFIRM"  # confirm / cancel buttons
    PREPAID_OFFER = "PREPAID_OFFER"        # discount to pay online
    PARTIAL_COD = "PARTIAL_COD"            # ₹99 token advance
    AGENT_CALLBACK = "AGENT_CALLBACK"      # human call for high-value orders
    HOLD = "HOLD"                          # pause dispatch (max 24h)
    CANCEL_REQUEST = "CANCEL_REQUEST"      # only a request; humans cancel


CUSTOMER_FACING = {
    Action.WHATSAPP_CONFIRM, Action.PREPAID_OFFER, Action.PARTIAL_COD, Action.AGENT_CALLBACK,
}


class Decision(BaseModel):
    action: Action
    offer_pct: float = Field(0, ge=0, le=100, description="Prepaid discount %, 0 if none")
    ask_address_details: bool = Field(
        False, description="Ask customer for house number / landmark"
    )
    reasoning: str = Field(description="2-3 sentences, cite sources like [1], [2]")
    cited_sources: list[int] = Field(default_factory=list)


class MessageDraft(BaseModel):
    language: str = Field(description="English or Hinglish")
    text: str = Field(
        description="WhatsApp message. Must start with 'Hi {first_name}'. Under 60 words."
    )
