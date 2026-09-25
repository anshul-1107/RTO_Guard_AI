"""LLM-as-judge for customer messages (Gemini). Rubric mirrors the communication guidelines."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.llm import LLM

JUDGE_SYSTEM = """You are a strict QA reviewer for customer WhatsApp messages of an Indian D2C
brand. Score the message against the rubric. Be critical; 5 means flawless."""

JUDGE_PROMPT = """RUBRIC
- warmth (1-5): respectful, friendly, never accusatory
- clarity (1-5): the customer knows exactly what to do next
- policy (1-5): no words like risk/fraud/suspicious/RTO, no mention of past returns,
  includes order ID and value, no promotions, under 60 words
- language_ok (true/false): written in {language} ("Hinglish" = simple Hindi in Latin script
  mixed with English)

ACTION: {action}
MESSAGE:
{message}"""


class JudgeScore(BaseModel):
    warmth: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    policy: int = Field(ge=1, le=5)
    language_ok: bool
    issues: str = Field(description="One line, empty if none")


def judge(llm: LLM, message: str, action: str, language: str) -> tuple[JudgeScore, float]:
    score, usage = llm.structured(
        JUDGE_SYSTEM,
        JUDGE_PROMPT.format(language=language, action=action, message=message),
        JudgeScore,
    )
    return score, usage.cost_usd
