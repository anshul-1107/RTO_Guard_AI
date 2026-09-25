"""
LLM layer.

GeminiLLM: structured JSON output validated by Pydantic, retries with backoff,
automatic fallback to a cheaper model, token + cost tracking per call.

FakeLLM: deterministic stand-in for tests/CI. It records every prompt so tests
can prove no PII ever reaches the model.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from agent.schemas import Action, Decision, MessageDraft
from rto_guard.config import settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class LLM(Protocol):
    def structured(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]: ...


def _cost(inp: int, out: int) -> float:
    return (inp * settings.llm_price_in_per_m + out * settings.llm_price_out_per_m) / 1e6


class GeminiLLM:
    def __init__(self, model: str | None = None, fallback: str | None = None):
        from google import genai

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (or use LLM_PROVIDER=fake)")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model or settings.gemini_llm_model
        self.fallback = fallback or settings.gemini_fallback_model

    def _call(self, model: str, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )
        from agent.tracing import generation

        with generation(model, system + "\n\n" + prompt) as gen:
            t0 = time.perf_counter()
            res = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
            latency = int((time.perf_counter() - t0) * 1000)
            parsed = res.parsed if isinstance(res.parsed, schema) else (
                schema.model_validate_json(res.text)
            )
            um = res.usage_metadata
            inp = int(getattr(um, "prompt_token_count", 0) or 0)
            out = int(getattr(um, "candidates_token_count", 0) or 0)
            gen.update(
                output=parsed.model_dump(mode="json"),
                usage_details={"input": inp, "output": out},
                cost_details={"total": _cost(inp, out)},
            )
        return parsed, Usage(model, inp, out, latency, _cost(inp, out))

    def structured(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        last: Exception | None = None
        for model in (self.model, self.fallback):
            for attempt in range(3):
                try:
                    parsed, usage = self._call(model, system, prompt, schema)
                    usage.fallback_used = model != self.model
                    return parsed, usage
                except Exception as e:  # rate limit, 5xx, invalid JSON
                    last = e
                    time.sleep(2**attempt)
        raise RuntimeError(f"All Gemini models failed: {last}") from last


@dataclass
class FakeLLM:
    """
    Rule-of-thumb stand-in. `decision_override` / `messages` let tests script
    misbehaviour (e.g. a 15% offer, or a message with banned words).
    """

    decision_override: Decision | None = None
    messages: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def structured(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        self.prompts.append(system + "\n" + prompt)
        usage = Usage("fake", len(prompt) // 4, 60, 1, 0.0)
        if schema is Decision:
            if self.decision_override:
                return self.decision_override, usage  # type: ignore[return-value]
            facts = json.loads(re.search(r"FACTS:\n(\{.*?\})\n", prompt, re.S).group(1))
            if facts["partial_cod_mandatory"]:
                d = Decision(action=Action.PARTIAL_COD, reasoning="Repeat RTO customer [1]")
            elif facts["is_high_value"]:
                d = Decision(action=Action.AGENT_CALLBACK, reasoning="High value COD [1]")
            elif facts["offer_eligible"]:
                d = Decision(
                    action=Action.PREPAID_OFFER, offer_pct=facts["max_auto_offer_pct"],
                    reasoning="Prepaid conversion removes RTO risk [1]",
                )
            else:
                d = Decision(action=Action.WHATSAPP_CONFIRM, reasoning="Verify COD order [1]")
            d.cited_sources = [1]
            return d, usage  # type: ignore[return-value]
        if schema is MessageDraft:
            text = self.messages.pop(0) if self.messages else None
            if text is None:
                m = re.search(r"REFERENCE_TEMPLATE: (.*)", prompt)
                text = m.group(1) if m else "Hi {first_name}"
            return MessageDraft(language="English", text=text), usage  # type: ignore[return-value]
        raise ValueError(f"FakeLLM has no rule for {schema}")


def get_llm() -> LLM:
    return FakeLLM() if settings.llm_provider == "fake" else GeminiLLM()
