"""
Optional Langfuse tracing. No keys -> every call here is a cheap no-op.

Trace per order:
  rto-guard-order (agent)
    ├─ score_risk (span)            ├─ enforce (guardrail)
    ├─ retrieve_policy (retriever)  ├─ draft_message (span)
    ├─ decide_action (chain)        │    └─ gemini generation(s)
    │    └─ gemini generation       └─ execute / finalize

Node *inputs* are never logged (the raw order contains PII); only node outputs,
which are already masked, and LLM prompts, which never contain PII.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache

from rto_guard.config import settings

NODE_TYPES = {
    "retrieve_policy": "retriever",
    "decide_action": "chain",
    "enforce": "guardrail",
}
NO_TRACE_NODES = {"human_approval"}  # interrupt() raises by design; keep it out of spans


@lru_cache(maxsize=1)
def client():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def enabled() -> bool:
    return client() is not None


class _Noop:
    def update(self, **_):
        pass


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs):
    lf = client()
    if lf is None:
        yield _Noop()
        return
    with lf.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def generation(model: str, prompt: str):
    return observe("gemini", as_type="generation", model=model, input=prompt)


def wrap_node(name: str, fn):
    if name in NO_TRACE_NODES:
        return fn

    def traced(state):
        if not enabled():
            return fn(state)
        with observe(name, as_type=NODE_TYPES.get(name, "span")) as obs:
            out = fn(state)
            obs.update(output={k: v for k, v in out.items() if k != "usage"})
            return out

    return traced


def order_trace(order_id: str):
    if not enabled():
        return nullcontext(_Noop())
    return observe("rto-guard-order", as_type="agent", input={"order_id": order_id},
                   metadata={"order_id": order_id})


def flush() -> None:
    lf = client()
    if lf is not None:
        lf.flush()
