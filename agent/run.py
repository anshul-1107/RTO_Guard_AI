"""
CLI to run the agent.

  python -m agent.run --order ORD0091234
  python -m agent.run --sample 5              # random recent COD orders
  python -m agent.run --sample 5 --auto-approve
  python -m agent.run --resume ORD0091234     # answer a pending approval later
"""

from __future__ import annotations

import argparse
import json
import time

from langgraph.types import Command

from agent.orders import get_order, sample_orders
from agent.runtime import get_agent
from agent.tracing import flush, order_trace


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _print_trace(state: dict) -> None:
    for e in state.get("events", []):
        extra = {k: v for k, v in e.items() if k not in {"node", "ms"}}
        print(f"  {e['node']:<16}{e['ms']:>6} ms  {json.dumps(extra, default=str)[:110]}")


def _ask(payload: dict, auto: bool) -> dict:
    print("\n  ⏸  APPROVAL NEEDED")
    for r in payload["reasons"]:
        print(f"     - {r}")
    d = payload["decision"]
    print(f"     action: {d['action']}  offer: {d['offer_pct']}%")
    print(f"     reasoning: {d['reasoning']}")
    if payload.get("message"):
        print(f"     message: {payload['message']}")
    if auto:
        print("     -> auto-approved")
        return {"decision": "approve", "by": "cli-auto"}
    choice = input("     [a]pprove / [r]eject / [e]dit message: ").strip().lower()
    if choice.startswith("e"):
        return {"decision": "edit", "message": input("     new message: "), "by": "cli"}
    if choice.startswith("r"):
        return {"decision": "reject", "by": "cli", "note": input("     reason: ")}
    return {"decision": "approve", "by": "cli"}


def process(order_id: str, auto: bool = False, resume: bool = False) -> dict:
    agent, deps = get_agent()
    if resume:
        run = deps.store.get_run(order_id)
        if not run:
            print(f"{order_id}: no run found")
            return {}
        cfg = _cfg(run["thread_id"])
        snap = agent.get_state(cfg)
        if not snap.interrupts:
            print(f"{order_id}: nothing pending")
            return snap.values
        state = agent.invoke(Command(resume=_ask(snap.interrupts[0].value, auto)), cfg)
    else:
        # one thread per run: re-processing an order never mixes with an old run's state
        cfg = _cfg(f"order-{order_id}-{int(time.time() * 1000)}")
        order = get_order(order_id)
        with order_trace(order_id) as trace:
            state = agent.invoke(
                {"order": order, "thread_id": cfg["configurable"]["thread_id"]}, cfg
            )
            while state.get("__interrupt__"):
                payload = state["__interrupt__"][0].value
                state = agent.invoke(Command(resume=_ask(payload, auto)), cfg)
            trace.update(output={"status": state.get("status"),
                                 "action": state.get("decision", {}).get("action")})
        flush()

    print(f"\n{order_id}  ->  {state.get('status')}")
    _print_trace(state)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--auto-approve", action="store_true")
    a = ap.parse_args()

    if a.resume:
        process(a.resume, a.auto_approve, resume=True)
    elif a.order:
        process(a.order, a.auto_approve)
    else:
        for oid in sample_orders(a.sample or 3):
            process(oid, a.auto_approve)


if __name__ == "__main__":
    main()
