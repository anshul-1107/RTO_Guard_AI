"""
RTO Guard dashboard (Streamlit).

Tabs: Overview · Run agent · Approvals · WhatsApp outbox · Model & evals
Run locally:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Streamlit Cloud secrets -> env vars, before settings are loaded
try:
    def _sync_secrets(obj, prefix=""):
        if hasattr(obj, "items"):
            for k, v in obj.items():
                _sync_secrets(v, k if not prefix else f"{prefix}_{k}")
        elif isinstance(obj, (str, int, float, bool)):
            os.environ[prefix.upper()] = str(obj)

    _sync_secrets(st.secrets)
except Exception:
    pass


import pandas as pd  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent.orders import get_order, sample_orders  # noqa: E402
from agent.runtime import get_agent  # noqa: E402
from agent.store import MemoryStore  # noqa: E402
from agent.tracing import flush, order_trace  # noqa: E402
from dashboard import data  # noqa: E402
from ml.economics import ECON  # noqa: E402
from rto_guard.config import settings  # noqa: E402

st.set_page_config(page_title="RTO Guard", page_icon="🛡️", layout="wide")

BAND_COLOR = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}
STATUS_ICON = {
    "SHIPPED": "📦", "CUSTOMER_CONTACTED": "💬", "PENDING_APPROVAL": "⏸️",
    "REJECTED": "🚫", "ON_HOLD": "⏳", "CANCELLED_BY_OPS": "❌",
}


@st.cache_resource(show_spinner="Loading model, policies and agent…")
def agent_and_deps():
    return get_agent()


def store():
    return agent_and_deps()[1].store


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_order(order_id: str) -> dict:
    import time

    agent, _ = agent_and_deps()
    tid = f"order-{order_id}-{int(time.time() * 1000)}"
    order = get_order(order_id)
    with order_trace(order_id):
        state = agent.invoke({"order": order, "thread_id": tid}, _cfg(tid))
    flush()
    state["_thread_id"] = tid
    return state


def resume(thread_id: str, answer: dict) -> dict:
    agent, _ = agent_and_deps()
    state = agent.invoke(Command(resume=answer), _cfg(thread_id))
    flush()
    return state


# ---------------------------------------------------------------- header
st.title("🛡️ RTO Guard")
st.caption(
    f"AI agent that cuts COD returns for **{settings.brand_name}** · "
    f"LLM: `{settings.llm_provider}` · embeddings: `{settings.embeddings_provider}` · "
    f"storage: `{'Postgres' if data.db_ok() else 'in-memory (demo)'}`"
)
if settings.llm_provider == "fake":
    st.info("Demo mode: a scripted LLM is used. Add `GEMINI_API_KEY` and set "
            "`LLM_PROVIDER=gemini` to use Gemini.", icon="ℹ️")

tab_over, tab_run, tab_appr, tab_out, tab_model = st.tabs(
    ["📊 Overview", "▶️ Run agent", "⏸️ Approvals", "💬 WhatsApp outbox", "🧪 Model & evals"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    runs = data.runs(store())
    if runs.empty:
        st.write("No orders processed yet. Go to **Run agent** and process a few orders.")
    else:
        est = data.estimated_savings(runs, ECON)
        llm_inr = runs["cost_usd"].fillna(0).sum() * settings.usd_inr
        c = st.columns(6)
        c[0].metric("Orders processed", len(runs))
        c[1].metric("Auto-shipped", f"{(runs.status == 'SHIPPED').mean():.0%}")
        c[2].metric("Customers contacted", int((runs.status == "CUSTOMER_CONTACTED").sum()))
        c[3].metric("Pending approval", int((runs.status == "PENDING_APPROVAL").sum()))
        c[4].metric("Est. net savings", f"₹{est:,.0f}")
        c[5].metric("LLM cost", f"₹{llm_inr:,.2f}")
        st.caption(
            "Est. savings = Σ over contacted orders of p(RTO)×success×RTO cost − intervention "
            "cost − friction on good customers (assumptions in `ml/economics.py`)."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Actions taken")
            st.bar_chart(runs["action"].value_counts())
        with right:
            st.subheader("Risk bands")
            st.bar_chart(runs["band"].value_counts())

        st.subheader("Recent runs")
        view = runs[["order_id", "band", "rto_probability", "action", "offer_pct", "status",
                     "cost_usd", "latency_ms"]].copy()
        view["band"] = view["band"].map(lambda b: f"{BAND_COLOR.get(b, '')} {b}")
        view["status"] = view["status"].map(lambda s: f"{STATUS_ICON.get(s, '')} {s}")
        st.dataframe(view, hide_index=True, width="stretch")

# ---------------------------------------------------------------- run agent
with tab_run:
    st.write("Pick recent COD orders (from the test period the model never saw) and run the agent.")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        n = st.number_input("How many random orders", 1, 20, 5)
        if st.button("🎲 Sample orders"):
            st.session_state["sample"] = sample_orders(int(n), seed=int(pd.Timestamp.now().value))
    with col_b:
        manual = st.text_input("…or type an order ID", placeholder="ORD0085506")

    queue = st.session_state.get("sample", [])
    if manual:
        queue = [manual.strip()]
    if queue:
        st.write("Orders: " + ", ".join(f"`{o}`" for o in queue))
        if st.button("▶️ Run agent", type="primary"):
            results = []
            prog = st.progress(0.0)
            for i, oid in enumerate(queue):
                try:
                    results.append(run_order(oid))
                except KeyError:
                    st.error(f"Order {oid} not found")
                prog.progress((i + 1) / len(queue))
            st.session_state["results"] = results

    for s in st.session_state.get("results", []):
        pred, dec = s.get("prediction", {}), s.get("decision", {})
        paused = bool(s.get("__interrupt__"))
        status = "PENDING_APPROVAL" if paused else s.get("status")
        band = s.get("facts", {}).get("band", pred.get("risk_band"))
        with st.expander(
            f"{STATUS_ICON.get(status, '')} {s['order']['order_id']} · {BAND_COLOR.get(band, '')} "
            f"{band} ({pred.get('rto_probability', 0):.0%}) · {dec.get('action')} · {status}"
        ):
            o = s["order"]
            st.markdown(
                f"**₹{o['order_value']:,.0f}** · {o['category']} · {o['payment_mode']} · "
                f"{o['state']} (tier {o['city_tier']}) · {o['courier']} · "
                f"prior orders {o['customer_prior_orders']}, prior RTO {o['customer_prior_rto']}"
            )
            if pred.get("reasons"):
                st.markdown("**Why risky:** " + ", ".join(r["text"] for r in pred["reasons"]))
            if dec.get("reasoning"):
                st.markdown(f"**Agent reasoning:** {dec['reasoning']}")
            if s.get("violations"):
                st.warning("Guardrails corrected the AI: " + "; ".join(s["violations"]))
            if s.get("message_template"):
                st.markdown("**Message draft:**")
                st.info(s["message_template"])
            if s.get("sources"):
                st.markdown("**Policy sources:** " + " · ".join(
                    f"[{i}] {src['citation']}" for i, src in enumerate(s["sources"], 1)))
            if paused:
                st.warning("Waiting for approval, see the **Approvals** tab.")
            st.markdown("**Trace**")
            st.dataframe(pd.DataFrame(s.get("events", [])), hide_index=True,
                         width="stretch")

# ---------------------------------------------------------------- approvals
with tab_appr:
    pending = data.pending(store())
    if pending.empty:
        st.success("No pending approvals 🎉")
    for _, r in pending.iterrows():
        with st.container(border=True):
            st.markdown(
                f"### {r.order_id} · {BAND_COLOR.get(r.band, '')} {r.band} "
                f"({r.rto_probability:.0%})"
            )
            st.markdown(f"**Proposed:** `{r.action}`"
                        + (f" with **{r.offer_pct:g}%** off" if r.offer_pct else ""))
            if r.violations:
                st.warning("Guardrail corrections: " + "; ".join(r.violations))
            msg = st.text_area("Message (edit if needed)", r.message or "",
                               key=f"msg-{r.order_id}")
            b1, b2, b3 = st.columns(3)
            answer = None
            if b1.button("✅ Approve", key=f"ok-{r.order_id}", type="primary"):
                answer = ({"decision": "edit", "message": msg, "by": "dashboard"}
                          if msg and msg != (r.message or "")
                          else {"decision": "approve", "by": "dashboard"})
            if b2.button("🚫 Reject", key=f"no-{r.order_id}"):
                answer = {"decision": "reject", "by": "dashboard"}
            b3.caption(f"thread `{r.thread_id}`")
            if answer:
                final = resume(r.thread_id, answer)
                st.toast(f"{r.order_id} → {final.get('status')}")
                st.rerun()

# ---------------------------------------------------------------- outbox
with tab_out:
    out = data.outbox(store())
    st.caption("WhatsApp is mocked: these messages would be sent by the WhatsApp Business API.")
    if out.empty:
        st.write("No messages yet.")
    for _, m in out.iterrows():
        with st.chat_message("assistant", avatar="🛍️"):
            st.markdown(f"**To {m.to_masked}** · order `{m.order_id}` · send at "
                        f"{pd.Timestamp(m.send_at).tz_convert('Asia/Kolkata'):%d %b, %I:%M %p} IST")
            st.write(m.body)

# ---------------------------------------------------------------- model & evals
with tab_model:
    meta_p = Path("ml/artifacts/meta.json")
    if meta_p.exists():
        m = json.loads(meta_p.read_text())["metrics"]
        st.subheader("ML model (time-based test set)")
        c = st.columns(5)
        c[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        c[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
        c[2].metric("Recall @ threshold", f"{m['recall']:.0%}")
        c[3].metric("Net savings (test)", f"₹{m['net_savings_inr']:,.0f}")
        c[4].metric("vs flag-all-COD", f"₹{m['baseline_all_cod_savings_inr']:,.0f}")
        st.caption(f"Decision threshold {m['threshold']} chosen to maximise net ₹ savings.")

    for name, title in [("retrieval", "Retrieval eval"), ("agent", "Agent eval")]:
        p = Path(f"evals/results/{name}.json")
        if p.exists():
            st.subheader(title)
            res = json.loads(p.read_text())
            if name == "retrieval":
                st.dataframe(pd.DataFrame(res).T, width="stretch")
            else:
                st.dataframe(pd.Series(res["metrics"], name="value").astype(str),
                             width="stretch")
        else:
            st.caption(f"{title}: run `make eval` to populate.")

if isinstance(store(), MemoryStore):
    st.sidebar.warning("No database connected: data resets when the app restarts.")
