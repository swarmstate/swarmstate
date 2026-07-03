#!/usr/bin/env python3
"""End-to-end demo: a support-triage agent workflow on swarmstate.

This is a *runnable* tour of the three things swarmstate does, wired into one
small LangGraph workflow. No API keys, no LLM calls: routing is deterministic,
so the whole demo runs offline and produces the same output every time.

    1. HandoffGraph     deterministic, LLM-free routing (resolved in Rust)
    2. SwarmStateSaver  a drop-in LangGraph checkpointer (replaces SqliteSaver)
    3. snapshot/restore time-travel over the whole checkpoint DB at once

Run it:

    pip install "swarmstate[langgraph]"        # or: uv add "swarmstate[langgraph]"
    python examples/support_triage.py

Expected output is at the bottom of this file.
"""

from __future__ import annotations

from typing import TypedDict

import swarmstate as ss
from swarmstate.integrations.langgraph import SwarmStateSaver


# --- 1. Deterministic routing -----------------------------------------------
# "Which agent handles this ticket?" is a rule over the ticket's fields, not a
# decision that needs an LLM. HandoffGraph resolves it natively in Rust. Edges
# are tried in order; the first matching `when` wins, and a plain edge is the
# default fallback.
router = ss.HandoffGraph()
router.add_edge("triage", "billing", when="category == 'billing'")
router.add_edge("triage", "technical", when="category == 'technical' and priority >= 2")
router.add_edge("triage", "human")  # unconditional default


class Ticket(TypedDict):
    subject: str
    category: str
    priority: int
    route: str
    resolution: str


# --- 2. A tiny LangGraph workflow -------------------------------------------
# The nodes are plain Python. `triage` asks the HandoffGraph where to go, and a
# conditional edge follows that decision. Every super-step is checkpointed by
# SwarmStateSaver.
def triage(state: Ticket) -> Ticket:
    nxt = router.route("triage", dict(state))
    return {"route": nxt}


def billing(state: Ticket) -> Ticket:
    return {"resolution": f"[billing] refunded: {state['subject']}"}


def technical(state: Ticket) -> Ticket:
    return {"resolution": f"[technical] escalated to on-call: {state['subject']}"}


def human(state: Ticket) -> Ticket:
    return {"resolution": f"[human] queued for an agent: {state['subject']}"}


def build_graph(saver: SwarmStateSaver):
    from langgraph.graph import END, START, StateGraph

    b = StateGraph(Ticket)
    b.add_node("triage", triage)
    b.add_node("billing", billing)
    b.add_node("technical", technical)
    b.add_node("human", human)

    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", lambda s: s["route"], ["billing", "technical", "human"])
    for leaf in ("billing", "technical", "human"):
        b.add_edge(leaf, END)

    return b.compile(checkpointer=saver)


TICKETS = [
    {"subject": "double charge on invoice #42", "category": "billing", "priority": 1},
    {"subject": "API returns 500 on /v1/run", "category": "technical", "priority": 3},
    {"subject": "how do I rotate my key?", "category": "other", "priority": 1},
]


def main() -> None:
    # One Store backs the checkpointer. Swap in DiskStore/RedisStore/PostgresStore
    # here for durable, shared checkpoints without touching anything else.
    saver = SwarmStateSaver()
    graph = build_graph(saver)

    print("== 1. Deterministic routing + checkpointed runs ==")
    for i, t in enumerate(TICKETS):
        cfg = {"configurable": {"thread_id": f"ticket-{i}"}}
        out = graph.invoke({**t, "route": "", "resolution": ""}, cfg)
        print(
            f"  {t['category']:<10} pri={t['priority']}  ->  {out['route']:<9}  {out['resolution']}"
        )

    # --- 3. Time-travel over the whole checkpoint DB --------------------------
    # snapshot() captures every thread at once (O(1), structural sharing). We can
    # keep running, then restore the entire checkpoint DB in one call.
    print("\n== 2. Snapshot the entire checkpoint DB (all threads) ==")
    snap = saver.store.snapshot()
    print(f"  snapshot #{snap.id}  keys={len(snap.keys)}  size={snap.size_bytes} bytes")

    # A late-breaking ticket comes in and is processed after the snapshot.
    cfg = {"configurable": {"thread_id": "ticket-late"}}
    graph.invoke(
        {
            "subject": "urgent: prod down",
            "category": "technical",
            "priority": 5,
            "route": "",
            "resolution": "",
        },
        cfg,
    )
    print(f"  after a new run, ticket-late resumes as: {graph.get_state(cfg).values['route']}")

    print("\n== 3. Restore rolls every thread back at once ==")
    saver.store.restore(snap)
    resumed = graph.get_state(cfg).values
    print(f"  ticket-late after restore: {resumed or '(gone: never happened)'}")
    # The three original threads are still there and still resume.
    still = graph.get_state({"configurable": {"thread_id": "ticket-0"}}).values["resolution"]
    print(f"  ticket-0 still resumes:    {still}")

    print(
        "\nDone. Routing was resolved in Rust; checkpoints lived in the Store; "
        "one restore() rewound the whole system."
    )


if __name__ == "__main__":
    main()

# --- Expected output ---------------------------------------------------------
# == 1. Deterministic routing + checkpointed runs ==
#   billing    pri=1  ->  billing    [billing] refunded: double charge on invoice #42
#   technical  pri=3  ->  technical  [technical] escalated to on-call: API returns 500 on /v1/run
#   other      pri=1  ->  human      [human] queued for an agent: how do I rotate my key?
#
# == 2. Snapshot the entire checkpoint DB (all threads) ==
#   snapshot #1  keys=39  size=8556 bytes
#   after a new run, ticket-late resumes as: technical
#
# == 3. Restore rolls every thread back at once ==
#   ticket-late after restore: (gone: never happened)
#   ticket-0 still resumes:    [billing] refunded: double charge on invoice #42
