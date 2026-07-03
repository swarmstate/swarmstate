"""M3 integration tests: SwarmStateSaver as a drop-in LangGraph checkpointer."""

import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402

import swarmstate as ss  # noqa: E402
from swarmstate.integrations.langgraph import SwarmStateSaver  # noqa: E402


class State(TypedDict):
    count: Annotated[int, operator.add]
    trail: Annotated[list, operator.add]


def make_graph(checkpointer):
    b = StateGraph(State)
    b.add_node("inc", lambda s: {"count": 1, "trail": ["inc"]})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)
    return b.compile(checkpointer=checkpointer)


def test_persist_and_resume():
    g = make_graph(SwarmStateSaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    r1 = g.invoke({"count": 0, "trail": []}, cfg)
    assert r1["count"] == 1

    # State was persisted through the checkpointer.
    assert g.get_state(cfg).values["count"] == 1
    # And there is checkpoint history (exercises list()).
    assert len(list(g.get_state_history(cfg))) >= 1

    # Resuming the same thread accumulates on the persisted state.
    r2 = g.invoke({"count": 0, "trail": []}, cfg)
    assert r2["count"] == 2
    assert r2["trail"] == ["inc", "inc"]


def test_drop_in_shared_store_persistence():
    """A brand-new saver over the same Store sees prior checkpoints."""
    store = ss.Store()
    make_graph(SwarmStateSaver(store)).invoke(
        {"count": 0, "trail": []}, {"configurable": {"thread_id": "shared"}}
    )
    g2 = make_graph(SwarmStateSaver(store))
    st = g2.get_state({"configurable": {"thread_id": "shared"}})
    assert st.values["count"] == 1


def test_equivalent_to_inmemory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = {"configurable": {"thread_id": "x"}}
    ours = make_graph(SwarmStateSaver()).invoke({"count": 0, "trail": []}, cfg)
    theirs = make_graph(InMemorySaver()).invoke({"count": 0, "trail": []}, cfg)
    assert ours == theirs


def test_store_snapshot_rolls_back_all_checkpoints():
    store = ss.Store()
    g = make_graph(SwarmStateSaver(store))
    cfg = {"configurable": {"thread_id": "t"}}

    g.invoke({"count": 0, "trail": []}, cfg)  # -> 1
    snap = store.snapshot()
    g.invoke({"count": 0, "trail": []}, cfg)  # -> 2
    assert g.get_state(cfg).values["count"] == 2

    store.restore(snap)  # roll the entire checkpoint DB back
    assert g.get_state(cfg).values["count"] == 1


def test_delete_thread():
    saver = SwarmStateSaver()
    g = make_graph(saver)
    cfg = {"configurable": {"thread_id": "gone"}}
    g.invoke({"count": 0, "trail": []}, cfg)
    assert g.get_state(cfg).values  # present
    saver.delete_thread("gone")
    assert g.get_state(cfg).values == {}  # cleared


def test_incremental_mode_roundtrip():
    """incremental=True reassembles channel_values correctly and resumes."""
    store = ss.Store()
    g = make_graph(SwarmStateSaver(store, incremental=True))
    cfg = {"configurable": {"thread_id": "inc"}}
    g.invoke({"count": 0, "trail": []}, cfg)
    g.invoke({"count": 0, "trail": []}, cfg)
    st = g.get_state(cfg)
    assert st.values["count"] == 2
    assert st.values["trail"] == ["inc", "inc"]

    # A fresh saver over the same store (incremental) still reads it.
    g2 = make_graph(SwarmStateSaver(store, incremental=True))
    assert g2.get_state(cfg).values["count"] == 2


def test_async_ainvoke_and_aget_state():
    import asyncio

    g = make_graph(SwarmStateSaver())
    cfg = {"configurable": {"thread_id": "async"}}

    async def run():
        await g.ainvoke({"count": 0, "trail": []}, cfg)
        snap = await g.aget_state(cfg)
        return snap.values["count"]

    assert asyncio.run(run()) == 1
