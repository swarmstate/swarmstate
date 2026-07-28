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


class _CountingStore:
    """Wraps a real Store, counting set vs set_many calls to prove batching."""

    def __init__(self):
        self._inner = ss.Store()
        self.set_calls = 0
        self.set_many_calls = 0

    def set_many(self, items):
        self.set_many_calls += 1
        self._inner.set_many(items)

    def set(self, ns, key, value):
        self.set_calls += 1
        self._inner.set(ns, key, value)

    def __getattr__(self, name):
        # Delegate everything else (get, get_many, contains, keys, ...).
        return getattr(self._inner, name)


def test_put_writes_batches_via_set_many():
    """Multiple pending writes flush in a single set_many, and round-trip."""
    store = _CountingStore()
    saver = SwarmStateSaver(store)
    cfg = {
        "configurable": {
            "thread_id": "w",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }
    writes = [("a", 1), ("b", 2), ("c", 3)]
    saver.put_writes(cfg, writes, task_id="task-1")

    # One batched call for three writes — not one set() per write.
    assert store.set_many_calls == 1
    assert store.set_calls == 0

    # The writes are readable back through the normal namespace/key layout.
    from swarmstate.integrations.langgraph import _writes_ns

    ns = _writes_ns("w", "", "cp-1")
    assert len(store.keys(ns)) == 3


def test_put_writes_idempotent_positional_retry():
    """A retry of the same positional writes stores nothing new."""
    store = _CountingStore()
    saver = SwarmStateSaver(store)
    cfg = {
        "configurable": {
            "thread_id": "w2",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }
    writes = [("a", 1), ("b", 2)]
    saver.put_writes(cfg, writes, task_id="task-1")
    from swarmstate.integrations.langgraph import _writes_ns

    ns = _writes_ns("w2", "", "cp-1")
    assert len(store.keys(ns)) == 2

    # Replaying the identical writes is a no-op (write-once positional keys):
    # every item is filtered out, so no empty batch is flushed either.
    before = store.set_many_calls
    saver.put_writes(cfg, writes, task_id="task-1")
    assert len(store.keys(ns)) == 2
    assert store.set_many_calls == before  # empty batch → no call


def test_put_writes_fallback_without_set_many():
    """A custom store lacking set_many still works via per-item set."""

    class NoBatchStore:
        def __init__(self):
            self._inner = ss.Store()

        def __getattr__(self, name):
            if name == "set_many" or name == "get_many":
                raise AttributeError(name)
            return getattr(self._inner, name)

    saver = SwarmStateSaver(NoBatchStore())
    cfg = {
        "configurable": {
            "thread_id": "w3",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }
    saver.put_writes(cfg, [("a", 1), ("b", 2)], task_id="task-1")
    from swarmstate.integrations.langgraph import _writes_ns

    assert len(saver.store.keys(_writes_ns("w3", "", "cp-1"))) == 2
