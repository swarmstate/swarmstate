"""Batch operations: Store.set_many / get_many and backend parity."""

import operator
from typing import Annotated, TypedDict

import swarmstate as ss


class _State(TypedDict):
    count: Annotated[int, operator.add]


def test_store_set_many_get_many_roundtrip():
    s = ss.Store()
    s.set_many([("a", "x", 1), ("a", "y", {"k": 2}), ("b", "z", [1, 2, 3])])
    assert len(s) == 3
    assert s.get_many([("a", "x"), ("b", "z"), ("a", "y")]) == [1, [1, 2, 3], {"k": 2}]


def test_get_many_preserves_order_and_missing():
    s = ss.Store()
    s.set("n", "a", 1)
    s.set("n", "c", 3)
    # Order preserved; missing keys come back as None.
    assert s.get_many([("n", "c"), ("n", "b"), ("n", "a")]) == [3, None, 1]
    assert s.get_many([]) == []


def test_set_many_overwrites_and_byte_accounting():
    s = ss.Store()
    s.set_many([("n", "k", {"v": 1})])
    s.set_many([("n", "k", {"v": 2})])  # overwrite
    assert len(s) == 1
    assert s.get("n", "k") == {"v": 2}
    # size_bytes stays consistent after overwrites.
    assert s.snapshot().size_bytes > 0


def test_set_many_matches_individual_sets():
    a, b = ss.Store(), ss.Store()
    items = [(f"ns{i % 4}", f"k{i}", {"i": i}) for i in range(50)]
    a.set_many(items)
    for ns, k, v in items:
        b.set(ns, k, v)
    assert len(a) == len(b) == 50
    assert a.get_many([(ns, k) for ns, k, _ in items]) == [b.get(ns, k) for ns, k, _ in items]


def test_disk_backend_batch(tmp_path):
    import pytest

    pytest.importorskip("msgpack")
    from swarmstate.backends.disk import DiskStore

    s = DiskStore(str(tmp_path / "b.db"))
    s.set_many([("a", "x", 1), ("a", "y", 2), ("b", "z", {"k": 3})])
    assert len(s) == 3
    assert s.get_many([("a", "x"), ("nope", "nope"), ("b", "z")]) == [1, None, {"k": 3}]
    s.close()


def test_saver_uses_get_many_and_still_resumes():
    import pytest

    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    from swarmstate.integrations.langgraph import SwarmStateSaver

    calls = {"n": 0}

    class CountingStore:
        """Delegating wrapper (ss.Store isn't subclassable) that counts get_many."""

        def __init__(self):
            self._s = ss.Store()

        def set(self, *a):
            return self._s.set(*a)

        def get(self, *a, **k):
            return self._s.get(*a, **k)

        def contains(self, *a):
            return self._s.contains(*a)

        def keys(self, ns):
            return self._s.keys(ns)

        def namespaces(self):
            return self._s.namespaces()

        def get_many(self, pairs):
            calls["n"] += 1
            return self._s.get_many(pairs)

    b = StateGraph(_State)
    b.add_node("inc", lambda s: {"count": 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)

    saver = SwarmStateSaver(CountingStore())
    graph = b.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"count": 0}, cfg)
    # The checkpoint resumes correctly...
    assert graph.get_state(cfg).values["count"] == 1
    # ...and the batch read path was exercised.
    assert calls["n"] > 0
