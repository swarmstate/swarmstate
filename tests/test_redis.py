"""M5 tests: the Redis-backed store (via fakeredis) + as a checkpointer backend."""

import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("msgpack")

from swarmstate.backends.redis import RedisStore  # noqa: E402


def make_store():
    return RedisStore(client=fakeredis.FakeStrictRedis(), prefix="ss-test")


def test_set_get_roundtrip_types():
    s = make_store()
    payload = {"step": 3, "ratio": 1.5, "tags": ["a", "b"], "nested": {"k": [1, 2]}, "n": None}
    s.set("wf", "a", payload)
    assert s.get("wf", "a") == payload
    assert s.get("wf", "missing") is None
    assert s.get("wf", "missing", default=42) == 42


def test_bytes_preserved():
    s = make_store()
    s.set("bin", "blob", b"\x00\x01\xff")
    assert s.get("bin", "blob") == b"\x00\x01\xff"


def test_contains_delete_keys_namespaces_len():
    s = make_store()
    s.set("a", "x", 1)
    s.set("a", "y", 2)
    s.set("b", "z", 3)

    assert len(s) == 3
    assert s.contains("a", "x")
    assert set(s.keys("a")) == {"x", "y"}
    assert set(s.namespaces()) == {"a", "b"}
    assert s.delete("a", "x") is True
    assert s.delete("a", "x") is False
    assert not s.contains("a", "x")
    assert len(s) == 2

    s.clear()
    assert len(s) == 0
    assert s.namespaces() == []


def test_snapshot_restore():
    s = make_store()
    s.set("wf", "a", {"step": 1})
    snap = s.snapshot()
    assert ("wf", "a") in snap.keys

    s.set("wf", "a", {"step": 2})
    s.set("wf", "b", {"step": 9})
    assert s.get("wf", "a") == {"step": 2}
    assert len(s) == 2

    s.restore(snap)
    assert s.get("wf", "a") == {"step": 1}
    assert len(s) == 1


def test_msgpack_wire_format_is_standard():
    """Values are plain msgpack — any msgpack reader can decode them."""
    import msgpack

    client = fakeredis.FakeStrictRedis()
    s = RedisStore(client=client, prefix="ss-test")
    s.set("ns", "k", {"hello": "world", "n": 7})
    raw = client.hget("ss-test:ns", "k")
    assert msgpack.unpackb(raw, raw=False) == {"hello": "world", "n": 7}


def test_as_langgraph_checkpointer_backend():
    """RedisStore is a drop-in backend for SwarmStateSaver (persistent checkpoints)."""
    lg = pytest.importorskip("langgraph")  # noqa: F841
    import operator
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph

    from swarmstate.integrations.langgraph import SwarmStateSaver

    class State(TypedDict):
        count: Annotated[int, operator.add]

    b = StateGraph(State)
    b.add_node("inc", lambda s: {"count": 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)

    store = make_store()
    graph = b.compile(checkpointer=SwarmStateSaver(store))
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"count": 0}, cfg)
    assert graph.get_state(cfg).values["count"] == 1

    # A fresh saver over the same Redis store resumes the thread (persistence).
    graph2 = b.compile(checkpointer=SwarmStateSaver(store))
    assert graph2.get_state(cfg).values["count"] == 1
