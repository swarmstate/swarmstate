"""Tests for the Postgres-backed store.

Runs only when a Postgres DSN is provided via ``SWARMSTATE_TEST_PG_DSN``
(e.g. in CI with a Postgres service). Skipped otherwise.
"""

import os
import uuid

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("msgpack")

DSN = os.environ.get("SWARMSTATE_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set SWARMSTATE_TEST_PG_DSN to run")

from swarmstate.backends.postgres import PostgresStore  # noqa: E402


def make_store():
    # Unique table per store so tests don't collide.
    return PostgresStore(DSN, table=f"kv_{uuid.uuid4().hex[:12]}")


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


def test_contains_delete_keys_namespaces_len_clear():
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
    assert len(s) == 2

    s.clear()
    assert len(s) == 0


def test_upsert_replaces():
    s = make_store()
    s.set("n", "k", {"v": 1})
    s.set("n", "k", {"v": 2})
    assert s.get("n", "k") == {"v": 2}
    assert len(s) == 1


def test_snapshot_restore():
    s = make_store()
    s.set("wf", "a", {"step": 1})
    snap = s.snapshot()
    assert ("wf", "a") in snap.keys

    s.set("wf", "a", {"step": 2})
    s.set("wf", "b", {"step": 9})
    assert len(s) == 2

    s.restore(snap)
    assert s.get("wf", "a") == {"step": 1}
    assert len(s) == 1


def test_msgpack_wire_format_is_standard():
    import msgpack

    s = make_store()
    s.set("ns", "k", {"hello": "world", "n": 7})
    row = s._conn.execute(f"SELECT v FROM {s.table} WHERE ns='ns' AND k='k'").fetchone()
    assert msgpack.unpackb(bytes(row[0]), raw=False) == {"hello": "world", "n": 7}


def test_invalid_table_name():
    with pytest.raises(ValueError):
        PostgresStore(DSN, table="bad; DROP TABLE x")


def test_as_langgraph_checkpointer_backend():
    pytest.importorskip("langgraph")
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
    cfg = {"configurable": {"thread_id": "t1"}}
    b.compile(checkpointer=SwarmStateSaver(store)).invoke({"count": 0}, cfg)

    # A fresh saver over the same table resumes the thread (durable).
    g2 = b.compile(checkpointer=SwarmStateSaver(PostgresStore(DSN, table=store.table)))
    assert g2.get_state(cfg).values["count"] == 1
