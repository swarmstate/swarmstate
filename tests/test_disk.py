"""Tests for the file-backed DiskStore (SQLite) + as a checkpointer backend."""

import pytest

pytest.importorskip("msgpack")

from swarmstate.backends.disk import DiskStore


def test_set_get_roundtrip_types(tmp_path):
    s = DiskStore(str(tmp_path / "s.db"))
    payload = {"step": 3, "ratio": 1.5, "tags": ["a", "b"], "nested": {"k": [1, 2]}, "n": None}
    s.set("wf", "a", payload)
    assert s.get("wf", "a") == payload
    assert s.get("wf", "missing") is None
    assert s.get("wf", "missing", default=42) == 42


def test_bytes_preserved(tmp_path):
    s = DiskStore(str(tmp_path / "s.db"))
    s.set("bin", "blob", b"\x00\x01\xff")
    assert s.get("bin", "blob") == b"\x00\x01\xff"


def test_contains_delete_keys_namespaces_len(tmp_path):
    s = DiskStore(str(tmp_path / "s.db"))
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


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "state.db")
    s1 = DiskStore(path)
    s1.set("wf", "a", {"step": 7})
    s1.close()

    # A brand-new store over the same file sees the data (survives "restart").
    s2 = DiskStore(path)
    assert s2.get("wf", "a") == {"step": 7}


def test_snapshot_restore(tmp_path):
    s = DiskStore(str(tmp_path / "s.db"))
    s.set("wf", "a", {"step": 1})
    snap = s.snapshot()
    assert ("wf", "a") in snap.keys

    s.set("wf", "a", {"step": 2})
    s.set("wf", "b", {"step": 9})
    assert len(s) == 2

    s.restore(snap)
    assert s.get("wf", "a") == {"step": 1}
    assert len(s) == 1


def test_msgpack_wire_format_is_standard(tmp_path):
    """Any SQLite + msgpack consumer can read the file, without swarmstate."""
    import sqlite3

    import msgpack

    path = str(tmp_path / "s.db")
    s = DiskStore(path)
    s.set("ns", "k", {"hello": "world", "n": 7})

    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT v FROM kv WHERE ns='ns' AND k='k'").fetchone()
    finally:
        conn.close()
    assert msgpack.unpackb(row[0], raw=False) == {"hello": "world", "n": 7}


def test_concurrent_threads_share_the_file(tmp_path):
    """Each thread gets its own connection; reads no longer serialize behind one."""
    import threading

    s = DiskStore(str(tmp_path / "conc.db"))
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            for i in range(50):
                s.set(f"t{tid}", str(i), {"tid": tid, "i": i})
            for i in range(50):
                assert s.get(f"t{tid}", str(i)) == {"tid": tid, "i": i}
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(s) == 200
    s.close()


def test_restore_is_atomic(tmp_path):
    """A failed restore must not leave the table wiped."""
    import sqlite3

    s = DiskStore(str(tmp_path / "atomic.db"))
    s.set("ns", "a", {"v": 1})
    snap = s.snapshot()
    snap._rows = [("ns", "a", b"\x01"), ("ns", "a", b"\x02")]  # duplicate PK -> fails

    with pytest.raises(sqlite3.IntegrityError):
        s.restore(snap)
    assert s.get("ns", "a") == {"v": 1}


def test_as_langgraph_checkpointer_backend(tmp_path):
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

    path = str(tmp_path / "ckpt.db")
    cfg = {"configurable": {"thread_id": "t1"}}

    graph = b.compile(checkpointer=SwarmStateSaver(DiskStore(path)))
    graph.invoke({"count": 0}, cfg)
    assert graph.get_state(cfg).values["count"] == 1

    # New process/store over the same file resumes the thread (durable).
    graph2 = b.compile(checkpointer=SwarmStateSaver(DiskStore(path)))
    assert graph2.get_state(cfg).values["count"] == 1
