"""Moving state between backends and through files (swarmstate.portability)."""

import io

import pytest

import swarmstate as ss

ENTRIES = [
    ("wf", "a", {"step": 3, "tags": ["x", "y"], "nested": {"k": [1, 2]}}),
    ("wf", "b", b"\x00\x01\xff"),
    ("wf", "none", None),  # a stored None is an entry, not a miss
    ("ck\x1ft1\x1f", "0001", {"cp": ["msgpack", b"\x80"]}),
    ("ck\x1ft2\x1f", "0001", 42),
]


def filled():
    store = ss.Store()
    store.set_many(ENTRIES)
    return store


def contents(store):
    return sorted(
        (ns, k, repr(store.get(ns, k))) for ns in store.namespaces() for k in store.keys(ns)
    )


def test_copy_store_between_memory_stores():
    src, dst = filled(), ss.Store()
    assert ss.copy_store(src, dst) == len(ENTRIES)
    assert contents(dst) == contents(src)
    assert dst.contains("wf", "none")


def test_copy_store_with_prefix_and_small_batches():
    dst = ss.Store()
    dst.set("keep", "me", 1)
    assert ss.copy_store(filled(), dst, prefix="ck\x1f", batch_size=1) == 2
    assert sorted(dst.namespaces()) == ["ck\x1ft1\x1f", "ck\x1ft2\x1f", "keep"]
    with pytest.raises(ValueError):
        ss.copy_store(filled(), dst, batch_size=0)


def test_copy_store_to_and_from_persistent_backends(tmp_path):
    pytest.importorskip("msgpack")
    from swarmstate.backends.disk import DiskStore

    disk = DiskStore(str(tmp_path / "copy.db"))
    ss.copy_store(filled(), disk)
    back = ss.Store()
    ss.copy_store(disk, back)
    assert contents(back) == contents(filled())


def test_dump_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.swst"
    assert ss.dump_store(filled(), path) == len(ENTRIES)
    assert not (tmp_path / "state.swst.tmp").exists()

    loaded = ss.load_store(path)
    assert isinstance(loaded, ss.Store)
    assert contents(loaded) == contents(filled())

    # Into an existing store, and through file objects.
    buf = io.BytesIO()
    ss.dump_store(filled(), buf, prefix="wf")
    existing = ss.Store()
    existing.set("other", "k", 1)
    ss.load_store(io.BytesIO(buf.getvalue()), existing)
    assert sorted(existing.namespaces()) == ["other", "wf"]


def test_load_rejects_what_is_not_a_dump():
    with pytest.raises(ValueError, match="header"):
        ss.load_store(io.BytesIO(b"not a dump at all"))
    with pytest.raises(ValueError, match="version"):
        ss.load_store(io.BytesIO(b"SWSTATE\x63"))
    buf = io.BytesIO()
    ss.dump_store(filled(), buf)
    with pytest.raises(ValueError, match="truncated"):
        ss.load_store(io.BytesIO(buf.getvalue()[:-3]))


def test_a_failed_dump_leaves_no_partial_file(tmp_path):
    class Exploding:
        def namespaces(self, prefix=None):
            raise RuntimeError("store went away")

    path = tmp_path / "state.swst"
    with pytest.raises(RuntimeError):
        ss.dump_store(Exploding(), path)
    assert list(tmp_path.iterdir()) == []


def test_langgraph_threads_survive_a_migration(tmp_path):
    """The point of a common format: checkpoints move, and the graph resumes."""
    pytest.importorskip("langgraph")
    import operator
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph

    from swarmstate.integrations.langgraph import SwarmStateSaver

    class State(TypedDict):
        n: Annotated[int, operator.add]

    def graph(store):
        b = StateGraph(State)
        b.add_node("inc", lambda s: {"n": 1})
        b.add_edge(START, "inc")
        b.add_edge("inc", END)
        return b.compile(checkpointer=SwarmStateSaver(store))

    cfg = {"configurable": {"thread_id": "t"}}
    memory = ss.Store()
    graph(memory).invoke({"n": 0}, cfg)
    graph(memory).invoke({"n": 0}, cfg)

    path = tmp_path / "threads.swst"
    ss.dump_store(memory, path)
    restored = ss.load_store(path)
    assert graph(restored).invoke({"n": 0}, cfg)["n"] == 3
