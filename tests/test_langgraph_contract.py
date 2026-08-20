"""M3 contract tests: SwarmStateSaver against the BaseCheckpointSaver spec.

Drives the checkpointer API directly (not just through a graph) to lock down the
drop-in contract: put/get_tuple round-trip, parent chains, list filtering,
pending writes (+ idempotency), nested checkpoint namespaces, and behavioural
equivalence with the reference InMemorySaver.
"""

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.base import (  # noqa: E402
    empty_checkpoint,
    get_checkpoint_id,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import swarmstate as ss  # noqa: E402
from swarmstate.integrations.langgraph import SwarmStateSaver  # noqa: E402


@pytest.fixture(params=["memory", "disk"])
def store(request, tmp_path):
    """A store of each "which checkpoint is latest" strategy.

    The in-memory store keeps a pointer row; the SQL backends answer max_key from
    an index and keep none. Both have to give the same answers, so the tests that
    pin that behaviour run against both.
    """
    if request.param == "memory":
        yield ss.Store()
        return
    pytest.importorskip("msgpack")
    from swarmstate.backends.disk import DiskStore

    made = DiskStore(str(tmp_path / "contract.db"))
    try:
        yield made
    finally:
        made.close()


def cfg(thread, ns="", cid=None):
    c = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    if cid:
        c["configurable"]["checkpoint_id"] = cid
    return c


def make_cp(cid, values, versions=None):
    cp = empty_checkpoint()
    cp["id"] = cid
    cp["channel_values"] = values
    # LangGraph keeps channel_versions and the versions passed to put() in sync;
    # incremental mode keys its blobs off them, so tests must too.
    cp["channel_versions"] = versions or {k: "1" for k in values}
    return cp


def put_seq(saver, thread, ids_values, ns=""):
    """Put a parent-linked sequence of checkpoints; return the last config."""
    parent = None
    for cid, values in ids_values:
        c = cfg(thread, ns, parent)
        cp = make_cp(cid, values)
        saver.put(c, cp, {"source": "loop", "step": int(cid)}, cp["channel_versions"])
        parent = cid
    return cfg(thread, ns)


def test_put_get_roundtrip_and_latest():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1}), ("2", {"x": 2}), ("3", {"x": 3})])

    # No checkpoint_id -> latest.
    t = s.get_tuple(cfg("t"))
    assert t.checkpoint["id"] == "3"
    assert t.checkpoint["channel_values"] == {"x": 3}
    assert t.metadata["step"] == 3

    # Explicit id -> that checkpoint.
    t2 = s.get_tuple(cfg("t", cid="2"))
    assert t2.checkpoint["id"] == "2"
    assert t2.checkpoint["channel_values"] == {"x": 2}


def test_parent_config_chain():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1}), ("2", {"x": 2})])

    latest = s.get_tuple(cfg("t"))
    assert latest.parent_config["configurable"]["checkpoint_id"] == "1"

    root = s.get_tuple(cfg("t", cid="1"))
    assert root.parent_config is None


def test_list_order_limit_before_and_filter():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1}), ("2", {"x": 2}), ("3", {"x": 3})])

    ids = [get_checkpoint_id(t.config) for t in s.list(cfg("t"))]
    assert ids == ["3", "2", "1"]  # newest first

    assert len(list(s.list(cfg("t"), limit=2))) == 2

    before = [get_checkpoint_id(t.config) for t in s.list(cfg("t"), before=cfg("t", cid="3"))]
    assert before == ["2", "1"]

    step2 = [get_checkpoint_id(t.config) for t in s.list(cfg("t"), filter={"step": 2})]
    assert step2 == ["2"]


def test_pending_writes_and_idempotency():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1})])
    c = cfg("t", cid="1")

    s.put_writes(c, [("messages", "a"), ("messages", "b")], task_id="task1")
    # Re-sending the same positional writes must not duplicate them.
    s.put_writes(c, [("messages", "a"), ("messages", "b")], task_id="task1")

    pending = s.get_tuple(c).pending_writes
    assert [(tid, ch, val) for tid, ch, val in pending] == [
        ("task1", "messages", "a"),
        ("task1", "messages", "b"),
    ]


def test_nested_checkpoint_ns_isolated():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1})], ns="")
    put_seq(s, "t", [("9", {"y": 9})], ns="sub")

    assert s.get_tuple(cfg("t", ns="")).checkpoint["id"] == "1"
    assert s.get_tuple(cfg("t", ns="sub")).checkpoint["id"] == "9"
    # Listing a thread with a specific ns only yields that ns.
    assert [get_checkpoint_id(t.config) for t in s.list(cfg("t", ns="sub"))] == ["9"]


@pytest.mark.parametrize("incremental", [False, True])
def test_delete_thread_clears_everything(incremental, store):
    """Nothing of a deleted thread may survive — blobs included."""
    s = SwarmStateSaver(store, incremental=incremental)
    put_seq(s, "t", [("1", {"x": 1})])
    s.put_writes(cfg("t", cid="1"), [("c", "v")], task_id="task1")

    s.delete_thread("t")
    assert s.get_tuple(cfg("t")) is None
    assert list(s.list(cfg("t"))) == []
    # Not just unreachable: actually gone from the store (incremental=True used
    # to leave the channel blobs, and the latest pointer, behind forever).
    assert len(store) == 0
    assert store.namespaces() == []


def test_latest_is_shared_across_savers_on_one_store(store):
    """A second saver's newer checkpoint must be visible to the first one.

    Latest is resolved from the store — a pointer row it publishes, or an indexed
    max_key query. A per-saver cache used to serve a stale checkpoint here, which
    is exactly the multi-worker setup the persistent backends are for.
    """
    a, b = SwarmStateSaver(store), SwarmStateSaver(store)

    a.put(cfg("t"), make_cp("1", {"x": 1}), {"source": "loop", "step": 1}, {"x": "1"})
    assert a.get_tuple(cfg("t")).checkpoint["id"] == "1"  # a now knows about "1"

    b.put(cfg("t", cid="1"), make_cp("2", {"x": 2}), {"source": "loop", "step": 2}, {"x": "2"})
    assert a.get_tuple(cfg("t")).checkpoint["id"] == "2"
    assert a.get_tuple(cfg("t")).checkpoint["channel_values"] == {"x": 2}


def test_out_of_order_put_keeps_max_as_latest(store):
    """`latest` is max(checkpoint_id), matching InMemorySaver, not last-written."""
    ours, ref = SwarmStateSaver(store), InMemorySaver()
    for s in (ours, ref):
        s.put(cfg("t"), make_cp("2", {"x": 2}), {"source": "loop", "step": 2}, {"x": "1"})
        s.put(cfg("t"), make_cp("1", {"x": 1}), {"source": "loop", "step": 1}, {"x": "1"})
    assert ours.get_tuple(cfg("t")).checkpoint["id"] == ref.get_tuple(cfg("t")).checkpoint["id"]


def test_latest_survives_a_cold_saver_over_an_existing_store(store):
    """A saver that never wrote to the store still finds the latest checkpoint."""
    put_seq(SwarmStateSaver(store), "t", [("1", {"x": 1}), ("2", {"x": 2})])
    assert SwarmStateSaver(store).get_tuple(cfg("t")).checkpoint["id"] == "2"


def test_rejects_a_useless_retention_limit():
    with pytest.raises(ValueError):
        SwarmStateSaver(max_checkpoints_per_thread=0)


@pytest.mark.parametrize("incremental", [False, True])
def test_retention_bounds_a_long_thread_and_still_resumes(incremental, store):
    """Without a limit a store grows for the life of the process."""
    s = SwarmStateSaver(store, incremental=incremental, max_checkpoints_per_thread=5)

    parent = None
    for step in range(200):
        cid, ver = f"{step:04d}", {"x": str(step)}
        s.put(cfg("t", cid=parent), make_cp(cid, {"x": step}, ver), {"step": step}, ver)
        s.put_writes(cfg("t", cid=cid), [("c", f"w{step}")], task_id="task1")
        parent = cid

    kept = [get_checkpoint_id(t.config) for t in s.list(cfg("t"))]
    # Trimming is batched, so the window sits at the limit plus the slack.
    assert 5 <= len(kept) <= 5 + 8
    assert kept[0] == "0199"

    # The newest checkpoint is intact: values, metadata and pending writes.
    latest = s.get_tuple(cfg("t"))
    assert latest.checkpoint["id"] == "0199"
    assert latest.checkpoint["channel_values"] == {"x": 199}
    assert [ch for _tid, ch, _v in latest.pending_writes] == ["c"]

    # Nothing of the dropped checkpoints is left behind, blobs included.
    live_keys = {(ns, k) for ns in store.namespaces() for k in store.keys(ns)}
    assert not [k for ns, k in live_keys if ns.startswith("ck") and k < "0100"]
    assert not [k for ns, k in live_keys if ns.startswith("wr") and "0000" in ns]
    if incremental:
        blob_versions = {k for ns, k in live_keys if ns.startswith("bl")}
        assert len(blob_versions) <= 5 + 8


def test_retention_keeps_blobs_a_forked_checkpoint_still_needs():
    """A newer checkpoint can reference an *older* channel version (a fork).

    Pruning therefore has to union the versions of every survivor: looking only
    at the oldest one would delete a blob the fork still reads through.
    """
    store = ss.Store()
    s = SwarmStateSaver(store, incremental=True, max_checkpoints_per_thread=2)

    # Ten checkpoints, one channel version each. Still within limit+slack.
    for step in range(1, 11):
        cid, ver = f"{step:04d}", {"x": str(step)}
        s.put(cfg("t", cid=f"{step - 1:04d}"), make_cp(cid, {"x": step}, ver), {"step": step}, ver)

    # The newest checkpoint forks from step 3, so it reads version "3" — older
    # than the version the oldest survivor ("0010") references. This put crosses
    # limit+slack and triggers the trim.
    fork_ver = {"x": "3"}
    s.put(cfg("t", cid="0003"), make_cp("0011", {"x": 3}, fork_ver), {"step": 11}, fork_ver)

    assert [get_checkpoint_id(t.config) for t in s.list(cfg("t"))] == ["0011", "0010"]
    assert s.get_tuple(cfg("t")).checkpoint["channel_values"] == {"x": 3}
    assert s.get_tuple(cfg("t", cid="0010")).checkpoint["channel_values"] == {"x": 10}


def _fields(t):
    return (
        t.checkpoint["id"],
        t.checkpoint["channel_values"],
        t.parent_config["configurable"]["checkpoint_id"] if t.parent_config else None,
        [(tid, ch, val) for tid, ch, val in t.pending_writes],
    )


def test_equivalent_to_inmemory_saver():
    seq = [("1", {"x": 1}), ("2", {"x": 2, "y": [1, 2]}), ("3", {"x": 3})]
    ours, ref = SwarmStateSaver(), InMemorySaver()
    for s in (ours, ref):
        put_seq(s, "t", seq)
        s.put_writes(cfg("t", cid="3"), [("m", "hi")], task_id="tk")

    assert _fields(ours.get_tuple(cfg("t"))) == _fields(ref.get_tuple(cfg("t")))
    assert [get_checkpoint_id(t.config) for t in ours.list(cfg("t"))] == [
        get_checkpoint_id(t.config) for t in ref.list(cfg("t"))
    ]
