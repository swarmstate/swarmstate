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

from swarmstate.integrations.langgraph import SwarmStateSaver  # noqa: E402


def cfg(thread, ns="", cid=None):
    c = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    if cid:
        c["configurable"]["checkpoint_id"] = cid
    return c


def make_cp(cid, values):
    cp = empty_checkpoint()
    cp["id"] = cid
    cp["channel_values"] = values
    cp["channel_versions"] = {k: "1" for k in values}
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


def test_delete_thread_clears_everything():
    s = SwarmStateSaver()
    put_seq(s, "t", [("1", {"x": 1})])
    s.put_writes(cfg("t", cid="1"), [("c", "v")], task_id="task1")

    s.delete_thread("t")
    assert s.get_tuple(cfg("t")) is None
    assert list(s.list(cfg("t"))) == []


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
