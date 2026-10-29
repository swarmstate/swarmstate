"""Thread management on SwarmStateSaver: retention, prune, copy, run deletion.

DeltaChannel (LangGraph 1.2+) keeps a channel out of most checkpoints and rebuilds
it from the writes of the checkpoints before, so anything that deletes
checkpoints has to know which ones that history lives in. Those tests skip on
LangGraph versions without it.
"""

import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from swarmstate.integrations.langgraph import SwarmStateSaver, _ckpt_ns


def _delta_channel():
    delta = pytest.importorskip("langgraph.channels.delta")
    return delta.DeltaChannel


def _concat(state, batches):
    out = list(state or [])
    for batch in batches:
        out.extend(batch if isinstance(batch, list) else [batch])
    return out


def delta_graph(saver, snapshot_frequency=1000):
    DeltaChannel = _delta_channel()

    class State(TypedDict):
        log: Annotated[list, DeltaChannel(_concat, snapshot_frequency=snapshot_frequency)]

    b = StateGraph(State)
    b.add_node("step", lambda s: {"log": ["x"]})
    b.add_edge(START, "step")
    b.add_edge("step", END)
    return b.compile(checkpointer=saver)


def plain_graph(saver):
    class State(TypedDict):
        log: Annotated[list, operator.add]

    b = StateGraph(State)
    b.add_node("step", lambda s: {"log": ["x"]})
    b.add_edge(START, "step")
    b.add_edge("step", END)
    return b.compile(checkpointer=saver)


def run(graph, thread, times, **config):
    cfg = {"configurable": {"thread_id": thread}, **config}
    for _ in range(times):
        graph.invoke({"log": []}, cfg)
    return cfg


def stored(saver, thread):
    return len(saver.store.keys(_ckpt_ns(thread, "")))


# ---------------------------------------------------------------- retention


@pytest.mark.parametrize("incremental", [False, True])
def test_retention_keeps_what_delta_channels_are_rebuilt_from(incremental):
    # Retention used to keep only the newest N checkpoints, which cut the chain
    # a DeltaChannel is replayed from: 30 entries came back as 3, silently.
    saver = SwarmStateSaver(incremental=incremental, max_checkpoints_per_thread=5)
    g = delta_graph(saver)
    cfg = run(g, "t", 30)
    assert len(g.get_state(cfg).values["log"]) == 30


def test_retention_still_bounds_a_delta_thread():
    # Snapshots every 4 updates: the chain back to the last one stays short, so
    # retention can still cut everything older.
    saver = SwarmStateSaver(max_checkpoints_per_thread=3)
    g = delta_graph(saver, snapshot_frequency=4)
    cfg = run(g, "t", 60)
    assert len(g.get_state(cfg).values["log"]) == 60
    assert stored(saver, "t") < 40  # 60 runs wrote ~180 checkpoints


def test_retention_of_plain_channels_is_unchanged():
    saver = SwarmStateSaver(max_checkpoints_per_thread=5)
    g = plain_graph(saver)
    cfg = run(g, "t", 30)
    assert len(g.get_state(cfg).values["log"]) == 30
    assert stored(saver, "t") <= 5 + 8


# -------------------------------------------------------------------- prune


@pytest.mark.parametrize("incremental", [False, True])
def test_prune_keep_latest(incremental):
    saver = SwarmStateSaver(incremental=incremental)
    g = plain_graph(saver)
    cfg = run(g, "t", 10)
    other = run(g, "u", 3)

    saver.prune(["t"])
    assert stored(saver, "t") == 1
    assert len(g.get_state(cfg).values["log"]) == 10
    assert stored(saver, "u") > 1  # untouched

    run(g, "t", 1)  # and the thread carries on
    assert len(g.get_state(cfg).values["log"]) == 11
    assert len(g.get_state(other).values["log"]) == 3


def test_prune_keep_latest_preserves_delta_state():
    saver = SwarmStateSaver()
    g = delta_graph(saver)
    cfg = run(g, "t", 10)
    saver.prune(["t"])
    assert len(g.get_state(cfg).values["log"]) == 10


def test_prune_delete_and_bad_strategy():
    saver = SwarmStateSaver()
    g = plain_graph(saver)
    cfg = run(g, "t", 2)
    saver.prune(["t"], strategy="delete")
    assert saver.get_tuple(cfg) is None
    with pytest.raises(ValueError, match="strategy"):
        saver.prune(["t"], strategy="keep_some")


# ------------------------------------------------------------------- copying


@pytest.mark.parametrize("incremental", [False, True])
def test_copy_thread_is_a_full_independent_copy(incremental):
    saver = SwarmStateSaver(incremental=incremental)
    g = plain_graph(saver)
    src = run(g, "src", 3)

    saver.copy_thread("src", "dst")
    dst = {"configurable": {"thread_id": "dst"}}
    assert g.get_state(dst).values == g.get_state(src).values
    history = [s.config["configurable"]["checkpoint_id"] for s in g.get_state_history(dst)]
    assert history == [s.config["configurable"]["checkpoint_id"] for s in g.get_state_history(src)]

    run(g, "dst", 2)  # the copy moves on; the source does not
    assert len(g.get_state(dst).values["log"]) == 5
    assert len(g.get_state(src).values["log"]) == 3


def test_copy_thread_carries_delta_history():
    saver = SwarmStateSaver()
    g = delta_graph(saver)
    run(g, "src", 7)
    saver.copy_thread("src", "dst")
    assert len(g.get_state({"configurable": {"thread_id": "dst"}}).values["log"]) == 7


# -------------------------------------------------------------- run deletion


def test_delete_for_runs():
    # Each invoke needs its own run_id: LangGraph reads a repeated one as
    # re-entering the same run, not as a new run.
    saver = SwarmStateSaver()
    g = plain_graph(saver)
    cfg = run(g, "t", 1, metadata={"run_id": "keep-1"})
    run(g, "t", 1, metadata={"run_id": "keep-2"})
    run(g, "t", 1, metadata={"run_id": "drop"})
    run(g, "other", 1, metadata={"run_id": "drop"})
    assert len(g.get_state(cfg).values["log"]) == 3

    saver.delete_for_runs(["drop"])

    runs = {t.metadata.get("run_id") for t in saver.list(None)}
    assert runs == {"keep-1", "keep-2"}
    # The thread falls back to its newest surviving checkpoint.
    assert len(g.get_state(cfg).values["log"]) == 2
    assert saver.get_tuple({"configurable": {"thread_id": "other"}}) is None


# ------------------------------------------------------------- delta history


@pytest.mark.parametrize("incremental", [False, True])
def test_delta_channel_history_matches_langgraph_default(incremental):
    saver = SwarmStateSaver(incremental=incremental)
    g = delta_graph(saver, snapshot_frequency=3)
    cfg = run(g, "t", 8)

    configs = [cfg, *[t.config for t in saver.list(cfg)][::5]]
    for config in configs:
        ours = saver.get_delta_channel_history(config=config, channels=["log"])
        reference = BaseCheckpointSaver.get_delta_channel_history(
            saver, config=config, channels=["log"]
        )
        assert ours == reference
    assert saver.get_delta_channel_history(config=cfg, channels=[]) == {}
    missing = {"configurable": {"thread_id": "nope"}}
    assert saver.get_delta_channel_history(config=missing, channels=["log"]) == {
        "log": {"writes": []}
    }
