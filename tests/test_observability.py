"""Metrics hooks for the checkpointer."""

import operator
from typing import Annotated, TypedDict

import pytest

from swarmstate.observability import InMemoryMetrics, MetricsSink, NullMetrics


class State(TypedDict):
    count: Annotated[int, operator.add]


def test_inmemory_records_counts_and_percentiles():
    m = InMemoryMetrics()
    for _ in range(5):
        m.record("put", 0.001, thread_id="t", ok=True)
    m.record("put", 0.002, thread_id="t", ok=False)
    s = m.summary()
    assert s["put"]["count"] == 6
    assert s["put"]["errors"] == 1
    assert s["put"]["p50_ms"] > 0
    m.reset()
    assert m.summary() == {}


def test_null_metrics_is_a_sink():
    assert isinstance(NullMetrics(), MetricsSink)
    assert NullMetrics().record("put", 0.0, thread_id="t", ok=True) is None


def test_saver_reports_ops_to_sink():
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    from swarmstate.integrations.langgraph import SwarmStateSaver

    b = StateGraph(State)
    b.add_node("inc", lambda s: {"count": 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)

    metrics = InMemoryMetrics()
    saver = SwarmStateSaver(metrics=metrics)
    graph = b.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"count": 0}, cfg)
    graph.get_state(cfg)

    s = metrics.summary()
    assert s.get("put", {}).get("count", 0) >= 1
    assert s.get("get_tuple", {}).get("count", 0) >= 1
    # Every recorded op succeeded.
    assert all(v["errors"] == 0 for v in s.values())


def test_no_metrics_by_default_has_no_sink():
    pytest.importorskip("langgraph")
    from swarmstate.integrations.langgraph import SwarmStateSaver

    assert SwarmStateSaver()._metrics is None
