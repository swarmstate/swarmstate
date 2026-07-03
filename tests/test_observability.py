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

    saver = SwarmStateSaver()
    assert saver._metrics is None
    assert saver._tracer is None


# --- tracing --------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class _FakeTracer:
    """Minimal OTel-shaped tracer so we can assert spans without opentelemetry."""

    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = _FakeSpan(name)
        self.spans.append(span)

        class _Ctx:
            def __enter__(self):
                return span

            def __exit__(self, *exc):
                return False

        return _Ctx()


def _build_graph(saver):
    from langgraph.graph import END, START, StateGraph

    b = StateGraph(State)
    b.add_node("inc", lambda s: {"count": 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)
    return b.compile(checkpointer=saver)


def test_saver_opens_spans_with_attributes():
    pytest.importorskip("langgraph")
    from swarmstate.integrations.langgraph import SwarmStateSaver

    tracer = _FakeTracer()
    saver = SwarmStateSaver(tracer=tracer)
    graph = _build_graph(saver)
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"count": 0}, cfg)
    graph.get_state(cfg)

    names = {s.name for s in tracer.spans}
    assert "swarmstate.checkpoint.put" in names
    assert "swarmstate.checkpoint.get_tuple" in names
    put_span = next(s for s in tracer.spans if s.name == "swarmstate.checkpoint.put")
    assert put_span.attributes.get("swarmstate.thread_id") == "t1"
    assert "swarmstate.checkpoint_id" in put_span.attributes


def test_metrics_and_tracer_compose():
    pytest.importorskip("langgraph")
    from swarmstate.integrations.langgraph import SwarmStateSaver

    tracer = _FakeTracer()
    metrics = InMemoryMetrics()
    saver = SwarmStateSaver(metrics=metrics, tracer=tracer)
    _build_graph(saver).invoke({"count": 0}, {"configurable": {"thread_id": "t2"}})

    assert tracer.spans  # spans opened
    assert metrics.summary().get("put", {}).get("count", 0) >= 1  # and timed


def test_span_records_error(monkeypatch):
    pytest.importorskip("langgraph")
    from swarmstate.integrations.langgraph import SwarmStateSaver

    tracer = _FakeTracer()
    saver = SwarmStateSaver(tracer=tracer)

    # Force the underlying write to fail; the span must capture it and re-raise.
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(saver, "_get_tuple_impl", boom)
    with pytest.raises(RuntimeError):
        saver.get_tuple({"configurable": {"thread_id": "t3"}})

    span = tracer.spans[-1]
    assert span.exceptions and isinstance(span.exceptions[0], RuntimeError)


def test_get_tracer_needs_otel():
    from swarmstate.observability import get_tracer

    otel = pytest.importorskip("opentelemetry")  # noqa: F841
    assert get_tracer() is not None
