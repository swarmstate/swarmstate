"""Optional metrics hooks for checkpoint operations.

The checkpointer ([`SwarmStateSaver`][swarmstate.integrations.langgraph.SwarmStateSaver])
can report the latency and outcome of each `put` / `put_writes` / `get_tuple` to a
**metrics sink**. This is opt-in and has **zero overhead when unused** (the default is
no sink at all).

A sink is anything with a ``record`` method::

    def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None

Three sinks ship here:

- [`NullMetrics`][swarmstate.observability.NullMetrics] - the no-op default.
- [`InMemoryMetrics`][swarmstate.observability.InMemoryMetrics] - accumulates counts and
  timings in process; handy for tests, notebooks and quick profiling.
- [`OpenTelemetryMetrics`][swarmstate.observability.OpenTelemetryMetrics] - emits an
  OpenTelemetry histogram + counter (lazy import; needs the ``[otel]`` extra).

Example::

    from swarmstate.integrations.langgraph import SwarmStateSaver
    from swarmstate.observability import InMemoryMetrics

    metrics = InMemoryMetrics()
    saver = SwarmStateSaver(metrics=metrics)
    ...
    print(metrics.summary())   # {'put': {'count': 12, 'p50_ms': 0.006, ...}, ...}
"""

from __future__ import annotations

import statistics
import threading
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsSink(Protocol):
    """Anything that can receive a per-operation measurement."""

    def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None:
        """Record one checkpoint operation.

        Args:
            op: operation name (``"put"``, ``"put_writes"``, ``"get_tuple"``).
            duration_s: wall-clock duration in seconds.
            thread_id: the LangGraph thread the operation belongs to.
            ok: ``True`` if the operation succeeded, ``False`` if it raised.
        """


class NullMetrics:
    """A sink that discards everything. Used as the explicit no-op."""

    def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None:
        return None


class InMemoryMetrics:
    """Accumulate per-operation counts and latency samples in process.

    Thread-safe. Useful for tests, notebooks and ad-hoc profiling without pulling
    in a metrics backend. Call :meth:`summary` for percentiles.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}
        self._errors: dict[str, int] = {}

    def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None:
        with self._lock:
            self._samples.setdefault(op, []).append(duration_s * 1000.0)
            if not ok:
                self._errors[op] = self._errors.get(op, 0) + 1

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{op: {count, errors, mean_ms, p50_ms, p99_ms}}``."""
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for op, ms in self._samples.items():
                s = sorted(ms)
                p99 = s[min(len(s) - 1, max(0, round(0.99 * len(s)) - 1))]
                out[op] = {
                    "count": len(s),
                    "errors": self._errors.get(op, 0),
                    "mean_ms": round(statistics.fmean(s), 5),
                    "p50_ms": round(statistics.median(s), 5),
                    "p99_ms": round(p99, 5),
                }
            return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._errors.clear()


class OpenTelemetryMetrics:
    """Emit OpenTelemetry metrics for each checkpoint operation.

    Records a histogram ``swarmstate.checkpoint.duration`` (milliseconds) and a
    counter ``swarmstate.checkpoint.operations``, both tagged with ``op`` and
    ``ok``. ``thread_id`` is deliberately **not** used as an attribute to avoid
    unbounded metric cardinality.

    Requires the ``[otel]`` extra (``pip install "swarmstate[otel]"``); the
    ``opentelemetry`` import is lazy so importing this module never fails.
    """

    def __init__(self, meter: Any = None) -> None:
        if meter is None:
            from opentelemetry import metrics as _otel_metrics

            meter = _otel_metrics.get_meter("swarmstate")
        self._duration = meter.create_histogram(
            name="swarmstate.checkpoint.duration",
            unit="ms",
            description="Latency of swarmstate checkpoint operations.",
        )
        self._ops = meter.create_counter(
            name="swarmstate.checkpoint.operations",
            unit="1",
            description="Count of swarmstate checkpoint operations.",
        )

    def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None:
        attrs = {"op": op, "ok": ok}
        self._duration.record(duration_s * 1000.0, attributes=attrs)
        self._ops.add(1, attributes=attrs)


__all__ = ["InMemoryMetrics", "MetricsSink", "NullMetrics", "OpenTelemetryMetrics"]
