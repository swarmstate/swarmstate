# Metrics & tracing

The checkpointer can report the latency and outcome of every `put` / `put_writes` /
`get_tuple`, and run each inside an OpenTelemetry span. Both are opt-in and cost nothing
when unused — the uninstrumented path does not even allocate a timer.

## Metrics

A sink is anything with a `record` method:

```python
def record(self, op: str, duration_s: float, *, thread_id: str, ok: bool) -> None: ...
```

Three ship with the package:

```python
from swarmstate.observability import InMemoryMetrics
from swarmstate.integrations.langgraph import SwarmStateSaver

metrics = InMemoryMetrics()
saver = SwarmStateSaver(metrics=metrics)

# ... run the graph ...
metrics.summary()
# {"put": {"count": 12, "errors": 0, "mean_ms": 0.007, "p50_ms": 0.006, "p99_ms": 0.014}, ...}
```

- `InMemoryMetrics` — accumulates counts and percentiles in process. Thread-safe; good for
  tests, notebooks and quick profiling.
- `OpenTelemetryMetrics` — a histogram `swarmstate.checkpoint.duration` (ms) and a counter
  `swarmstate.checkpoint.operations`, tagged with `op` and `ok`. `thread_id` is
  deliberately *not* an attribute, to keep cardinality bounded. Needs `swarmstate[otel]`.
- `NullMetrics` — the explicit no-op.

Failures are recorded too: a sink sees `ok=False` for an operation that raised, and the
exception propagates unchanged.

## Tracing

```python
from swarmstate.observability import get_tracer

saver = SwarmStateSaver(tracer=get_tracer())      # needs swarmstate[otel]
```

Each operation becomes a `swarmstate.checkpoint.<op>` span carrying
`swarmstate.thread_id`, `swarmstate.checkpoint_ns` and `swarmstate.checkpoint_id`; a
failure records the exception and sets the span status to ERROR. Instrumentation never
masks the real error — a misbehaving span is swallowed, the original exception is not.

Metrics and tracing compose:

```python
saver = SwarmStateSaver(metrics=InMemoryMetrics(), tracer=get_tracer())
```
