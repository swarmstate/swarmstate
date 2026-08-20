# LangGraph checkpointer

`SwarmStateSaver` implements LangGraph's `BaseCheckpointSaver` interface — `put`,
`put_writes`, `get_tuple`, `list`, `delete_thread` and the async variants — so it replaces
`SqliteSaver` or `InMemorySaver` in one line.

```bash
pip install "swarmstate[langgraph]"
```

```python
from swarmstate.integrations.langgraph import SwarmStateSaver

graph = builder.compile(checkpointer=SwarmStateSaver())
```

## Why reads stay flat

Resuming a thread means answering "which checkpoint is the newest?". The reference savers
scan the thread's keys for the maximum, so a thread that has run for a week costs more per
resume than a fresh one. `SwarmStateSaver` resolves it by lookup:

- **in-memory and Redis stores** — the saver publishes a *latest pointer* alongside each
  checkpoint, in the same batched write, and reads it in O(1).
- **SQL backends** (`DiskStore`, `PostgresStore`) — no pointer at all: they answer
  `max_key` from the `(ns, key)` primary-key index, which is both always current and one
  row less to write per step.

Either way the answer is `max(checkpoint_id)`, matching `InMemorySaver`, and it is read
from the store — so two savers, or two processes, sharing one backend always agree on the
newest checkpoint.

## Sharing one store

A single `Store` can back several graphs, which makes the whole checkpoint database one
snapshot-able unit:

```python
import swarmstate as ss

store = ss.Store()
saver = SwarmStateSaver(store)

snap = store.snapshot()      # checkpoint the entire checkpoint DB
...                          # run the graphs
store.restore(snap)          # rewind every thread at once
```

## Retention

Checkpointers keep every step by default. For a service that never restarts, that is
unbounded growth:

```python
saver = SwarmStateSaver(max_checkpoints_per_thread=8)
```

Older checkpoints are dropped with their pending writes and, in incremental mode, the
channel blobs no surviving checkpoint still references. On a 300-invocation thread this is
0.5 MB instead of 28 MB.

Two things to know:

- Trimming happens in batches, so a thread sits slightly above the limit before it is cut
  back. The limit is a bound, not an exact length.
- Time travel is limited to the retained window. Size it to the history you actually want
  to be able to rewind to.

## Incremental mode

By default each checkpoint stores its whole channel payload. With `incremental=True`, each
channel value is stored once per version and the checkpoint references it:

```python
saver = SwarmStateSaver(incremental=True)
```

This pays off on long threads with large, mostly-stable channels (a growing message
history that changes by one message per step). The cost is one extra read per channel on
`get_tuple`.

## Async

The async methods offload to a worker thread. Since the store releases the GIL on its hot
paths, that work overlaps with the event loop instead of blocking it:

```python
await graph.ainvoke({"messages": [...]}, config)
state = await graph.aget_state(config)
```

## Durability

`SwarmStateSaver()` defaults to an in-memory `Store`, which does not survive a restart.
Pass a [persistent backend](backends.md) for durable checkpoints — no other code changes:

```python
from swarmstate.backends.disk import DiskStore

saver = SwarmStateSaver(DiskStore("checkpoints.db"))
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `store` | `Store()` | The backing store; shareable across savers and graphs. |
| `serde` | LangGraph's default | Serializer for checkpoint payloads. |
| `incremental` | `False` | Store channel values once per version instead of per step. |
| `max_checkpoints_per_thread` | `None` | Keep only the newest N checkpoints per thread. |
| `metrics` | `None` | A [metrics sink](observability.md) for per-operation latency. |
| `tracer` | `None` | An OpenTelemetry tracer; each operation becomes a span. |
