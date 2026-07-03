# Swap your LangGraph checkpointer and cut write latency ~12x

*Draft launch post. Target: HN / a technical blog. Keep it short, lead with the number,
end with a one-line install. No hype, just the benchmark and the diff.*

---

If you run multi-agent systems in production, you have probably filed some version of
this ticket: **checkpoint latency**. LangGraph checkpoints every super-step, and the
default persistent option, `SqliteSaver`, commits to disk on every one of them. At scale
that per-step commit shows up in your p99.

I built **swarmstate**: a state and checkpointing backend with a **Rust core** and a
Python API. It is not another agent framework and it does not replace LangGraph or
CrewAI. It sits *underneath* them, the way DuckDB, Arrow or Polars sit under data apps.
It does three things.

## 1. A faster LangGraph checkpointer, one line

`SwarmStateSaver` implements LangGraph's real `BaseCheckpointSaver` interface, so it is a
drop-in for `SqliteSaver`:

```python
from swarmstate.integrations.langgraph import SwarmStateSaver

graph = builder.compile(checkpointer=SwarmStateSaver())   # was: SqliteSaver(...)
```

Per-operation latency through that interface (Apple Silicon, Python 3.14, warm cache,
seed 7, reproducible via `benchmarks/run.py`):

| Checkpointer | `put` p50 | `put` throughput | `get_tuple` p50 |
| --- | --- | --- | --- |
| **SwarmStateSaver** | **5.8 µs** | **~158k ops/s** | **7.5 µs** |
| InMemorySaver | 4.0 µs | ~203k ops/s | 81.8 µs |
| SqliteSaver (file) | 74.0 µs | ~9.5k ops/s | 14.5 µs |

**`put` is ~12.8x faster than `SqliteSaver`** (no per-step disk commit), and `get_tuple`
of the latest checkpoint is ~11x faster than the reference in-memory saver, because
swarmstate keeps an O(1) "latest" pointer instead of scanning keys.

The caveat, stated up front: `SwarmStateSaver` is in-memory by default, so this
`put` comparison is against SQLite's persistence cost. If you need durability, point it at
a persistent backend (SQLite file, Redis, or Postgres) with no other code change:

```python
from swarmstate.backends.postgres import PostgresStore
saver = SwarmStateSaver(PostgresStore("postgresql://user:pass@host/db"))
```

## 2. O(1) snapshots, so time-travel is free

The store uses persistent (structurally-shared) data structures, so `snapshot()` is O(1)
regardless of state size, versus an O(n) `deepcopy`:

| entries in state | `Store.snapshot()` | `dict` deepcopy | speedup |
| --- | --- | --- | --- |
| 1,000 | 0.0002 ms | 0.86 ms | ~4,000x |
| 50,000 | 0.0002 ms | 52 ms | **~236,000x** |

One `snapshot()` captures *every thread in the checkpoint DB at once*, and one `restore()`
rolls the whole system back. That is the basis for cheap rewind/replay.

## 3. Deterministic routing that does not spend tokens

Many "which agent gets control next" decisions are rules over the state, not judgment
calls. Paying an LLM for them is slow and non-deterministic. `HandoffGraph` resolves them
natively in Rust with a bounded, safe condition evaluator (no `eval`):

```python
import swarmstate as ss

g = ss.HandoffGraph()
g.add_edge("triage", "billing", when="category == 'billing'")
g.add_edge("triage", "technical", when="category == 'technical' and priority >= 2")
g.add_edge("triage", "human")                     # default fallback
g.route("triage", {"category": "billing"})        # -> "billing", in microseconds
```

## Why Rust, why msgpack

Hot paths (serialization, snapshot diffs, graph traversal) live entirely in Rust, and the
GIL is released on operations that do not touch Python objects. State serializes to plain
**msgpack** bytes, a stable, cross-language format, so state written by one framework can
be read by another. No lock-in, no bespoke format.

## The end-to-end demo

The [`examples/support_triage.py`](../examples/support_triage.py) demo wires all three
together into one small LangGraph workflow: HandoffGraph decides the route, SwarmStateSaver
checkpoints each step, and a snapshot/restore rewinds the entire checkpoint DB. It runs
offline, no API keys, deterministic output.

## Install

```bash
pip install swarmstate            # prebuilt abi3 wheels, no compiler
uv add swarmstate                 # or with uv
```

`cp39-abi3` wheels for Linux (x86_64/aarch64), macOS (x86_64/arm64) and Windows.
Docs and the full, reproducible benchmark: **https://swarmstate.github.io**.

MIT licensed.
