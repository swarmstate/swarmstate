# Your LangGraph checkpointer gets slower as the thread grows. Ours doesn't.

*Draft launch post. Target: HN / a technical blog. Keep it short, lead with the number,
end with a one-line install. No hype, just the benchmark and the diff.*

---

If you run multi-agent systems in production, you have probably filed some version of
this ticket: **checkpoint latency**. LangGraph checkpoints every super-step, and every
resume has to find the newest checkpoint of a thread. The reference savers find it by
scanning that thread's keys, so the cost of resuming a conversation grows with its
length — the longer a thread lives, the more each step pays.

I built **swarmstate**: a state and checkpointing backend with a **Rust core** and a
Python API. It is not another agent framework and it does not replace LangGraph or
CrewAI. It sits *underneath* them, the way DuckDB, Arrow or Polars sit under data apps.
It does three things.

## 1. A drop-in LangGraph checkpointer whose read latency does not drift

`SwarmStateSaver` implements LangGraph's real `BaseCheckpointSaver` interface, so it is a
drop-in for `SqliteSaver`:

```python
from swarmstate.integrations.langgraph import SwarmStateSaver

graph = builder.compile(checkpointer=SwarmStateSaver())   # was: SqliteSaver(...)
```

Reading the latest checkpoint takes the same time no matter how long the thread is
(Apple Silicon M-series, Python 3.14, release build, warm cache, `benchmarks/run.py`,
3 000 iterations):

| checkpoints in the thread | `SwarmStateSaver` `get_tuple` p50 | `InMemorySaver` `get_tuple` p50 |
| --- | --- | --- |
| 5 | 7.4 µs | 5.5 µs |
| 50 | 7.3 µs | 6.2 µs |
| 500 | 7.3 µs | 13.9 µs |
| 2 000 | **7.5 µs** | 40.0 µs |

swarmstate resolves "latest" by lookup — an O(1) pointer for in-memory stores, an indexed
`max(key)` for the SQL-backed ones — instead of scanning. Flat line versus a rising one.

On writes, be clear about what is being compared. Durable against durable, both SQLite
files in WAL mode:

| Checkpointer (durable) | `put` p50 | `put` p99 | `get_tuple` p50 |
| --- | --- | --- | --- |
| **`SwarmStateSaver(DiskStore(...))`**, `synchronous=NORMAL` | **34.1 µs** | 87.6 µs | 15.7 µs |
| `SqliteSaver`, same `synchronous=NORMAL` | 34.7 µs | 75.6 µs | 14.5 µs |
| `SqliteSaver`, shipped `synchronous=FULL` | 63.9 µs | 211.9 µs | 14.4 µs |

So: **on par with `SqliteSaver` when both fsync the same way**, and about 1.9× faster than
its shipped default — which buys stronger durability, not nothing. If you have seen
"10x faster than SqliteSaver" claims for a state backend (including in an earlier draft of
this post), they compare an in-memory store against a file-backed one. That measures the
price of persistence, not the implementation.

Purely in memory, against LangGraph's own in-memory saver, the trade is explicit: writes
are **slower** (6.2 µs vs 4.1 µs) because swarmstate serializes to msgpack bytes on the
way in, and that is exactly what buys portable state, cheap snapshots and the flat read
line above.

## 2. O(1) snapshots, so time-travel is free

The store uses persistent (structurally-shared) data structures, so `snapshot()` is O(1)
regardless of state size, versus an O(n) `deepcopy`:

| entries in state | `Store.snapshot()` | `dict` deepcopy | speedup |
| --- | --- | --- | --- |
| 1,000 | 0.00045 ms | 0.82 ms | ~1,800x |
| 50,000 | 0.00047 ms | 50.2 ms | **~107,000x** |

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
The full, reproducible benchmark — method, hardware and raw numbers: **`benchmarks/`**.

MIT licensed.
