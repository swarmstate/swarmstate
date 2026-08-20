# swarmstate

> Drop-in state backend for LangGraph, CrewAI & custom agent loops - Rust core, framework-agnostic, built for production.

> **Constant-time checkpoint reads** — `get_tuple` stays at ~7 µs whether a thread holds 5 or
> 2 000 checkpoints, where LangGraph's `InMemorySaver` climbs from ~5 µs to ~40 µs — and **O(1)**
> state snapshots: ~0.5 µs at any size, against 50 ms to `deepcopy` a 50 000-entry state.
> Durable writes land on par with `SqliteSaver` at the same fsync policy (~1.9× faster than
> its shipped default, which buys stronger durability).
> Method, hardware and raw numbers → **[`benchmarks/`](benchmarks/)**.

`swarmstate` is a **state and checkpointing backend** with a Rust core and a Python API for multi-agent
systems. It does not compete with visible agent frameworks; it acts as low-level infrastructure - much
like engines such as DuckDB, ClickHouse, Arrow, or Polars sit underneath data applications without
replacing them.

It solves three production pains:

1. **State lock-in across frameworks** - a framework-agnostic store so migrating frameworks doesn't lose state.
2. **Checkpoint reads that get slower as threads grow** - a Rust-backed implementation of LangGraph's
   checkpointer interface that resolves "the latest checkpoint" by lookup instead of scanning a thread's keys.
3. **Deterministic routing paid for in tokens** - a native handoff graph that resolves rule-based transitions in microseconds.

## Installation

```bash
pip install swarmstate            # prebuilt abi3 wheels, no compiler required
uv add swarmstate                 # or with uv
```

Optional extras: `swarmstate[langgraph]`, `swarmstate[crewai]`, `swarmstate[redis]`,
`swarmstate[disk]`, `swarmstate[postgres]`, `swarmstate[otel]`, `swarmstate[all]`.

## Usage

```python
import swarmstate as ss

store = ss.Store()                              # in-memory, msgpack codec
store.set("workflow", "onboarding", {"step": 3, "data": {...}})
snap = store.snapshot()                          # cheap, immutable snapshot
store.set("workflow", "onboarding", {"step": 4})
store.restore(snap)                              # rollback
store.get("workflow", "onboarding")              # -> {"step": 3, "data": {...}}

snap2 = store.snapshot()
snap2.diff(snap)                                 # {"added": [...], "removed": [...], "changed": [...]}

# Retention is opt-in: a snapshot the store keeps pins the state it saw
hist = ss.Store(max_history=10)                  # 0 (default) keeps none, None keeps all
hist.history()                                   # -> [Snapshot, ...], oldest first

# Batch ops: one GIL release / round-trip for the whole set
store.set_many([("workflow", "a", {...}), ("workflow", "b", {...})])
store.get_many([("workflow", "a"), ("workflow", "b")])   # -> [..., ...], order preserved

# Deterministic, LLM-free routing (resolved natively in Rust)
g = ss.HandoffGraph()
g.add_edge("triage", "billing", when="category == 'billing'")
g.add_edge("triage", "human")                    # unconditional default
g.route("triage", {"category": "billing"})       # -> "billing"
```

Drop-in LangGraph checkpointer (`pip install "swarmstate[langgraph]"`):

```python
from swarmstate.integrations.langgraph import SwarmStateSaver

graph = builder.compile(checkpointer=SwarmStateSaver())   # replaces SqliteSaver, 1 line
```

Bounded memory for long-running threads — checkpointers keep every step by
default, which for a service that never restarts means growth without end:

```python
saver = SwarmStateSaver(max_checkpoints_per_thread=8)     # keep the newest N per thread
```

Older checkpoints are dropped with their pending writes and channel blobs. On a
300-invocation thread that is **0.5 MB instead of 28 MB**, and the thread still
resumes; time travel is limited to the retained window, so size it to taste.

Optional metrics on checkpoint operations (opt-in, zero overhead when unused):

```python
from swarmstate.observability import InMemoryMetrics       # or OpenTelemetryMetrics

metrics = InMemoryMetrics()
saver = SwarmStateSaver(metrics=metrics)
# ... run the graph ...
metrics.summary()   # {"put": {"count": 12, "p50_ms": 0.006, ...}, "get_tuple": {...}}
```

OpenTelemetry tracing (each checkpoint op becomes a `swarmstate.checkpoint.<op>` span):

```python
from swarmstate.observability import get_tracer     # needs swarmstate[otel]

saver = SwarmStateSaver(tracer=get_tracer())         # composes with metrics=...
```

## Status

Early development.

- **M0 (scaffolding)** ✅ - Rust core builds; `import swarmstate` works.
- **M1 (Rust store)** ✅ - concurrent KV store, msgpack codec, O(1) immutable snapshots,
  incremental diffs, GIL released on hot paths.
- **M2 (HandoffGraph)** ✅ - deterministic conditional routing with a safe Rust condition
  evaluator (no `eval`), cycle detection.
- **M3 (LangGraph adapter)** ✅ - `SwarmStateSaver`, a drop-in `BaseCheckpointSaver`
  backed by the `Store`; snapshot/roll back the whole checkpoint DB at once.
- **M4 (Benchmarks)** ✅ - durable-vs-durable and in-memory-vs-in-memory comparisons on
  LangGraph's interface, read latency as a thread grows, `Store.snapshot()` vs `deepcopy`,
  and concurrency scaling. Reproducible: [`benchmarks/run.py`](benchmarks/run.py); method
  and results in [`benchmarks/README.md`](benchmarks/README.md).
- **M5 (CrewAI adapter + backends)** ✅ - persistent, drop-in checkpointer backends
  `RedisStore`, `DiskStore` (SQLite) and `PostgresStore`, all msgpack wire-format, plus
  `SwarmStateStorage` (portable memory backed by a shared `Store`).
- **M6 (docs · wheels · PyPI)** ✅ - full docs site, benchmarks, cross-platform abi3
  wheels, and PyPI publishing via Trusted Publishing (OIDC).
- **Observability** ✅ - opt-in metrics hooks and OpenTelemetry **tracing** on checkpoint
  ops (`put` / `put_writes` / `get_tuple`): an in-memory sink, an OpenTelemetry metrics
  sink, and per-op spans (`swarmstate[otel]`). Zero overhead when unused. Strict `mypy` in CI.
- **Free-threaded (no-GIL) ready** ✅ - the Rust core declares free-threaded support, so on
  a free-threaded CPython build (`cp313t`) the store **doesn't collapse under threads the way
  the GIL build does**: on a set+get workload at 8 threads it sustains **~1.8M ops/s vs ~130k
  on GIL Python (over 10x)**, where the GIL build gets *much slower* as threads are added.
  (These workloads are allocation-bound, so neither scales linearly with cores; the win is
  avoiding the GIL's collapse.) Version-specific `cp313t` and `cp314t` wheels ship for Linux
  (x86_64/aarch64), macOS (arm64) and Windows (x64) alongside the abi3 ones.
- **Batch API** ✅ - `Store.set_many` / `get_many` (and on every backend) amortize the
  per-call overhead over a batch: one GIL release for the in-memory core, one round-trip for
  networked backends. On free-threaded at 8 threads, `set_many` is ~3x the throughput of
  individual sets. `SwarmStateSaver` uses it internally: `put_writes` (and the `incremental`
  channel blobs) flush all writes of a step in a single `set_many`, so fan-out steps that emit
  many pending writes pay one lock/round-trip instead of one per write.

## Examples

Runnable, offline, deterministic demos in [`examples/`](examples/):

- [`support_triage.py`](examples/support_triage.py) - a LangGraph workflow tying together
  `HandoffGraph` routing, `SwarmStateSaver` checkpointing and snapshot/restore time-travel.
- [`state_portability.py`](examples/state_portability.py) - state as standard msgpack
  bytes, read back and cross-checked against the `msgpack` package.

## Documentation

Guide and API reference: **[swarmstate.github.io/swarmstate](https://swarmstate.github.io/swarmstate/)**
— the [store and snapshots](docs/guide/store.md), the
[LangGraph checkpointer](docs/guide/langgraph.md), the
[persistent backends](docs/guide/backends.md), the
[handoff graph](docs/guide/handoff-graph.md), and the
[benchmark method](docs/benchmarks.md).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release     # compile the Rust core and install it locally
cargo test                    # Rust core tests
pytest -q                     # Python API tests

pip install -e ".[docs]"
./scripts/build-docs.sh --serve   # docs at http://127.0.0.1:8000
```

## Citation

If you use swarmstate in academic work, please cite it:

```bibtex
@software{salmeron_swarmstate,
  author    = {Salmeron, Jose L.},
  title     = {{swarmstate}: A state and checkpointing backend for multi-agent
               systems with a Rust core},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXXX},
  url       = {https://github.com/swarmstate/swarmstate}
}
```

> The DOI is minted when the first release is archived on Zenodo; replace the
> placeholder in this block, in [`docs/index.md`](docs/index.md) and in
> [`CITATION.cff`](CITATION.cff) once it exists.

## License

MIT
