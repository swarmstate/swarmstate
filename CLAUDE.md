# CLAUDE.md — swarmstate

> Master guide for building **swarmstate** from scratch. This file lives at the repository root and is the
> single source of truth for any development session with Claude Code. Read it fully before writing any code.

---

## 0. How to use this file

- **Package name:** `swarmstate` (PyPI + import). If it needs to change, change it **only here** and propagate:
  `PACKAGE_NAME=swarmstate`, `IMPORT_NAME=swarmstate`, `RUST_CRATE=swarmstate_core`, `GH_REPO=<org>/swarmstate`.
- The repository will be hosted under a **dedicated GitHub organization** (e.g. `github.com/swarmstate/...`).
  The org and an **empty repo** are created by the maintainer beforehand; do not scaffold until they exist and
  this file has been placed at the repo root.
- Work in small, verifiable increments. After each milestone, run the test suite and the benchmarks.
- Do not invent LangGraph/CrewAI APIs: whenever you touch their interface, **verify the real signatures** against
  the version pinned in `pyproject.toml` before implementing an adapter.
- The top priority is that `pip install swarmstate` works with **no compiler** (prebuilt `abi3` wheels). Compiling
  Rust is only for development (`maturin develop`).

---

## 1. What swarmstate is (vision)

`swarmstate` is a **state and checkpointing backend with a Rust core and a Python API** for multi-agent systems.
It is not an orchestration framework and does **not** compete with LangGraph, CrewAI, or AutoGen: it is the fast
engine that sits **underneath** them, the same way Polars became the fast engine underneath workloads that used
to run on pandas.

**The pain it solves (production engineering teams):**

1. **State lock-in across frameworks.** Today, if you migrate from CrewAI to LangGraph you lose the accumulated
   state. `swarmstate` provides a store with a framework-agnostic format: any agent reads/writes the same state.
2. **Checkpoint reads that get slower as threads grow.** Every resume has to find a thread's newest
   checkpoint, and the reference savers find it by scanning that thread's keys — so a long-lived
   conversation pays more per step than a fresh one. `swarmstate` implements LangGraph's checkpointer
   interface with a Rust backend and resolves "latest" by lookup (an O(1) pointer in memory, an indexed
   `max(key)` on the SQL backends), so read latency is flat: ~7 µs at 5 checkpoints and at 2 000, where
   `InMemorySaver` climbs from ~5 µs to ~40 µs. Snapshots of the whole checkpoint DB are O(1) via
   structural sharing. **Writes are a different story and must not be oversold:** durable writes land on
   par with `SqliteSaver` at a matched fsync policy, and in-memory writes are *slower* than
   `InMemorySaver` because state is serialized to msgpack — which is what buys the portability and the
   cheap snapshots.
3. **Deterministic routing that is currently paid for in tokens.** Many "which agent gets control next" decisions
   don't need an LLM — they are rules over a dependency graph. `swarmstate` exposes a native Rust handoff graph
   that resolves those transitions in microseconds.

**Target user:** the engineer with a *"checkpoint latency"* ticket or an orchestration bill that doesn't add up.
The sales pitch is **a number in a benchmark**, not an idea — and the number has to survive a reviewer who
compares like with like (durable against durable, same fsync policy). Claims that only hold because an
in-memory store was measured against a file-backed one are worth less than no claim at all.

**Positioning (one line):**
> Drop-in state backend for LangGraph, CrewAI & custom agent loops — Rust core, framework-agnostic, built for production.

---

## 2. Reference library model (turboswarm)

We deliberately mirror the DNA of `turboswarm` (same intended author/style):

- **Rust compute core, Python API** via **PyO3 + maturin**.
- **`cp39-abi3`** cross-platform wheels (Linux x86_64/aarch64, macOS x86_64/arm64, Windows amd64) — one build
  serves Python 3.9+.
- README structure: title + one sentence → **Installation** → **Usage** (short runnable examples) →
  **parameter table** → **Result object** → **Integrations** (drop-in with the existing stack) →
  **Documentation** → **License (MIT)**.
- **Optional extras** with lazy imports: `swarmstate[langgraph]`, `[crewai]`, `[redis]`, `[all]`.
- **Documentation with MkDocs Material**: narrative guide + API reference, served via a `scripts/build-docs.sh --serve` script.
- Clear technical prose, reproducible `seed`-based examples, focus on **variant comparison and visible benchmarks**.
- **MIT** license.

---

## 3. Architecture

```
swarmstate/                      # repo root (GH_REPO)
├─ Cargo.toml                    # Rust core crate (RUST_CRATE = swarmstate_core)
├─ pyproject.toml                # maturin as build backend
├─ rust/                         # or src/ at root depending on maturin layout
│  └─ src/
│     ├─ lib.rs                  # #[pymodule] — exports classes to Python
│     ├─ store.rs                # concurrent KV store + incremental snapshots
│     ├─ checkpoint.rs           # Checkpoint/CheckpointTuple types, serialization
│     ├─ graph.rs                # handoff/dependency graph (DAG), cycle detection
│     └─ codec.rs                # serialization (msgpack/bincode) — fast and stable
├─ python/
│  └─ swarmstate/
│     ├─ __init__.py             # public API: Store, Checkpointer, HandoffGraph
│     ├─ _core.pyi               # type stubs for the Rust module (typing for users)
│     ├─ integrations/
│     │  ├─ __init__.py
│     │  ├─ langgraph.py         # SwarmStateSaver(BaseCheckpointSaver) — DROP-IN, flagship piece
│     │  ├─ crewai.py            # state/memory adapter for CrewAI
│     │  └─ redis.py             # optional persistent backend
│     └─ py.typed
├─ tests/
├─ benchmarks/                   # comparisons vs SqliteSaver / in-memory dict
├─ docs/                         # MkDocs Material
├─ scripts/
│  └─ build-docs.sh
├─ .github/workflows/
│  ├─ ci.yml                     # Rust + Python tests on every push
│  └─ release.yml                # maturin build of abi3 wheels + publish to PyPI (OIDC)
├─ README.md
├─ LICENSE                       # MIT
└─ CLAUDE.md                     # this file
```

**Rust core design principles:**

- Hot logic (serialization, snapshot diffs, graph traversal) lives **entirely in Rust**.
- Release the **GIL** (`py.allow_threads`) on any operation that doesn't touch Python objects.
- State serializes to bytes with a stable codec (**msgpack**) so it is readable from any language/framework.
- The Python API is thin: ergonomic wrappers over the PyO3 classes, with complete type hints.

---

## 4. Target public API (first design; iterate)

```python
import swarmstate as ss

# --- Framework-agnostic store -----------------------------------------------
store = ss.Store()                       # in-memory; ss.Store(backend="redis", url=...) optional
store.set("workflow", "onboarding", {"step": 3, "data": {...}})
snap = store.snapshot()                  # immutable, cheap snapshot
val = store.get("workflow", "onboarding")
store.restore(snap)                      # rollback

# --- Handoff graph (deterministic transitions, no LLM) ----------------------
g = ss.HandoffGraph()
g.add_edge("triage", "billing", when="category == 'billing'")
nxt = g.route("triage", state={"category": "billing"})   # -> "billing", resolved in Rust

# --- Drop-in checkpointer for LangGraph -------------------------------------
from swarmstate.integrations.langgraph import SwarmStateSaver
graph = builder.compile(checkpointer=SwarmStateSaver())   # replaces SqliteSaver, 1 line
```

**Parameter table (turboswarm style) — document like this in the README:**

| Component         | Parameter    | Default        | Description                                             |
| ----------------- | ------------ | -------------- | ------------------------------------------------------- |
| `Store`           | `backend`    | `"memory"`     | `"memory"`, `"redis"` (extra), future `"disk"`          |
| `Store`           | `codec`      | `"msgpack"`    | state serialization (stable across languages)           |
| `Store`           | `max_history`| `0`            | retained snapshots: `0` none, `n` last `n`, `None` all  |
| `HandoffGraph`    | `on_cycle`   | `"error"`      | `"error"` or `"allow"` on cycle detection               |
| `SwarmStateSaver` | `store`      | `Store()`      | underlying store; shareable across graphs               |
| `SwarmStateSaver` | `serde`      | auto           | (de)serialization compatible with LangGraph             |
| `SwarmStateSaver` | `incremental`| `False`        | store channel values once per version instead of per step |
| `SwarmStateSaver` | `max_checkpoints_per_thread` | `None` | retention: keep the newest N per thread (None = all) |

**`Snapshot` object:** `.id`, `.timestamp`, `.keys`, `.size_bytes`, `.parent` (for incremental diffs).

---

## 5. Implementation roadmap (milestones)

Work in this order. Do not move to the next milestone without green tests.

**M0 — Scaffolding**
- [ ] Initialize a maturin project (`maturin new`, mixed Rust+Python layout) with `IMPORT_NAME`.
- [ ] `pyproject.toml` with maturin backend, `abi3-py39`, extras `[langgraph]/[crewai]/[redis]/[docs]/[all]`.
- [ ] `maturin develop` compiles and `import swarmstate` works. A trivial test passes.

**M1 — Rust store**
- [ ] Concurrent KV store (namespace + key -> value bytes) with msgpack codec.
- [ ] Cheap immutable snapshots + `restore()`. Incremental diffs between snapshots.
- [ ] Release the GIL on set/get/snapshot. Concurrency tests.

**M2 — HandoffGraph in Rust**
- [ ] DAG with conditional edges; a simple, safe condition evaluator (no Python `eval`).
- [ ] Deterministic `route()`; cycle detection; traversal in Rust.

**M3 — LangGraph adapter (flagship piece)**
- [ ] `SwarmStateSaver` implements the real `BaseCheckpointSaver` interface
      (`put`, `put_writes`, `get_tuple`, `list`, and async variants). **Verify signatures against the pinned version.**
- [ ] Integration test: a minimal LangGraph graph persists and resumes with `SwarmStateSaver`.

**M4 — Benchmarks (the selling argument)**
- [ ] `benchmarks/`: `SwarmStateSaver` vs `SqliteSaver` (checkpoint latency p50/p99, throughput).
- [ ] Store vs pure in-memory dict and vs Redis. Produce reproducible charts.
- [ ] Compare like with like: durable against durable at a matched fsync policy, in-memory against
      in-memory. Report the losses too (in-memory writes are slower than `InMemorySaver`).
- [ ] The claims to communicate are the ones that survive that: **flat read latency** as a thread grows,
      **O(1) snapshots** of the whole checkpoint DB, and durable writes **on par** with `SqliteSaver`.

**M5 — CrewAI adapter + persistence**
- [ ] CrewAI state/memory adapter that shares the same `Store` (demonstrate state portability).
- [ ] Optional `redis` backend under the extra.

**M6 — Docs + CI + release**
- [ ] MkDocs Material: guide + API reference + benchmarks page.
- [ ] CI: Rust tests (`cargo test`) + Python tests (`pytest`) on Linux/macOS/Windows.
- [ ] `release.yml`: `maturin build` of cross-platform abi3 wheels + publish to PyPI via **Trusted Publishing (OIDC)**.
- [ ] Publish `0.1.0`. Polished README with the benchmark highlighted at the top.

---

## 6. Toolchain and commands

```bash
# development environment
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release        # compiles the Rust core and installs it

# tests
cargo test                       # Rust core
pytest -q                        # Python API + integrations

# benchmarks
python benchmarks/run.py         # generates metrics and charts

# docs
pip install -e ".[docs]"
./scripts/build-docs.sh --serve  # http://127.0.0.1:8000

# release (local sanity check)
maturin build --release          # produces abi3 wheels
```

**Versions/standards:**
- Python `>=3.9`, `cp39-abi3` wheels.
- PyO3 + maturin (latest stable). Rust edition 2021+.
- Complete type hints + `py.typed`. `ruff` for Python-side lint/format.
- Pin `langgraph` and `crewai` versions in the extras and in CI; the adapters depend on their APIs.

---

## 7. Quality rules

- **Zero install friction:** if a user needs a Rust compiler for `pip install`, it's a bug.
- **Reproducibility:** every example and benchmark accepts a `seed`/deterministic config.
- **Never break the drop-in:** `SwarmStateSaver` must be swappable for `SqliteSaver` with no other code change.
- **Honest benchmarks:** document hardware, versions, and whether the cache is warm. No inflated numbers.
- **Stable state format:** the codec must not change incompatibly across minor versions.
- **Security:** `HandoffGraph` conditions are evaluated in a bounded mini-evaluator in Rust, never with `eval()`.

---

## 8. Launch strategy (context, non-blocking)

- The first public artifact should be a **short technical post with the benchmark**, not a buried README:
  "your LangGraph checkpointer gets slower as the thread grows; this one doesn't" — a different asymptote
  is a better story than a speed multiplier, and it holds up under scrutiny.
- Shipping a working end-to-end example adapter is what drives shares on HN/Twitter.
- Publish wheels for all platforms from `0.1.0`: friction kills early traction.

---

## 9. Pre-release checklist

- [ ] `cargo test` and `pytest` green on all 3 platforms.
- [ ] abi3 wheels built for Linux (x86_64/aarch64), macOS (x86_64/arm64), Windows (amd64).
- [ ] `import swarmstate` works in a clean venv **without** a Rust toolchain.
- [ ] README examples run without error.
- [ ] Benchmark updated and linked from the README.
- [ ] Version bumped in `Cargo.toml` and `pyproject.toml` (must match).
- [ ] Docs deployed.