# swarmstate

> Drop-in state backend for LangGraph, CrewAI & custom agent loops — Rust core, framework-agnostic, built for production.

`swarmstate` is a **state and checkpointing backend** with a Rust core and a Python API for multi-agent
systems. It is not an orchestration framework: it is the fast engine that sits *underneath* LangGraph,
CrewAI, and custom agent loops — the same way a fast columnar engine sits underneath data workloads.

It solves three production pains:

1. **State lock-in across frameworks** — a framework-agnostic store so migrating frameworks doesn't lose state.
2. **Checkpointing cost and latency** — a Rust-backed implementation of LangGraph's checkpointer interface.
3. **Deterministic routing paid for in tokens** — a native handoff graph that resolves rule-based transitions in microseconds.

## Installation

```bash
pip install swarmstate            # prebuilt abi3 wheels, no compiler required
```

Optional extras: `swarmstate[langgraph]`, `swarmstate[crewai]`, `swarmstate[redis]`, `swarmstate[all]`.

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

## Status

Early development.

- **M0 (scaffolding)** ✅ — Rust core builds; `import swarmstate` works.
- **M1 (Rust store)** ✅ — concurrent KV store, msgpack codec, O(1) immutable snapshots,
  incremental diffs, GIL released on hot paths.
- **M2 (HandoffGraph)** ✅ — deterministic conditional routing with a safe Rust condition
  evaluator (no `eval`), cycle detection.
- **M3 (LangGraph adapter)** ✅ — `SwarmStateSaver`, a drop-in `BaseCheckpointSaver`
  backed by the `Store`; snapshot/roll back the whole checkpoint DB at once.
- **M4 (Benchmarks)** — next.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release     # compile the Rust core and install it locally
cargo test                    # Rust core tests
pytest -q                     # Python API tests
```

## License

MIT
