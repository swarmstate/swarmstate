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

## Status

Early development. Milestone **M0 (scaffolding)** is complete: the Rust core builds and
`import swarmstate` works. Store (M1), HandoffGraph (M2), and the LangGraph adapter (M3) are next.

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
