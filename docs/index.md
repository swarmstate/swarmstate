# swarmstate

Drop-in state backend for LangGraph, CrewAI & custom agent loops — Rust core,
framework-agnostic, built for production.

`swarmstate` is a **state and checkpointing backend**, not an agent framework. It sits
underneath LangGraph, CrewAI or your own loop the way DuckDB or Arrow sit underneath data
applications: you keep your orchestration, it takes over storing and moving the state.

## Install

```bash
pip install swarmstate            # prebuilt abi3 wheels, no compiler required
uv add swarmstate                 # or with uv
```

Optional extras: `swarmstate[langgraph]`, `[crewai]`, `[redis]`, `[disk]`, `[postgres]`,
`[otel]`, `[all]`.

## Three things it does

**A LangGraph checkpointer whose read latency does not drift.** Resuming a thread means
finding its newest checkpoint. The reference savers scan the thread's keys for it, so the
cost grows with the conversation; `swarmstate` looks it up instead — flat at ~7 µs whether
the thread holds 5 checkpoints or 2 000.

```python
from swarmstate.integrations.langgraph import SwarmStateSaver

graph = builder.compile(checkpointer=SwarmStateSaver())   # was: SqliteSaver(...)
```

**A store with O(1) snapshots.** State lives in persistent (structurally shared) data
structures, so a snapshot of everything costs the same at any size — ~0.5 µs against 50 ms
to `deepcopy` a 50 000-entry state. One `restore()` rewinds every thread at once.

```python
import swarmstate as ss

store = ss.Store()
store.set("workflow", "onboarding", {"step": 3})
snap = store.snapshot()          # cheap, immutable
store.set("workflow", "onboarding", {"step": 4})
store.restore(snap)              # back to step 3
```

**Deterministic routing that spends no tokens.** Many "which agent goes next" decisions are
rules over the state, not judgment calls.

```python
g = ss.HandoffGraph()
g.add_edge("triage", "billing", when="category == 'billing'")
g.add_edge("triage", "human")                      # unconditional default
g.route("triage", {"category": "billing"})          # -> "billing", resolved in Rust
```

## What it does not do

- **It is not faster at writing than SQLite.** Durable writes land on par with
  `SqliteSaver` at a matched fsync policy; in-memory writes are *slower* than LangGraph's
  `InMemorySaver`, because state is serialized to msgpack on the way in. That
  serialization is what makes the state portable and the snapshots cheap. The
  [benchmarks](benchmarks.md) page has the full table, including the losses.
- **It is not durable by default.** `Store()` is in-memory; point the saver at a
  [persistent backend](guide/backends.md) for state that survives a restart.
- **It does not do semantic search.** The CrewAI adapter is lexical recall, not a vector
  store — see [CrewAI memory](guide/crewai.md).

## Citing swarmstate

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

!!! note "DOI pending"
    The DOI above is a placeholder until the first release is archived on Zenodo.
    `CITATION.cff` in the repository carries the same metadata, which is what
    GitHub's "Cite this repository" button and Zenodo read.

## Where to go next

- [Store & snapshots](guide/store.md) — the framework-agnostic core.
- [LangGraph checkpointer](guide/langgraph.md) — the drop-in, retention, incremental mode.
- [Persistent backends](guide/backends.md) — SQLite file, Redis, Postgres.
- [Benchmarks](benchmarks.md) — method, hardware, numbers.
