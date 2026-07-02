# Benchmarks

Reproducible benchmarks for swarmstate.

```bash
pip install -e ".[langgraph]" langgraph-checkpoint-sqlite matplotlib
python benchmarks/run.py --iters 5000 --seed 7
```

Outputs `results.json` and two SVG charts under `benchmarks/charts/`.

## What is measured

1. **Checkpointer latency** (LangGraph `BaseCheckpointSaver` interface): `put` and
   `get_tuple` p50/p99 and throughput for `SwarmStateSaver`, `InMemorySaver`, and
   `SqliteSaver`.
2. **Snapshot cost vs state size**: `Store.snapshot()` vs `copy.deepcopy` of an
   equivalent dict — the two ways to get an independent, mutable copy of state.

## Notes on methodology

- **Build matters.** Numbers are from a **release** build (`maturin develop --release`
  or the published wheels). Debug builds are several times slower.
- **In-memory vs persistent.** `SwarmStateSaver` and `InMemorySaver` are in-memory;
  `SqliteSaver` is **file-backed** (it persists to disk, they don't). The `put`
  comparison reflects the cost of SQLite-backed persistence as commonly configured. A
  persistent swarmstate backend (redis/disk) is on the roadmap (M5).
- **Warm cache**, single process. Hardware, versions, seed and payload are recorded in
  `results.json` — always read them alongside any number.
- No number here is hand-picked; regenerate with the command above.
