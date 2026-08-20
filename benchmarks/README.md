# Benchmarks

Reproducible benchmarks for swarmstate.

```bash
pip install -e ".[langgraph,disk]" langgraph-checkpoint-sqlite matplotlib
python benchmarks/run.py --iters 5000 --seed 7
```

Outputs `results.json` and four SVG charts under `benchmarks/charts/`.

## What is measured

1. **Checkpointer latency** (LangGraph `BaseCheckpointSaver` interface), split into two
   groups that are never mixed:
   - **durable** — `SwarmStateSaver` over `DiskStore` vs `SqliteSaver`. Both are SQLite
     files in WAL mode. The headline pairs them at the same `synchronous=NORMAL` fsync
     policy; `SqliteSaver` is also measured at its shipped `synchronous=FULL`, which is a
     stronger guarantee and costs about twice as much per write.
   - **in-memory** — `SwarmStateSaver` vs `InMemorySaver`. Neither survives a restart.
2. **Read latency as a thread grows** — `get_tuple` p50 at 5 / 50 / 500 / 2000 checkpoints
   in one thread. This is the "resume this thread" path, and it separates savers that find
   the latest checkpoint by scanning from those that look it up.
3. **Snapshot cost vs state size** — `Store.snapshot()` vs `copy.deepcopy` of an
   equivalent dict: the two ways to get an independent, mutable copy of state.
4. **Concurrency scaling** — set+get throughput vs thread count, against a dict behind a
   `Lock`.

## Notes on methodology

- **Durable is compared with durable.** An in-memory saver measured against a file-backed
  one is measuring the cost of persistence, not the quality of an implementation. Earlier
  revisions of this file reported such a pairing as a speedup; the durable groups above
  replace it.
- **Same fsync policy, or say so.** `DiskStore` runs WAL + `synchronous=NORMAL`;
  `SqliteSaver` ships WAL + `synchronous=FULL`. Both settings appear in the table so the
  difference between "faster" and "weaker durability" stays visible.
- **Build matters.** Numbers are from a **release** build (`maturin develop --release` or
  the published wheels). Debug builds are several times slower.
- **Warm cache**, single process, no competing load. Hardware, versions, seed and payload
  are recorded in `results.json` — always read them alongside any number.
- Percentiles come from per-call `perf_counter` samples: p50 and p99 of the individual
  operations, not an average of a batch.
- No number here is hand-picked; regenerate with the command above.

## What the numbers say

On the machine recorded in `results.json`, in short:

- Durable writes land **on par with `SqliteSaver`** at a matched fsync policy, and about
  **1.9× faster** than its shipped default (which buys stronger durability).
- Reading the latest checkpoint is **constant time** — flat at ~7 µs whether the thread
  holds 5 or 2000 checkpoints, where `InMemorySaver` grows from ~5 µs to ~37 µs, because
  it scans the thread's keys for a maximum.
- In-memory writes are **slower** than `InMemorySaver` (~0.66×): swarmstate serializes to
  msgpack bytes on the way in, which is what makes state portable and snapshots cheap.
- `Store.snapshot()` is **O(1)** — ~0.5 µs at any size, against 51 ms to `deepcopy` a
  50 000-entry state.
