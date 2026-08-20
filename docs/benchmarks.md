# Benchmarks

Every number here is reproducible with the script in the repository, and every comparison
pairs like with like — durable against durable, in-memory against in-memory. An in-memory
saver measured against a file-backed one is measuring the cost of persistence, not the
quality of an implementation.

```bash
pip install -e ".[langgraph,disk]" langgraph-checkpoint-sqlite matplotlib
python benchmarks/run.py --iters 5000 --seed 7
```

The run below: Apple Silicon, macOS, Python 3.14, release build, swarmstate 0.10.4,
3 000 iterations after 300 warm-up, warm cache, single process, no competing load. The full
output, including hardware and versions, is written to `benchmarks/results.json`.

## Reading the latest checkpoint, as a thread grows

The operation every resume performs. A saver that finds the newest checkpoint by scanning
the thread's keys gets slower as the thread lives; one that looks it up does not.

| checkpoints in the thread | `SwarmStateSaver` | `InMemorySaver` |
| --- | --- | --- |
| 5 | 7.4 µs | 5.5 µs |
| 50 | 7.3 µs | 6.2 µs |
| 500 | 7.3 µs | 13.9 µs |
| 2 000 | **7.5 µs** | 40.0 µs |

Flat versus rising: at five checkpoints the reference saver is faster, and by two thousand
it is five times slower. This is the difference that shows up in a long-running deployment.

## Checkpoint latency, durable

Both are SQLite files in WAL mode. The first two rows use the same `synchronous=NORMAL`
fsync policy; the third is `SqliteSaver` as it ships, which is a stronger guarantee.

| Checkpointer | `put` p50 | `put` p99 | `get_tuple` p50 |
| --- | --- | --- | --- |
| `SwarmStateSaver(DiskStore(...))` · `synchronous=NORMAL` | **34.1 µs** | 87.6 µs | 15.7 µs |
| `SqliteSaver` · same `synchronous=NORMAL` | 34.7 µs | 75.6 µs | 14.5 µs |
| `SqliteSaver` · shipped `synchronous=FULL` | 63.9 µs | 211.9 µs | 14.4 µs |

Durable writes are **on par** with `SqliteSaver` at a matched fsync policy (1.02×), and
about 1.9× faster than its shipped default — which buys stronger durability, not nothing.

## Checkpoint latency, in-memory

Neither of these survives a restart.

| Checkpointer | `put` p50 | `put` p99 | `get_tuple` p50 |
| --- | --- | --- | --- |
| `SwarmStateSaver` | 6.2 µs | 11.7 µs | **7.8 µs** |
| `InMemorySaver` | **4.1 µs** | 7.5 µs | 63.5 µs |

Writes are **slower** (0.66×): swarmstate serializes state to msgpack bytes on the way in,
which is what makes it portable across frameworks and makes snapshots O(1). Reads are
8× faster here, and the gap is a function of thread length — see the first table.

## Snapshot cost vs state size

`Store.snapshot()` against `copy.deepcopy` of an equivalent dict: the two ways to get an
independent, mutable copy of state.

| entries in state | `Store.snapshot()` | `dict` deepcopy | ratio |
| --- | --- | --- | --- |
| 100 | 0.00049 ms | 0.085 ms | ~170× |
| 1 000 | 0.00045 ms | 0.82 ms | ~1 800× |
| 10 000 | 0.00052 ms | 8.9 ms | ~17 000× |
| 50 000 | 0.00047 ms | 50.2 ms | **~107 000×** |

Constant, because persistent data structures share structure instead of copying it. One
`snapshot()` captures every thread in the checkpoint database, and one `restore()` rolls the
whole system back.

## Concurrency

Set+get throughput against a plain dict behind a `threading.Lock`, on a GIL build. The
store releases the GIL around the map operation, but under a GIL the interpreter still
serializes the Python-side work, and allocation dominates this workload — a locked dict
stays ahead here. The store's advantage is on a **free-threaded** build, where it keeps
scaling instead of collapsing.

Reproduce both with `python benchmarks/run.py`; the free-threaded numbers need a `cp313t`
interpreter.

## Method notes

- **Build matters.** Release build only (`maturin develop --release` or a published wheel);
  debug builds are several times slower.
- **Percentiles are per call.** p50/p99 of individual operations, not an average of a batch.
- **Losses are reported.** Where the reference implementation wins, the table says so.
- Earlier revisions of this project advertised "~12.8× faster writes than `SqliteSaver`".
  That number compared an in-memory store against a file-backed one; the durable table
  above replaces it.
