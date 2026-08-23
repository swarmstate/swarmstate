# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The **stored msgpack format is stable**: bytes written by any released version are readable
by every later one. Changes that would break that are not shipped in a minor release, and
the wire format is pinned by byte-for-byte tests against an independent msgpack
implementation at every width boundary.

## [0.11.0] - 2026-08-23

### Fixed

- **`Store.set` no longer aborts the process on a deeply nested or self-referential value.**
  The codec walkers recursed without a depth limit, so a runaway structure exhausted the
  native stack and killed the interpreter with `SIGSEGV`. Nesting is now capped at 128
  levels in both directions and raises `ValueError`.
- **`SwarmStateSaver.get_tuple` no longer returns a stale checkpoint when a store is
  shared.** The "latest checkpoint" was cached per saver instance and only validated for
  existence, so a newer checkpoint written by another saver — or another process on a
  persistent backend — was invisible. Latest is now resolved from the store itself.
- **`SwarmStateSaver.delete_thread` now deletes a thread's channel blobs.** With
  `incremental=True` the `bl` namespace survived deletion, so the bulk of a deleted
  thread's bytes stayed in the store forever.
- **`SwarmStateStorage` (CrewAI) no longer overwrites memories.** Keys were derived from
  the entry *count*, so any deletion made the next `save` collide with an existing entry.
  Keys are now a monotonic counter plus a random suffix, assigned under a lock.
- **`Store.delete` drops a namespace when its last key goes**, instead of leaving empty
  namespaces to accumulate in `namespaces()`.
- **`DiskStore.restore` is atomic.** The `DELETE` committed on its own before the rows
  went back in, so an interrupted restore left the table emptied.
- A panic while a lock was held no longer poisons the store for the rest of the process.

### Added

- `Store.history()` / `Store.clear_history()`, and `max_history` now means what it says:
  `0` (the new default) retains nothing, `n` the last `n`, `None` everything. Retention was
  previously unreachable from Python while still pinning every snapshot's state — 2 000
  snapshots of a 5 KB value held ~31 MB that nothing could read or free.
- `SwarmStateSaver(max_checkpoints_per_thread=N)`: keep the newest N checkpoints of each
  thread, dropping older ones with their pending writes and unreferenced channel blobs. On
  a 300-invocation thread, 0.5 MB instead of 28 MB.
- `Store.max_key(namespace)`, and `namespaces(prefix=...)` / `keys(namespace, prefix=...)`
  on every store — filtering inside the store instead of copying every name out first.
- `swarmstate.protocols` with `StoreLike` / `SnapshotLike`, the store contract written down
  for type checking, plus a conformance suite that runs the same assertions against every
  bundled backend.
- `Store.__contains__`, so `namespace in store` works on the Rust store as it already did
  on the backends.
- Uniform snapshot metadata (`id`, `timestamp`, `parent`, `diff`) on the `DiskStore`,
  `RedisStore` and `PostgresStore` snapshots.
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and this changelog. The documentation
  site (in `swarmstate/swarmstate.github.io`) gains a Citing section, corrected benchmark
  pages, and coverage of everything new here.
- CI: the LangGraph adapters are tested against both ends of the declared range (the `0.2`
  floor and current), coverage is gated at 80%, free-threaded 3.13t **and** 3.14t are
  exercised, and the benchmark script is run so it cannot rot unnoticed.
- Fuzz-style decoder tests: 20 000 pseudo-random inputs plus every prefix and single-byte
  mutation of a valid encoding, asserting the decoder errors rather than crashing, and that
  anything it accepts re-encodes and decodes again.

### Changed

- **PyO3 0.23 → 0.28**, keeping `abi3-py39` (0.29 is the first release to drop it, so
  moving further would mean dropping Python 3.9). Python 3.14 and free-threaded 3.14 are
  supported, and `cp314t` wheels now ship alongside `cp313t` for Linux
  (x86_64/aarch64), macOS (arm64) and Windows (x64).
- **`im` → `imbl`**, the maintained fork of the persistent-collections crate (`im` has had
  no release since 2021). Not only maintenance: iteration-heavy paths got faster —
  `delete_thread` over 2 000 threads by 3.4×, `list()` by 1.9×, `namespaces()` by 1.4×.
- `PostgresStore` uses a `psycopg_pool` connection pool by default, so concurrent workers
  are not serialized through one connection. Passing `conn=` keeps the previous behaviour;
  an install without `psycopg_pool` falls back to it.
- `DiskStore` opens one connection per thread (WAL lets readers run concurrently with a
  writer) and batches multi-statement writes into transactions.
- **Benchmarks compare like with like.** The previous headline (`~12.8× faster writes than
  SqliteSaver`) measured an in-memory store against a file-backed one. Durable is now
  compared with durable at a matched fsync policy — where writes are *on par* — and the
  in-memory pairing reports that swarmstate's writes are slower than `InMemorySaver`. The
  claims that survive: flat read latency as a thread grows, and O(1) snapshots.

### Performance

Measured on Apple Silicon, Python 3.14, release build; reproduce with `benchmarks/run.py`.

- `HandoffGraph.is_dag()` on a 3 000-node chain: 546 ms → **298 µs**. Kahn's algorithm,
  iterative, so a deep graph can no longer overflow the stack either.
- `HandoffGraph.route()` with a realistic agent state: 165 µs → **0.58 µs**. Only the state
  paths a condition actually names are materialized.
- `Snapshot.diff` over 2 000 namespaces with one change: 582 µs → **11 µs**, by using
  structural sharing to skip untouched shards, namespaces and values.
- `Store.get` of a 200 KB value: 24.8 µs → **4.9 µs**. Values are `Arc<[u8]>`, so handing
  one out is a refcount bump rather than a copy.
- Decoding a 50-message checkpoint payload: 33 µs → **16.6 µs**. Both codec directions
  stream, with no intermediate `rmpv::Value` tree.
- `SwarmStateSaver.list()` / `delete_thread()` on a store with 2 000 threads: 5.3× and 3.1×
  faster, via prefix scans.

## [0.10.4] and earlier

See the git history; releases before this changelog was started are not itemized here.

[0.11.0]: https://github.com/swarmstate/swarmstate/releases/tag/v0.11.0
