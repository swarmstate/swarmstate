# Persistent backends

Anything that takes a store accepts any of these interchangeably — including
[`SwarmStateSaver`](langgraph.md). They all serialize with **msgpack**, the same wire
format as the Rust core, so the bytes stay readable from other languages and other
backends.

| Backend | Extra | Storage | Snapshots |
| --- | --- | --- | --- |
| `Store` | — | in-process (Rust) | O(1), structural sharing |
| `DiskStore` | `[disk]` | one SQLite file | O(n) copy |
| `RedisStore` | `[redis]` | Redis hashes | O(n) copy |
| `PostgresStore` | `[postgres]` | one Postgres table | O(n) copy |

The persistent backends *are* the persistence, so their snapshots exist for point-in-time
rollback, not for taking one per step.

## SQLite file

```bash
pip install "swarmstate[disk]"
```

```python
from swarmstate.backends.disk import DiskStore
from swarmstate.integrations.langgraph import SwarmStateSaver

store = DiskStore("state.db")
graph = builder.compile(checkpointer=SwarmStateSaver(store))
```

No server, no extra service. Layout is a single `kv(ns, k, v BLOB)` table keyed by
`(ns, k)`, so any SQLite + msgpack consumer can read it:

```python
import sqlite3, msgpack

conn = sqlite3.connect("state.db")
ns, k, v = conn.execute("SELECT ns, k, v FROM kv LIMIT 1").fetchone()
msgpack.unpackb(v, raw=False)
```

Operational notes:

- **WAL + `synchronous=NORMAL`.** Crash-safe (never a corrupt file); a power loss can cost
  the last transactions. That is a weaker guarantee than SQLite's `FULL` default, and it is
  why the [benchmarks](../benchmarks.md) compare against `SqliteSaver` at both settings.
- **One connection per thread.** In WAL mode readers run concurrently with a writer, so
  the store does not funnel them through a single mutex. `busy_timeout` (from `timeout=`,
  5 s by default) makes concurrent writers wait rather than error.
- Multi-statement writes (`set_many`, `restore`) run in a transaction, so an interrupted
  restore cannot leave the table half-emptied.
- Call `close()` when done; it closes every thread's connection.

## Redis

```bash
pip install "swarmstate[redis]"
```

```python
from swarmstate.backends.redis import RedisStore

store = RedisStore("redis://localhost:6379/0")            # or client=<redis.Redis>
```

Each namespace is a Redis hash at `{prefix}:{namespace}`, fields are the keys. Batched
operations pipeline. Redis cannot order hash fields server-side, so this backend keeps the
saver's latest pointer rather than scanning for a maximum.

## Postgres

```bash
pip install "swarmstate[postgres]"
```

```python
from swarmstate.backends.postgres import PostgresStore

store = PostgresStore("postgresql://user:pass@host/db")    # pooled by default
store = PostgresStore(conn=my_connection)                  # or bring your own
```

Layout is `(ns text, k text, v bytea, primary key (ns, k))`. Connections come from a
`psycopg_pool` pool (`max_size=8` by default) so concurrent workers are not serialized
through one connection; passing `conn=` opts into single-connection mode, and an install
without `psycopg_pool` falls back to it automatically.

## The contract

All four satisfy the same duck-typed interface, written down as a typing protocol:

```python
from swarmstate.protocols import StoreLike

def build_saver(store: StoreLike) -> SwarmStateSaver:
    return SwarmStateSaver(store)
```

`tests/test_store_conformance.py` runs the same assertions against every bundled backend —
key/value semantics, prefix scans, `max_key`, snapshot metadata, diffs and restore — so
"drop-in" is checked rather than asserted. A custom store only has to implement that
interface; optional capabilities (like an index-backed `max_key`, advertised with
`indexed_max_key = True`) are detected, never assumed.
