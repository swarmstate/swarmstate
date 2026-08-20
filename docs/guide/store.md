# Store & snapshots

`Store` is the framework-agnostic core: a concurrent key/value store keyed by
`(namespace, key)`, with values serialized to **msgpack** bytes by the Rust codec. Any
msgpack reader in any language can read what it writes, which is the point — state written
under one framework stays readable from another.

```python
import swarmstate as ss

store = ss.Store()                                  # in-memory, msgpack codec
store.set("workflow", "onboarding", {"step": 3, "user": {"tier": "gold"}})
store.get("workflow", "onboarding")                 # -> {"step": 3, ...}
store.get("workflow", "missing", "fallback")        # -> "fallback"

store.keys("workflow")                              # -> ["onboarding"]
store.namespaces()                                  # -> ["workflow"]
"workflow" in store                                 # -> True
len(store)                                          # -> 1
```

## Batching

`set_many` / `get_many` amortize the per-call cost over a batch: one GIL release and one
lock acquisition per shard for the in-memory store, one round trip for the networked
backends.

```python
store.set_many([("wf", "a", {"step": 1}), ("wf", "b", {"step": 2})])
store.get_many([("wf", "a"), ("wf", "b"), ("wf", "gone")])   # -> [{...}, {...}, None]
```

Order is preserved and missing pairs come back as `None`.

## Prefix scans

Namespaces are often structured (`tenant:thread:...`), so both listings can filter inside
the store instead of copying every name out first:

```python
store.namespaces(prefix="wf\x1ft42\x1f")   # only this thread's namespaces
store.keys("wf", prefix="task-")           # only keys under that prefix
store.max_key("wf")                        # greatest key, without building a list
```

The filtering is literal: a `_`, `%`, `*` or `[` inside a thread id is never treated as a
wildcard, on any backend.

## Snapshots

`snapshot()` captures an immutable point-in-time view. The backing maps are persistent, so
this is **O(1)** — it shares structure with the live store rather than copying it, and later
writes cannot reach into it.

```python
snap = store.snapshot()
store.set("workflow", "onboarding", {"step": 4})

snap.id            # monotonic id from the originating store
snap.timestamp     # seconds since the epoch
snap.parent        # the previous snapshot's id, for chaining
snap.size_bytes    # total serialized bytes it holds
snap.keys          # [(namespace, key), ...]

store.restore(snap)   # roll everything back at once
```

### Diffs

`diff` reports what changed between two snapshots. Structural sharing is used to skip
whole shards and namespaces that were never touched, so the cost tracks the *changes*, not
the size of the store.

```python
base = store.snapshot()
store.set("wf", "new", 1)
store.delete("wf", "old")
now = store.snapshot()

now.diff(base)
# {"added": [("wf", "new")], "removed": [("wf", "old")], "changed": [...]}
```

### Retention

Snapshots you hold are always valid. Whether the *store* keeps its own list of them is
opt-in, because a retained snapshot pins every value it saw:

```python
ss.Store()                      # max_history=0: retains none (the default)
ss.Store(max_history=10)        # the last 10, reachable via history()
ss.Store(max_history=None)      # every one — grows without bound, on purpose

store = ss.Store(max_history=10)
store.snapshot()
store.history()                 # -> [Snapshot, ...] oldest first
store.clear_history()           # release what they pin
```

## Concurrency

Namespaces are hashed across 16 independent locks, so writers to different namespaces do
not contend, and the GIL is released around every map operation — only (de)serialization
runs under it. On a free-threaded (no-GIL) build the store keeps scaling where a
lock-guarded dict collapses.

```python
import threading

store = ss.Store()
threads = [
    threading.Thread(target=lambda i=i: store.set(f"t{i}", "k", {"i": i}))
    for i in range(8)
]
for t in threads: t.start()
for t in threads: t.join()
```

## Supported value types

`None`, `bool`, `int` (64-bit), `float`, `str`, `bytes`, `list`, `tuple` (returned as
`list`), and `dict`. Anything else raises `TypeError` rather than being silently pickled —
the stored bytes have to stay readable from other languages.

Nesting is capped at 128 levels: deeper than that (or self-referential) raises `ValueError`
instead of exhausting the native stack.

```python
ss.dumps({"a": [1, 2, {"b": b"raw"}]})   # -> msgpack bytes
ss.loads(_)                              # -> the same value back
```
