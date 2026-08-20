# CrewAI memory

`SwarmStateStorage` is a small, dependency-free keyword memory backed by a swarmstate
store. Its point is **portability**: crew memories live in the same store as your LangGraph
checkpoints, in the same msgpack format, readable by anything else you run.

```python
import swarmstate as ss
from swarmstate.integrations.crewai import SwarmStateStorage

store = ss.Store()                                    # or DiskStore / RedisStore / PostgresStore
mem = SwarmStateStorage(store, namespace="crew:research")

mem.save("The Q2 churn rate was 4.1%", {"agent": "analyst"})
mem.save("Refund policy is 30 days", {"agent": "billing"})

mem.search("churn rate")
# [{"context": "The Q2 churn rate was 4.1%", "metadata": {"agent": "analyst"}, "score": 1.0}]
```

Results are scored by token overlap with the query, sorted by score and then by recency,
and filtered by `score_threshold`. `reset()` clears the namespace.

!!! important "This is not a drop-in for CrewAI's built-in memory"

    As of CrewAI 1.x the native `StorageBackend` protocol is **embedding-based**
    (`save(list[MemoryRecord])`, `search(query_embedding, ...)`) — a vector store, which
    swarmstate is not. `SwarmStateStorage` is a lightweight *lexical* alternative you wire
    in yourself, for example from a task callback or your own loop, when you want durable,
    portable, dependency-free recall. For semantic RAG recall, use CrewAI's own storage.
    Verified against crewai 1.15.

## Sharing state across frameworks

Because the memory is just namespaced entries in a store, anything else can read it —
which is the migration story in miniature:

```python
store = ss.Store()
mem = SwarmStateStorage(store, namespace="crew:research")
mem.save("finding: latency dropped", {"step": 1})

# Another system, no swarmstate-specific knowledge beyond the namespace:
key = store.keys("crew:research")[0]
store.get("crew:research", key)["value"]     # -> "finding: latency dropped"
```

Point it at a [persistent backend](backends.md) and the same memories survive a restart, or
become visible to another process:

```python
from swarmstate.backends.redis import RedisStore

mem = SwarmStateStorage(RedisStore("redis://localhost:6379/0"), namespace="crew:research")
```

Keys are a zero-padded counter plus a random suffix, so entries stay in insertion order
and two writers — threads, or processes on a shared backend — can never claim the same key.
