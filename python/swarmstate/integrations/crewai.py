"""Durable, portable memory for CrewAI crews, backed by a swarmstate ``Store``.

``SwarmStateStorage`` is a small, dependency-free **keyword** memory store:
``save(value, metadata)`` / ``search(query, limit, score_threshold)`` / ``reset()``,
persisted in a swarmstate ``Store`` (in-memory, or ``RedisStore``/``DiskStore`` for
durability). Its point is **state portability**: crew memories live in the same
store as your LangGraph checkpoints and can be read by any other system.

    import swarmstate as ss
    from swarmstate.integrations.crewai import SwarmStateStorage

    store = ss.Store()                       # or RedisStore(...) / DiskStore(...)
    mem = SwarmStateStorage(store, namespace="crew:research")
    mem.save("The Q2 churn rate was 4.1%", {"agent": "analyst"})
    mem.search("churn rate")                 # lexical (token-overlap) recall

.. important::
    This is **not** a drop-in for CrewAI's built-in memory. As of CrewAI 1.x, the
    native ``StorageBackend`` protocol is **embedding-based** (``save(list[MemoryRecord])``,
    ``search(query_embedding, ...)``) — a vector store, which swarmstate is not.
    ``SwarmStateStorage`` is a lightweight *lexical* alternative you wire in yourself
    (e.g. from a task callback or your own loop) when you want durable, portable,
    dependency-free recall; for semantic RAG recall, use CrewAI's own storage.
    Verified against crewai 1.15.
"""

from __future__ import annotations

import re
import threading
import uuid
from typing import Any, Optional

from .. import Store

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class SwarmStateStorage:
    """A lightweight, portable keyword memory store backed by a ``Store``.

    Not CrewAI's embedding-based ``StorageBackend`` — a simple lexical store you
    wire in yourself for durable, shareable recall (see the module docstring).

    Args:
        store: the backing store (in-memory ``Store`` or a persistent backend
            such as ``RedisStore``/``DiskStore``). Defaults to a fresh ``Store()``.
        namespace: the store namespace to keep this memory under.
    """

    def __init__(self, store: Optional[Store] = None, *, namespace: str = "crewai") -> None:
        self.store = store if store is not None else Store()
        self.namespace = namespace
        self._lock = threading.Lock()
        self._seq: Optional[int] = None

    def _next_key(self) -> str:
        """Return a fresh, insertion-ordered key.

        Zero-padded counter (so keys sort by insertion order) plus a random
        suffix. The counter is derived from the highest existing key rather than
        from the entry *count*: counting collided with existing keys after any
        deletion and silently overwrote a stored memory. The suffix keeps two
        writers (threads, or processes on a shared backend) from claiming the
        same key, so a race costs ordering, never an entry.
        """
        with self._lock:
            if self._seq is None:
                self._seq = self._highest_seq()
            self._seq += 1
            seq = self._seq
        return f"{seq:012d}-{uuid.uuid4().hex[:8]}"

    def _highest_seq(self) -> int:
        highest = 0
        for key in self.store.keys(self.namespace):
            try:
                highest = max(highest, int(key.split("-", 1)[0]))
            except ValueError:  # a key we did not write; ignore it for ordering
                continue
        return highest

    def save(self, value: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        """Persist a memory ``value`` with optional ``metadata``."""
        self.store.set(
            self.namespace,
            self._next_key(),
            {"value": _as_text(value), "metadata": metadata or {}},
        )

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` entries scored by token overlap with ``query``.

        Each result is ``{"context": str, "metadata": dict, "score": float}``,
        sorted by score (desc) then recency. Lexical, not semantic.
        """
        q = _tokens(query)
        results = []
        for key in self.store.keys(self.namespace):
            entry = self.store.get(self.namespace, key)
            if not entry:
                continue
            text = entry.get("value", "")
            score = (len(q & _tokens(text)) / len(q)) if q else 0.0
            if score >= score_threshold:
                results.append((key, score, entry))
        results.sort(key=lambda r: (r[1], r[0]), reverse=True)
        return [
            {"context": e["value"], "metadata": e.get("metadata", {}), "score": round(s, 4)}
            for _, s, e in results[:limit]
        ]

    def reset(self) -> None:
        """Clear all stored memory in this namespace."""
        for key in self.store.keys(self.namespace):
            self.store.delete(self.namespace, key)

    def __len__(self) -> int:
        return len(self.store.keys(self.namespace))

    def __repr__(self) -> str:
        return f"SwarmStateStorage(namespace='{self.namespace}', entries={len(self)})"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)
