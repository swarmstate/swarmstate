"""CrewAI memory storage backed by a swarmstate :class:`~swarmstate.Store`.

``SwarmStateStorage`` implements CrewAI's storage protocol — ``save(value,
metadata)``, ``search(query, limit, score_threshold)``, ``reset()`` — so a
CrewAI crew's memory persists in a swarmstate ``Store`` and is therefore
**shareable and portable** across processes and frameworks (same store as your
LangGraph checkpoints; see the portability guide).

Because it only *implements the protocol* (it does not import CrewAI), it is
independent of any CrewAI version — wire it into your crew's memory/storage
where CrewAI accepts a storage object:

    import swarmstate as ss
    from swarmstate.integrations.crewai import SwarmStateStorage

    store = ss.Store()                       # or RedisStore(...) for persistence
    storage = SwarmStateStorage(store, namespace="crew:research")
    # pass `storage` to CrewAI's memory (e.g. ExternalMemory/RAGStorage slot)

!!! note
    Search here is **lexical** (token-overlap), not embedding-based semantic
    retrieval. It is deterministic and dependency-free; for semantic recall use
    CrewAI's RAG storage. A first-class, embedding-aware adapter may follow.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .. import Store

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class SwarmStateStorage:
    """A CrewAI-compatible storage object persisting entries in a ``Store``.

    Args:
        store: the backing store (in-memory ``Store`` or a persistent backend
            such as ``RedisStore``). Defaults to a fresh ``Store()``.
        namespace: the store namespace to keep this memory under.
    """

    def __init__(self, store: Optional[Store] = None, *, namespace: str = "crewai") -> None:
        self.store = store if store is not None else Store()
        self.namespace = namespace

    def _next_key(self) -> str:
        # Zero-padded monotonic keys keep insertion order sortable.
        n = len(self.store.keys(self.namespace))
        return f"{n:012d}"

    def save(self, value: Any, metadata: Optional[dict] = None) -> None:
        """Persist a memory ``value`` with optional ``metadata`` (CrewAI protocol)."""
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
    ) -> list[dict]:
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
        """Clear all stored memory in this namespace (CrewAI protocol)."""
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
