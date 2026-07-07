"""Type stubs for the native ``swarmstate._core`` module (built from Rust).

Kept in sync by hand with ``rust/src/``. Extended as milestones land.
"""

from typing import Any, Optional

__version__: str

def core_version() -> str:
    """Return the version string of the compiled Rust core."""
    ...

def dumps(obj: Any) -> bytes:
    """Serialize a Python object to msgpack bytes (stable, cross-language codec)."""
    ...

def loads(data: bytes) -> Any:
    """Deserialize msgpack bytes back into a Python object."""
    ...

class Snapshot:
    """A cheap, immutable point-in-time view of a :class:`Store`."""

    @property
    def id(self) -> int:
        """Monotonic id assigned by the originating store."""
        ...

    @property
    def timestamp(self) -> float:
        """Seconds since the Unix epoch when the snapshot was taken."""
        ...

    @property
    def parent(self) -> Optional[int]:
        """Id of the previous snapshot from the same store, if any."""
        ...

    @property
    def size_bytes(self) -> int:
        """Total size in bytes of all stored (serialized) values."""
        ...

    @property
    def keys(self) -> list[tuple[str, str]]:
        """All ``(namespace, key)`` pairs present in the snapshot."""
        ...

    def diff(self, base: "Snapshot") -> dict[str, list[tuple[str, str]]]:
        """Return ``{"added", "removed", "changed"}`` -> ``(namespace, key)`` lists.

        Describes how to go from ``base`` to ``self``.
        """
        ...

class Store:
    """Framework-agnostic state store with immutable snapshots."""

    def __init__(
        self,
        backend: str = "memory",
        codec: str = "msgpack",
        max_history: Optional[int] = None,
    ) -> None: ...
    @property
    def codec(self) -> str: ...
    @property
    def max_history(self) -> Optional[int]: ...
    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store ``value`` under ``(namespace, key)``."""
        ...

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Return the value at ``(namespace, key)`` or ``default`` if absent."""
        ...

    def set_many(self, items: list[tuple[str, str, Any]]) -> None:
        """Store many ``(namespace, key, value)`` triples in one call.

        Encodes under the GIL, then writes with the GIL released, locking each
        shard once for the whole batch.
        """
        ...

    def get_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        """Fetch many ``(namespace, key)`` pairs, in order; missing -> ``None``."""
        ...

    def contains(self, namespace: str, key: str) -> bool: ...
    def delete(self, namespace: str, key: str) -> bool:
        """Delete ``(namespace, key)``; return True if a value was removed."""
        ...

    def keys(self, namespace: str) -> list[str]: ...
    def namespaces(self) -> list[str]: ...
    def clear(self) -> None: ...
    def snapshot(self) -> Snapshot:
        """Capture a cheap, immutable snapshot of the current state."""
        ...

    def restore(self, snapshot: Snapshot) -> None:
        """Roll the store back to a previously captured snapshot."""
        ...

    def __len__(self) -> int: ...

class HandoffGraph:
    """A deterministic, LLM-free routing graph over named nodes.

    Edges carry an optional ``when`` condition written in a small, safe
    mini-language (never Python ``eval``): literals, dotted state paths,
    ``== != < <= > >=``, ``in``, ``and``/``or``/``not`` and parentheses.
    """

    def __init__(self, on_cycle: str = "error") -> None:
        """``on_cycle``: ``"error"`` (default) or ``"allow"`` on cycle detection."""
        ...

    @property
    def on_cycle(self) -> str: ...
    def add_node(self, name: str) -> None:
        """Register a node with no edges."""
        ...

    def add_edge(self, from_node: str, to: str, when: Optional[str] = None) -> None:
        """Add a directed edge ``from_node -> to``, optionally guarded by ``when``.

        Raises ``ValueError`` on an invalid condition, or (when
        ``on_cycle="error"``) if the edge would create a cycle.
        """
        ...

    def route(self, node: str, state: Optional[dict[str, Any]] = None) -> Optional[str]:
        """Return the next node from ``node`` given ``state``.

        Evaluates outgoing edges in insertion order and returns the first whose
        condition holds (an edge with no condition always matches); ``None`` if
        none match.
        """
        ...

    def nodes(self) -> list[str]:
        """All nodes, sorted."""
        ...

    def edges(self, node: str) -> list[tuple[str, Optional[str]]]:
        """Outgoing edges of ``node`` as ``(to, when)`` pairs, in insertion order."""
        ...

    def has_node(self, node: str) -> bool: ...
    def is_dag(self) -> bool:
        """Whether the graph is currently acyclic."""
        ...

    def __len__(self) -> int: ...
    def __contains__(self, node: str) -> bool: ...
