"""The store contract every swarmstate backend implements.

Anything that takes a store — :class:`~swarmstate.integrations.langgraph.SwarmStateSaver`
above all — accepts the Rust :class:`swarmstate.Store` and the persistent backends
(``DiskStore``, ``RedisStore``, ``PostgresStore``) interchangeably, because they
all satisfy the same duck-typed interface. These protocols write that interface
down so it can be type-checked in your own code::

    from swarmstate.protocols import StoreLike

    def build_saver(store: StoreLike) -> SwarmStateSaver:
        return SwarmStateSaver(store)

They are typing constructs, not base classes: nothing inherits from them, and a
store is conformant by having the methods. ``tests/test_store_conformance.py``
exercises the behaviour behind them against every bundled backend.

Two documented differences between implementations, both of which callers can act
on rather than guess at:

* **Snapshot cost.** The Rust store shares structure, so ``snapshot()`` is O(1);
  the persistent backends copy their rows, so theirs is O(n) and is meant for
  point-in-time rollback rather than for taking one per step.
* **``max_key`` cost.** Every store answers it, but only some do so from an
  index. A store sets the optional class attribute ``indexed_max_key = True``
  when it does (the SQL backends), and
  :class:`~swarmstate.integrations.langgraph.SwarmStateSaver` uses that to decide
  whether to ask for the newest checkpoint directly or to maintain its own
  pointer. Leaving the attribute off is always safe.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class SnapshotLike(Protocol):
    """A point-in-time view of a store, as returned by ``store.snapshot()``."""

    #: Monotonic id assigned by the originating store.
    id: int
    #: Seconds since the Unix epoch when the snapshot was taken.
    timestamp: float
    #: Id of the previous snapshot from the same store, if any.
    parent: Optional[int]
    #: Total size in bytes of the serialized values it holds.
    size_bytes: int

    @property
    def keys(self) -> "list[tuple[str, str]]":
        """All ``(namespace, key)`` pairs present in the snapshot."""

    def diff(self, base: Any) -> "dict[str, list[tuple[str, str]]]":
        """``{"added", "removed", "changed"}`` -> ``(namespace, key)`` lists.

        Describes how to get from ``base`` to this snapshot.
        """


class StoreLike(Protocol):
    """The key/value store interface swarmstate components program against."""

    def set(self, namespace: str, key: str, value: Any) -> None: ...
    def get(self, namespace: str, key: str, default: Any = None) -> Any: ...
    def set_many(self, items: "list[tuple[str, str, Any]]") -> None: ...
    def get_many(self, pairs: "list[tuple[str, str]]") -> "list[Any]": ...
    def contains(self, namespace: str, key: str) -> bool: ...
    def delete(self, namespace: str, key: str) -> bool: ...
    def keys(self, namespace: str, prefix: Optional[str] = None) -> "list[str]": ...
    def max_key(self, namespace: str) -> Optional[str]: ...
    def namespaces(self, prefix: Optional[str] = None) -> "list[str]": ...
    def clear(self) -> None: ...
    def snapshot(self) -> SnapshotLike: ...
    def restore(self, snapshot: Any) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, namespace: str) -> bool: ...


__all__ = ["SnapshotLike", "StoreLike"]
