"""Type stubs for the native ``swarmstate._core`` module (built from Rust).

Kept in sync by hand with ``rust/src/``. Extended as milestones land.
"""

from typing import Any, Optional

__version__: str

def core_version() -> str:
    """Return the version string of the compiled Rust core."""
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
