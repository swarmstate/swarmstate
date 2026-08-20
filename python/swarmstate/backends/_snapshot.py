"""Snapshot bookkeeping shared by the copy-based backends.

The Rust :class:`swarmstate.Snapshot` carries ``id`` / ``timestamp`` / ``parent``
and can ``diff`` against another snapshot. The persistent backends copy rows
instead of sharing structure, but callers should not have to care: this module
gives them the same surface, so a snapshot is a snapshot whichever store made it
(see :mod:`swarmstate.protocols`).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional


class SnapshotMeta:
    """Hands out monotonic ``(id, timestamp, parent)`` triples for one store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._last: Optional[int] = None

    def next(self) -> "tuple[int, float, Optional[int]]":
        with self._lock:
            self._seq += 1
            snapshot_id, parent = self._seq, self._last
            self._last = snapshot_id
        return snapshot_id, time.time(), parent


class CopySnapshot:
    """A snapshot holding a copy of every ``(namespace, key, value bytes)`` row.

    O(n) in the size of the store, unlike the Rust store's O(1) structural
    sharing — the persistent backends *are* the persistence, so their snapshots
    exist for point-in-time rollback rather than for taking one per step.
    """

    def __init__(
        self,
        rows: "list[tuple[str, str, bytes]]",
        snapshot_id: int = 0,
        timestamp: float = 0.0,
        parent: Optional[int] = None,
    ):
        self._rows = rows
        self.id = snapshot_id
        self.timestamp = timestamp
        self.parent = parent
        self.size_bytes = sum(len(v) for _, _, v in rows)

    @property
    def keys(self) -> "list[tuple[str, str]]":
        return [(ns, k) for ns, k, _ in self._rows]

    def diff(self, base: Any) -> "dict[str, list[tuple[str, str]]]":
        """``{"added", "removed", "changed"}`` -> ``(namespace, key)`` lists."""
        mine = {(ns, k): v for ns, k, v in self._rows}
        theirs = {(ns, k): v for ns, k, v in base._rows}
        return {
            "added": [p for p in mine if p not in theirs],
            "removed": [p for p in theirs if p not in mine],
            "changed": [p for p, v in mine.items() if p in theirs and theirs[p] != v],
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.id}, size_bytes={self.size_bytes}, "
            f"parent={self.parent})"
        )
