"""A file-backed store with the same interface as :class:`swarmstate.Store`.

`DiskStore` persists state to a single **SQLite** file (no server, no extra service),
serializing values with **msgpack** — the same wire format as the Rust core — so state
survives process restarts and is readable by any msgpack + SQLite consumer, in any
language.

    from swarmstate.backends.disk import DiskStore
    from swarmstate.integrations.langgraph import SwarmStateSaver

    store = DiskStore("state.db")
    graph = builder.compile(checkpointer=SwarmStateSaver(store))   # durable checkpoints

Requires the ``disk`` extra: ``pip install "swarmstate[disk]"`` (SQLite is stdlib; the
extra just pulls in ``msgpack``).

Layout: a single table ``kv(ns, k, v BLOB)`` keyed by ``(ns, k)``; ``v`` is msgpack
bytes. ``snapshot``/``restore`` copy the data (O(n)) — the file *is* the persistence,
so these are for point-in-time rollback rather than the Rust store's O(1) snapshots.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, cast

import msgpack


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


class DiskSnapshot:
    """A copy-based snapshot of a :class:`DiskStore` (O(n))."""

    def __init__(self, rows: list[tuple[str, str, bytes]]):
        self._rows = rows
        self.size_bytes = sum(len(v) for _, _, v in rows)

    @property
    def keys(self) -> list[tuple[str, str]]:
        return [(ns, k) for ns, k, _ in self._rows]


class DiskStore:
    """SQLite-backed store implementing the :class:`swarmstate.Store` interface."""

    def __init__(self, path: str = "swarmstate.db", *, codec: str = "msgpack") -> None:
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        self.path = path
        self.codec = codec
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        # WAL + synchronous=NORMAL is the recommended durable-yet-fast config: it
        # avoids an fsync on every checkpoint commit (the hot path here) while
        # remaining crash-safe (only the last transactions can be lost on power
        # loss, never a corrupt file).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT NOT NULL, k TEXT NOT NULL, "
            "v BLOB NOT NULL, PRIMARY KEY (ns, k))"
        )

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (ns, k, v) VALUES (?, ?, ?)",
                (namespace, key, _pack(value)),
            )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT v FROM kv WHERE ns = ? AND k = ?", (namespace, key)
            ).fetchone()
        return default if row is None else _unpack(row[0])

    def contains(self, namespace: str, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM kv WHERE ns = ? AND k = ? LIMIT 1", (namespace, key)
            ).fetchone()
        return row is not None

    def delete(self, namespace: str, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM kv WHERE ns = ? AND k = ?", (namespace, key))
        return cur.rowcount > 0

    def keys(self, namespace: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT k FROM kv WHERE ns = ?", (namespace,)).fetchall()
        return [r[0] for r in rows]

    def namespaces(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT ns FROM kv").fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv")

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0])

    def __contains__(self, namespace: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM kv WHERE ns = ? LIMIT 1", (namespace,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> DiskSnapshot:
        with self._lock:
            rows = self._conn.execute("SELECT ns, k, v FROM kv").fetchall()
        return DiskSnapshot([(ns, k, bytes(v)) for ns, k, v in rows])

    def restore(self, snapshot: DiskSnapshot) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv")
            self._conn.executemany("INSERT INTO kv (ns, k, v) VALUES (?, ?, ?)", snapshot._rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f"DiskStore(path={self.path!r}, codec='{self.codec}')"
