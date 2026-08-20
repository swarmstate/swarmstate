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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional, cast

import msgpack

from ._snapshot import CopySnapshot, SnapshotMeta


# SQLite's default bound-parameter ceiling on older builds; queries stay below it.
_MAX_PARAMS = 900


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


class DiskSnapshot(CopySnapshot):
    """A copy-based snapshot of a :class:`DiskStore` (O(n))."""


class DiskStore:
    """SQLite-backed store implementing the :class:`swarmstate.Store` interface."""

    #: ``max_key`` is answered from the ``(ns, k)`` primary-key index, so callers
    #: after the newest entry can just ask instead of keeping their own pointer.
    indexed_max_key = True

    def __init__(
        self, path: str = "swarmstate.db", *, codec: str = "msgpack", timeout: float = 5.0
    ) -> None:
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        self.path = path
        self.codec = codec
        self.timeout = timeout
        # One connection per thread instead of one shared connection behind a
        # mutex: in WAL mode SQLite lets readers run concurrently with a writer,
        # so a single serializing lock was throttling reads for no benefit.
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._snap_meta = SnapshotMeta()
        self._connect()  # fail fast on an unusable path

    # ------------------------------------------------------------ connections
    def _connect(self) -> sqlite3.Connection:
        """Return this thread's connection, opening and configuring it if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        # check_same_thread=False so close() can shut down every thread's
        # connection; each one is still only *used* by its own thread.
        conn = sqlite3.connect(
            self.path, timeout=self.timeout, check_same_thread=False, isolation_level=None
        )
        # WAL + synchronous=NORMAL is the recommended durable-yet-fast config: it
        # avoids an fsync on every checkpoint commit (the hot path here) while
        # remaining crash-safe (only the last transactions can be lost on power
        # loss, never a corrupt file). busy_timeout keeps concurrent writers (other
        # threads, other processes) waiting their turn instead of erroring out.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT NOT NULL, k TEXT NOT NULL, "
            "v BLOB NOT NULL, PRIMARY KEY (ns, k))"
        )
        self._local.conn = conn
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a multi-statement write atomically (autocommit is on otherwise)."""
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        self._connect().execute(
            "INSERT OR REPLACE INTO kv (ns, k, v) VALUES (?, ?, ?)",
            (namespace, key, _pack(value)),
        )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        row = (
            self._connect()
            .execute("SELECT v FROM kv WHERE ns = ? AND k = ?", (namespace, key))
            .fetchone()
        )
        return default if row is None else _unpack(row[0])

    def set_many(self, items: list[tuple[str, str, Any]]) -> None:
        if not items:
            return
        rows = [(ns, k, _pack(v)) for ns, k, v in items]
        # One transaction for the batch: in autocommit each row would be its own.
        with self._transaction() as conn:
            conn.executemany("INSERT OR REPLACE INTO kv (ns, k, v) VALUES (?, ?, ?)", rows)

    def get_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        """Fetch many pairs, preserving input order; missing ones come back None.

        One query per namespace (chunked), rather than one per key: callers ask
        for a whole namespace's worth of keys at a time — the LangGraph adapter
        reads a checkpoint's pending writes that way — and the per-statement
        overhead dominated at that size.
        """
        if not pairs:
            return []
        by_ns: dict[str, list[str]] = {}
        for ns, key in pairs:
            by_ns.setdefault(ns, []).append(key)

        conn = self._connect()
        found: dict[tuple[str, str], bytes] = {}
        for ns, keys in by_ns.items():
            # Chunked to stay under SQLite's bound-parameter limit, which is 999
            # on older builds.
            for start in range(0, len(keys), _MAX_PARAMS - 1):
                chunk = keys[start : start + _MAX_PARAMS - 1]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT k, v FROM kv WHERE ns = ? AND k IN ({placeholders})",
                    (ns, *chunk),
                ).fetchall()
                for key, value in rows:
                    found[(ns, key)] = value
        return [None if p not in found else _unpack(found[p]) for p in pairs]

    def contains(self, namespace: str, key: str) -> bool:
        row = (
            self._connect()
            .execute("SELECT 1 FROM kv WHERE ns = ? AND k = ? LIMIT 1", (namespace, key))
            .fetchone()
        )
        return row is not None

    def delete(self, namespace: str, key: str) -> bool:
        cur = self._connect().execute("DELETE FROM kv WHERE ns = ? AND k = ?", (namespace, key))
        return cur.rowcount > 0

    def keys(self, namespace: str, prefix: Optional[str] = None) -> list[str]:
        conn = self._connect()
        if prefix is None:
            rows = conn.execute("SELECT k FROM kv WHERE ns = ?", (namespace,)).fetchall()
        else:
            # substr, not LIKE: keys carry user data, and LIKE would read a '_'
            # or '%' inside a thread id as a wildcard.
            rows = conn.execute(
                "SELECT k FROM kv WHERE ns = ? AND substr(k, 1, ?) = ?",
                (namespace, len(prefix), prefix),
            ).fetchall()
        return [r[0] for r in rows]

    def max_key(self, namespace: str) -> Optional[str]:
        """The greatest key in ``namespace``, via the primary-key index."""
        row = self._connect().execute("SELECT max(k) FROM kv WHERE ns = ?", (namespace,)).fetchone()
        return None if row is None else cast(Optional[str], row[0])

    def namespaces(self, prefix: Optional[str] = None) -> list[str]:
        conn = self._connect()
        if prefix is None:
            rows = conn.execute("SELECT DISTINCT ns FROM kv").fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT ns FROM kv WHERE substr(ns, 1, ?) = ?",
                (len(prefix), prefix),
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        self._connect().execute("DELETE FROM kv")

    def __len__(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM kv").fetchone()[0])

    def __contains__(self, namespace: str) -> bool:
        row = (
            self._connect()
            .execute("SELECT 1 FROM kv WHERE ns = ? LIMIT 1", (namespace,))
            .fetchone()
        )
        return row is not None

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> DiskSnapshot:
        rows = self._connect().execute("SELECT ns, k, v FROM kv").fetchall()
        return DiskSnapshot([(ns, k, bytes(v)) for ns, k, v in rows], *self._snap_meta.next())

    def restore(self, snapshot: DiskSnapshot) -> None:
        # Atomic: an interrupted restore used to leave the table emptied, since
        # the DELETE committed on its own before the rows went back in.
        with self._transaction() as conn:
            conn.execute("DELETE FROM kv")
            conn.executemany("INSERT INTO kv (ns, k, v) VALUES (?, ?, ?)", snapshot._rows)

    def close(self) -> None:
        """Close every thread's connection."""
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            conn.close()
        self._local = threading.local()

    def __repr__(self) -> str:
        return f"DiskStore(path={self.path!r}, codec='{self.codec}')"
