"""A PostgreSQL-backed store with the same interface as :class:`swarmstate.Store`.

`PostgresStore` persists state to a Postgres table, serializing values with
**msgpack** (the same wire format as the Rust core). It is a drop-in backend for
anything that takes a store, including
:class:`~swarmstate.integrations.langgraph.SwarmStateSaver` -- giving durable,
shared, networked checkpoints backed by your existing Postgres.

    from swarmstate.backends.postgres import PostgresStore
    from swarmstate.integrations.langgraph import SwarmStateSaver

    store = PostgresStore("postgresql://user:pass@host/db")
    graph = builder.compile(checkpointer=SwarmStateSaver(store))

Requires the ``postgres`` extra: ``pip install "swarmstate[postgres]"``.

Layout: a single table ``(ns text, k text, v bytea, primary key (ns, k))``; ``v``
is msgpack bytes. ``snapshot``/``restore`` copy the data (O(n)); Postgres is the
persistence, so they are for point-in-time rollback rather than the Rust store's
O(1) snapshots.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import msgpack

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, use_bin_type=True)


def _unpack(raw) -> Any:
    return msgpack.unpackb(bytes(raw), raw=False, strict_map_key=False)


class PostgresSnapshot:
    """A copy-based snapshot of a :class:`PostgresStore` (O(n))."""

    def __init__(self, rows: list[tuple[str, str, bytes]]):
        self._rows = rows
        self.size_bytes = sum(len(v) for _, _, v in rows)

    @property
    def keys(self) -> list[tuple[str, str]]:
        return [(ns, k) for ns, k, _ in self._rows]


class PostgresStore:
    """Postgres-backed store implementing the :class:`swarmstate.Store` interface."""

    def __init__(
        self,
        dsn: str = "postgresql:///swarmstate",
        *,
        conn: Any = None,
        table: str = "swarmstate_kv",
        codec: str = "msgpack",
    ) -> None:
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.table = table
        self.codec = codec
        self._lock = threading.Lock()
        if conn is None:
            import psycopg

            conn = psycopg.connect(dsn, autocommit=True)
        self._conn = conn
        with self._lock:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(ns text NOT NULL, k text NOT NULL, v bytea NOT NULL, PRIMARY KEY (ns, k))"
            )

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self.table} (ns, k, v) VALUES (%s, %s, %s) "
                "ON CONFLICT (ns, k) DO UPDATE SET v = EXCLUDED.v",
                (namespace, key, _pack(value)),
            )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                f"SELECT v FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            ).fetchone()
        return default if row is None else _unpack(row[0])

    def contains(self, namespace: str, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            ).fetchone()
        return row is not None

    def delete(self, namespace: str, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            )
            return cur.rowcount > 0

    def keys(self, namespace: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT k FROM {self.table} WHERE ns = %s", (namespace,)
            ).fetchall()
        return [r[0] for r in rows]

    def namespaces(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(f"SELECT DISTINCT ns FROM {self.table}").fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute(f"DELETE FROM {self.table}")

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def __contains__(self, namespace: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self.table} WHERE ns = %s LIMIT 1", (namespace,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> PostgresSnapshot:
        with self._lock:
            rows = self._conn.execute(f"SELECT ns, k, v FROM {self.table}").fetchall()
        return PostgresSnapshot([(ns, k, bytes(v)) for ns, k, v in rows])

    def restore(self, snapshot: PostgresSnapshot) -> None:
        with self._lock:
            with self._conn.transaction():
                self._conn.execute(f"DELETE FROM {self.table}")
                self._conn.cursor().executemany(
                    f"INSERT INTO {self.table} (ns, k, v) VALUES (%s, %s, %s)",
                    snapshot._rows,
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        return f"PostgresStore(table={self.table!r}, codec='{self.codec}')"
