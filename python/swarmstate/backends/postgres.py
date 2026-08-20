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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional, cast

import msgpack

from ._snapshot import CopySnapshot, SnapshotMeta

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: Any) -> Any:
    return msgpack.unpackb(bytes(raw), raw=False, strict_map_key=False)


class PostgresSnapshot(CopySnapshot):
    """A copy-based snapshot of a :class:`PostgresStore` (O(n))."""


class PostgresStore:
    """Postgres-backed store implementing the :class:`swarmstate.Store` interface."""

    #: ``max_key`` is answered from the ``(ns, k)`` primary-key index (see
    #: :class:`~swarmstate.backends.disk.DiskStore`).
    indexed_max_key = True

    def __init__(
        self,
        dsn: str = "postgresql:///swarmstate",
        *,
        conn: Any = None,
        pool: Any = None,
        table: str = "swarmstate_kv",
        codec: str = "msgpack",
        max_size: int = 8,
    ) -> None:
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.table = table
        self.codec = codec
        # Single-connection mode keeps a mutex, so concurrent callers queue up on
        # one connection; a pool lets them work in parallel, which is what a
        # multi-worker service needs. An injected `conn` opts out on purpose.
        self._lock = threading.Lock()
        self._snap_meta = SnapshotMeta()
        self._conn: Any = conn
        self._pool: Any = pool
        if self._conn is None and self._pool is None:
            self._pool = self._make_pool(dsn, max_size)
        with self._connection() as c:
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(ns text NOT NULL, k text NOT NULL, v bytea NOT NULL, PRIMARY KEY (ns, k))"
            )

    def _make_pool(self, dsn: str, max_size: int) -> Any:
        """Open a psycopg connection pool, or fall back to a single connection.

        ``psycopg_pool`` is a separate distribution, so an install without it
        still works — just serialized through one connection, as before.
        """
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            import psycopg

            self._conn = psycopg.connect(dsn, autocommit=True)
            return None
        return ConnectionPool(
            dsn, min_size=1, max_size=max_size, open=True, kwargs={"autocommit": True}
        )

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Yield a connection: one from the pool, or the shared one under lock."""
        if self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            with self._lock:
                yield self._conn

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._connection() as conn:
            conn.execute(
                f"INSERT INTO {self.table} (ns, k, v) VALUES (%s, %s, %s) "
                "ON CONFLICT (ns, k) DO UPDATE SET v = EXCLUDED.v",
                (namespace, key, _pack(value)),
            )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT v FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            ).fetchone()
        return default if row is None else _unpack(row[0])

    def set_many(self, items: list[tuple[str, str, Any]]) -> None:
        if not items:
            return
        rows = [(ns, k, _pack(v)) for ns, k, v in items]
        with self._connection() as conn:
            conn.cursor().executemany(
                f"INSERT INTO {self.table} (ns, k, v) VALUES (%s, %s, %s) "
                "ON CONFLICT (ns, k) DO UPDATE SET v = EXCLUDED.v",
                rows,
            )

    def get_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        if not pairs:
            return []
        nss = [p[0] for p in pairs]
        ks = [p[1] for p in pairs]
        # unnest two arrays positionally, then join: one round-trip for the batch.
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT t.ns, t.k, t.v FROM {self.table} t "
                "JOIN unnest(%s::text[], %s::text[]) AS q(ns, k) "
                "ON t.ns = q.ns AND t.k = q.k",
                (nss, ks),
            ).fetchall()
        found = {(ns, k): v for ns, k, v in rows}
        return [None if (p := (ns, k)) not in found else _unpack(found[p]) for ns, k in pairs]

    def contains(self, namespace: str, key: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            ).fetchone()
        return row is not None

    def delete(self, namespace: str, key: str) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                f"DELETE FROM {self.table} WHERE ns = %s AND k = %s", (namespace, key)
            )
            return bool(cur.rowcount > 0)

    def keys(self, namespace: str, prefix: Optional[str] = None) -> list[str]:
        sql = f"SELECT k FROM {self.table} WHERE ns = %s"
        params: tuple[Any, ...] = (namespace,)
        if prefix is not None:
            # starts_with, not LIKE: keys carry user data that may contain
            # LIKE wildcards.
            sql += " AND starts_with(k, %s)"
            params += (prefix,)
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def max_key(self, namespace: str) -> Optional[str]:
        """The greatest key in ``namespace``, via the primary-key index."""
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT max(k) FROM {self.table} WHERE ns = %s", (namespace,)
            ).fetchone()
        return None if row is None else cast(Optional[str], row[0])

    def namespaces(self, prefix: Optional[str] = None) -> list[str]:
        sql = f"SELECT DISTINCT ns FROM {self.table}"
        params: tuple[Any, ...] = ()
        if prefix is not None:
            sql += " WHERE starts_with(ns, %s)"
            params = (prefix,)
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute(f"DELETE FROM {self.table}")

    def __len__(self) -> int:
        with self._connection() as conn:
            return int(conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])

    def __contains__(self, namespace: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self.table} WHERE ns = %s LIMIT 1", (namespace,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> PostgresSnapshot:
        with self._connection() as conn:
            rows = conn.execute(f"SELECT ns, k, v FROM {self.table}").fetchall()
        return PostgresSnapshot([(ns, k, bytes(v)) for ns, k, v in rows], *self._snap_meta.next())

    def restore(self, snapshot: PostgresSnapshot) -> None:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(f"DELETE FROM {self.table}")
                conn.cursor().executemany(
                    f"INSERT INTO {self.table} (ns, k, v) VALUES (%s, %s, %s)",
                    snapshot._rows,
                )

    def close(self) -> None:
        """Close the pool (or the single connection)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        elif self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None

    def __repr__(self) -> str:
        return f"PostgresStore(table={self.table!r}, codec='{self.codec}')"
