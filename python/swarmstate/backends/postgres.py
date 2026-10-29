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

Every operation also has an async twin (``aget``, ``aset_many``, ...), which the
saver's async methods use, so an async graph never blocks a worker thread on
Postgres. A store built from a DSN opens its async pool lazily, one per event loop.

Requires the ``postgres`` extra: ``pip install "swarmstate[postgres]"``.

Layout: a single table ``(ns text, k text, v bytea, primary key (ns, k))``, with
``ns`` and ``k`` in ``COLLATE "C"`` (code-point order); ``v`` is msgpack bytes.
``snapshot``/``restore`` copy the data (O(n)); Postgres is the persistence, so they
are for point-in-time rollback rather than the Rust store's O(1) snapshots.
"""

from __future__ import annotations

import asyncio
import re
import threading
import weakref
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional, cast

import msgpack

from ._snapshot import CopySnapshot, SnapshotMeta

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: Any) -> Any:
    return msgpack.unpackb(bytes(raw), raw=False, strict_map_key=False)


class _Queries:
    """The SQL for one table, shared by the sync and the async methods."""

    def __init__(self, table: str) -> None:
        t = table
        # COLLATE "C" orders by code point, like every other backend, so max_key
        # agrees with them. It is also what lets starts_with() seek the primary
        # key (Postgres 15+) instead of scanning the whole table, which the
        # LangGraph adapter's list() and delete_thread() rely on. Tables created
        # before this keep the database collation; running
        # ALTER COLUMN ... TYPE text COLLATE "C" on ns and k upgrades them.
        self.create = (
            f"CREATE TABLE IF NOT EXISTS {t} "
            '(ns text COLLATE "C" NOT NULL, k text COLLATE "C" NOT NULL, '
            "v bytea NOT NULL, PRIMARY KEY (ns, k))"
        )
        self.upsert = (
            f"INSERT INTO {t} (ns, k, v) VALUES (%s, %s, %s) "
            "ON CONFLICT (ns, k) DO UPDATE SET v = EXCLUDED.v"
        )
        self.get = f"SELECT v FROM {t} WHERE ns = %s AND k = %s"
        # unnest two arrays positionally, then join: one round-trip for the batch.
        self.get_many = (
            f"SELECT t.ns, t.k, t.v FROM {t} t "
            "JOIN unnest(%s::text[], %s::text[]) AS q(ns, k) "
            "ON t.ns = q.ns AND t.k = q.k"
        )
        self.contains = f"SELECT 1 FROM {t} WHERE ns = %s AND k = %s"
        self.delete = f"DELETE FROM {t} WHERE ns = %s AND k = %s"
        self.delete_many = (
            f"DELETE FROM {t} t "
            "USING unnest(%s::text[], %s::text[]) AS q(ns, k) "
            "WHERE t.ns = q.ns AND t.k = q.k"
        )
        self.delete_namespace = f"DELETE FROM {t} WHERE ns = %s"
        self.keys = f"SELECT k FROM {t} WHERE ns = %s"
        # starts_with, not LIKE: keys carry user data that may contain LIKE
        # wildcards.
        self.keys_prefix = f"SELECT k FROM {t} WHERE ns = %s AND starts_with(k, %s)"
        self.max_key = f"SELECT max(k) FROM {t} WHERE ns = %s"
        self.namespaces = f"SELECT DISTINCT ns FROM {t}"
        self.namespaces_prefix = f"SELECT DISTINCT ns FROM {t} WHERE starts_with(ns, %s)"
        self.clear = f"DELETE FROM {t}"
        self.count = f"SELECT count(*) FROM {t}"
        self.has_namespace = f"SELECT 1 FROM {t} WHERE ns = %s LIMIT 1"
        self.all_rows = f"SELECT ns, k, v FROM {t}"
        self.insert = f"INSERT INTO {t} (ns, k, v) VALUES (%s, %s, %s)"


def _unzip(pairs: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _in_order(pairs: list[tuple[str, str]], rows: list[Any]) -> list[Any]:
    found = {(ns, k): v for ns, k, v in rows}
    return [None if (p := (ns, k)) not in found else _unpack(found[p]) for ns, k in pairs]


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
        apool: Any = None,
        table: str = "swarmstate_kv",
        codec: str = "msgpack",
        max_size: int = 8,
    ) -> None:
        """
        Args:
            dsn: connection string, used for the connections this store opens.
            conn: a ready psycopg connection (autocommit) to use for everything sync.
            pool: a ready ``psycopg_pool.ConnectionPool`` for the sync methods.
            apool: a ready ``psycopg_pool.AsyncConnectionPool`` for the async ones.
                Without it, a store built from ``dsn`` opens its own per event
                loop, and a store given only ``conn``/``pool`` has no async support.
            table: table name (a plain identifier).
            codec: only ``"msgpack"``.
            max_size: size of the pools this store opens.
        """
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.table = table
        self.codec = codec
        self._q = _Queries(table)
        # Single-connection mode keeps a mutex, so concurrent callers queue up on
        # one connection; a pool lets them work in parallel, which is what a
        # multi-worker service needs. An injected `conn` opts out on purpose.
        self._lock = threading.Lock()
        self._snap_meta = SnapshotMeta()
        self._conn: Any = conn
        self._pool: Any = pool
        self._max_size = max_size
        self._dsn: Optional[str] = None
        if self._conn is None and self._pool is None:
            self._dsn = dsn
            self._pool = self._make_pool(dsn, max_size)
        # Async connections belong to the event loop that opened them.
        self._apool: Any = apool
        self._apools: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
        with self._connection() as c:
            c.execute(self._q.create)

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
            conn.execute(self._q.upsert, (namespace, key, _pack(value)))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        with self._connection() as conn:
            row = conn.execute(self._q.get, (namespace, key)).fetchone()
        return default if row is None else _unpack(row[0])

    def set_many(self, items: list[tuple[str, str, Any]]) -> None:
        if not items:
            return
        rows = [(ns, k, _pack(v)) for ns, k, v in items]
        # One transaction for the batch. The connections are in autocommit, so
        # without it every row committed on its own: a failure partway kept the
        # rows before it, and a batch paid one commit per row.
        with self._connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(self._q.upsert, rows)

    def get_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        if not pairs:
            return []
        with self._connection() as conn:
            rows = conn.execute(self._q.get_many, _unzip(pairs)).fetchall()
        return _in_order(pairs, rows)

    def contains(self, namespace: str, key: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(self._q.contains, (namespace, key)).fetchone()
        return row is not None

    def delete(self, namespace: str, key: str) -> bool:
        with self._connection() as conn:
            return bool(conn.execute(self._q.delete, (namespace, key)).rowcount > 0)

    def delete_many(self, pairs: list[tuple[str, str]]) -> int:
        """Delete many pairs in one statement; returns how many existed."""
        if not pairs:
            return 0
        with self._connection() as conn:
            return int(conn.execute(self._q.delete_many, _unzip(pairs)).rowcount)

    def delete_namespace(self, namespace: str) -> int:
        """Delete every key in ``namespace``; returns how many were removed."""
        with self._connection() as conn:
            return int(conn.execute(self._q.delete_namespace, (namespace,)).rowcount)

    def keys(self, namespace: str, prefix: Optional[str] = None) -> list[str]:
        with self._connection() as conn:
            if prefix is None:
                rows = conn.execute(self._q.keys, (namespace,)).fetchall()
            else:
                rows = conn.execute(self._q.keys_prefix, (namespace, prefix)).fetchall()
        return [r[0] for r in rows]

    def max_key(self, namespace: str) -> Optional[str]:
        """The greatest key in ``namespace``, via the primary-key index."""
        with self._connection() as conn:
            row = conn.execute(self._q.max_key, (namespace,)).fetchone()
        return None if row is None else cast(Optional[str], row[0])

    def namespaces(self, prefix: Optional[str] = None) -> list[str]:
        with self._connection() as conn:
            if prefix is None:
                rows = conn.execute(self._q.namespaces, ()).fetchall()
            else:
                rows = conn.execute(self._q.namespaces_prefix, (prefix,)).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute(self._q.clear)

    def __len__(self) -> int:
        with self._connection() as conn:
            return int(conn.execute(self._q.count).fetchone()[0])

    def __contains__(self, namespace: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(self._q.has_namespace, (namespace,)).fetchone()
        return row is not None

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> PostgresSnapshot:
        with self._connection() as conn:
            rows = conn.execute(self._q.all_rows).fetchall()
        return PostgresSnapshot([(ns, k, bytes(v)) for ns, k, v in rows], *self._snap_meta.next())

    def restore(self, snapshot: PostgresSnapshot) -> None:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(self._q.clear)
                conn.cursor().executemany(self._q.insert, snapshot._rows)

    def close(self) -> None:
        """Close the pool (or the single connection)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        elif self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None

    # --------------------------------------------------------------- async API

    @property
    def supports_async(self) -> bool:
        """Whether the ``a*`` methods can run (an async pool is given or buildable)."""
        return self._apool is not None or self._dsn is not None

    async def _open_async(self) -> tuple[Any, Any, Any]:
        """``(pool, conn, lock)`` for this event loop: a pool, or one locked connection."""
        if self._dsn is None:
            raise RuntimeError(
                "this PostgresStore has no async connection: pass apool=, or build it from a DSN"
            )
        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError:
            import psycopg

            conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
            return None, conn, asyncio.Lock()
        pool = AsyncConnectionPool(
            self._dsn,
            min_size=1,
            max_size=self._max_size,
            open=False,
            kwargs={"autocommit": True},
        )
        await pool.open()
        return pool, None, None

    @asynccontextmanager
    async def _aconnection(self) -> AsyncIterator[Any]:
        """Async twin of :meth:`_connection`."""
        if self._apool is not None:
            async with self._apool.connection() as conn:
                yield conn
            return
        loop = asyncio.get_running_loop()
        entry = self._apools.get(loop)
        if entry is None:
            made = await self._open_async()
            # Two tasks can race to open the first one; keep a single winner.
            entry = self._apools.setdefault(loop, made)
            if entry is not made:
                await self._close_async(made)
        pool, conn, lock = entry
        if pool is not None:
            async with pool.connection() as pooled:
                yield pooled
        else:
            async with lock:
                yield conn

    @staticmethod
    async def _close_async(entry: tuple[Any, Any, Any]) -> None:
        pool, conn, _ = entry
        if pool is not None:
            await pool.close()
        else:
            await conn.close()

    async def aset(self, namespace: str, key: str, value: Any) -> None:
        async with self._aconnection() as conn:
            await conn.execute(self._q.upsert, (namespace, key, _pack(value)))

    async def aget(self, namespace: str, key: str, default: Any = None) -> Any:
        async with self._aconnection() as conn:
            row = await (await conn.execute(self._q.get, (namespace, key))).fetchone()
        return default if row is None else _unpack(row[0])

    async def aset_many(self, items: list[tuple[str, str, Any]]) -> None:
        if not items:
            return
        rows = [(ns, k, _pack(v)) for ns, k, v in items]
        async with self._aconnection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.executemany(self._q.upsert, rows)

    async def aget_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        if not pairs:
            return []
        async with self._aconnection() as conn:
            rows = await (await conn.execute(self._q.get_many, _unzip(pairs))).fetchall()
        return _in_order(pairs, rows)

    async def acontains(self, namespace: str, key: str) -> bool:
        async with self._aconnection() as conn:
            row = await (await conn.execute(self._q.contains, (namespace, key))).fetchone()
        return row is not None

    async def adelete(self, namespace: str, key: str) -> bool:
        async with self._aconnection() as conn:
            return bool((await conn.execute(self._q.delete, (namespace, key))).rowcount > 0)

    async def adelete_many(self, pairs: list[tuple[str, str]]) -> int:
        if not pairs:
            return 0
        async with self._aconnection() as conn:
            return int((await conn.execute(self._q.delete_many, _unzip(pairs))).rowcount)

    async def adelete_namespace(self, namespace: str) -> int:
        async with self._aconnection() as conn:
            return int((await conn.execute(self._q.delete_namespace, (namespace,))).rowcount)

    async def akeys(self, namespace: str, prefix: Optional[str] = None) -> list[str]:
        async with self._aconnection() as conn:
            if prefix is None:
                cur = await conn.execute(self._q.keys, (namespace,))
            else:
                cur = await conn.execute(self._q.keys_prefix, (namespace, prefix))
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def amax_key(self, namespace: str) -> Optional[str]:
        async with self._aconnection() as conn:
            row = await (await conn.execute(self._q.max_key, (namespace,))).fetchone()
        return None if row is None else cast(Optional[str], row[0])

    async def anamespaces(self, prefix: Optional[str] = None) -> list[str]:
        async with self._aconnection() as conn:
            if prefix is None:
                cur = await conn.execute(self._q.namespaces, ())
            else:
                cur = await conn.execute(self._q.namespaces_prefix, (prefix,))
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def aclose(self) -> None:
        """Close the async pool of the running event loop (an injected one too)."""
        if self._apool is not None:
            await self._apool.close()
            self._apool = None
            return
        entry = self._apools.pop(asyncio.get_running_loop(), None)
        if entry is not None:
            await self._close_async(entry)

    def __repr__(self) -> str:
        return f"PostgresStore(table={self.table!r}, codec='{self.codec}')"
