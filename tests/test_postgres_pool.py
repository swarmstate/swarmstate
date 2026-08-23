"""Connection handling in PostgresStore: pooled vs single connection.

The rest of the Postgres suite needs a live server (``SWARMSTATE_TEST_PG_DSN``).
These tests inject a stub pool/connection instead, so the path a real deployment
takes — a connection per operation, from a pool — stays covered everywhere.
"""

from contextlib import contextmanager

import pytest

pytest.importorskip("msgpack")

from swarmstate.backends.postgres import PostgresStore


class RecordingCursor:
    rowcount = 0

    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []

    def executemany(self, sql, rows):
        return None


class RecordingConn:
    def __init__(self):
        self.statements: list[str] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.statements.append(sql)
        # Aggregates must come back with a row; lookups report "not found".
        return RecordingCursor((0,) if "count(*)" in sql else None)

    def cursor(self):
        return RecordingCursor()

    def close(self):
        self.closed = True


class RecordingPool:
    def __init__(self):
        self.conn = RecordingConn()
        self.checkouts = 0
        self.closed = False

    @contextmanager
    def connection(self):
        self.checkouts += 1
        yield self.conn

    def close(self):
        self.closed = True


def test_pooled_store_takes_one_connection_per_operation():
    pool = RecordingPool()
    store = PostgresStore(pool=pool)

    assert pool.checkouts == 1  # CREATE TABLE IF NOT EXISTS
    assert "CREATE TABLE IF NOT EXISTS" in pool.conn.statements[0]

    store.set("ns", "k", {"v": 1})
    store.get("ns", "k")
    store.contains("ns", "k")
    store.keys("ns")
    store.namespaces()
    len(store)
    assert pool.checkouts == 7  # every op borrows and returns a connection


def test_pooled_store_closes_the_pool():
    pool = RecordingPool()
    store = PostgresStore(pool=pool)
    store.close()
    assert pool.closed
    store.close()  # idempotent


def test_injected_connection_bypasses_the_pool():
    conn = RecordingConn()
    store = PostgresStore(conn=conn)

    assert store._pool is None
    store.set("ns", "k", {"v": 1})
    assert any("INSERT INTO" in sql for sql in conn.statements)

    store.close()
    assert conn.closed
