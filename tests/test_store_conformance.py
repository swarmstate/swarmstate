"""One contract, every backend.

``SwarmStateSaver`` (and any user code) treats the Rust ``Store`` and the
persistent backends interchangeably, so they have to behave identically. This
suite runs the same assertions against all of them — it is the executable version
of :mod:`swarmstate.protocols`. Backends whose dependency or service is missing
are skipped, never silently dropped from the matrix.
"""

import os

import pytest

import swarmstate as ss

PG_DSN = os.environ.get("SWARMSTATE_TEST_PG_DSN")


def _memory():
    return ss.Store()


def _disk(tmp_path):
    pytest.importorskip("msgpack")
    from swarmstate.backends.disk import DiskStore

    return DiskStore(str(tmp_path / "conformance.db"))


def _redis():
    pytest.importorskip("msgpack")
    fakeredis = pytest.importorskip("fakeredis")
    from swarmstate.backends.redis import RedisStore

    return RedisStore(client=fakeredis.FakeStrictRedis(), prefix=f"conf{os.getpid()}")


def _postgres():
    pytest.importorskip("msgpack")
    pytest.importorskip("psycopg")
    if not PG_DSN:
        pytest.skip("set SWARMSTATE_TEST_PG_DSN to run the Postgres backend")
    import uuid

    from swarmstate.backends.postgres import PostgresStore

    return PostgresStore(PG_DSN, table=f"kv_{uuid.uuid4().hex[:12]}")


@pytest.fixture(params=["memory", "disk", "redis", "postgres"])
def store(request, tmp_path):
    # Explicit branches, not `a or b`: an empty store is falsy (it defines
    # __len__), so short-circuiting would build the wrong backend.
    if request.param == "memory":
        made = _memory()
    elif request.param == "disk":
        made = _disk(tmp_path)
    elif request.param == "redis":
        made = _redis()
    else:
        made = _postgres()
    try:
        yield made
    finally:
        made.clear()
        close = getattr(made, "close", None)
        if close is not None:
            close()


PAYLOAD = {"step": 3, "ratio": 1.5, "tags": ["a", "b"], "nested": {"k": [1, 2]}, "n": None}


# ----------------------------------------------------------------- key/value


def test_set_get_and_default(store):
    store.set("wf", "a", PAYLOAD)
    assert store.get("wf", "a") == PAYLOAD
    assert store.get("wf", "missing") is None
    assert store.get("nope", "missing", "fallback") == "fallback"


def test_set_overwrites(store):
    store.set("wf", "a", {"v": 1})
    store.set("wf", "a", {"v": 2})
    assert store.get("wf", "a") == {"v": 2}
    assert len(store) == 1


def test_batch_ops_preserve_order_and_report_misses(store):
    store.set_many([("a", "x", 1), ("a", "y", 2), ("b", "z", 3)])
    assert len(store) == 3
    assert store.get_many([("a", "y"), ("b", "z"), ("a", "nope"), ("a", "x")]) == [2, 3, None, 1]
    store.set_many([])  # a no-op, not an error
    assert store.get_many([]) == []


def test_contains_delete_and_membership(store):
    store.set("wf", "a", 1)
    assert store.contains("wf", "a")
    assert not store.contains("wf", "b")
    assert "wf" in store
    assert "other" not in store

    assert store.delete("wf", "a") is True
    assert store.delete("wf", "a") is False  # already gone
    assert not store.contains("wf", "a")


def test_emptied_namespace_is_no_longer_listed(store):
    """Deleted threads must not leave namespaces behind: callers scan this list."""
    store.set("wf", "a", 1)
    store.set("other", "b", 1)
    assert sorted(store.namespaces()) == ["other", "wf"]

    store.delete("wf", "a")
    assert store.namespaces() == ["other"]
    assert "wf" not in store


def test_keys_namespaces_len_and_clear(store):
    store.set_many([("a", "x", 1), ("a", "y", 2), ("b", "z", 3)])
    assert sorted(store.keys("a")) == ["x", "y"]
    assert store.keys("unknown") == []
    assert sorted(store.namespaces()) == ["a", "b"]
    assert len(store) == 3

    store.clear()
    assert len(store) == 0
    assert store.namespaces() == []


def test_prefix_filtering_on_namespaces_and_keys(store):
    store.set_many(
        [
            ("ck\x1ft1\x1f", "a", 1),
            ("ck\x1ft2\x1f", "b", 2),
            ("wr\x1ft1\x1f", "c", 3),
        ]
    )
    assert sorted(store.namespaces(prefix="ck\x1f")) == ["ck\x1ft1\x1f", "ck\x1ft2\x1f"]
    assert store.namespaces(prefix="ck\x1ft1\x1f") == ["ck\x1ft1\x1f"]
    assert store.namespaces(prefix="nope") == []
    assert store.namespaces(prefix=None) == store.namespaces()

    store.set_many([("ns", "task-1", 1), ("ns", "task-2", 2), ("ns", "other", 3)])
    assert sorted(store.keys("ns", prefix="task-")) == ["task-1", "task-2"]
    assert store.keys("ns", prefix="zzz") == []


def test_prefix_filtering_treats_wildcards_literally(store):
    """Thread ids are user data: SQL LIKE and Redis glob syntax must not leak in."""
    store.set_many(
        [
            ("ck\x1fa_b\x1f", "k", 1),
            ("ck\x1faXb\x1f", "k", 2),
            ("ck\x1f100%\x1f", "k", 3),
            ("ck\x1f1000\x1f", "k", 4),
            ("ck\x1fa[b]*?\x1f", "k", 5),
        ]
    )
    assert store.namespaces(prefix="ck\x1fa_b\x1f") == ["ck\x1fa_b\x1f"]
    assert store.namespaces(prefix="ck\x1f100%\x1f") == ["ck\x1f100%\x1f"]
    assert store.namespaces(prefix="ck\x1fa[b]*?\x1f") == ["ck\x1fa[b]*?\x1f"]

    store.set_many([("ns", "a_b", 1), ("ns", "aXb", 2)])
    assert store.keys("ns", prefix="a_") == ["a_b"]


#: The store interface, spelled out. Checked against `swarmstate.protocols` below so the
#: two cannot drift, and against every backend so "drop-in" is verified, not asserted.
STORE_INTERFACE = (
    "set",
    "get",
    "set_many",
    "get_many",
    "contains",
    "delete",
    "keys",
    "max_key",
    "namespaces",
    "clear",
    "snapshot",
    "restore",
    "__len__",
    "__contains__",
)


def test_satisfies_the_documented_store_interface(store):
    """Every bundled backend has the surface `swarmstate.protocols` writes down."""
    from swarmstate.protocols import StoreLike

    declared = {name for name in vars(StoreLike) if not name.startswith("_")}
    declared |= {"__len__", "__contains__"}
    assert declared == set(STORE_INTERFACE), "StoreLike and this test have drifted apart"

    for name in STORE_INTERFACE:
        assert callable(getattr(store, name)), f"{type(store).__name__} is missing {name}"


def test_max_key(store):
    """Every store answers "newest key"; only some do it from an index."""
    assert store.max_key("empty") is None

    store.set_many([("ns", "0002", 2), ("ns", "0001", 1), ("ns", "0003", 3)])
    assert store.max_key("ns") == "0003"

    store.delete("ns", "0003")
    assert store.max_key("ns") == "0002"
    # The flag is a promise about cost, not about the answer.
    assert isinstance(getattr(store, "indexed_max_key", False), bool)


def test_bytes_survive_the_roundtrip(store):
    store.set("wf", "raw", {"blob": b"\x00\x01\xff"})
    assert store.get("wf", "raw") == {"blob": b"\x00\x01\xff"}


# ------------------------------------------------------------------ snapshot


def test_snapshot_metadata_is_uniform(store):
    store.set("wf", "a", PAYLOAD)
    first = store.snapshot()
    second = store.snapshot()

    assert isinstance(first.id, int)
    assert first.timestamp > 0
    assert first.parent is None  # nothing preceded it
    assert second.parent == first.id  # chained for incremental diffs
    assert first.size_bytes > 0
    assert first.keys == [("wf", "a")]


def test_snapshot_diff_reports_added_removed_changed(store):
    store.set_many([("n", "keep", 1), ("n", "drop", 1)])
    base = store.snapshot()

    store.delete("n", "drop")
    store.set("n", "keep", 2)
    store.set("n", "new", 2)
    now = store.snapshot()

    d = now.diff(base)
    assert d["added"] == [("n", "new")]
    assert d["removed"] == [("n", "drop")]
    assert d["changed"] == [("n", "keep")]


def test_restore_rolls_state_back(store):
    store.set("wf", "a", {"step": 1})
    snap = store.snapshot()

    store.set("wf", "a", {"step": 2})
    store.set("wf", "b", {"step": 9})
    assert len(store) == 2

    store.restore(snap)
    assert store.get("wf", "a") == {"step": 1}
    assert store.get("wf", "b") is None
    assert len(store) == 1


def test_rejects_an_unsupported_codec(store):
    with pytest.raises(ValueError):
        type(store)(codec="pickle")
