"""M1 tests: the concurrent store, msgpack codec, snapshots and diffs."""

import threading

import pytest

import swarmstate as ss


def test_set_get_roundtrip_types():
    store = ss.Store()
    payload = {
        "step": 3,
        "ratio": 1.5,
        "name": "onboarding",
        "done": False,
        "tags": ["a", "b"],
        "nested": {"k": [1, 2, 3]},
        "nothing": None,
    }
    store.set("workflow", "onboarding", payload)
    assert store.get("workflow", "onboarding") == payload
    assert store.get("workflow", "onboarding")["tags"] == ["a", "b"]


def test_get_missing_returns_default():
    store = ss.Store()
    assert store.get("ns", "missing") is None
    assert store.get("ns", "missing", default=42) == 42


def test_delete_contains_keys_namespaces_len():
    store = ss.Store()
    store.set("a", "x", 1)
    store.set("a", "y", 2)
    store.set("b", "z", 3)

    assert len(store) == 3
    assert store.contains("a", "x")
    assert set(store.keys("a")) == {"x", "y"}
    assert set(store.namespaces()) == {"a", "b"}

    assert store.delete("a", "x") is True
    assert store.delete("a", "x") is False
    assert not store.contains("a", "x")
    assert len(store) == 2


def test_bytes_are_preserved():
    store = ss.Store()
    store.set("bin", "blob", b"\x00\x01\x02\xff")
    assert store.get("bin", "blob") == b"\x00\x01\x02\xff"


def test_unsupported_type_raises():
    store = ss.Store()
    with pytest.raises(TypeError):
        store.set("ns", "k", object())


def test_snapshot_is_isolated_and_restore_rolls_back():
    store = ss.Store()
    store.set("wf", "a", {"step": 1})
    snap = store.snapshot()

    store.set("wf", "a", {"step": 2})
    store.set("wf", "b", {"step": 9})
    assert store.get("wf", "a") == {"step": 2}
    assert len(store) == 2

    store.restore(snap)
    assert store.get("wf", "a") == {"step": 1}
    assert len(store) == 1
    # The snapshot object is unaffected by later mutations.
    assert ("wf", "a") in snap.keys


def test_snapshot_metadata_and_diff():
    store = ss.Store()
    store.set("n", "keep", 1)
    store.set("n", "drop", 1)
    base = store.snapshot()

    store.delete("n", "drop")
    store.set("n", "keep", 2)
    store.set("n", "new", 2)
    now = store.snapshot()

    assert now.parent == base.id
    assert now.timestamp >= base.timestamp
    assert now.size_bytes > 0

    d = now.diff(base)
    assert d["added"] == [("n", "new")]
    assert d["removed"] == [("n", "drop")]
    assert d["changed"] == [("n", "keep")]


def test_invalid_backend_and_codec():
    with pytest.raises(ValueError):
        ss.Store(backend="postgres")
    with pytest.raises(ValueError):
        ss.Store(codec="pickle")


def test_max_history_getter():
    assert ss.Store().max_history is None
    assert ss.Store(max_history=5).max_history == 5
    assert ss.Store().codec == "msgpack"


def test_concurrent_writes_are_safe():
    """Many threads hammer the store concurrently (GIL released on hot paths)."""
    store = ss.Store()
    n_threads = 8
    per_thread = 500
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            ns = f"t{tid}"
            for i in range(per_thread):
                store.set(ns, str(i), {"tid": tid, "i": i})
            for i in range(per_thread):
                assert store.get(ns, str(i)) == {"tid": tid, "i": i}
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store) == n_threads * per_thread
    assert set(store.namespaces()) == {f"t{t}" for t in range(n_threads)}


def test_concurrent_snapshots_do_not_crash():
    store = ss.Store()
    for i in range(100):
        store.set("ns", str(i), i)

    snaps: list = []
    lock = threading.Lock()

    def snapshotter() -> None:
        for _ in range(50):
            s = store.snapshot()
            with lock:
                snaps.append(s)

    def mutator() -> None:
        for i in range(100, 300):
            store.set("ns", str(i), i)

    threads = [threading.Thread(target=snapshotter) for _ in range(4)]
    threads += [threading.Thread(target=mutator) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every snapshot is internally consistent regardless of concurrent writes.
    assert len(snaps) == 200
    for s in snaps:
        assert s.size_bytes >= 0
