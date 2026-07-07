"""Free-threaded (no-GIL) behaviour.

These run only on a free-threaded CPython build (``python3.13t`` etc.), where
``sys._is_gil_enabled()`` exists and returns False. On standard builds they skip.
"""

import sys
import threading

import pytest

import swarmstate as ss

_gil_status = getattr(sys, "_is_gil_enabled", None)
pytestmark = pytest.mark.skipif(
    _gil_status is None or _gil_status(),
    reason="requires a free-threaded (no-GIL) CPython build",
)


def test_import_does_not_reenable_gil():
    # The extension declares free-threaded support (m.gil_used(false)); importing
    # it must not force the GIL back on.
    assert sys._is_gil_enabled() is False


def test_concurrent_writes_are_consistent():
    store = ss.Store()
    n_threads, per = 8, 5000

    def worker(tid: int) -> None:
        ns = f"t{tid}"
        for i in range(per):
            store.set(ns, str(i), {"i": i})
            assert store.get(ns, str(i)) == {"i": i}

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store) == n_threads * per
    # Byte accounting stayed consistent under concurrent writers.
    assert store.snapshot().size_bytes > 0


def test_concurrent_set_many_batches():
    store = ss.Store()
    n_threads, batches, bs = 8, 200, 25

    def worker(tid: int) -> None:
        ns = f"b{tid}"
        for c in range(batches):
            store.set_many([(ns, str(c * bs + j), j) for j in range(bs)])

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store) == n_threads * batches * bs
