"""Micro-benchmark: put_writes batching (set_many) vs per-write set.

Simulates a LangGraph fan-out step that emits many pending writes per
checkpoint, which is the case where batching the store writes pays off most.

Run:  python benchmarks/bench_put_writes.py [--writes 32] [--steps 2000]
"""

from __future__ import annotations

import argparse
import statistics
import time

import swarmstate as ss
from swarmstate.integrations.langgraph import SwarmStateSaver, _writes_ns


class _PerWriteStore:
    """A Store that hides set_many, forcing the saver's per-item fallback.

    This reproduces the pre-batching code path so the two strategies can be
    compared on identical data with the same underlying Rust store.
    """

    def __init__(self) -> None:
        self._inner = ss.Store()

    def __getattr__(self, name: str):
        if name in ("set_many", "get_many"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def _run(store, n_writes: int, n_steps: int) -> float:
    saver = SwarmStateSaver(store)
    writes = [(f"ch{i}", {"v": i, "payload": "x" * 32}) for i in range(n_writes)]
    t0 = time.perf_counter()
    for step in range(n_steps):
        cfg = {
            "configurable": {
                "thread_id": "bench",
                "checkpoint_ns": "",
                "checkpoint_id": f"cp-{step:08d}",
            }
        }
        saver.put_writes(cfg, writes, task_id=f"task-{step}")
    elapsed = time.perf_counter() - t0
    # Sanity: the last step actually stored all writes.
    assert len(store.keys(_writes_ns("bench", "", f"cp-{n_steps - 1:08d}"))) == n_writes
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--writes", type=int, default=32, help="writes per step (fan-out width)")
    ap.add_argument("--steps", type=int, default=2000, help="checkpoint steps")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    batched, per_write = [], []
    for _ in range(args.repeat):
        batched.append(_run(ss.Store(), args.writes, args.steps))
        per_write.append(_run(_PerWriteStore(), args.writes, args.steps))

    b = statistics.median(batched)
    p = statistics.median(per_write)
    total = args.writes * args.steps
    print(f"put_writes: {args.steps} steps x {args.writes} writes = {total} writes\n")
    print(f"  per-write set() : {p * 1e3:8.1f} ms  ({total / p:10.0f} writes/s)")
    print(f"  batched set_many: {b * 1e3:8.1f} ms  ({total / b:10.0f} writes/s)")
    print(f"  speedup         : {p / b:8.2f}x")


if __name__ == "__main__":
    main()
