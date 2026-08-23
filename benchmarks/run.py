#!/usr/bin/env python3
"""swarmstate benchmarks - reproducible.

Four measurements, each comparing like with like:

1. **Checkpoint latency**, LangGraph checkpointer interface, in two groups that
   are never mixed: *durable* (``SwarmStateSaver`` over ``DiskStore`` vs
   ``SqliteSaver``, both SQLite files, paired at the same ``synchronous`` setting
   and also shown at SqliteSaver's shipped one) and *in-memory*
   (``SwarmStateSaver`` vs ``InMemorySaver``, neither of which survives a
   restart). Comparing an in-memory saver against a file-backed one measures the
   cost of durability, not the quality of an implementation.
2. **Read latency as a thread grows** - the cost of "resume this thread" once it
   holds thousands of checkpoints, which is where finding the latest by scanning
   diverges from looking it up.
3. **Snapshot cost vs state size** - ``Store.snapshot()`` (O(1), structural
   sharing) vs ``copy.deepcopy`` of an equivalent dict (the O(n) way to get an
   independent, mutable copy).
4. **Concurrency scaling** - set+get throughput vs thread count.

Everything is seeded and parameterised. Hardware, versions and payload sizes are
recorded in ``results.json``. Run:

    python benchmarks/run.py --iters 5000 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import swarmstate as ss

# --- optional deps (only needed to run the benchmark) ----------------------
try:
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    from swarmstate.backends.disk import DiskStore
    from swarmstate.integrations.langgraph import SwarmStateSaver

    HAVE_LG = True
except Exception:  # pragma: no cover
    HAVE_LG = False


def percentiles(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    q = statistics.quantiles(s, n=100)  # q[i] ~ (i+1)th percentile
    return {
        "p50_ms": round(statistics.median(s), 4),
        "p99_ms": round(q[98], 4),
        "mean_ms": round(statistics.fmean(s), 4),
        "min_ms": round(s[0], 4),
        "ops_per_s": round(1000.0 / statistics.fmean(s), 1),
    }


def make_checkpoint(payload: dict):
    """A realistic checkpoint + the config/metadata/versions a saver expects."""
    cp = empty_checkpoint()
    cp["channel_values"] = payload
    cp["channel_versions"] = {k: "1" for k in payload}
    new_versions = dict(cp["channel_versions"])
    metadata = {"source": "loop", "step": 1, "writes": {}}
    return cp, metadata, new_versions


def bench_checkpointer(name: str, saver, payload: dict, iters: int, warmup: int) -> dict:
    cp, metadata, new_versions = make_checkpoint(payload)
    thread = "bench-thread"

    def cfg(cid: str | None = None):
        c = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
        if cid:
            c["configurable"]["checkpoint_id"] = cid
        return c

    # PUT latency: each put is a fresh checkpoint id (parented to the previous).
    put_samples: list[float] = []
    prev = None
    for i in range(iters + warmup):
        cid = f"{i:032d}"
        cp["id"] = cid
        c = cfg(prev)
        t0 = time.perf_counter()
        saver.put(c, cp, metadata, new_versions)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            put_samples.append(dt)
        prev = cid

    # GET latency: fetch the latest tuple repeatedly.
    get_samples: list[float] = []
    gc = cfg()
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        saver.get_tuple(gc)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            get_samples.append(dt)

    return {"put": percentiles(put_samples), "get_tuple": percentiles(get_samples)}


def bench_read_scaling(thread_lengths: list[int], iters: int, warmup: int) -> dict:
    """`get_tuple` latency as a thread accumulates checkpoints.

    Reading "the latest checkpoint" is the operation an agent performs on every
    resume. A saver that finds it by scanning the thread's keys gets slower as the
    thread grows; one that keeps an index does not. That difference — not raw
    per-call overhead — is what shows up in a long-running deployment, so it is
    measured separately from the fixed-size latency above.
    """
    payload = {"messages": [{"role": "user", "content": "message " * 8} for _ in range(20)]}
    cp, metadata, new_versions = make_checkpoint(payload)
    out: dict = {"thread_lengths": thread_lengths, "series": {}}

    def fill(saver, n):
        prev = None
        for i in range(n):
            cid = f"{i:032d}"
            cp["id"] = cid
            c = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
            if prev:
                c["configurable"]["checkpoint_id"] = prev
            saver.put(c, cp, metadata, new_versions)
            prev = cid

    for n in thread_lengths:
        for label, saver in (
            ("SwarmStateSaver", SwarmStateSaver()),
            ("InMemorySaver", InMemorySaver()),
        ):
            fill(saver, n)
            gc = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
            samples = []
            for i in range(iters + warmup):
                t0 = time.perf_counter()
                saver.get_tuple(gc)
                if i >= warmup:
                    samples.append((time.perf_counter() - t0) * 1000.0)
            out["series"].setdefault(label, []).append(round(statistics.median(samples), 5))
    return out


def bench_snapshot(sizes: list[int], iters: int, warmup: int) -> dict:
    out = {"sizes": sizes, "store_snapshot_ms": [], "dict_deepcopy_ms": []}
    for k in sizes:
        payload = {str(i): {"v": i, "s": "x" * 16} for i in range(k)}

        store = ss.Store()
        for key, val in payload.items():
            store.set("s", key, val)
        samples = []
        for i in range(iters + warmup):
            t0 = time.perf_counter()
            store.snapshot()
            if i >= warmup:
                samples.append((time.perf_counter() - t0) * 1000.0)
        out["store_snapshot_ms"].append(round(statistics.fmean(samples), 5))

        d = dict(payload)
        samples = []
        for i in range(iters + warmup):
            t0 = time.perf_counter()
            copy.deepcopy(d)
            if i >= warmup:
                samples.append((time.perf_counter() - t0) * 1000.0)
        out["dict_deepcopy_ms"].append(round(statistics.fmean(samples), 5))
    return out


def bench_concurrency(thread_counts: list[int], ops: int) -> dict:
    """Throughput (ops/s) of set+get under N threads: swarmstate Store vs a
    plain dict guarded by a Lock. The Store releases the GIL on the map op, so
    work overlaps across threads where the locked dict fully serializes."""
    import threading

    payload = {"v": 1, "s": "x" * 32}
    out = {"threads": thread_counts, "store_ops_s": [], "dict_lock_ops_s": []}

    for n in thread_counts:
        store = ss.Store()

        def store_worker(tid, store=store):
            ns = f"t{tid}"
            for i in range(ops):
                store.set(ns, str(i), payload)
                store.get(ns, str(i))

        threads = [threading.Thread(target=store_worker, args=(t,)) for t in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out["store_ops_s"].append(round(n * ops * 2 / (time.perf_counter() - t0)))

        d: dict = {}
        lock = threading.Lock()

        def dict_worker(tid, d=d, lock=lock):
            for i in range(ops):
                with lock:
                    d[(tid, i)] = payload
                with lock:
                    d.get((tid, i))

        threads = [threading.Thread(target=dict_worker, args=(t,)) for t in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out["dict_lock_ops_s"].append(round(n * ops * 2 / (time.perf_counter() - t0)))
    return out


# Validated dark-mode categorical palette (dataviz skill: all checks pass on
# surface #0d1117 - blue / aqua / orange).
_COLORS = {
    # Colour follows the entity: swarmstate keeps the same blue in both panels.
    "SwarmStateSaver": "#3987e5",
    "SwarmStateSaver+DiskStore": "#3987e5",
    "InMemorySaver": "#199e70",
    "SqliteSaver": "#d95926",
    "SqliteSaver (synchronous=FULL)": "#d95926",
}
_INK, _MUTED, _GRID = "#e6edf3", "#8b97a5", "#2c2c2a"

# The two comparisons that belong on the same axes, as (title, series) pairs. The
# shipped-pragma SqliteSaver row stays out of the chart and in the data table:
# three bars would put two shades of one hue side by side.
_PANELS = [
    ("durable (survives a restart)", ["SwarmStateSaver+DiskStore", "SqliteSaver"]),
    ("in-memory (does not)", ["SwarmStateSaver", "InMemorySaver"]),
]


def _style_axes(ax):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def make_charts(results: dict, outdir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Chart 1: checkpoint latency (put/get p50), µs ---
    # One panel per comparison rather than six bars on shared axes: durable and
    # in-memory savers are an order of magnitude apart, and putting them on one
    # scale would both flatten the in-memory pair and invite reading across
    # groups that are not comparable. Each panel keeps its own axis, and every
    # bar is labelled with its value so nothing rests on bar height alone.
    metrics = [("put", "write (put)"), ("get_tuple", "read (get_tuple)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    fig.patch.set_alpha(0)
    for ax, (panel_title, series) in zip(axes, _PANELS):
        series = [s for s in series if s in results["checkpointer"]]
        xb = range(len(metrics))
        w = 0.30
        for i, name in enumerate(series):
            vals = [results["checkpointer"][name][m]["p50_ms"] * 1000 for m, _ in metrics]
            xs = [x + (i - (len(series) - 1) / 2) * w for x in xb]
            bars = ax.bar(xs, vals, width=w * 0.8, color=_COLORS[name], label=name)
            for rect, v in zip(bars, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    v,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    color=_INK,
                    fontsize=8,
                )
        ax.set_xticks(list(xb))
        ax.set_xticklabels([lbl for _, lbl in metrics], color=_INK, fontsize=9)
        ax.set_title(panel_title, color=_INK, fontsize=10, pad=8)
        ax.set_ylabel("latency p50 (µs), lower is better", color=_MUTED, fontsize=9)
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=_INK,
            loc="upper center",
            ncol=1,
            bbox_to_anchor=(0.5, -0.14),
        )
        _style_axes(ax)
    fig.suptitle(
        "Checkpointer latency (LangGraph interface) — each panel has its own scale",
        color=_INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outdir / "checkpoint_latency.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: snapshot cost vs state size (log-log) ---
    snap = results["snapshot"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.patch.set_alpha(0)
    ax.loglog(
        snap["sizes"],
        snap["store_snapshot_ms"],
        "-o",
        color=_COLORS["SwarmStateSaver"],
        linewidth=2,
        markersize=6,
        label="Store.snapshot()  (O(1))",
    )
    ax.loglog(
        snap["sizes"],
        snap["dict_deepcopy_ms"],
        "-o",
        color=_COLORS["SqliteSaver"],
        linewidth=2,
        markersize=6,
        label="dict deepcopy  (O(n))",
    )
    ax.set_xlabel("entries in state", color=_MUTED, fontsize=9)
    ax.set_ylabel("snapshot time (ms), lower is better", color=_MUTED, fontsize=9)
    ax.set_title("Snapshot cost vs state size", color=_INK, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")
    _style_axes(ax)
    ax.grid(True, which="both", color=_GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "snapshot_scaling.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 3: read latency as a thread accumulates checkpoints ---
    read = results.get("read_scaling")
    if read:
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        fig.patch.set_alpha(0)
        for label, values in read["series"].items():
            ax.plot(
                read["thread_lengths"],
                [v * 1000 for v in values],
                "-o",
                color=_COLORS[label],
                linewidth=2,
                markersize=6,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("checkpoints in the thread", color=_MUTED, fontsize=9)
        ax.set_ylabel("get_tuple p50 (µs), lower is better", color=_MUTED, fontsize=9)
        ax.set_title(
            "Reading the latest checkpoint, as a thread grows", color=_INK, fontsize=12, pad=12
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(outdir / "read_scaling.svg", transparent=True, bbox_inches="tight")
        plt.close(fig)

    # --- Chart 4: concurrency scaling (throughput vs threads) ---
    conc = results.get("concurrency")
    if conc:
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        fig.patch.set_alpha(0)
        ax.plot(
            conc["threads"],
            conc["store_ops_s"],
            "-o",
            color=_COLORS["SwarmStateSaver"],
            linewidth=2,
            markersize=6,
            label="swarmstate Store (GIL released)",
        )
        ax.plot(
            conc["threads"],
            conc["dict_lock_ops_s"],
            "-o",
            color=_COLORS["SqliteSaver"],
            linewidth=2,
            markersize=6,
            label="dict + Lock (pure Python)",
        )
        ax.set_xlabel("threads", color=_MUTED, fontsize=9)
        ax.set_ylabel("throughput (set+get ops/s), higher is better", color=_MUTED, fontsize=9)
        ax.set_title("Concurrency scaling", color=_INK, fontsize=12, pad=12)
        ax.set_xticks(conc["threads"])
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(outdir / "concurrency.svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote charts to {outdir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="swarmstate benchmarks")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--payload-msgs", type=int, default=20)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    if not HAVE_LG:
        sys.exit(
            "Install deps first: pip install '.[langgraph]' langgraph-checkpoint-sqlite matplotlib"
        )

    # A realistic checkpoint payload: a message history channel.
    payload = {
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i} " * 8}
            for i in range(args.payload_msgs)
        ],
        "step": 1,
    }

    results: dict = {
        "meta": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
            "swarmstate": ss.__version__,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
            "payload_msgs": args.payload_msgs,
            "note": (
                "Two separate comparisons. Durable: SwarmStateSaver over DiskStore vs "
                "SqliteSaver, both SQLite files in WAL mode; the headline pairs them at the "
                "same synchronous=NORMAL fsync policy, and SqliteSaver is also measured at "
                "its shipped synchronous=FULL. In-memory: SwarmStateSaver vs InMemorySaver, "
                "neither of which survives a restart. Warm cache, single process, no "
                "concurrent load."
            ),
        },
        "checkpointer": {},
    }

    print(f"# swarmstate benchmarks  (iters={args.iters}, warmup={args.warmup})")
    print(
        f"# {results['meta']['platform']} · py{results['meta']['python']} · swarmstate {ss.__version__}\n"
    )

    # --- checkpointer latency -------------------------------------------------
    # Two comparisons, deliberately kept apart. A checkpointer that survives a
    # restart does strictly more work than one that does not, so the headline is
    # durable-vs-durable: SwarmStateSaver over DiskStore against SqliteSaver, both
    # SQLite files in WAL mode at the same `synchronous` setting. SqliteSaver is
    # measured at its shipped setting too, so the reader can see the win is not a
    # pragma trick.
    def record(label: str, saver, *, durable: bool, config: str) -> None:
        entry = bench_checkpointer(label, saver, payload, args.iters, args.warmup)
        entry["durable"] = durable
        entry["config"] = config
        results["checkpointer"][label] = entry

    with tempfile.TemporaryDirectory() as td:
        disk = DiskStore(str(Path(td) / "swarmstate.db"))
        try:
            record(
                "SwarmStateSaver+DiskStore",
                SwarmStateSaver(disk),
                durable=True,
                config="SQLite file · WAL · synchronous=NORMAL",
            )
        finally:
            disk.close()

        for label, sync in (
            ("SqliteSaver", "NORMAL"),
            ("SqliteSaver (synchronous=FULL)", "FULL"),
        ):
            with SqliteSaver.from_conn_string(str(Path(td) / f"cp-{sync}.sqlite")) as sq:
                sq.setup()
                # SqliteSaver already runs WAL; only `synchronous` differs, and at
                # NORMAL it matches what DiskStore guarantees.
                sq.conn.execute(f"PRAGMA synchronous={sync}")
                record(
                    label,
                    sq,
                    durable=True,
                    config=f"SQLite file · WAL · synchronous={sync}",
                )

    record("SwarmStateSaver", SwarmStateSaver(), durable=False, config="in-process, not durable")
    record("InMemorySaver", InMemorySaver(), durable=False, config="in-process, not durable")

    header = f"{'checkpointer':<32} {'durable':>7} {'put p50':>9} {'put p99':>9} {'get p50':>9}"
    print(header)
    for name, r in results["checkpointer"].items():
        print(
            f"{name:<32} {'yes' if r['durable'] else 'no':>7} {r['put']['p50_ms']:>9.4f} "
            f"{r['put']['p99_ms']:>9.4f} {r['get_tuple']['p50_ms']:>9.4f}"
        )

    ck = results["checkpointer"]
    speedups = {
        "durable_put": ck["SqliteSaver"]["put"]["p50_ms"]
        / ck["SwarmStateSaver+DiskStore"]["put"]["p50_ms"],
        "durable_get": ck["SqliteSaver"]["get_tuple"]["p50_ms"]
        / ck["SwarmStateSaver+DiskStore"]["get_tuple"]["p50_ms"],
        "in_memory_put": ck["InMemorySaver"]["put"]["p50_ms"]
        / ck["SwarmStateSaver"]["put"]["p50_ms"],
        "in_memory_get": ck["InMemorySaver"]["get_tuple"]["p50_ms"]
        / ck["SwarmStateSaver"]["get_tuple"]["p50_ms"],
    }
    results["meta"]["speedups"] = {k: round(v, 2) for k, v in speedups.items()}
    print(
        f"\n  durable, same fsync policy: put {speedups['durable_put']:.1f}x and "
        f"get_tuple {speedups['durable_get']:.1f}x vs SqliteSaver"
    )
    print(
        f"  in-memory vs InMemorySaver: put {speedups['in_memory_put']:.2f}x, "
        f"get_tuple {speedups['in_memory_get']:.2f}x\n"
    )

    # --- read latency as a thread grows ---
    results["read_scaling"] = bench_read_scaling([5, 50, 500, 2000], iters=500, warmup=100)
    print(
        f"{'checkpoints in thread':>21} "
        + " ".join(f"{k:>16}" for k in results["read_scaling"]["series"])
    )
    for i, n in enumerate(results["read_scaling"]["thread_lengths"]):
        row = " ".join(f"{v[i]:>16.5f}" for v in results["read_scaling"]["series"].values())
        print(f"{n:>21} {row}")
    print()

    # --- snapshot scaling ---
    results["snapshot"] = bench_snapshot([100, 1000, 10000, 50000], iters=200, warmup=20)
    print(f"{'state keys':>10} {'Store.snapshot ms':>18} {'dict deepcopy ms':>18}")
    for k, a, b in zip(
        results["snapshot"]["sizes"],
        results["snapshot"]["store_snapshot_ms"],
        results["snapshot"]["dict_deepcopy_ms"],
    ):
        print(f"{k:>10,} {a:>18.5f} {b:>18.5f}")

    # --- concurrency scaling ---
    results["concurrency"] = bench_concurrency([1, 2, 4, 8], ops=20000)
    print(f"\n{'threads':>8} {'Store ops/s':>14} {'dict+lock ops/s':>16}")
    for t, a, b in zip(
        results["concurrency"]["threads"],
        results["concurrency"]["store_ops_s"],
        results["concurrency"]["dict_lock_ops_s"],
    ):
        print(f"{t:>8} {a:>14,} {b:>16,}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.outdir / 'results.json'}")

    try:
        make_charts(results, args.outdir / "charts")
    except Exception as e:  # pragma: no cover - charts are optional
        print(f"(charts skipped: {e})")


if __name__ == "__main__":
    main()
