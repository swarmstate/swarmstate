#!/usr/bin/env python3
"""swarmstate benchmarks — reproducible.

Measures two things that matter for production checkpointing:

1. **Checkpoint latency & throughput** of the LangGraph checkpointer interface:
   ``SwarmStateSaver`` vs ``InMemorySaver`` (both in-memory) vs ``SqliteSaver``
   (file-backed, the common way to get persistence today).
2. **Snapshot cost vs state size** — ``Store.snapshot()`` (O(1), structural
   sharing) vs ``copy.deepcopy`` of an equivalent dict (the O(n) way to get an
   independent, mutable copy).

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


# Validated dark-mode categorical palette (dataviz skill: all checks pass on
# surface #0d1117 — blue / aqua / orange).
_COLORS = {"SwarmStateSaver": "#3987e5", "InMemorySaver": "#199e70", "SqliteSaver": "#d95926"}
_INK, _MUTED, _GRID = "#e6edf3", "#8b97a5", "#2c2c2a"


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
    names = list(results["checkpointer"].keys())

    # --- Chart 1: checkpoint latency (put/get p50), grouped bars, µs ---
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.patch.set_alpha(0)
    metrics = [("put", "write (put)"), ("get_tuple", "read (get_tuple)")]
    xb = range(len(metrics))
    w = 0.26
    for i, name in enumerate(names):
        vals = [results["checkpointer"][name][m]["p50_ms"] * 1000 for m, _ in metrics]
        xs = [x + (i - 1) * w for x in xb]
        bars = ax.bar(xs, vals, width=w * 0.9, color=_COLORS[name], label=name)
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
    ax.set_xticklabels([lbl for _, lbl in metrics], color=_INK, fontsize=10)
    ax.set_ylabel("latency p50 (µs) — lower is better", color=_MUTED, fontsize=9)
    ax.set_title("Checkpointer latency (LangGraph interface)", color=_INK, fontsize=12, pad=12)
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=_INK,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
    )
    _style_axes(ax)
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
    ax.set_ylabel("snapshot time (ms) — lower is better", color=_MUTED, fontsize=9)
    ax.set_title("Snapshot cost vs state size", color=_INK, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")
    _style_axes(ax)
    ax.grid(True, which="both", color=_GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "snapshot_scaling.svg", transparent=True, bbox_inches="tight")
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
            "note": "SwarmStateSaver and InMemorySaver are in-memory; SqliteSaver is file-backed (persistent). Warm cache.",
        },
        "checkpointer": {},
    }

    print(f"# swarmstate benchmarks  (iters={args.iters}, warmup={args.warmup})")
    print(
        f"# {results['meta']['platform']} · py{results['meta']['python']} · swarmstate {ss.__version__}\n"
    )

    # --- checkpointer latency ---
    results["checkpointer"]["SwarmStateSaver"] = bench_checkpointer(
        "SwarmStateSaver", SwarmStateSaver(), payload, args.iters, args.warmup
    )
    results["checkpointer"]["InMemorySaver"] = bench_checkpointer(
        "InMemorySaver", InMemorySaver(), payload, args.iters, args.warmup
    )
    with tempfile.TemporaryDirectory() as td:
        with SqliteSaver.from_conn_string(str(Path(td) / "cp.sqlite")) as sq:
            sq.setup()
            results["checkpointer"]["SqliteSaver"] = bench_checkpointer(
                "SqliteSaver", sq, payload, args.iters, args.warmup
            )

    print(f"{'checkpointer':<18} {'put p50':>9} {'put p99':>9} {'put ops/s':>11} {'get p50':>9}")
    for name, r in results["checkpointer"].items():
        print(
            f"{name:<18} {r['put']['p50_ms']:>9.4f} {r['put']['p99_ms']:>9.4f} "
            f"{r['put']['ops_per_s']:>11,.0f} {r['get_tuple']['p50_ms']:>9.4f}"
        )
    sq_put = results["checkpointer"]["SqliteSaver"]["put"]["p50_ms"]
    ss_put = results["checkpointer"]["SwarmStateSaver"]["put"]["p50_ms"]
    results["meta"]["put_speedup_vs_sqlite"] = round(sq_put / ss_put, 1)
    print(f"\n  SwarmStateSaver put p50 is {sq_put / ss_put:.1f}x faster than SqliteSaver (file)\n")

    # --- snapshot scaling ---
    results["snapshot"] = bench_snapshot([100, 1000, 10000, 50000], iters=200, warmup=20)
    print(f"{'state keys':>10} {'Store.snapshot ms':>18} {'dict deepcopy ms':>18}")
    for k, a, b in zip(
        results["snapshot"]["sizes"],
        results["snapshot"]["store_snapshot_ms"],
        results["snapshot"]["dict_deepcopy_ms"],
    ):
        print(f"{k:>10,} {a:>18.5f} {b:>18.5f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.outdir / 'results.json'}")

    try:
        make_charts(results, args.outdir / "charts")
    except Exception as e:  # pragma: no cover - charts are optional
        print(f"(charts skipped: {e})")


if __name__ == "__main__":
    main()
