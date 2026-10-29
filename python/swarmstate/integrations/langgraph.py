"""Drop-in LangGraph checkpointer backed by a swarmstate :class:`~swarmstate.Store`.

``SwarmStateSaver`` implements LangGraph's :class:`BaseCheckpointSaver` interface
(``put``, ``put_writes``, ``get_tuple``, ``list`` and their async variants), so it
is a **one-line replacement** for ``SqliteSaver`` / ``InMemorySaver``:

    from swarmstate.integrations.langgraph import SwarmStateSaver

    graph = builder.compile(checkpointer=SwarmStateSaver())

Checkpoints are stored in a swarmstate ``Store`` (Rust core), which means the same
store can be shared across graphs and snapshotted/rolled back as a whole:

    saver = SwarmStateSaver()
    snap = saver.store.snapshot()      # checkpoint the whole checkpoint DB
    ...
    saver.store.restore(snap)          # roll every thread back at once

It also implements the thread-management half of the interface — ``copy_thread``,
``delete_for_runs``, ``prune`` — and a direct ``get_delta_channel_history`` for
graphs that use ``DeltaChannel``.

:func:`handoff_router` turns a :class:`~swarmstate.HandoffGraph` into the routing
function ``add_conditional_edges`` takes, so rule-based handoffs cost no LLM call.

Requires the ``langgraph`` extra: ``pip install "swarmstate[langgraph]"``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, TypeVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from .. import HandoffGraph, Store
from ..observability import MetricsSink
from ..protocols import AsyncStoreLike

# Unit-separator delimiter: never appears in thread ids / namespaces.
_SEP = "\x1f"

# Namespace holding the "latest checkpoint id" pointer per (thread, ns). It lives
# in the store rather than in the saver so that every saver and every process
# sharing a backend agrees on which checkpoint is the latest one.
_LATEST_NS = "lt"

# Cap for the write-side hint dicts (see SwarmStateSaver._max_seen): they only
# avoid work, so dropping them wholesale when they grow is harmless.
_MAX_HINTS = 4096

# Retention runs in batches: a thread is trimmed back to the limit only once it
# is this far above it (or a quarter of the limit, whichever is larger). Pruning
# has to look at the surviving checkpoints, so batching keeps that off the hot
# path — the cost lands on one put in every `slack`, not on all of them.
_PRUNE_SLACK = 8

# Checkpoint metadata LangGraph writes for graphs with DeltaChannel state: its
# keys name the thread's delta channels. Those channels are absent from
# `channel_values` except at snapshot points, and are rebuilt by replaying the
# writes of the checkpoints before — which retention must therefore keep.
_DELTA_COUNTERS = "counters_since_delta_snapshot"

_PRUNE_STRATEGIES = ("keep_latest", "delete")

_T = TypeVar("_T")


def _mark_span_error(span: Any, exc: BaseException) -> None:
    """Record an exception on an OTel span and set its status to ERROR.

    Defensive: instrumentation must never mask the real failure, so any problem
    here (missing opentelemetry, a misbehaving span) is swallowed.
    """
    try:
        span.record_exception(exc)
    except Exception:
        pass
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass


def _ckpt_ns(thread_id: str, checkpoint_ns: str) -> str:
    return f"ck{_SEP}{thread_id}{_SEP}{checkpoint_ns}"


def _writes_ns(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"wr{_SEP}{thread_id}{_SEP}{checkpoint_ns}{_SEP}{checkpoint_id}"


def _blobs_ns(thread_id: str, checkpoint_ns: str) -> str:
    return f"bl{_SEP}{thread_id}{_SEP}{checkpoint_ns}"


def _latest_key(thread_id: str, checkpoint_ns: str) -> str:
    return f"{thread_id}{_SEP}{checkpoint_ns}"


def _write_order(key: str) -> tuple[str, int]:
    """Sort key for a pending-write key: ``(task_id, idx)`` with ``idx`` numeric.

    The keys are ``task_id<SEP>idx`` strings, and sorting them as text put write
    10 before write 2 — reordering a task's writes on resume once it had more
    than ten. The reference savers order by ``(task_id, idx)`` as an integer.
    """
    task_id, _, idx = key.rpartition(_SEP)
    return task_id, int(idx)


def _config(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def _bounded_set(hints: dict[Any, Any], key: Any, value: Any) -> None:
    if len(hints) >= _MAX_HINTS:
        hints.clear()
    hints[key] = value


class SwarmStateSaver(BaseCheckpointSaver[str]):  # type: ignore[misc]  # base is Any (no stubs)
    """A LangGraph checkpointer backed by a swarmstate :class:`~swarmstate.Store`.

    Args:
        store: underlying store; defaults to a fresh in-memory ``Store()``.
            Share one ``Store`` across savers/graphs for a unified checkpoint DB:
            the "latest checkpoint" pointer is kept in the store itself, so every
            saver (and every process, on a networked backend) sees the newest
            checkpoint regardless of which one wrote it.
        serde: optional LangGraph serializer (defaults to ``JsonPlusSerializer``).
        incremental: if True, store each channel value once per version (dedup)
            instead of the whole checkpoint blob per step. Saves storage and
            serialization for long threads with large, mostly-stable channels,
            at the cost of extra reads on ``get_tuple`` (one per channel). The
            default (False) keeps ``get_tuple`` at a single read.
        max_checkpoints_per_thread: keep only the newest N checkpoints of each
            thread, dropping older ones (with their pending writes, and their
            channel blobs when ``incremental``). ``None`` — the default —
            keeps every checkpoint, which means an in-memory store grows for as
            long as the process runs. Trimming happens in batches, so a thread
            can sit slightly above N before it is cut back; time travel past the
            retained window is no longer possible, so size it accordingly. For
            graphs with ``DeltaChannel`` state, the checkpoints back to each
            channel's last snapshot are kept as well, since the newest state is
            rebuilt from them.
        metrics: optional :class:`~swarmstate.observability.MetricsSink` that
            receives the latency and outcome of each ``put`` / ``put_writes`` /
            ``get_tuple``. Defaults to ``None`` (no measurement, zero overhead).
        tracer: optional OpenTelemetry ``Tracer``. When set, each ``put`` /
            ``put_writes`` / ``get_tuple`` runs inside a
            ``swarmstate.checkpoint.<op>`` span with thread/checkpoint attributes.
            Get one from :func:`swarmstate.observability.get_tracer`. Defaults to
            ``None`` (no spans, zero overhead).

    The async methods talk to the store natively when it supports it (a store
    whose ``supports_async`` is true, such as ``RedisStore`` or ``PostgresStore``
    built from a URL); otherwise they run the sync method in a worker thread.
    """

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        serde: Optional[SerializerProtocol] = None,
        incremental: bool = False,
        max_checkpoints_per_thread: Optional[int] = None,
        metrics: Optional[MetricsSink] = None,
        tracer: Any = None,
    ) -> None:
        super().__init__(serde=serde)
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be >= 1 (or None to keep all)")
        self.store: Store = store if store is not None else Store()
        self.incremental = incremental
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._metrics = metrics
        self._tracer = tracer
        # Write-side hint: the highest checkpoint id this saver has published per
        # (thread_id, checkpoint_ns). Used only to skip re-publishing the latest
        # pointer for an out-of-order put, never to answer a read — reads go to
        # the pointer in the store so that savers sharing a backend agree.
        self._max_seen: dict[tuple[str, str], str] = {}
        # Retention hint: how many checkpoints the last trim of a thread kept.
        # Delta channels can make that well above the limit, and without the hint
        # every put would trigger another (fruitless) trim of the same thread.
        self._prune_floor: dict[tuple[str, str], int] = {}
        # Whether the store can filter namespaces/keys by prefix itself. Assumed,
        # then turned off for good the first time a store rejects the argument.
        self._store_takes_prefix = True
        # How "which checkpoint is the latest" gets answered. A store whose
        # max_key is index-backed (the SQL backends) is asked directly, which is
        # both always-current and one row less to write per put. Everything else
        # gets the pointer below, because scanning a long thread's keys for a
        # maximum costs far more than reading one extra row.
        self._store_indexes_max_key = bool(getattr(self.store, "indexed_max_key", False))
        self._native_async = bool(getattr(self.store, "supports_async", False))
        # The same store, typed for the a* methods: only used when _native_async.
        self._astore = cast(AsyncStoreLike, self.store)

    # --------------------------------------------------------- instrumentation

    @contextmanager
    def _measure(self, op: str, thread_id: str, attrs: dict[str, Any]) -> Iterator[None]:
        """Time ``op`` for the metrics sink and wrap it in a span for the tracer.

        Entered only when at least one of tracer/metrics is set; the callers keep
        a fast-path guard so the uninstrumented default allocates nothing here.
        """
        tracer = self._tracer
        metrics = self._metrics
        t0 = time.perf_counter()
        ok = True
        span_cm = (
            tracer.start_as_current_span(f"swarmstate.checkpoint.{op}")
            if tracer is not None
            else nullcontext(None)
        )
        with span_cm as span:
            if span is not None:
                try:
                    for key, value in attrs.items():
                        if value is not None:
                            span.set_attribute(f"swarmstate.{key}", value)
                except Exception:
                    pass
            try:
                yield
            except BaseException as exc:
                ok = False
                if span is not None:
                    _mark_span_error(span, exc)
                raise
            finally:
                if metrics is not None:
                    metrics.record(op, time.perf_counter() - t0, thread_id=thread_id, ok=ok)

    def _run(self, op: str, thread_id: str, attrs: dict[str, Any], fn: Callable[[], _T]) -> _T:
        with self._measure(op, thread_id, attrs):
            return fn()

    def _put_attrs(self, config: RunnableConfig, checkpoint: Checkpoint) -> dict[str, Any]:
        cfg = config["configurable"]
        return {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint["id"],
            "incremental": self.incremental,
        }

    @staticmethod
    def _writes_attrs(
        config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str
    ) -> dict[str, Any]:
        cfg = config["configurable"]
        return {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": cfg.get("checkpoint_id"),
            "task_id": task_id,
            "writes": len(writes),
        }

    @staticmethod
    def _get_attrs(config: RunnableConfig) -> dict[str, Any]:
        cfg = config["configurable"]
        return {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": get_checkpoint_id(config),
        }

    # ------------------------------------------------------------------ sync

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if self._metrics is None and self._tracer is None:
            return self._put_impl(config, checkpoint, metadata, new_versions)
        return self._run(
            "put",
            config["configurable"]["thread_id"],
            self._put_attrs(config, checkpoint),
            lambda: self._put_impl(config, checkpoint, metadata, new_versions),
        )

    def _put_impl(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, rows, blobs = self._put_rows(
            config, checkpoint, metadata, new_versions
        )
        if blobs:
            # Skip blobs already stored: the same (channel, version) is the same value.
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            self._set_many([(bl_ns, k, v) for k, v in blobs if not self.store.contains(bl_ns, k)])
        self._set_many(rows)
        if self.max_checkpoints_per_thread is not None:
            self._prune_thread(thread_id, checkpoint_ns)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def _put_rows(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> tuple[str, str, List[tuple[str, str, Any]], List[tuple[str, Any]]]:
        """Encode a put: its rows, and its candidate channel blobs (incremental).

        No store access — shared by the sync and the native async path.
        """
        cfg = config["configurable"]
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        cp_to_store = checkpoint
        blobs: List[tuple[str, Any]] = []
        if self.incremental:
            # Store each new channel value once, keyed by (channel, version);
            # serialize the checkpoint without its inline channel_values.
            cp_to_store = {**checkpoint}
            values = cp_to_store.pop("channel_values", {})
            for ch, ver in new_versions.items():
                if ch in values:
                    vt, vb = self.serde.dumps_typed(values[ch])
                    blobs.append((f"{ch}{_SEP}{ver}", ["v", vt, vb]))
                else:
                    blobs.append((f"{ch}{_SEP}{ver}", ["empty"]))

        cp_type, cp_bytes = self.serde.dumps_typed(cp_to_store)
        md_type, md_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        rows: List[tuple[str, str, Any]] = [
            (
                _ckpt_ns(thread_id, checkpoint_ns),
                checkpoint_id,
                {
                    "cp": [cp_type, cp_bytes],
                    "md": [md_type, md_bytes],
                    "parent": cfg.get("checkpoint_id"),
                },
            )
        ]
        # Publish the latest pointer alongside the checkpoint, in the same
        # set_many (one lock / one round-trip for both). The hint keeps an
        # out-of-order put from moving the pointer backwards, matching the
        # reference savers' max(checkpoint_id) semantics without an extra read.
        # Stores that index max_key need no pointer at all.
        if not self._store_indexes_max_key:
            key = (thread_id, checkpoint_ns)
            seen = self._max_seen.get(key)
            if seen is None or checkpoint_id > seen:
                if len(self._max_seen) >= _MAX_HINTS:
                    self._max_seen.clear()
                self._max_seen[key] = checkpoint_id
                rows.append((_LATEST_NS, _latest_key(thread_id, checkpoint_ns), checkpoint_id))
        return thread_id, checkpoint_ns, rows, blobs

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if self._metrics is None and self._tracer is None:
            return self._put_writes_impl(config, writes, task_id, task_path)
        return self._run(
            "put_writes",
            config["configurable"]["thread_id"],
            self._writes_attrs(config, writes, task_id),
            lambda: self._put_writes_impl(config, writes, task_id, task_path),
        )

    def _put_writes_impl(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        cfg = config["configurable"]
        ns = _writes_ns(cfg["thread_id"], cfg.get("checkpoint_ns", ""), cfg["checkpoint_id"])
        # All writes land in the same namespace (one shard), so collect them and
        # flush with a single set_many: one lock acquisition / GIL release for
        # the whole batch instead of one per write (matters on fan-out steps that
        # emit many writes). Idempotency is preserved — the contains() checks run
        # first and only the writes that actually go through are serialized.
        batch = []
        for idx, (channel, value) in enumerate(writes):
            widx = WRITES_IDX_MAP.get(channel, idx)
            key = f"{task_id}{_SEP}{widx}"
            # Positional writes are write-once (idempotent retries); special
            # negative-index writes may overwrite.
            if widx >= 0 and self.store.contains(ns, key):
                continue
            v_type, v_bytes = self.serde.dumps_typed(value)
            batch.append((ns, key, [task_id, channel, [v_type, v_bytes], task_path]))
        self._set_many(batch)

    def _write_rows(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> tuple[str, List[tuple[str, Any, bool]]]:
        """Encode pending writes as ``(key, value, positional)`` rows, no store access."""
        cfg = config["configurable"]
        ns = _writes_ns(cfg["thread_id"], cfg.get("checkpoint_ns", ""), cfg["checkpoint_id"])
        rows: List[tuple[str, Any, bool]] = []
        for idx, (channel, value) in enumerate(writes):
            widx = WRITES_IDX_MAP.get(channel, idx)
            v_type, v_bytes = self.serde.dumps_typed(value)
            # Special (negative-index) writes may overwrite; positional ones may not.
            rows.append(
                (
                    f"{task_id}{_SEP}{widx}",
                    [task_id, channel, [v_type, v_bytes], task_path],
                    widx >= 0,
                )
            )
        return ns, rows

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if self._metrics is None and self._tracer is None:
            return self._get_tuple_impl(config)
        return self._run(
            "get_tuple",
            config["configurable"]["thread_id"],
            self._get_attrs(config),
            lambda: self._get_tuple_impl(config),
        )

    def _get_tuple_impl(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        resolved = self._resolve(config)
        if resolved is None:
            return None
        thread_id, checkpoint_ns, checkpoint_id, saved = resolved
        return self._build_tuple(thread_id, checkpoint_ns, checkpoint_id, saved)

    def _resolve(self, config: RunnableConfig) -> Optional[tuple[str, str, str, dict[str, Any]]]:
        """Find the checkpoint ``config`` names (or the latest): ids and stored record."""
        cfg = config["configurable"]
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        ns = _ckpt_ns(thread_id, checkpoint_ns)

        checkpoint_id = get_checkpoint_id(config)
        saved = None
        if not checkpoint_id and self._store_indexes_max_key:
            checkpoint_id = self.store.max_key(ns)
            if not checkpoint_id:
                return None
        elif not checkpoint_id:
            # Fast path: the latest pointer from the store — shared, so a
            # checkpoint written by another saver or process is visible here.
            # Reading the checkpoint it names doubles as the check that it still
            # exists, which saves a round-trip per resume on a networked store.
            latest_key = _latest_key(thread_id, checkpoint_ns)
            checkpoint_id = self.store.get(_LATEST_NS, latest_key)
            if checkpoint_id:
                saved = self.store.get(ns, checkpoint_id)
            if saved is None:
                # No pointer (store written by an older version) or it dangles
                # (a restore rolled the thread back past it): scan and republish.
                keys = self.store.keys(ns)
                if not keys:
                    return None
                checkpoint_id = max(keys)
                self.store.set(_LATEST_NS, latest_key, checkpoint_id)

        if saved is None:
            saved = self.store.get(ns, checkpoint_id)
        if saved is None:
            return None
        return thread_id, checkpoint_ns, checkpoint_id, saved

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        want_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        n = 0
        for thread_id, checkpoint_ns, ns in self._list_targets(config):
            # A requested id needs no listing of the thread: a missing one just
            # reads back as None below.
            ids = [want_id] if want_id else sorted(self.store.keys(ns), reverse=True)
            for checkpoint_id in ids:
                if before_id and checkpoint_id >= before_id:
                    continue
                saved = self.store.get(ns, checkpoint_id)
                metadata = self._metadata_if_matching(saved, filter)
                if metadata is None:
                    continue
                yield self._build_tuple(thread_id, checkpoint_ns, checkpoint_id, saved, metadata)
                n += 1
                if limit is not None and n >= limit:
                    return

    def _list_targets(self, config: Optional[RunnableConfig]) -> List[tuple[str, str, str]]:
        """The ``(thread_id, checkpoint_ns, namespace)`` triples ``list`` scans.

        Asking the store for the matching prefix keeps this off the "copy every
        namespace name in the store" path, which a busy store makes expensive.
        """
        if config is None:
            return [
                (p[1], p[2], ns)
                for ns in self._namespaces(f"ck{_SEP}")
                if len(p := ns.split(_SEP)) == 3
            ]
        thread_id = config["configurable"]["thread_id"]
        want_ns = config["configurable"].get("checkpoint_ns")
        targets = []
        for ns in self._namespaces(f"ck{_SEP}{thread_id}{_SEP}"):
            parts = ns.split(_SEP)
            if len(parts) != 3 or parts[1] != thread_id:
                continue
            if want_ns is not None and parts[2] != want_ns:
                continue
            targets.append((parts[1], parts[2], ns))
        return targets

    def _metadata_if_matching(
        self, saved: Optional[dict[str, Any]], filter: Optional[dict[str, Any]]
    ) -> Optional[Any]:
        """The decoded metadata of ``saved``, or None if it is missing or filtered out.

        Filtering on the metadata alone, before the checkpoint and its pending
        writes are decoded, spares that work for every entry the filter drops.
        """
        if saved is None:
            return None
        metadata = self.serde.loads_typed(tuple(saved["md"]))
        if filter and not all(metadata.get(k) == v for k, v in filter.items()):
            return None
        return metadata

    def delete_thread(self, thread_id: str) -> None:
        # "bl" too: the incremental channel blobs are the bulk of a thread's
        # bytes, and leaving them behind means the data is not really gone.
        for kind in ("ck", "wr", "bl"):
            for ns in self._thread_namespaces(kind, thread_id):
                self._delete_namespace(ns)
        if not self._store_indexes_max_key:
            self._delete_many(
                [(_LATEST_NS, k) for k in self._keys(_LATEST_NS, f"{thread_id}{_SEP}")]
            )
        for hints in (self._max_seen, self._prune_floor):
            for k in [k for k in hints if k[0] == thread_id]:
                del hints[k]

    # ------------------------------------------------------- thread management

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy every checkpoint, pending write and channel blob of a thread.

        The whole history is copied, parent chain included, so the target can be
        resumed, time-travelled and — for ``DeltaChannel`` state — rebuilt exactly
        like the source. Rows already in the target with the same ids are
        overwritten; the target's latest checkpoint is its newest one afterwards.
        """
        source, target = source_thread_id, target_thread_id
        if source == target:
            return
        for kind in ("ck", "wr", "bl"):
            for ns in self._thread_namespaces(kind, source):
                parts = ns.split(_SEP)
                parts[1] = target
                target_ns = _SEP.join(parts)
                keys = self.store.keys(ns)
                rows = []
                for key, value in zip(keys, self._get_many(ns, keys)):
                    if value is None:
                        continue
                    if kind == "ck":
                        value = self._retarget(value, source, target)
                    rows.append((target_ns, key, value))
                self._set_many(rows)
        if not self._store_indexes_max_key:
            for ns in self._thread_namespaces("ck", target):
                newest = self.store.max_key(ns)
                if newest:
                    checkpoint_ns = ns.split(_SEP)[2]
                    self.store.set(_LATEST_NS, _latest_key(target, checkpoint_ns), newest)
                    _bounded_set(self._max_seen, (target, checkpoint_ns), newest)

    def _retarget(self, saved: dict[str, Any], source: str, target: str) -> dict[str, Any]:
        """A checkpoint record whose metadata names ``target`` where it named ``source``."""
        metadata = self.serde.loads_typed(tuple(saved["md"]))
        if not isinstance(metadata, dict) or metadata.get("thread_id") != source:
            return saved
        md_type, md_bytes = self.serde.dumps_typed({**metadata, "thread_id": target})
        return {**saved, "md": [md_type, md_bytes]}

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete every checkpoint (with its writes) whose metadata ``run_id`` is listed.

        LangGraph records ``run_id`` from ``config["metadata"]``; checkpoints
        written without one are never matched. This reads the metadata of every
        checkpoint in the store, so it is an administrative operation, not a hot
        path. As LangGraph warns, deleting a run whose writes a ``DeltaChannel``
        of a still-live thread replays from loses that history.
        """
        wanted = set(run_ids)
        if not wanted:
            return
        for thread_id, checkpoint_ns, ns in self._list_targets(None):
            ids = self.store.keys(ns)
            doomed: set[str] = set()
            for checkpoint_id, saved in zip(ids, self._get_many(ns, ids)):
                if saved is None:
                    continue
                metadata = self.serde.loads_typed(tuple(saved["md"]))
                if isinstance(metadata, dict) and metadata.get("run_id") in wanted:
                    doomed.add(checkpoint_id)
            if doomed:
                kept = [i for i in ids if i not in doomed]
                self._drop_checkpoints(thread_id, checkpoint_ns, sorted(doomed), kept)

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Prune the given threads.

        ``"keep_latest"`` keeps the newest checkpoint of each checkpoint
        namespace — plus, for ``DeltaChannel`` state, the checkpoints back to each
        channel's last snapshot, which that state is rebuilt from. ``"delete"``
        removes the threads entirely.
        """
        if strategy not in _PRUNE_STRATEGIES:
            raise ValueError(f"strategy must be one of {_PRUNE_STRATEGIES}, got {strategy!r}")
        for thread_id in thread_ids:
            if strategy == "delete":
                self.delete_thread(thread_id)
                continue
            for ns in self._thread_namespaces("ck", thread_id):
                self._trim(thread_id, ns.split(_SEP)[2], sorted(self.store.keys(ns)), 1)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        """Walk the parent chain for ``channels``: their writes, and the nearest seed.

        Same result as LangGraph's default, which rebuilds a full
        ``CheckpointTuple`` — every channel value, all metadata, every pending
        write — for each ancestor, and the chain runs up to a channel's
        ``snapshot_frequency`` (1000 by default) long. This reads the stored
        records directly and decodes only the requested channels' writes and
        values.
        """
        if not channels:
            return {}
        collected: dict[str, List[Any]] = {ch: [] for ch in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)

        resolved = self._resolve(config)
        thread_id, checkpoint_ns, _, target = resolved if resolved else ("", "", "", {})
        ns = _ckpt_ns(thread_id, checkpoint_ns)
        cursor = target.get("parent")
        while cursor and remaining:
            saved = self.store.get(ns, cursor)
            if saved is None:
                break
            for write in reversed(
                self._pending_writes(thread_id, checkpoint_ns, cursor, remaining)
            ):
                collected[write[1]].append(write)
            present = self._channel_values(thread_id, checkpoint_ns, saved, remaining)
            for ch, value in present.items():
                seeds[ch] = value
                remaining.discard(ch)
            cursor = saved.get("parent")

        result: dict[str, Any] = {}
        for ch in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[ch]))}
            if ch in seeds:
                entry["seed"] = seeds[ch]
            result[ch] = entry
        return result

    # ------------------------------------------------------------------ async
    # With a store that supports async natively, the hot paths (put, put_writes,
    # get_tuple, list) await it directly. Otherwise each call runs in a worker
    # thread: the Rust store releases the GIL, so that keeps the event loop
    # responsive and lets store work overlap with it. Administrative operations
    # always take the worker-thread route.

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if not self._native_async:
            return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)
        if self._metrics is None and self._tracer is None:
            return await self._aput_impl(config, checkpoint, metadata, new_versions)
        cfg = config["configurable"]
        with self._measure("put", cfg["thread_id"], self._put_attrs(config, checkpoint)):
            return await self._aput_impl(config, checkpoint, metadata, new_versions)

    async def _aput_impl(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, rows, blobs = self._put_rows(
            config, checkpoint, metadata, new_versions
        )
        if blobs:
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            stored = await asyncio.gather(*(self._astore.acontains(bl_ns, k) for k, _ in blobs))
            new_blobs = [(bl_ns, k, v) for (k, v), done in zip(blobs, stored) if not done]
            if new_blobs:
                await self._astore.aset_many(new_blobs)
        await self._astore.aset_many(rows)
        if self.max_checkpoints_per_thread is not None:
            await asyncio.to_thread(self._prune_thread, thread_id, checkpoint_ns)
        return _config(thread_id, checkpoint_ns, checkpoint["id"])

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not self._native_async:
            await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)
            return
        if self._metrics is None and self._tracer is None:
            await self._aput_writes_impl(config, writes, task_id, task_path)
            return
        cfg = config["configurable"]
        with self._measure(
            "put_writes", cfg["thread_id"], self._writes_attrs(config, writes, task_id)
        ):
            await self._aput_writes_impl(config, writes, task_id, task_path)

    async def _aput_writes_impl(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        ns, rows = self._write_rows(config, writes, task_id, task_path)
        existing: set[str] = set()
        if any(positional for _, _, positional in rows):
            existing = set(await self._astore.akeys(ns, prefix=f"{task_id}{_SEP}"))
        batch = [(ns, k, v) for k, v, positional in rows if not (positional and k in existing)]
        if batch:
            await self._astore.aset_many(batch)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if not self._native_async:
            return await asyncio.to_thread(self.get_tuple, config)
        if self._metrics is None and self._tracer is None:
            return await self._aget_tuple_impl(config)
        with self._measure(
            "get_tuple", config["configurable"]["thread_id"], self._get_attrs(config)
        ):
            return await self._aget_tuple_impl(config)

    async def _aget_tuple_impl(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        resolved = await self._aresolve(config)
        return None if resolved is None else await self._abuild_tuple(*resolved)

    async def _aresolve(
        self, config: RunnableConfig
    ) -> Optional[tuple[str, str, str, dict[str, Any]]]:
        """Async twin of :meth:`_resolve`."""
        cfg = config["configurable"]
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        ns = _ckpt_ns(thread_id, checkpoint_ns)

        checkpoint_id = get_checkpoint_id(config)
        saved = None
        if not checkpoint_id and self._store_indexes_max_key:
            checkpoint_id = await self._astore.amax_key(ns)
            if not checkpoint_id:
                return None
        elif not checkpoint_id:
            latest_key = _latest_key(thread_id, checkpoint_ns)
            checkpoint_id = await self._astore.aget(_LATEST_NS, latest_key)
            if checkpoint_id:
                saved = await self._astore.aget(ns, checkpoint_id)
            if saved is None:
                keys = await self._astore.akeys(ns)
                if not keys:
                    return None
                checkpoint_id = max(keys)
                await self._astore.aset(_LATEST_NS, latest_key, checkpoint_id)

        if saved is None:
            saved = await self._astore.aget(ns, checkpoint_id)
        if saved is None:
            return None
        return thread_id, checkpoint_ns, checkpoint_id, saved

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if not self._native_async:
            items = await asyncio.to_thread(
                lambda: list(self.list(config, filter=filter, before=before, limit=limit))
            )
            for item in items:
                yield item
            return

        want_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        targets = await asyncio.to_thread(self._list_targets, config)
        n = 0
        for thread_id, checkpoint_ns, ns in targets:
            ids = [want_id] if want_id else sorted(await self._astore.akeys(ns), reverse=True)
            for checkpoint_id in ids:
                if before_id and checkpoint_id >= before_id:
                    continue
                saved = await self._astore.aget(ns, checkpoint_id)
                metadata = self._metadata_if_matching(saved, filter)
                if metadata is None:
                    continue
                yield await self._abuild_tuple(
                    thread_id, checkpoint_ns, checkpoint_id, saved, metadata
                )
                n += 1
                if limit is not None and n >= limit:
                    return

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await asyncio.to_thread(self.copy_thread, source_thread_id, target_thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await asyncio.to_thread(self.delete_for_runs, run_ids)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await asyncio.to_thread(lambda: self.prune(thread_ids, strategy=strategy))

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            lambda: self.get_delta_channel_history(config=config, channels=channels)
        )

    # -------------------------------------------------------------- retention

    def _prune_thread(self, thread_id: str, checkpoint_ns: str) -> None:
        """Trim a thread back to ``max_checkpoints_per_thread`` newest checkpoints.

        Runs only once a thread is ``_PRUNE_SLACK`` (or a quarter of what the last
        trim kept) checkpoints above that, so the survivor scan is amortized.
        """
        limit = self.max_checkpoints_per_thread
        if limit is None:
            return
        key = (thread_id, checkpoint_ns)
        floor = max(limit, self._prune_floor.get(key, limit))
        ids = self.store.keys(_ckpt_ns(thread_id, checkpoint_ns))
        if len(ids) <= floor + max(_PRUNE_SLACK, floor // 4):
            return
        kept = self._trim(thread_id, checkpoint_ns, sorted(ids), limit)
        _bounded_set(self._prune_floor, key, kept)

    def _trim(self, thread_id: str, checkpoint_ns: str, ids: Sequence[str], keep: int) -> int:
        """Drop all but the newest ``keep`` of ``ids`` (sorted) and what they need.

        Returns how many checkpoints were kept: the newest ``keep``, plus the
        ancestors their ``DeltaChannel`` state is rebuilt from.
        """
        survivors = list(ids[-keep:])
        ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
        records = dict(zip(survivors, self._get_many(ck_ns, survivors)))
        kept = set(survivors) | self._delta_ancestors(thread_id, checkpoint_ns, survivors, records)
        doomed = [i for i in ids if i not in kept]
        if doomed:
            self._drop_checkpoints(thread_id, checkpoint_ns, doomed, sorted(kept), records)
        return len(kept)

    def _delta_ancestors(
        self,
        thread_id: str,
        checkpoint_ns: str,
        survivors: Sequence[str],
        records: dict[str, Any],
    ) -> set[str]:
        """Ancestors of ``survivors`` that their ``DeltaChannel`` values depend on.

        A delta channel missing from a checkpoint's ``channel_values`` is rebuilt
        by walking its parents, collecting their writes, up to the nearest one
        that holds a value for it. Every checkpoint on that walk has to stay.
        ``records`` caches the stored rows read along the way.
        """
        ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
        needed: set[str] = set()
        for checkpoint_id in survivors:
            saved = records.get(checkpoint_id)
            if saved is None:
                continue
            metadata = self.serde.loads_typed(tuple(saved["md"]))
            delta = metadata.get(_DELTA_COUNTERS) if isinstance(metadata, dict) else None
            if not delta:
                continue
            missing = set(delta) - set(
                self._channel_values(thread_id, checkpoint_ns, saved, set(delta))
            )
            cursor = saved.get("parent")
            while missing and cursor:
                if cursor not in records:
                    records[cursor] = self.store.get(ck_ns, cursor)
                ancestor = records[cursor]
                if ancestor is None:
                    break
                needed.add(cursor)
                missing -= set(self._channel_values(thread_id, checkpoint_ns, ancestor, missing))
                cursor = ancestor.get("parent")
        return needed

    def _drop_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        doomed: Sequence[str],
        kept: Sequence[str],
        records: Optional[dict[str, Any]] = None,
    ) -> None:
        """Delete ``doomed`` checkpoints with their writes and orphaned blobs.

        ``kept`` are the checkpoints that stay; they decide which incremental
        blobs are still referenced and where the latest pointer ends up.
        """
        ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
        if self.incremental:
            # Blobs are shared by version, and a checkpoint forked from an older
            # one can reference a version older than its siblings' — so the keep
            # set is the union over *every* survivor, not just the oldest.
            keep = self._referenced_blobs(thread_id, checkpoint_ns, kept, records)
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            self._delete_many([(bl_ns, b) for b in self.store.keys(bl_ns) if b not in keep])
        for checkpoint_id in doomed:
            self._delete_namespace(_writes_ns(thread_id, checkpoint_ns, checkpoint_id))
        self._delete_many([(ck_ns, checkpoint_id) for checkpoint_id in doomed])

        if not self._store_indexes_max_key:
            latest_key = _latest_key(thread_id, checkpoint_ns)
            if self.store.get(_LATEST_NS, latest_key) in set(doomed):
                if kept:
                    self.store.set(_LATEST_NS, latest_key, max(kept))
                else:
                    self.store.delete(_LATEST_NS, latest_key)
                self._max_seen.pop((thread_id, checkpoint_ns), None)

    def _referenced_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_ids: Sequence[str],
        records: Optional[dict[str, Any]] = None,
    ) -> "set[str]":
        """Blob keys still referenced by the given checkpoints' channel versions."""
        records = records or {}
        missing = [i for i in checkpoint_ids if i not in records]
        if missing:
            ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
            records = {**records, **dict(zip(missing, self._get_many(ck_ns, missing)))}
        keep: set[str] = set()
        for checkpoint_id in checkpoint_ids:
            saved = records.get(checkpoint_id)
            if saved is None:
                continue
            checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
            for channel, version in checkpoint.get("channel_versions", {}).items():
                keep.add(f"{channel}{_SEP}{version}")
        return keep

    # ---------------------------------------------------------------- decoding

    def _build_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: dict[str, Any],
        metadata: Any = None,
    ) -> CheckpointTuple:
        """Assemble a ``CheckpointTuple``; pass ``metadata`` if already decoded.

        This is the read path behind every resume, so it is written out inline
        rather than through the helpers below: at a few microseconds per read,
        each extra Python call shows up in the latency.
        """
        loads = self.serde.loads_typed
        checkpoint = loads(tuple(saved["cp"]))
        if self.incremental and "channel_values" not in checkpoint:
            versions = list(checkpoint.get("channel_versions", {}).items())
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            blobs = self._get_many(bl_ns, [f"{ch}{_SEP}{ver}" for ch, ver in versions])
            self._attach_values(checkpoint, versions, blobs)
        if metadata is None:
            metadata = loads(tuple(saved["md"]))

        writes_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
        keys = self.store.keys(writes_ns)
        if len(keys) > 1:
            keys.sort(key=_write_order)
        pending_writes = []
        for saved_write in self._get_many(writes_ns, keys):
            if saved_write is None:
                continue
            task_id, channel, tv, _task_path = saved_write
            pending_writes.append((task_id, channel, loads(tuple(tv))))

        parent_id = saved.get("parent")
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=pending_writes,
        )

    async def _abuild_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: dict[str, Any],
        metadata: Any = None,
    ) -> CheckpointTuple:
        """Async twin of :meth:`_build_tuple`."""
        checkpoint, versions = self._decode_checkpoint(saved)
        writes_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
        if versions:
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            blobs, write_keys = await asyncio.gather(
                self._astore.aget_many([(bl_ns, f"{ch}{_SEP}{ver}") for ch, ver in versions]),
                self._astore.akeys(writes_ns),
            )
            self._attach_values(checkpoint, versions, blobs)
        else:
            write_keys = await self._astore.akeys(writes_ns)
        if metadata is None:
            metadata = self.serde.loads_typed(tuple(saved["md"]))
        write_keys = sorted(write_keys, key=_write_order)
        raw = (
            await self._astore.aget_many([(writes_ns, k) for k in write_keys]) if write_keys else []
        )
        pending = self._decode_writes(raw)
        return self._tuple(
            thread_id, checkpoint_ns, checkpoint_id, saved, checkpoint, metadata, pending
        )

    def _decode_checkpoint(self, saved: dict[str, Any]) -> tuple[dict[str, Any], List[Any]]:
        """Decode a stored checkpoint, plus the ``(channel, version)`` blobs it needs.

        The list is empty unless the checkpoint was stored incrementally, in which
        case its ``channel_values`` still have to be attached from those blobs.
        """
        checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
        if self.incremental and "channel_values" not in checkpoint:
            return checkpoint, list(checkpoint.get("channel_versions", {}).items())
        return checkpoint, []

    def _attach_values(
        self, checkpoint: dict[str, Any], versions: Sequence[Any], blobs: Sequence[Any]
    ) -> None:
        values: dict[str, Any] = {}
        for (ch, _ver), blob in zip(versions, blobs):
            if blob and blob[0] == "v":
                values[ch] = self.serde.loads_typed((blob[1], blob[2]))
        checkpoint["channel_values"] = values

    def _channel_values(
        self, thread_id: str, checkpoint_ns: str, saved: dict[str, Any], channels: set[str]
    ) -> dict[str, Any]:
        """The values ``saved`` holds for ``channels`` (absent ones are left out)."""
        checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
        if "channel_values" in checkpoint:
            stored = checkpoint["channel_values"]
            return {ch: stored[ch] for ch in channels if ch in stored}
        # Incremental: read just these channels' blobs.
        versions = [
            (ch, ver)
            for ch, ver in checkpoint.get("channel_versions", {}).items()
            if ch in channels
        ]
        bl_ns = _blobs_ns(thread_id, checkpoint_ns)
        blobs = self._get_many(bl_ns, [f"{ch}{_SEP}{ver}" for ch, ver in versions])
        return {
            ch: self.serde.loads_typed((blob[1], blob[2]))
            for (ch, _ver), blob in zip(versions, blobs)
            if blob and blob[0] == "v"
        }

    def _pending_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channels: Optional[set[str]] = None,
    ) -> List[tuple[str, str, Any]]:
        """A checkpoint's pending writes in order, optionally only to ``channels``."""
        writes_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
        keys = sorted(self.store.keys(writes_ns), key=_write_order)
        return self._decode_writes(self._get_many(writes_ns, keys), channels)

    def _decode_writes(
        self, raw: Sequence[Any], channels: Optional[set[str]] = None
    ) -> List[tuple[str, str, Any]]:
        writes = []
        for saved_write in raw:
            if saved_write is None:
                continue
            task_id, channel, tv, _task_path = saved_write
            if channels is not None and channel not in channels:
                continue  # never decoded: the caller does not want it
            writes.append((task_id, channel, self.serde.loads_typed(tuple(tv))))
        return writes

    @staticmethod
    def _tuple(
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: dict[str, Any],
        checkpoint: Any,
        metadata: Any,
        pending_writes: List[tuple[str, str, Any]],
    ) -> CheckpointTuple:
        parent_id = saved.get("parent")
        return CheckpointTuple(
            config=_config(thread_id, checkpoint_ns, checkpoint_id),
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=_config(thread_id, checkpoint_ns, parent_id) if parent_id else None,
            pending_writes=pending_writes,
        )

    # ---------------------------------------------------------------- helpers

    def _thread_namespaces(self, kind: str, thread_id: str) -> List[str]:
        """The ``kind`` (ck/wr/bl) namespaces of one thread, every checkpoint ns."""
        return [
            ns
            for ns in self._namespaces(f"{kind}{_SEP}{thread_id}{_SEP}")
            if ns.split(_SEP)[1:2] == [thread_id]
        ]

    def _namespaces(self, prefix: str) -> List[str]:
        """Namespaces starting with ``prefix``, filtered by the store when it can.

        The bundled stores filter internally, so only the matching names are
        copied out. A custom store with the older no-argument signature still
        works — it just copies everything and filters here.
        """
        if self._store_takes_prefix:
            try:
                matched: List[str] = self.store.namespaces(prefix=prefix)
                return matched
            except TypeError:
                self._store_takes_prefix = False
        return [ns for ns in self.store.namespaces() if ns.startswith(prefix)]

    def _keys(self, namespace: str, prefix: str) -> List[str]:
        """Keys of ``namespace`` starting with ``prefix`` (see :meth:`_namespaces`)."""
        if self._store_takes_prefix:
            try:
                matched: List[str] = self.store.keys(namespace, prefix=prefix)
                return matched
            except TypeError:
                self._store_takes_prefix = False
        return [key for key in self.store.keys(namespace) if key.startswith(prefix)]

    # ``list`` (the LangGraph method) shadows the builtin in this class's method
    # annotations, so batch helpers spell their list types with ``List``.
    def _get_many(self, ns: str, keys: Sequence[str]) -> List[Any]:
        """Batch-read ``keys`` from one namespace, falling back to per-key gets.

        Uses the store's ``get_many`` when available (Rust core and the bundled
        backends), which is one GIL release / round-trip for the whole batch.
        Custom stores without it still work via per-key ``get``.
        """
        getter = getattr(self.store, "get_many", None)
        if getter is not None:
            batched: List[Any] = getter([(ns, k) for k in keys])
            return batched
        return [self.store.get(ns, k) for k in keys]

    def _set_many(self, items: Sequence[tuple[str, str, Any]]) -> None:
        """Batch-write ``(namespace, key, value)`` triples in one call.

        Uses the store's ``set_many`` when available (Rust core and the bundled
        backends): one lock acquisition per shard and one GIL release for the
        whole batch, instead of one per item. Custom stores without it still
        work via per-item ``set``.
        """
        if not items:
            return
        setter = getattr(self.store, "set_many", None)
        if setter is not None:
            setter(list(items))
        else:
            for ns, key, value in items:
                self.store.set(ns, key, value)

    def _delete_many(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Batch-delete pairs, falling back to per-key deletes (see :meth:`_set_many`)."""
        if not pairs:
            return
        deleter = getattr(self.store, "delete_many", None)
        if deleter is not None:
            deleter(list(pairs))
        else:
            for ns, key in pairs:
                self.store.delete(ns, key)

    def _delete_namespace(self, ns: str) -> None:
        """Delete a whole namespace, falling back to deleting its keys one by one."""
        deleter = getattr(self.store, "delete_namespace", None)
        if deleter is not None:
            deleter(ns)
        else:
            self._delete_many([(ns, key) for key in self.store.keys(ns)])


def handoff_router(
    graph: HandoffGraph, node: str, *, default: Optional[str] = None
) -> Callable[[Any], str]:
    """A routing function for ``add_conditional_edges``, resolved by ``graph``.

    The returned callable evaluates the outgoing edges of ``node`` against the
    LangGraph state (a dict, a dataclass or a Pydantic model) in Rust and
    returns the first match — no LLM call, no Python ``eval``. When no edge
    matches it returns ``default``, which defaults to LangGraph's ``END``::

        g = HandoffGraph()
        g.add_edge("triage", "billing", when="category in ['billing', 'refund']")
        g.add_edge("triage", "support")

        builder.add_conditional_edges(
            "triage", handoff_router(g, "triage"), ["billing", "support", END]
        )
    """
    if default is None:
        from langgraph.graph import END

        default = END
    fallback: str = default

    def route(state: Any) -> str:
        target = graph.route(node, state)
        return fallback if target is None else target

    route.__name__ = f"route_{node}"
    return route
