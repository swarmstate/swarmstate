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

Requires the ``langgraph`` extra: ``pip install "swarmstate[langgraph]"``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, List, Optional, TypeVar

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

from .. import Store
from ..observability import MetricsSink

# Unit-separator delimiter: never appears in thread ids / namespaces.
_SEP = "\x1f"

# Namespace holding the "latest checkpoint id" pointer per (thread, ns). It lives
# in the store rather than in the saver so that every saver and every process
# sharing a backend agrees on which checkpoint is the latest one.
_LATEST_NS = "lt"

# Cap for the write-side hint dict (see SwarmStateSaver._max_seen): it only
# avoids a read, so dropping it wholesale when it grows is harmless.
_MAX_HINTS = 4096

# Retention runs in batches: a thread is trimmed back to the limit only once it
# is this far above it (or a quarter of the limit, whichever is larger). Pruning
# has to look at the surviving checkpoints, so batching keeps that off the hot
# path — the cost lands on one put in every `slack`, not on all of them.
_PRUNE_SLACK = 8

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
            retained window is no longer possible, so size it accordingly.
        metrics: optional :class:`~swarmstate.observability.MetricsSink` that
            receives the latency and outcome of each ``put`` / ``put_writes`` /
            ``get_tuple``. Defaults to ``None`` (no measurement, zero overhead).
        tracer: optional OpenTelemetry ``Tracer``. When set, each ``put`` /
            ``put_writes`` / ``get_tuple`` runs inside a
            ``swarmstate.checkpoint.<op>`` span with thread/checkpoint attributes.
            Get one from :func:`swarmstate.observability.get_tracer`. Defaults to
            ``None`` (no spans, zero overhead).
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
        # Whether the store can filter namespaces/keys by prefix itself. Assumed,
        # then turned off for good the first time a store rejects the argument.
        self._store_takes_prefix = True
        # How "which checkpoint is the latest" gets answered. A store whose
        # max_key is index-backed (the SQL backends) is asked directly, which is
        # both always-current and one row less to write per put. Everything else
        # gets the pointer below, because scanning a long thread's keys for a
        # maximum costs far more than reading one extra row.
        self._store_indexes_max_key = bool(getattr(self.store, "indexed_max_key", False))

    def _run(self, op: str, thread_id: str, attrs: dict[str, Any], fn: Callable[[], _T]) -> _T:
        """Run ``fn`` under the configured span (tracer) and timer (metrics).

        Reached only when at least one of tracer/metrics is set; the callers keep
        a fast-path guard so the uninstrumented default allocates nothing here.
        """
        tracer = self._tracer
        metrics = self._metrics
        if tracer is None:
            if metrics is None:
                return fn()
            t0 = time.perf_counter()
            ok = True
            try:
                return fn()
            except BaseException:
                ok = False
                raise
            finally:
                metrics.record(op, time.perf_counter() - t0, thread_id=thread_id, ok=ok)

        t0 = time.perf_counter()
        ok = True
        with tracer.start_as_current_span(f"swarmstate.checkpoint.{op}") as span:
            try:
                for key, value in attrs.items():
                    if value is not None:
                        span.set_attribute(f"swarmstate.{key}", value)
            except Exception:
                pass
            try:
                return fn()
            except BaseException as exc:
                ok = False
                _mark_span_error(span, exc)
                raise
            finally:
                if metrics is not None:
                    metrics.record(op, time.perf_counter() - t0, thread_id=thread_id, ok=ok)

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
        cfg = config["configurable"]
        attrs = {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint["id"],
            "incremental": self.incremental,
        }
        return self._run(
            "put",
            cfg["thread_id"],
            attrs,
            lambda: self._put_impl(config, checkpoint, metadata, new_versions),
        )

    def _put_impl(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        cfg = config["configurable"]
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        cp_to_store = checkpoint
        if self.incremental:
            # Store each new channel value once, keyed by (channel, version);
            # serialize the checkpoint without its inline channel_values.
            cp_to_store = {**checkpoint}
            values = cp_to_store.pop("channel_values", {})
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            # Same namespace for every blob → collect and flush in one set_many.
            blob_batch = []
            for ch, ver in new_versions.items():
                bkey = f"{ch}{_SEP}{ver}"
                if self.store.contains(bl_ns, bkey):
                    continue  # this exact value/version is already stored
                if ch in values:
                    vt, vb = self.serde.dumps_typed(values[ch])
                    blob_batch.append((bl_ns, bkey, ["v", vt, vb]))
                else:
                    blob_batch.append((bl_ns, bkey, ["empty"]))
            self._set_many(blob_batch)

        cp_type, cp_bytes = self.serde.dumps_typed(cp_to_store)
        md_type, md_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        batch: List[tuple[str, str, Any]] = [
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
                batch.append((_LATEST_NS, _latest_key(thread_id, checkpoint_ns), checkpoint_id))
        self._set_many(batch)
        if self.max_checkpoints_per_thread is not None:
            self._prune_thread(thread_id, checkpoint_ns)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if self._metrics is None and self._tracer is None:
            return self._put_writes_impl(config, writes, task_id, task_path)
        cfg = config["configurable"]
        attrs = {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": cfg.get("checkpoint_id"),
            "task_id": task_id,
            "writes": len(writes),
        }
        return self._run(
            "put_writes",
            cfg["thread_id"],
            attrs,
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
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        checkpoint_id = cfg["checkpoint_id"]
        ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)

        # All writes land in the same namespace (one shard), so collect them and
        # flush with a single set_many: one lock acquisition / GIL release for
        # the whole batch instead of one per write (matters on fan-out steps that
        # emit many writes). Idempotency is preserved — the contains() checks run
        # first and only the writes that actually go through are batched.
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if self._metrics is None and self._tracer is None:
            return self._get_tuple_impl(config)
        cfg = config["configurable"]
        attrs = {
            "thread_id": cfg["thread_id"],
            "checkpoint_ns": cfg.get("checkpoint_ns", ""),
            "checkpoint_id": get_checkpoint_id(config),
        }
        return self._run("get_tuple", cfg["thread_id"], attrs, lambda: self._get_tuple_impl(config))

    def _get_tuple_impl(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        cfg = config["configurable"]
        thread_id = cfg["thread_id"]
        checkpoint_ns = cfg.get("checkpoint_ns", "")
        ns = _ckpt_ns(thread_id, checkpoint_ns)

        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id and self._store_indexes_max_key:
            checkpoint_id = self.store.max_key(ns)
            if not checkpoint_id:
                return None
        elif not checkpoint_id:
            # Fast path: the latest pointer from the store — shared, so a
            # checkpoint written by another saver or process is visible here.
            latest_key = _latest_key(thread_id, checkpoint_ns)
            checkpoint_id = self.store.get(_LATEST_NS, latest_key)
            if not checkpoint_id or not self.store.contains(ns, checkpoint_id):
                # No pointer (store written by an older version) or it dangles
                # (a restore rolled the thread back past it): scan and republish.
                keys = self.store.keys(ns)
                if not keys:
                    return None
                checkpoint_id = max(keys)
                self.store.set(_LATEST_NS, latest_key, checkpoint_id)

        saved = self.store.get(ns, checkpoint_id)
        if saved is None:
            return None
        return self._build_tuple(thread_id, checkpoint_ns, checkpoint_id, saved)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        # Determine which (thread_id, checkpoint_ns) namespaces to scan. Asking the
        # store for the matching prefix keeps this off the "copy every namespace
        # name in the store" path, which a busy store makes expensive.
        if config is not None:
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
        else:
            targets = [
                (p[1], p[2], ns)
                for ns in self._namespaces(f"ck{_SEP}")
                if len(p := ns.split(_SEP)) == 3
            ]

        want_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None

        n = 0
        for thread_id, checkpoint_ns, ns in targets:
            for checkpoint_id in sorted(self.store.keys(ns), reverse=True):
                if want_id and checkpoint_id != want_id:
                    continue
                if before_id and checkpoint_id >= before_id:
                    continue
                saved = self.store.get(ns, checkpoint_id)
                if saved is None:
                    continue
                tup = self._build_tuple(thread_id, checkpoint_ns, checkpoint_id, saved)
                if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
                yield tup
                n += 1
                if limit is not None and n >= limit:
                    return

    def delete_thread(self, thread_id: str) -> None:
        # "bl" too: the incremental channel blobs are the bulk of a thread's
        # bytes, and leaving them behind means the data is not really gone.
        for kind in ("ck", "wr", "bl"):
            for ns in self._namespaces(f"{kind}{_SEP}{thread_id}{_SEP}"):
                parts = ns.split(_SEP)
                if len(parts) < 2 or parts[1] != thread_id:
                    continue
                for key in self.store.keys(ns):
                    self.store.delete(ns, key)
        if not self._store_indexes_max_key:
            for key in self._keys(_LATEST_NS, f"{thread_id}{_SEP}"):
                self.store.delete(_LATEST_NS, key)
        for k in [k for k in self._max_seen if k[0] == thread_id]:
            del self._max_seen[k]

    # ------------------------------------------------------------------ async
    # The store releases the GIL on its hot paths, so offloading each call to a
    # worker thread keeps the event loop responsive and lets store work run
    # concurrently with it (rather than blocking inline).

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    # -------------------------------------------------------------- retention

    def _prune_thread(self, thread_id: str, checkpoint_ns: str) -> None:
        """Trim a thread back to ``max_checkpoints_per_thread`` newest checkpoints.

        Runs only once a thread is ``_PRUNE_SLACK`` (or a quarter of the limit)
        checkpoints above the limit, so the survivor scan below is amortized.
        """
        limit = self.max_checkpoints_per_thread
        if limit is None:
            return
        ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
        ids = self.store.keys(ck_ns)
        if len(ids) <= limit + max(_PRUNE_SLACK, limit // 4):
            return

        ids.sort()
        cut = len(ids) - limit
        doomed, survivors = ids[:cut], ids[cut:]

        if self.incremental:
            # Blobs are shared by version, and a checkpoint forked from an older
            # one can reference a version older than its siblings' — so the keep
            # set is the union over *every* survivor, not just the oldest.
            keep = self._referenced_blobs(thread_id, checkpoint_ns, survivors)
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            for bkey in self.store.keys(bl_ns):
                if bkey not in keep:
                    self.store.delete(bl_ns, bkey)

        for checkpoint_id in doomed:
            wr_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
            for key in self.store.keys(wr_ns):
                self.store.delete(wr_ns, key)
            self.store.delete(ck_ns, checkpoint_id)

    def _referenced_blobs(
        self, thread_id: str, checkpoint_ns: str, checkpoint_ids: Sequence[str]
    ) -> "set[str]":
        """Blob keys still referenced by the given checkpoints' channel versions."""
        ck_ns = _ckpt_ns(thread_id, checkpoint_ns)
        keep: set[str] = set()
        for saved in self._get_many(ck_ns, checkpoint_ids):
            if saved is None:
                continue
            checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
            for channel, version in checkpoint.get("channel_versions", {}).items():
                keep.add(f"{channel}{_SEP}{version}")
        return keep

    # ---------------------------------------------------------------- helpers

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

    def _build_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, saved: dict[str, Any]
    ) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
        metadata = self.serde.loads_typed(tuple(saved["md"]))
        parent_id = saved.get("parent")

        if self.incremental and "channel_values" not in checkpoint:
            # Reassemble channel_values from the per-(channel, version) blobs.
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            versions = list(checkpoint.get("channel_versions", {}).items())
            blobs = self._get_many(bl_ns, [f"{ch}{_SEP}{ver}" for ch, ver in versions])
            values: dict[str, Any] = {}
            for (ch, _ver), blob in zip(versions, blobs):
                if blob and blob[0] == "v":
                    values[ch] = self.serde.loads_typed((blob[1], blob[2]))
            checkpoint["channel_values"] = values

        writes_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
        keys = sorted(self.store.keys(writes_ns))
        pending_writes = []
        for saved_write in self._get_many(writes_ns, keys):
            if saved_write is None:
                continue
            task_id, channel, tv, _task_path = saved_write
            pending_writes.append((task_id, channel, self.serde.loads_typed(tuple(tv))))

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
