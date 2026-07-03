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
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

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

# Unit-separator delimiter: never appears in thread ids / namespaces.
_SEP = "\x1f"


def _ckpt_ns(thread_id: str, checkpoint_ns: str) -> str:
    return f"ck{_SEP}{thread_id}{_SEP}{checkpoint_ns}"


def _writes_ns(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"wr{_SEP}{thread_id}{_SEP}{checkpoint_ns}{_SEP}{checkpoint_id}"


def _blobs_ns(thread_id: str, checkpoint_ns: str) -> str:
    return f"bl{_SEP}{thread_id}{_SEP}{checkpoint_ns}"


class SwarmStateSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpointer backed by a swarmstate :class:`~swarmstate.Store`.

    Args:
        store: underlying store; defaults to a fresh in-memory ``Store()``.
            Share one ``Store`` across savers/graphs for a unified checkpoint DB.
        serde: optional LangGraph serializer (defaults to ``JsonPlusSerializer``).
        incremental: if True, store each channel value once per version (dedup)
            instead of the whole checkpoint blob per step. Saves storage and
            serialization for long threads with large, mostly-stable channels,
            at the cost of extra reads on ``get_tuple`` (one per channel). The
            default (False) keeps ``get_tuple`` at a single read.
    """

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        serde: Optional[SerializerProtocol] = None,
        incremental: bool = False,
    ) -> None:
        super().__init__(serde=serde)
        self.store: Store = store if store is not None else Store()
        self.incremental = incremental
        # O(1) "latest checkpoint id" cache per (thread_id, checkpoint_ns).
        # A best-effort fast path for get_tuple; always falls back to a scan.
        self._latest: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------ sync

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        cp_to_store = checkpoint
        if self.incremental:
            # Store each new channel value once, keyed by (channel, version);
            # serialize the checkpoint without its inline channel_values.
            cp_to_store = {**checkpoint}
            values = cp_to_store.pop("channel_values", {})
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            for ch, ver in new_versions.items():
                bkey = f"{ch}{_SEP}{ver}"
                if self.store.contains(bl_ns, bkey):
                    continue  # this exact value/version is already stored
                if ch in values:
                    vt, vb = self.serde.dumps_typed(values[ch])
                    self.store.set(bl_ns, bkey, ["v", vt, vb])
                else:
                    self.store.set(bl_ns, bkey, ["empty"])

        cp_type, cp_bytes = self.serde.dumps_typed(cp_to_store)
        md_type, md_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        self.store.set(
            _ckpt_ns(thread_id, checkpoint_ns),
            checkpoint_id,
            {
                "cp": [cp_type, cp_bytes],
                "md": [md_type, md_bytes],
                "parent": config["configurable"].get("checkpoint_id"),
            },
        )
        key = (thread_id, checkpoint_ns)
        cur = self._latest.get(key)
        if cur is None or checkpoint_id > cur:
            self._latest[key] = checkpoint_id
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
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)

        for idx, (channel, value) in enumerate(writes):
            widx = WRITES_IDX_MAP.get(channel, idx)
            key = f"{task_id}{_SEP}{widx}"
            # Positional writes are write-once (idempotent retries); special
            # negative-index writes may overwrite.
            if widx >= 0 and self.store.contains(ns, key):
                continue
            v_type, v_bytes = self.serde.dumps_typed(value)
            self.store.set(ns, key, [task_id, channel, [v_type, v_bytes], task_path])

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        ns = _ckpt_ns(thread_id, checkpoint_ns)

        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            # Fast path: cached latest id; fall back to a scan (cold saver, or
            # after a store.restore invalidated the cache).
            cache_key = (thread_id, checkpoint_ns)
            checkpoint_id = self._latest.get(cache_key)
            if not checkpoint_id or not self.store.contains(ns, checkpoint_id):
                keys = self.store.keys(ns)
                if not keys:
                    return None
                checkpoint_id = max(keys)
                self._latest[cache_key] = checkpoint_id

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
        # Determine which (thread_id, checkpoint_ns) namespaces to scan.
        if config is not None:
            thread_id = config["configurable"]["thread_id"]
            want_ns = config["configurable"].get("checkpoint_ns")
            targets = []
            for ns in self.store.namespaces():
                parts = ns.split(_SEP)
                if len(parts) != 3 or parts[0] != "ck" or parts[1] != thread_id:
                    continue
                if want_ns is not None and parts[2] != want_ns:
                    continue
                targets.append((parts[1], parts[2], ns))
        else:
            targets = [
                (p[1], p[2], ns)
                for ns in self.store.namespaces()
                if len(p := ns.split(_SEP)) == 3 and p[0] == "ck"
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
        for ns in self.store.namespaces():
            parts = ns.split(_SEP)
            if len(parts) >= 2 and parts[0] in ("ck", "wr") and parts[1] == thread_id:
                for key in self.store.keys(ns):
                    self.store.delete(ns, key)
        for k in [k for k in self._latest if k[0] == thread_id]:
            del self._latest[k]

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

    # ---------------------------------------------------------------- helpers

    def _build_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, saved: dict
    ) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed(tuple(saved["cp"]))
        metadata = self.serde.loads_typed(tuple(saved["md"]))
        parent_id = saved.get("parent")

        if self.incremental and "channel_values" not in checkpoint:
            # Reassemble channel_values from the per-(channel, version) blobs.
            bl_ns = _blobs_ns(thread_id, checkpoint_ns)
            values = {}
            for ch, ver in checkpoint.get("channel_versions", {}).items():
                blob = self.store.get(bl_ns, f"{ch}{_SEP}{ver}")
                if blob and blob[0] == "v":
                    values[ch] = self.serde.loads_typed((blob[1], blob[2]))
            checkpoint["channel_values"] = values

        writes_ns = _writes_ns(thread_id, checkpoint_ns, checkpoint_id)
        pending_writes = []
        for key in sorted(self.store.keys(writes_ns)):
            task_id, channel, tv, _task_path = self.store.get(writes_ns, key)
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
