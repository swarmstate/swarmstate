"""A Redis-backed store with the same interface as :class:`swarmstate.Store`.

Values are serialized with **msgpack** — the same wire format as the Rust core —
so state written here is readable by any msgpack consumer, in any language. This
makes checkpoints and state **persistent** and **shareable across processes**,
while keeping the exact API the rest of swarmstate expects:

    from swarmstate.backends.redis import RedisStore
    from swarmstate.integrations.langgraph import SwarmStateSaver

    store = RedisStore("redis://localhost:6379/0")
    graph = builder.compile(checkpointer=SwarmStateSaver(store))   # persistent!

Requires the ``redis`` extra: ``pip install "swarmstate[redis]"``.

Layout: each namespace is a Redis hash at ``{prefix}:{namespace}`` whose fields
are the keys and whose values are msgpack bytes. ``snapshot``/``restore`` copy
the data (O(n)) — Redis persists rather than offering the Rust store's O(1)
structural-sharing snapshots.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional, cast

import msgpack

from ._snapshot import CopySnapshot, SnapshotMeta

_DEFAULT_URL = "redis://localhost:6379/0"


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def _glob_escape(text: str) -> str:
    """Escape Redis glob metacharacters so a prefix matches literally."""
    out = []
    for ch in text:
        if ch in "\\*?[]^":
            out.append("\\")
        out.append(ch)
    return "".join(out)


class RedisSnapshot(CopySnapshot):
    """A copy-based snapshot of a :class:`RedisStore` (O(n))."""

    def __init__(self, data: dict[str, dict[str, bytes]], *meta: Any):
        # Keeps the per-namespace mapping for restore()'s pipelined HSETs, and
        # feeds the flat rows the shared snapshot surface works from.
        self._data = data
        super().__init__(
            [(ns, k, v) for ns, kv in data.items() for k, v in kv.items()],
            *meta,
        )


class RedisStore:
    """Redis-backed store implementing the :class:`swarmstate.Store` interface."""

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        *,
        client: Any = None,
        prefix: str = "swarmstate",
        codec: str = "msgpack",
    ) -> None:
        if codec != "msgpack":
            raise ValueError(f"codec '{codec}' is not supported (only 'msgpack')")
        if client is None:
            import redis  # top-level dependency (extra)

            client = redis.Redis.from_url(url)
        self._r = client
        self._prefix = prefix
        self.codec = codec
        self._snap_meta = SnapshotMeta()

    # ------------------------------------------------------------- helpers
    def _hkey(self, namespace: str) -> str:
        return f"{self._prefix}:{namespace}"

    def _iter_hkeys(self, ns_prefix: Optional[str] = None) -> Iterator[str]:
        pattern = f"{_glob_escape(self._prefix)}:"
        pattern += f"{_glob_escape(ns_prefix)}*" if ns_prefix else "*"
        for raw in self._r.scan_iter(match=pattern):
            yield raw.decode() if isinstance(raw, bytes) else raw

    def _ns_of(self, hkey: str) -> str:
        return hkey[len(self._prefix) + 1 :]

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        self._r.hset(self._hkey(namespace), key, _pack(value))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        raw = self._r.hget(self._hkey(namespace), key)
        return default if raw is None else _unpack(raw)

    def set_many(self, items: list[tuple[str, str, Any]]) -> None:
        if not items:
            return
        pipe = self._r.pipeline(transaction=False)
        for ns, k, v in items:
            pipe.hset(self._hkey(ns), k, _pack(v))
        pipe.execute()

    def get_many(self, pairs: list[tuple[str, str]]) -> list[Any]:
        if not pairs:
            return []
        pipe = self._r.pipeline(transaction=False)
        for ns, k in pairs:
            pipe.hget(self._hkey(ns), k)
        return [None if raw is None else _unpack(raw) for raw in pipe.execute()]

    def contains(self, namespace: str, key: str) -> bool:
        return bool(self._r.hexists(self._hkey(namespace), key))

    def delete(self, namespace: str, key: str) -> bool:
        return bool(self._r.hdel(self._hkey(namespace), key) > 0)

    def keys(self, namespace: str, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            raw_keys: Iterable[Any] = self._r.hkeys(self._hkey(namespace))
        else:
            raw_keys = self._r.hscan_iter(self._hkey(namespace), match=f"{_glob_escape(prefix)}*")
            raw_keys = (field for field, _ in raw_keys)
        return [k.decode() if isinstance(k, bytes) else k for k in raw_keys]

    def max_key(self, namespace: str) -> Optional[str]:
        """The greatest field name in ``namespace``.

        Redis cannot order hash fields server-side, so this reads them all;
        ``indexed_max_key`` is therefore left off and callers that need an O(1)
        answer keep their own pointer.
        """
        return max(self.keys(namespace), default=None)

    def namespaces(self, prefix: Optional[str] = None) -> list[str]:
        return [self._ns_of(hk) for hk in self._iter_hkeys(prefix)]

    def clear(self) -> None:
        hkeys = list(self._iter_hkeys())
        if hkeys:
            self._r.delete(*hkeys)

    def __len__(self) -> int:
        return sum(self._r.hlen(hk) for hk in self._iter_hkeys())

    def __contains__(self, namespace: str) -> bool:
        return bool(self._r.exists(self._hkey(namespace)) > 0)

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> RedisSnapshot:
        data: dict[str, dict[str, bytes]] = {}
        for hk in self._iter_hkeys():
            ns = self._ns_of(hk)
            data[ns] = {
                (f.decode() if isinstance(f, bytes) else f): v
                for f, v in self._r.hgetall(hk).items()
            }
        return RedisSnapshot(data, *self._snap_meta.next())

    def restore(self, snapshot: RedisSnapshot) -> None:
        self.clear()
        pipe = self._r.pipeline()
        for ns, kv in snapshot._data.items():
            if kv:
                pipe.hset(self._hkey(ns), mapping=kv)
        pipe.execute()

    def __repr__(self) -> str:
        return f"RedisStore(prefix='{self._prefix}', codec='{self.codec}')"
