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

from collections.abc import Iterator
from typing import Any, cast

import msgpack

_DEFAULT_URL = "redis://localhost:6379/0"


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


class RedisSnapshot:
    """A copy-based snapshot of a :class:`RedisStore` (O(n))."""

    def __init__(self, data: dict[str, dict[str, bytes]]):
        self._data = data
        self.size_bytes = sum(len(v) for ns in data.values() for v in ns.values())

    @property
    def keys(self) -> list[tuple[str, str]]:
        return [(ns, k) for ns, kv in self._data.items() for k in kv]


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

    # ------------------------------------------------------------- helpers
    def _hkey(self, namespace: str) -> str:
        return f"{self._prefix}:{namespace}"

    def _iter_hkeys(self) -> Iterator[str]:
        for raw in self._r.scan_iter(match=f"{self._prefix}:*"):
            yield raw.decode() if isinstance(raw, bytes) else raw

    def _ns_of(self, hkey: str) -> str:
        return hkey[len(self._prefix) + 1 :]

    # ------------------------------------------------------------- core API
    def set(self, namespace: str, key: str, value: Any) -> None:
        self._r.hset(self._hkey(namespace), key, _pack(value))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        raw = self._r.hget(self._hkey(namespace), key)
        return default if raw is None else _unpack(raw)

    def contains(self, namespace: str, key: str) -> bool:
        return bool(self._r.hexists(self._hkey(namespace), key))

    def delete(self, namespace: str, key: str) -> bool:
        return bool(self._r.hdel(self._hkey(namespace), key) > 0)

    def keys(self, namespace: str) -> list[str]:
        return [
            k.decode() if isinstance(k, bytes) else k for k in self._r.hkeys(self._hkey(namespace))
        ]

    def namespaces(self) -> list[str]:
        return [self._ns_of(hk) for hk in self._iter_hkeys()]

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
        return RedisSnapshot(data)

    def restore(self, snapshot: RedisSnapshot) -> None:
        self.clear()
        pipe = self._r.pipeline()
        for ns, kv in snapshot._data.items():
            if kv:
                pipe.hset(self._hkey(ns), mapping=kv)
        pipe.execute()

    def __repr__(self) -> str:
        return f"RedisStore(prefix='{self._prefix}', codec='{self.codec}')"
