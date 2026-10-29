"""Move state between stores, and in and out of files.

Every backend speaks the same interface (:mod:`swarmstate.protocols`), so state
written through one can be carried to any other — from the in-memory ``Store`` to
Postgres when a prototype goes to production, from Redis to a SQLite file for a
bug report, or out to disk and back across a restart::

    import swarmstate as ss
    from swarmstate.backends.postgres import PostgresStore

    ss.copy_store(memory_store, PostgresStore(dsn))       # migrate a whole store
    ss.copy_store(src, dst, prefix="ck\\x1fthread-42")    # or just some namespaces

    ss.dump_store(store, "state.swst")                    # to a file...
    store = ss.load_store("state.swst")                   # ...and back

LangGraph checkpoints included: a ``SwarmStateSaver`` over the destination store
resumes every copied thread.

The dump format is a fixed header (``SWSTATE`` magic and a version byte) followed
by one frame per entry: a 4-byte big-endian length, then the msgpack encoding of
``[namespace, key, value]``. It is written with the same codec as the stores, so
any msgpack reader can parse it.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from typing import IO, Any, Optional, Union, cast

from ._core import Store, dumps, loads
from .protocols import StoreLike

_MAGIC = b"SWSTATE"
_FORMAT_VERSION = 1
_FRAME = struct.Struct(">I")
_BATCH = 1000

PathOrFile = Union[str, "os.PathLike[str]", IO[bytes]]


def _entries(
    store: StoreLike, prefix: Optional[str], batch_size: int
) -> Iterator["list[tuple[str, str, Any]]"]:
    """Yield the store's entries in batches of up to ``batch_size``."""
    namespaces = store.namespaces() if prefix is None else store.namespaces(prefix=prefix)
    for ns in namespaces:
        keys = store.keys(ns)
        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            values = store.get_many([(ns, k) for k in chunk])
            # get_many reports a missing key and a stored None alike; tell them
            # apart only for the Nones, so a None that was stored is carried over.
            yield [
                (ns, k, v) for k, v in zip(chunk, values) if v is not None or store.contains(ns, k)
            ]


def copy_store(
    src: StoreLike,
    dst: StoreLike,
    *,
    prefix: Optional[str] = None,
    batch_size: int = _BATCH,
) -> int:
    """Copy every entry of ``src`` into ``dst``; returns how many were copied.

    Args:
        src: the store to read from. It is not modified.
        dst: the store to write to. Entries with the same ``(namespace, key)``
            are overwritten; everything else in it is left alone.
        prefix: copy only namespaces starting with this.
        batch_size: entries read and written per round-trip.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    copied = 0
    for batch in _entries(src, prefix, batch_size):
        if batch:
            dst.set_many(batch)
            copied += len(batch)
    return copied


def dump_store(
    store: StoreLike,
    target: PathOrFile,
    *,
    prefix: Optional[str] = None,
) -> int:
    """Write every entry of ``store`` to a file; returns how many were written.

    ``target`` is a path or a binary file object. A path is written to a
    temporary file next to it first and moved into place at the end, so an
    interrupted dump never leaves a truncated file behind.
    """
    if not isinstance(target, (str, os.PathLike)):
        return _write(store, target, prefix)
    path = os.fspath(target)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as f:
            count = _write(store, f, prefix)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return count


def _write(store: StoreLike, f: IO[bytes], prefix: Optional[str]) -> int:
    f.write(_MAGIC + bytes([_FORMAT_VERSION]))
    count = 0
    for batch in _entries(store, prefix, _BATCH):
        for ns, key, value in batch:
            frame = dumps([ns, key, value])
            f.write(_FRAME.pack(len(frame)))
            f.write(frame)
            count += 1
    return count


def load_store(source: PathOrFile, store: Optional[StoreLike] = None) -> StoreLike:
    """Read a :func:`dump_store` file into ``store`` (a new in-memory ``Store`` if None).

    Returns the store. Entries already in it with the same ``(namespace, key)``
    are overwritten.
    """
    # The Rust Store satisfies StoreLike at runtime; its stub's narrower
    # restore() signature is what mypy objects to.
    target = store if store is not None else cast(StoreLike, Store())
    if isinstance(source, (str, os.PathLike)):
        with open(os.fspath(source), "rb") as f:
            _read(f, target)
    else:
        _read(source, target)
    return target


def _read(f: IO[bytes], store: StoreLike) -> None:
    header = f.read(len(_MAGIC) + 1)
    if header[: len(_MAGIC)] != _MAGIC:
        raise ValueError("not a swarmstate dump (bad header)")
    if header[len(_MAGIC)] != _FORMAT_VERSION:
        raise ValueError(f"unsupported swarmstate dump version {header[len(_MAGIC)]}")
    batch: list[tuple[str, str, Any]] = []
    while True:
        size = f.read(_FRAME.size)
        if not size:
            break
        if len(size) != _FRAME.size:
            raise ValueError("truncated swarmstate dump")
        (length,) = _FRAME.unpack(size)
        frame = f.read(length)
        if len(frame) != length:
            raise ValueError("truncated swarmstate dump")
        entry = loads(frame)
        if not (isinstance(entry, list) and len(entry) == 3):
            raise ValueError("corrupt swarmstate dump entry")
        batch.append((entry[0], entry[1], entry[2]))
        if len(batch) >= _BATCH:
            store.set_many(batch)
            batch = []
    if batch:
        store.set_many(batch)


__all__ = ["copy_store", "dump_store", "load_store"]
