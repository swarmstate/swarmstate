"""Prefix filters SQLite can answer from the primary-key index.

``substr(col, 1, n) = ?`` is exact but opaque to SQLite's planner, so every prefix
lookup scanned the whole table — and the LangGraph adapter runs one on each
``list()`` and ``delete_thread()``. The same filter written as a range,
``col >= prefix AND col < upper``, is an index seek, because SQLite's default
``BINARY`` collation compares by code point. (Postgres needs no rewrite: on a
``COLLATE "C"`` column its planner turns ``starts_with`` into that range itself.)
"""

from __future__ import annotations

from typing import Optional

_MAX_CODE_POINT = 0x10FFFF
_SURROGATES = range(0xD800, 0xE000)


def prefix_upper_bound(prefix: str) -> Optional[str]:
    """The smallest string greater than every string that starts with ``prefix``.

    Bumps the last character to the next code point, dropping trailing
    characters that are already at the maximum. Surrogates are skipped: they
    cannot be stored as UTF-8 text. ``None`` means there is no upper bound
    (``prefix`` is empty or made only of U+10FFFF), so the caller must fall back
    to an exact prefix test.
    """
    chars = list(prefix)
    while chars:
        cp = ord(chars[-1]) + 1
        if cp in _SURROGATES:
            cp = _SURROGATES.stop
        if cp <= _MAX_CODE_POINT:
            chars[-1] = chr(cp)
            return "".join(chars)
        chars.pop()
    return None
