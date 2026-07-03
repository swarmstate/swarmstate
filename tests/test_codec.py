"""Codec tests: swarmstate.dumps/loads round-trip + cross-implementation interop.

These prove the "stable, cross-language msgpack" claim: the Rust core's codec
(`rmpv`) and an independent Python msgpack implementation agree on the bytes.
"""

import pytest

import swarmstate as ss

CASES = [
    None,
    True,
    False,
    0,
    -7,
    2**40,
    3.14,
    "hello ünïcode 🚀",
    b"\x00\x01\xff",
    [1, 2, [3, "x"], None],
    {"a": 1, "b": [1, 2.5, "x"], "c": b"\x00\xff", "n": None, "t": True},
    {"nested": {"deep": {"k": [1, {"z": 9}]}}},
]


@pytest.mark.parametrize("obj", CASES)
def test_dumps_loads_roundtrip(obj):
    assert ss.loads(ss.dumps(obj)) == obj


@pytest.mark.parametrize("obj", CASES)
def test_cross_language_with_python_msgpack(obj):
    """Two independent msgpack implementations must agree both ways."""
    msgpack = pytest.importorskip("msgpack")

    # Rust core bytes decode with the Python msgpack C library.
    from_rust = ss.dumps(obj)
    assert msgpack.unpackb(from_rust, raw=False, strict_map_key=False) == obj

    # Python msgpack bytes decode with the Rust core.
    from_py = msgpack.packb(obj, use_bin_type=True)
    assert ss.loads(from_py) == obj


def test_dumps_returns_bytes():
    assert isinstance(ss.dumps({"a": 1}), bytes)


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        ss.dumps(object())
