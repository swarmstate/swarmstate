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


# Every msgpack width boundary, so a codec change cannot quietly alter the wire
# format: the stored bytes must stay readable by past and future versions.
BOUNDARIES = [
    0,
    127,
    128,
    255,
    256,
    65535,
    65536,
    2**32 - 1,
    2**32,
    2**63 - 1,
    -1,
    -32,
    -33,
    -128,
    -129,
    -32768,
    -32769,
    -(2**31),
    -(2**31) - 1,
    1.0,
    -0.5,
    "",
    "x" * 31,
    "x" * 32,
    "x" * 255,
    "x" * 256,
    "x" * 65535,
    "x" * 65536,
    b"",
    b"b" * 255,
    b"b" * 256,
    b"b" * 65535,
    b"b" * 65536,
    [],
    list(range(15)),
    list(range(16)),
    list(range(65535)),
    list(range(65536)),
    {},
    {str(i): i for i in range(15)},
    {str(i): i for i in range(16)},
    {str(i): i for i in range(65536)},
]


@pytest.mark.parametrize("obj", BOUNDARIES, ids=lambda o: f"{type(o).__name__}:{len(repr(o))}")
def test_wire_format_is_byte_identical_to_reference_msgpack(obj):
    msgpack = pytest.importorskip("msgpack")
    assert ss.dumps(obj) == msgpack.packb(obj, use_bin_type=True)
    assert ss.loads(msgpack.packb(obj, use_bin_type=True)) == obj


def test_float32_input_is_widened():
    """A float32 written by another implementation still decodes."""
    msgpack = pytest.importorskip("msgpack")
    raw = msgpack.packb(1.5, use_single_float=True)
    assert raw[0] == 0xCA  # float32 marker
    assert ss.loads(raw) == 1.5


def test_truncated_input_raises():
    with pytest.raises(ValueError):
        ss.loads(b"\x93\x01")  # array of 3, only one element present


def test_extension_types_are_rejected():
    msgpack = pytest.importorskip("msgpack")
    raw = msgpack.packb(msgpack.ExtType(42, b"payload"))
    with pytest.raises(ValueError):
        ss.loads(raw)


def test_deeply_nested_input_is_rejected():
    with pytest.raises(ValueError):
        ss.loads(b"\x91" * 5000 + b"\xc0")  # 5000 nested 1-element arrays


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        ss.dumps(object())


def _nest(levels: int) -> dict:
    root: dict = {}
    cur = root
    for _ in range(levels):
        nxt: dict = {}
        cur["a"] = nxt
        cur = nxt
    return root


def test_reasonable_nesting_roundtrips():
    obj = _nest(64)
    assert ss.loads(ss.dumps(obj)) == obj


def test_too_deep_raises_instead_of_crashing():
    """A runaway nesting depth must raise, not overflow the native stack.

    Before the depth guard this aborted the interpreter with SIGSEGV.
    """
    with pytest.raises(ValueError):
        ss.dumps(_nest(5000))


def test_self_referential_value_raises():
    d: dict = {}
    d["self"] = d
    with pytest.raises(ValueError):
        ss.dumps(d)


def test_deep_store_set_raises_instead_of_crashing():
    store = ss.Store()
    with pytest.raises(ValueError):
        store.set("ns", "deep", _nest(5000))
    assert len(store) == 0


def test_deep_route_state_raises_instead_of_crashing():
    """A referenced path holding runaway nesting must raise, not crash."""
    g = ss.HandoffGraph()
    g.add_edge("a", "b", when="x == 1")
    with pytest.raises(ValueError):
        g.route("a", {"x": _nest(5000)})


def test_deep_state_outside_the_condition_paths_is_never_touched():
    """route() only reads the paths its conditions name, however deep the rest is."""
    g = ss.HandoffGraph()
    g.add_edge("a", "b", when="x == 1")
    assert g.route("a", {"x": 1, "unrelated": _nest(5000)}) == "b"
