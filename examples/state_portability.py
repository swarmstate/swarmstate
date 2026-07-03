#!/usr/bin/env python3
"""Framework-agnostic state: write once, read anywhere.

swarmstate serializes state as plain **msgpack** bytes. That means state written
by one framework (or language) can be read by another, and moved between backends
(memory / SQLite / Redis / Postgres) with no re-encoding. This demo needs no
extras and no LLM:

    pip install swarmstate        # or: uv add swarmstate
    python examples/state_portability.py
"""

from __future__ import annotations

import swarmstate as ss


def main() -> None:
    # Framework A writes accumulated workflow state.
    store = ss.Store()
    state = {"step": 3, "history": ["greet", "collect", "verify"], "user": {"id": 7, "vip": True}}
    store.set("workflow", "onboarding", state)

    # The wire format is standard msgpack: dumps()/loads() round-trip through the
    # exact same encoding the Store uses internally, so any msgpack reader can
    # consume it, in any language.
    raw = ss.dumps(store.get("workflow", "onboarding"))
    print(f"encoded state: {len(raw)} bytes of standard msgpack")
    assert ss.loads(raw) == state

    # Framework B (or a different process, or a different language) reads it back
    # byte-for-byte, no lock-in, no bespoke format.
    store_b = ss.Store()
    store_b.set("workflow", "onboarding", ss.loads(raw))
    print("re-read by another store:", store_b.get("workflow", "onboarding"))

    # Optional cross-check against the standalone `msgpack` package, proving the
    # format is not swarmstate-specific.
    try:
        import msgpack

        assert msgpack.unpackb(raw, raw=False) == state
        print("verified against the standalone `msgpack` package: identical")
    except ImportError:
        print("(install `msgpack` to cross-check the wire format)")

    print("\nSame bytes, any backend, any framework. That is the anti-lock-in guarantee.")


if __name__ == "__main__":
    main()
