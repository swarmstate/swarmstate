"""swarmstate — a fast state & checkpointing backend for multi-agent systems.

Rust core (via PyO3) with a thin, fully typed Python API. This is the public
entry point; import the native extension lazily-friendly names here.

Public API (built out across milestones):
    - ``Store``          framework-agnostic KV store with cheap snapshots (M1)
    - ``HandoffGraph``   deterministic, LLM-free routing over a DAG (M2)
    - ``SwarmStateSaver`` (in ``swarmstate.integrations.langgraph``, M3)
"""

from __future__ import annotations

from . import _core
from ._core import HandoffGraph, Snapshot, Store, dumps, loads

__all__ = [
    "HandoffGraph",
    "Snapshot",
    "Store",
    "__version__",
    "core_version",
    "dumps",
    "loads",
]

#: Version of the installed ``swarmstate`` package.
__version__ = _core.__version__


def core_version() -> str:
    """Return the version string reported by the compiled Rust core."""
    return _core.core_version()
