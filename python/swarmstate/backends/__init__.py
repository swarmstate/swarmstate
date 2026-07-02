"""Alternative, persistent backends for swarmstate.

The default :class:`swarmstate.Store` is an in-memory Rust store. These backends
offer the same duck-typed interface (``set``/``get``/``contains``/``delete``/
``keys``/``namespaces``/``snapshot``/``restore``) over external systems, so they
drop into anything that takes a store — including
:class:`~swarmstate.integrations.langgraph.SwarmStateSaver`.

Each backend uses lazy/optional imports and its own extra
(e.g. ``swarmstate[redis]``).
"""
