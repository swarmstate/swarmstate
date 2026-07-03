# Examples

Runnable, offline (no API keys, no LLM calls), deterministic.

| Example | What it shows | Install |
| --- | --- | --- |
| [`support_triage.py`](support_triage.py) | A LangGraph support-triage workflow using **HandoffGraph** for deterministic routing, **SwarmStateSaver** as the checkpointer, and **snapshot/restore** to time-travel the whole checkpoint DB at once. | `pip install "swarmstate[langgraph]"` |
| [`state_portability.py`](state_portability.py) | State as standard **msgpack** bytes: write it once, read it back from another store, verify against the standalone `msgpack` package. The anti-lock-in guarantee. | `pip install swarmstate` |

```bash
# with pip
pip install "swarmstate[langgraph]"
python examples/support_triage.py

# or with uv
uv add "swarmstate[langgraph]"
uv run python examples/support_triage.py
```

Each file ends with its expected output, so you can diff a run against it.
