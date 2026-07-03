"""Smoke-run the runnable examples so they can't silently bit-rot.

`state_portability` needs only the base package; `support_triage` needs the
langgraph extra and is skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_state_portability_runs(capsys):
    _load("state_portability").main()
    assert "any framework" in capsys.readouterr().out


def test_support_triage_runs(capsys):
    pytest.importorskip("langgraph")
    _load("support_triage").main()
    out = capsys.readouterr().out
    assert "-> " not in out or "billing" in out
    # The restore must actually rewind the late ticket.
    assert "gone: never happened" in out
