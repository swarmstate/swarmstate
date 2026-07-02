"""M2 tests: the deterministic HandoffGraph and its safe condition evaluator."""

import pytest

import swarmstate as ss


def test_readme_example():
    g = ss.HandoffGraph()
    g.add_edge("triage", "billing", when="category == 'billing'")
    assert g.route("triage", state={"category": "billing"}) == "billing"


def test_deterministic_first_match_with_default():
    g = ss.HandoffGraph()
    g.add_edge("triage", "billing", when="category == 'billing'")
    g.add_edge("triage", "support", when="category == 'support'")
    g.add_edge("triage", "human")  # unconditional default (added last)

    assert g.route("triage", {"category": "billing"}) == "billing"
    assert g.route("triage", {"category": "support"}) == "support"
    assert g.route("triage", {"category": "anything"}) == "human"


def test_route_no_match_returns_none():
    g = ss.HandoffGraph()
    g.add_edge("a", "b", when="x == 1")
    assert g.route("a", {}) is None
    assert g.route("a") is None
    assert g.route("unknown-node") is None


def test_condition_operators():
    g = ss.HandoffGraph()
    g.add_edge("n", "hit", when="priority >= 3 and vip and status != 'closed'")

    assert g.route("n", {"priority": 5, "vip": True, "status": "open"}) == "hit"
    assert g.route("n", {"priority": 2, "vip": True, "status": "open"}) is None
    assert g.route("n", {"priority": 5, "vip": False, "status": "open"}) is None
    assert g.route("n", {"priority": 5, "vip": True, "status": "closed"}) is None


def test_dotted_path_and_membership():
    g = ss.HandoffGraph()
    g.add_edge("n", "esc", when="user.tier == 'gold' or 'urgent' in tags")

    assert g.route("n", {"user": {"tier": "gold"}, "tags": []}) == "esc"
    assert g.route("n", {"user": {"tier": "free"}, "tags": ["urgent", "x"]}) == "esc"
    assert g.route("n", {"user": {"tier": "free"}, "tags": ["x"]}) is None
    # Missing nested key evaluates to false rather than raising.
    assert g.route("n", {"tags": ["x"]}) is None


def test_cross_int_float_comparison():
    g = ss.HandoffGraph()
    g.add_edge("n", "ok", when="score == 5")
    assert g.route("n", {"score": 5.0}) == "ok"


def test_cycle_detection_default_errors():
    g = ss.HandoffGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    with pytest.raises(ValueError, match="cycle"):
        g.add_edge("c", "a")
    with pytest.raises(ValueError, match="cycle"):
        g.add_edge("a", "a")  # self-loop
    assert g.is_dag()


def test_cycle_allowed_when_configured():
    g = ss.HandoffGraph(on_cycle="allow")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    assert not g.is_dag()
    assert g.on_cycle == "allow"


def test_invalid_on_cycle_and_condition():
    with pytest.raises(ValueError):
        ss.HandoffGraph(on_cycle="whatever")
    g = ss.HandoffGraph()
    with pytest.raises(ValueError, match="condition"):
        g.add_edge("a", "b", when="x = 1")  # single '=' is invalid
    with pytest.raises(ValueError, match="condition"):
        g.add_edge("a", "b", when="(unbalanced")


def test_introspection():
    g = ss.HandoffGraph()
    g.add_edge("triage", "billing", when="category == 'billing'")
    g.add_edge("triage", "human")
    g.add_node("orphan")

    assert set(g.nodes()) == {"triage", "billing", "human", "orphan"}
    assert "triage" in g
    assert g.has_node("orphan")
    assert len(g) == 4
    assert g.edges("triage") == [("billing", "category == 'billing'"), ("human", None)]


def test_no_python_eval_side_effects():
    """Conditions are data, not code — identifiers are just state lookups."""
    g = ss.HandoffGraph()
    # '__import__' is only ever treated as a (missing) state key, never executed.
    g.add_edge("a", "b", when="__import__ == 'os'")
    assert g.route("a", {"__import__": "os"}) == "b"
    assert g.route("a", {}) is None
