"""M5 tests: the CrewAI-compatible storage adapter (save/search/reset)."""

import swarmstate as ss
from swarmstate.integrations.crewai import SwarmStateStorage


def test_save_search_reset_roundtrip():
    mem = SwarmStateStorage(namespace="crew")
    mem.save("The invoice total was 79 euros", {"agent": "billing"})
    mem.save("User asked how to export data to CSV", {"agent": "support"})
    mem.save("Refund policy is 30 days", {"agent": "billing"})
    assert len(mem) == 3

    hits = mem.search("invoice total", limit=2)
    assert hits[0]["context"].startswith("The invoice total")
    assert hits[0]["metadata"] == {"agent": "billing"}
    assert hits[0]["score"] > 0

    # score_threshold filters out weak matches.
    assert mem.search("quantum physics", score_threshold=0.5) == []

    mem.reset()
    assert len(mem) == 0
    assert mem.search("invoice") == []


def test_limit_and_ordering():
    mem = SwarmStateStorage()
    for i in range(5):
        mem.save(f"alpha beta item {i}")
    hits = mem.search("alpha beta", limit=3)
    assert len(hits) == 3
    # All fully match query tokens -> tie broken by recency (newest first).
    assert hits[0]["context"] == "alpha beta item 4"


def test_shares_the_same_store_as_other_agents():
    """State portability: one Store, CrewAI memory + arbitrary app state."""
    store = ss.Store()
    mem = SwarmStateStorage(store, namespace="crew:research")
    mem.save("finding: latency dropped 12x", {"step": 1})

    # Another system reads the crew's memory straight from the shared store.
    key = store.keys("crew:research")[0]
    assert "12x" in store.get("crew:research", key)["value"]


def test_deleting_an_entry_does_not_overwrite_another_on_the_next_save():
    """Keys used to come from the entry *count*, so a deletion caused a clash."""
    mem = SwarmStateStorage(namespace="crew")
    mem.save("first")
    mem.save("second")
    mem.store.delete("crew", sorted(mem.store.keys("crew"))[0])

    mem.save("third")
    kept = {mem.store.get("crew", k)["value"] for k in mem.store.keys("crew")}
    assert kept == {"second", "third"}


def test_concurrent_saves_keep_every_entry():
    import threading

    mem = SwarmStateStorage(namespace="crew")
    threads = [
        threading.Thread(target=lambda i=i: [mem.save(f"item {i}-{j}") for j in range(20)])
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(mem) == 80


def test_dict_value_is_indexed():
    mem = SwarmStateStorage()
    mem.save({"topic": "billing", "note": "duplicate charge"})
    assert mem.search("duplicate charge")[0]["score"] > 0
