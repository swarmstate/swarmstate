"""Native async on the networked backends, and SwarmStateSaver's use of it.

Redis runs against fakeredis; Postgres only when ``SWARMSTATE_TEST_PG_DSN`` is set.
Each test drives its own event loop with ``asyncio.run``, as the stores open their
async connections per loop.
"""

import asyncio
import os
import uuid

import pytest

pytest.importorskip("msgpack")

PG_DSN = os.environ.get("SWARMSTATE_TEST_PG_DSN")


def _redis():
    fakeredis = pytest.importorskip("fakeredis")
    from swarmstate.backends.redis import RedisStore

    server = fakeredis.FakeServer()
    return RedisStore(
        client=fakeredis.FakeRedis(server=server),
        aclient=fakeredis.FakeAsyncRedis(server=server),
        prefix=f"async{uuid.uuid4().hex[:8]}",
    )


def _postgres():
    pytest.importorskip("psycopg")
    if not PG_DSN:
        pytest.skip("set SWARMSTATE_TEST_PG_DSN to run the Postgres backend")
    from swarmstate.backends.postgres import PostgresStore

    return PostgresStore(PG_DSN, table=f"kv_{uuid.uuid4().hex[:12]}")


@pytest.fixture(params=["redis", "postgres"])
def store(request):
    made = _redis() if request.param == "redis" else _postgres()
    try:
        yield made
    finally:
        made.clear()
        close = getattr(made, "close", None)
        if close is not None:
            close()


def run(store, body):
    """Run ``body(store)`` on a fresh loop, closing the store's async side after."""

    async def main():
        try:
            return await body(store)
        finally:
            await store.aclose()

    return asyncio.run(main())


def test_async_api_matches_the_sync_one(store):
    assert store.supports_async

    async def body(s):
        await s.aset("wf", "a", {"step": 1, "blob": b"\x00\xff"})
        await s.aset_many([("wf", "task-1", 1), ("wf", "task-2", None), ("other", "x", 3)])
        assert await s.aget("wf", "a") == {"step": 1, "blob": b"\x00\xff"}
        assert await s.aget("wf", "missing", default=7) == 7
        assert await s.aget_many([("wf", "task-1"), ("nope", "x"), ("other", "x")]) == [
            1,
            None,
            3,
        ]
        assert await s.acontains("wf", "task-2")
        assert sorted(await s.akeys("wf", prefix="task-")) == ["task-1", "task-2"]
        assert await s.amax_key("wf") == "task-2"
        assert await s.amax_key("empty") is None
        assert sorted(await s.anamespaces()) == ["other", "wf"]
        assert await s.anamespaces(prefix="ot") == ["other"]

        assert await s.adelete("wf", "a") is True
        assert await s.adelete("wf", "a") is False
        assert await s.adelete_many([("wf", "task-1"), ("wf", "gone")]) == 1
        assert await s.adelete_namespace("other") == 1

    run(store, body)
    # Async writes are the same rows the sync API reads.
    assert store.keys("wf") == ["task-2"]
    assert store.namespaces() == ["wf"]


def test_a_store_without_an_async_client_says_so():
    fakeredis = pytest.importorskip("fakeredis")
    from swarmstate.backends.redis import RedisStore

    store = RedisStore(client=fakeredis.FakeRedis())
    assert not store.supports_async


def _graph(saver):
    pytest.importorskip("langgraph")
    import operator
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        n: Annotated[int, operator.add]

    b = StateGraph(State)
    b.add_node("inc", lambda s: {"n": 1})
    b.add_edge(START, "inc")
    b.add_edge("inc", END)
    return b.compile(checkpointer=saver)


@pytest.mark.parametrize("incremental", [False, True])
def test_saver_awaits_the_store_natively(store, monkeypatch, incremental):
    pytest.importorskip("langgraph")
    import swarmstate.integrations.langgraph as lg
    from swarmstate.observability import InMemoryMetrics

    metrics = InMemoryMetrics()
    saver = lg.SwarmStateSaver(store, incremental=incremental, metrics=metrics)
    assert saver._native_async

    # The hot paths must not fall back to running the sync code in a thread.
    def no_threads(*args, **kwargs):
        raise AssertionError("fell back to asyncio.to_thread")

    monkeypatch.setattr(lg.asyncio, "to_thread", no_threads)
    g = _graph(saver)
    cfg = {"configurable": {"thread_id": "t"}}

    async def body(_):
        await g.ainvoke({"n": 0}, cfg)
        await g.ainvoke({"n": 0}, cfg)
        state = await g.aget_state(cfg)
        latest = await saver.aget_tuple(cfg)
        return state.values["n"], latest

    n, latest = run(store, body)
    assert n == 2
    # Same answer through the sync API, which reads the same rows.
    assert saver.get_tuple(cfg).checkpoint == latest.checkpoint
    assert {"put", "put_writes", "get_tuple"} <= set(metrics.summary())


def test_native_alist_matches_list(store):
    pytest.importorskip("langgraph")
    from swarmstate.integrations.langgraph import SwarmStateSaver

    saver = SwarmStateSaver(store)
    g = _graph(saver)
    cfg = {"configurable": {"thread_id": "t"}}
    for _ in range(3):
        g.invoke({"n": 0}, cfg)

    def ids(tuples):
        return [t.config["configurable"]["checkpoint_id"] for t in tuples]

    async def body(_):
        everything = [t async for t in saver.alist(cfg)]
        limited = [t async for t in saver.alist(cfg, limit=2, filter={"source": "loop"})]
        return everything, limited

    everything, limited = run(store, body)
    assert ids(everything) == ids(saver.list(cfg))
    assert ids(limited) == ids(saver.list(cfg, limit=2, filter={"source": "loop"}))
