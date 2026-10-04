# tests/test_async_storage.py
"""Tests for AsyncInMemoryStorage.

Uses explicit @pytest.mark.asyncio markers throughout, per project
convention (not asyncio_mode="auto") -- requires pytest-asyncio as a
dev dependency; 
"""

from __future__ import annotations

import asyncio

import pytest

from rlimit.storage import AsyncInMemoryStorage


@pytest.mark.asyncio
async def test_get_missing_key_returns_none() -> None:
    storage = AsyncInMemoryStorage()
    assert await storage.get("missing") is None


@pytest.mark.asyncio
async def test_set_then_get_roundtrips() -> None:
    storage = AsyncInMemoryStorage()
    await storage.set("k", {"count": 5})
    assert await storage.get("k") == {"count": 5}


@pytest.mark.asyncio
async def test_set_overwrites_existing_value() -> None:
    storage = AsyncInMemoryStorage()
    await storage.set("k", {"count": 5})
    await storage.set("k", {"count": 9})
    assert await storage.get("k") == {"count": 9}


@pytest.mark.asyncio
async def test_lock_is_reentrant_safe_context_manager() -> None:
    # Same name as the sync test flagged as misnamed -- kept
    # consistent for now rather than silently renaming it; this test
    # actually checks basic acquire/release, not reentrancy, same
    # caveat as noted before.
    storage = AsyncInMemoryStorage()
    async with storage.lock("k"):
        pass
    async with storage.lock("k"):
        pass


@pytest.mark.asyncio
async def test_locks_for_different_keys_do_not_block_each_other() -> None:
    storage = AsyncInMemoryStorage()
    order: list[str] = []

    async def hold_lock_a() -> None:
        async with storage.lock("a"):
            order.append("a-start")
            await asyncio.sleep(0.05)
            order.append("a-end")

    async def hold_lock_b() -> None:
        async with storage.lock("b"):
            order.append("b-start")
            await asyncio.sleep(0.01)
            order.append("b-end")

    await asyncio.gather(hold_lock_a(), hold_lock_b())

    # b should finish before a since it holds a shorter sleep and the
    # two keys' locks are independent -- if they contended with each
    # other, b-start would be delayed until a-end.
    assert order.index("b-start") < order.index("a-end")


@pytest.mark.asyncio
async def test_lock_serializes_access_to_same_key() -> None:
    """Proves the per-key asyncio.Lock actually serializes coroutines,
    the async equivalent of the threading-lock serialization guarantee
    relied on throughout the sync algorithms.

    Does a manual read-sleep-increment-write cycle under the lock, with
    an artificial `await asyncio.sleep(0)` between the read and the
    write to give the event loop a chance to switch to another
    coroutine if the lock were NOT actually preventing it. If locking
    is broken, concurrent increments interleave and the final count
    ends up less than num_increments (a classic lost-update race).
    """
    storage = AsyncInMemoryStorage()
    key = "counter"
    await storage.set(key, {"count": 0})
    num_increments = 50

    async def increment() -> None:
        async with storage.lock(key):
            state = await storage.get(key)
            assert state is not None
            count = state["count"]
            await asyncio.sleep(0)  # yield point -- the danger zone
            await storage.set(key, {"count": count + 1})

    await asyncio.gather(*[increment() for _ in range(num_increments)])

    final_state = await storage.get(key)
    assert final_state is not None
    assert final_state["count"] == num_increments
