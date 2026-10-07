# tests/test_async_redis_lua_sliding_window_counter.py
"""Async mirror of test_redis_lua_sliding_window_counter.py.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from limivault.algorithms.async_redis_lua_sliding_window_counter import (
    AsyncRedisLuaSlidingWindowCounter,
)
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key() -> str:
    return f"async-swc-{uuid.uuid4().hex[:10]}"


async def _redis_time(redis_client: redis_async.Redis) -> float:
    seconds, microseconds = await redis_client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


@pytest.mark.asyncio
async def test_allows_up_to_limit_within_first_window(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaSlidingWindowCounter(
        async_redis_client,
        limit=3,
        period=60.0,
    )
    key = _fresh_key()

    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaSlidingWindowCounter(
        async_redis_client,
        limit=5,
        period=60.0,
    )

    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)


@pytest.mark.asyncio
async def test_weighted_count_decays_across_a_real_window_rollover(
    async_redis_client: redis_async.Redis,
) -> None:
    """Verify weighted sliding-window decay using Redis's actual clock.

    The Lua implementation uses Redis TIME and epoch-aligned windows.

    The test fills a window completely, confirms immediate denial, then
    waits long enough for the previous window's weighted contribution to
    decay completely before verifying that a new request is admitted.
    """
    period = 6.0

    limiter = AsyncRedisLuaSlidingWindowCounter(
        async_redis_client,
        limit=10,
        period=period,
    )
    key = _fresh_key()

    # Fill the current window.
    for _ in range(10):
        assert await limiter.allow(key) is True

    # The quota must be exhausted immediately after the fill.
    assert await limiter.allow(key) is False

    # Read Redis's actual server clock. The limiter uses this same clock.
    now = await _redis_time(async_redis_client)

    current_window = (now // period) * period
    remaining_in_window = (current_window + period) - now

    # First cross the current Redis-aligned window boundary.
    await asyncio.sleep(remaining_in_window + 0.25)

    # We are in the next window, but the previous window still contributes
    # to the weighted sliding-window count, so admission is not guaranteed
    # immediately after the boundary.
    assert await limiter.allow(key) is False

    # Wait for the previous window's weighted contribution to disappear.
    # At this point we are more than one complete window past the original
    # fill, matching the semantic contract used by the synchronous test.
    await asyncio.sleep(period + 0.25)

    assert await limiter.allow(key) is True


@pytest.mark.asyncio
async def test_allow_wait_then_retry_succeeds(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaSlidingWindowCounter(
        async_redis_client,
        limit=4,
        period=2.0,
    )
    key = _fresh_key()

    assert await limiter.allow(key, cost=4) is True

    wait = await limiter.allow_wait(key, cost=2)

    assert wait > 0.0

    await asyncio.sleep(wait + 0.05)

    assert await limiter.allow(key, cost=2) is True