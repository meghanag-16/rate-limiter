# tests/test_async_redis_lua_leaky_bucket.py
"""Async mirror of test_redis_lua_leaky_bucket.py. REQUIRES DOCKER --
see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from limivault.algorithms.async_redis_lua_leaky_bucket import (
    AsyncRedisLuaLeakyBucketMeter,
    AsyncRedisLuaLeakyBucketQueue,
)
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key() -> str:
    return f"async-lb-{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_meter_allows_up_to_capacity(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketMeter(
        async_redis_client, capacity=3, leak_rate=1.0
    )
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_meter_leaks_over_real_elapsed_time(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketMeter(
        async_redis_client, capacity=10, leak_rate=10.0
    )
    key = _fresh_key()
    await limiter.allow(key, cost=10)
    assert await limiter.remaining(key) == 0
    await asyncio.sleep(0.5)
    remaining = await limiter.remaining(key)
    assert 3 <= remaining <= 7


@pytest.mark.asyncio
async def test_meter_allow_wait_raises_when_cost_exceeds_capacity(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketMeter(
        async_redis_client, capacity=5, leak_rate=1.0
    )
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)


@pytest.mark.asyncio
async def test_queue_allows_up_to_capacity(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketQueue(
        async_redis_client, capacity=3, leak_rate=1.0
    )
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_queue_drains_whole_items_only(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketQueue(
        async_redis_client, capacity=5, leak_rate=5.0
    )
    key = _fresh_key()
    await limiter.allow(key, cost=5)
    assert await limiter.remaining(key) == 0
    await asyncio.sleep(0.45)
    assert await limiter.remaining(key) == 2


@pytest.mark.asyncio
async def test_queue_allow_wait_raises_when_cost_exceeds_capacity(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaLeakyBucketQueue(
        async_redis_client, capacity=5, leak_rate=1.0
    )
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)
