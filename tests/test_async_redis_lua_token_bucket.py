# tests/test_async_redis_lua_token_bucket.py
"""Async mirror of test_redis_lua_token_bucket.py. REQUIRES DOCKER --
see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from rlimit.algorithms.async_redis_lua_token_bucket import AsyncRedisGcraTokenBucket
from rlimit.base import UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key() -> str:
    return f"async-tb-{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_bucket_starts_full(async_redis_client: redis_async.Redis) -> None:
    limiter = AsyncRedisGcraTokenBucket(async_redis_client, capacity=5, refill_rate=1.0)
    assert await limiter.remaining(_fresh_key()) == 5


@pytest.mark.asyncio
async def test_allow_consumes_tokens(async_redis_client: redis_async.Redis) -> None:
    limiter = AsyncRedisGcraTokenBucket(async_redis_client, capacity=5, refill_rate=1.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.remaining(key) == 4


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_capacity(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisGcraTokenBucket(async_redis_client, capacity=5, refill_rate=1.0)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)


@pytest.mark.asyncio
async def test_refill_rate_zero_allows_exactly_capacity_then_denies_forever(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisGcraTokenBucket(async_redis_client, capacity=3, refill_rate=0.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_refill_over_real_elapsed_time(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisGcraTokenBucket(async_redis_client, capacity=5, refill_rate=5.0)
    key = _fresh_key()
    for _ in range(5):
        await limiter.allow(key)
    assert await limiter.allow(key) is False
    await asyncio.sleep(0.5)
    assert await limiter.allow(key) is True
