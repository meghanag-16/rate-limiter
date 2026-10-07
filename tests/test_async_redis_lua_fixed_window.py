# tests/test_async_redis_lua_fixed_window.py
"""Async mirror of test_redis_lua_fixed_window.py. REQUIRES DOCKER --
see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from limivault.algorithms.async_redis_lua_fixed_window import AsyncRedisLuaFixedWindow
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key() -> str:
    return f"async-fw-{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_allows_up_to_limit_within_window(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaFixedWindow(async_redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_remaining_decreases_with_each_allow(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaFixedWindow(async_redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert await limiter.remaining(key) == 3
    await limiter.allow(key)
    assert await limiter.remaining(key) == 2


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaFixedWindow(async_redis_client, limit=5, period=60.0)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)


@pytest.mark.asyncio
async def test_window_rolls_over_after_real_period_elapses(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaFixedWindow(async_redis_client, limit=1, period=2.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False
    await asyncio.sleep(2.2)
    assert await limiter.allow(key) is True
