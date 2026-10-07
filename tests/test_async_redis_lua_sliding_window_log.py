# tests/test_async_redis_lua_sliding_window_log.py
"""Async mirror of test_redis_lua_sliding_window_log.py. REQUIRES
DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from limivault.algorithms.async_redis_lua_sliding_window_log import (
    AsyncRedisLuaSlidingWindowLog,
)
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key() -> str:
    return f"async-swl-{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_allows_up_to_limit(async_redis_client: redis_async.Redis) -> None:
    limiter = AsyncRedisLuaSlidingWindowLog(async_redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit(
    async_redis_client: redis_async.Redis,
) -> None:
    limiter = AsyncRedisLuaSlidingWindowLog(async_redis_client, limit=5, period=60.0)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait(_fresh_key(), cost=6)


@pytest.mark.asyncio
async def test_entry_at_the_period_boundary_is_expired(
    async_redis_client: redis_async.Redis,
) -> None:
    """See test_redis_lua_sliding_window_log.py's sync equivalent for
    the full rationale on the inclusive-cutoff behavior being tested
    here."""
    limiter = AsyncRedisLuaSlidingWindowLog(async_redis_client, limit=1, period=2.0)
    key = _fresh_key()
    assert await limiter.allow(key) is True
    assert await limiter.allow(key) is False
    await asyncio.sleep(2.05)
    assert await limiter.allow(key) is True
