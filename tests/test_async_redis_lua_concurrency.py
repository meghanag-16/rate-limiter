# tests/test_async_redis_lua_concurrency.py
"""Async concurrency correctness for all six Redis Lua/GCRA
algorithms -- the review's #15 item ("async needs its own
concurrency tests"). Unlike the in-memory async backend
(AsyncInMemoryStorage, whose get()/set() contain no real `await` and
therefore need YieldingAsyncStorage to force genuine interleaving --
see tests/async_test_helpers.py), every one of these classes' EVAL
calls IS a real network round trip with a real `await` in it, so
asyncio.gather() gives every coroutine a genuine chance to interleave
with no wrapper needed.

Two scenarios, mirroring test_redis_lua_concurrency.py's sync file:

1. SAME-KEY RACE: limit/capacity=1, 50 coroutines via asyncio.gather,
   one attempt each, same key. Exactly one must be admitted.
2. FULL BURST: limit/capacity=40, 60 coroutines, one attempt each,
   same key -- admits exactly min(60, 40) == 40.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import redis.asyncio as redis_async

from limivault.algorithms.async_redis_lua_fixed_window import AsyncRedisLuaFixedWindow
from limivault.algorithms.async_redis_lua_leaky_bucket import (
    AsyncRedisLuaLeakyBucketMeter,
    AsyncRedisLuaLeakyBucketQueue,
)
from limivault.algorithms.async_redis_lua_sliding_window_counter import (
    AsyncRedisLuaSlidingWindowCounter,
)
from limivault.algorithms.async_redis_lua_sliding_window_log import (
    AsyncRedisLuaSlidingWindowLog,
)
from limivault.algorithms.async_redis_lua_token_bucket import AsyncRedisGcraTokenBucket
from limivault.base import AsyncRateLimiter
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]


def _fresh_key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _make_limiters(
    client: redis_async.Redis, capacity: int
) -> list[tuple[str, AsyncRateLimiter]]:
    return [
        (
            "AsyncRedisLuaFixedWindow",
            AsyncRedisLuaFixedWindow(client, limit=capacity, period=3600.0),
        ),
        (
            "AsyncRedisGcraTokenBucket",
            AsyncRedisGcraTokenBucket(client, capacity=capacity, refill_rate=0.0),
        ),
        (
            "AsyncRedisLuaSlidingWindowLog",
            AsyncRedisLuaSlidingWindowLog(client, limit=capacity, period=3600.0),
        ),
        (
            "AsyncRedisLuaSlidingWindowCounter",
            AsyncRedisLuaSlidingWindowCounter(client, limit=capacity, period=3600.0),
        ),
        (
            "AsyncRedisLuaLeakyBucketMeter",
            AsyncRedisLuaLeakyBucketMeter(client, capacity=capacity, leak_rate=0.0),
        ),
        (
            "AsyncRedisLuaLeakyBucketQueue",
            AsyncRedisLuaLeakyBucketQueue(client, capacity=capacity, leak_rate=0.0),
        ),
    ]


@pytest.mark.asyncio
async def test_same_key_race_admits_exactly_one_of_fifty(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client, capacity=1):
        key = _fresh_key("async-race")
        results = await asyncio.gather(*[limiter.allow(key) for _ in range(50)])
        assert sum(results) == 1, name


@pytest.mark.asyncio
async def test_full_burst_admits_exactly_the_limit(
    async_redis_client: redis_async.Redis,
) -> None:
    capacity = 40
    for name, limiter in _make_limiters(async_redis_client, capacity=capacity):
        key = _fresh_key("async-burst")
        results = await asyncio.gather(*[limiter.allow(key) for _ in range(60)])
        assert sum(results) == capacity, name


def test_all_six_async_algorithms_are_covered_by_this_file() -> None:
    import redis.asyncio as _ra

    dummy = _ra.Redis()
    assert len(_make_limiters(dummy, capacity=1)) == 6
