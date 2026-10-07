# tests/test_async_redis_lua_cost_contract.py
"""Async mirror of test_redis_lua_cost_contract.py -- same shared
cost-handling contract, same factory-table pattern, across all six
async Redis Lua Redis-native algorithms.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

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
from limivault.base import AsyncRateLimiter, UnsatisfiableRequestError
from tests.redis_test_helpers import (
    async_redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = ["redis_container", "async_redis_client", "redis_connection_params"]

_CAPACITY = 5


def _fresh_key() -> str:
    return f"async-cost-contract-{uuid.uuid4().hex[:10]}"


def _make_limiters(client: redis_async.Redis) -> list[tuple[str, AsyncRateLimiter]]:
    return [
        (
            "AsyncRedisLuaFixedWindow",
            AsyncRedisLuaFixedWindow(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "AsyncRedisGcraTokenBucket",
            AsyncRedisGcraTokenBucket(client, capacity=_CAPACITY, refill_rate=1.0),
        ),
        (
            "AsyncRedisLuaSlidingWindowLog",
            AsyncRedisLuaSlidingWindowLog(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "AsyncRedisLuaSlidingWindowCounter",
            AsyncRedisLuaSlidingWindowCounter(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "AsyncRedisLuaLeakyBucketMeter",
            AsyncRedisLuaLeakyBucketMeter(client, capacity=_CAPACITY, leak_rate=1.0),
        ),
        (
            "AsyncRedisLuaLeakyBucketQueue",
            AsyncRedisLuaLeakyBucketQueue(client, capacity=_CAPACITY, leak_rate=1.0),
        ),
    ]


@pytest.mark.asyncio
async def test_zero_cost_returns_true_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=0) is True, name


@pytest.mark.asyncio
async def test_zero_cost_allow_wait_is_zero_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow_wait(_fresh_key(), cost=0) == 0.0, name


@pytest.mark.asyncio
async def test_zero_cost_does_not_prevent_a_later_full_cost_call(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        key = _fresh_key()
        assert await limiter.allow(key, cost=0) is True, name
        assert await limiter.allow(key, cost=_CAPACITY) is True, name


@pytest.mark.asyncio
async def test_float_cost_rejected_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=2.0) is False, name  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bool_cost_rejected_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=True) is False, name


@pytest.mark.asyncio
async def test_nan_cost_rejected_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=float("nan")) is False, name  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_infinite_cost_rejected_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=float("inf")) is False, name  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_float_cost_raises_value_error_on_allow_wait_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for _name, limiter in _make_limiters(async_redis_client):
        with pytest.raises(ValueError):
            await limiter.allow_wait(_fresh_key(), cost=2.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bool_cost_raises_value_error_on_allow_wait_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for _name, limiter in _make_limiters(async_redis_client):
        with pytest.raises(ValueError):
            await limiter.allow_wait(_fresh_key(), cost=True)


@pytest.mark.asyncio
async def test_negative_cost_still_denied_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=-1) is False, name


@pytest.mark.asyncio
async def test_negative_cost_still_raises_on_allow_wait_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for _name, limiter in _make_limiters(async_redis_client):
        with pytest.raises(ValueError):
            await limiter.allow_wait(_fresh_key(), cost=-1)


@pytest.mark.asyncio
async def test_cost_exactly_at_capacity_is_allowed_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=_CAPACITY) is True, name


@pytest.mark.asyncio
async def test_cost_one_above_capacity_denied_by_allow_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        assert await limiter.allow(_fresh_key(), cost=_CAPACITY + 1) is False, name


@pytest.mark.asyncio
async def test_cost_one_above_capacity_raises_unsatisfiable_on_allow_wait_for_every_algorithm(  # noqa: E501
    async_redis_client: redis_async.Redis,
) -> None:
    for _name, limiter in _make_limiters(async_redis_client):
        with pytest.raises(UnsatisfiableRequestError):
            await limiter.allow_wait(_fresh_key(), cost=_CAPACITY + 1)


@pytest.mark.asyncio
async def test_cost_two_consumed_atomically_for_every_algorithm(
    async_redis_client: redis_async.Redis,
) -> None:
    for name, limiter in _make_limiters(async_redis_client):
        key = _fresh_key()
        assert await limiter.allow(key, cost=2) is True, name
        assert await limiter.remaining(key) == _CAPACITY - 2, name
