"""Unit coverage for async Redis Lua paths that need no Redis server.

The integration tests exercise actual Lua scripts. These tests isolate
the Python wrappers so error handling and allow_wait branches are covered
even when Docker-backed tests are unavailable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import redis.exceptions

from rlimit.algorithms.async_redis_lua_fixed_window import (
    AsyncRedisLuaFixedWindow,
)
from rlimit.algorithms.async_redis_lua_leaky_bucket import (
    AsyncRedisLuaLeakyBucketMeter,
    AsyncRedisLuaLeakyBucketQueue,
)
from rlimit.algorithms.async_redis_lua_sliding_window_log import (
    AsyncRedisLuaSlidingWindowLog,
)
from rlimit.algorithms.async_redis_lua_token_bucket import AsyncRedisGcraTokenBucket
from rlimit.base import UnsatisfiableRequestError
from rlimit.exceptions import BackendUnavailableError


class FakeRedis:
    """Minimal async Redis surface used by the Python wrapper methods."""

    def __init__(self) -> None:
        self.script_result: Any = [1, 0, 4, 100.0]
        self.script_error: BaseException | None = None
        self.read_error: BaseException | None = None
        self.now = (100, 0)
        self.hash_values: list[bytes | None] = [None, None]
        self.string_value: bytes | None = None
        self.sorted_entries: list[tuple[bytes | str, float]] = []

    def register_script(self, _script: str) -> Callable[..., Any]:
        async def run(**_kwargs: Any) -> Any:
            if self.script_error is not None:
                raise self.script_error
            return self.script_result

        return run

    async def time(self) -> tuple[int, int]:
        if self.read_error is not None:
            raise self.read_error
        return self.now

    async def hmget(self, *_args: Any) -> list[bytes | None]:
        if self.read_error is not None:
            raise self.read_error
        return self.hash_values

    async def get(self, _key: str) -> bytes | None:
        if self.read_error is not None:
            raise self.read_error
        return self.string_value

    async def zremrangebyscore(self, *_args: Any) -> int:
        if self.read_error is not None:
            raise self.read_error
        return 0

    async def zrange(
        self, *_args: Any, **_kwargs: Any
    ) -> list[tuple[bytes | str, float]]:
        if self.read_error is not None:
            raise self.read_error
        return self.sorted_entries


def _limiters(client: Any) -> list[Any]:
    return [
        AsyncRedisLuaFixedWindow(client, limit=5, period=10.0),
        AsyncRedisGcraTokenBucket(client, capacity=5, refill_rate=1.0),
        AsyncRedisLuaSlidingWindowLog(client, limit=5, period=10.0),
        AsyncRedisLuaLeakyBucketMeter(client, capacity=5, leak_rate=1.0),
        AsyncRedisLuaLeakyBucketQueue(client, capacity=5, leak_rate=1.0),
    ]


@pytest.mark.asyncio
async def test_script_paths_and_cost_shortcuts() -> None:
    client: Any = FakeRedis()
    for limiter in _limiters(client):
        assert await limiter.allow("key", cost=1) is True
        client.script_result = [0, 0, 0, 100.0]
        assert await limiter.allow("key", cost=1) is False
        assert await limiter.allow("key", cost=1.5) is False
        assert await limiter.allow("key", cost=-1) is False
        assert await limiter.allow("key", cost=0) is True
        assert await limiter.allow_wait("key", cost=0) == 0.0
        assert await limiter.remaining("key") == 0
        client.script_result = [1, 0, 4, 100.0]


@pytest.mark.asyncio
async def test_allow_reports_backend_errors_even_when_server_time_fails() -> None:
    client: Any = FakeRedis()
    limiter = AsyncRedisLuaFixedWindow(client, limit=5, period=10.0)
    client.script_error = redis.exceptions.ConnectionError("script offline")
    client.read_error = OSError("clock offline")
    with pytest.raises(BackendUnavailableError):
        await limiter.allow("key")
    client.script_error = redis.exceptions.ConnectionError("script offline")
    with pytest.raises(BackendUnavailableError):
        await limiter.remaining("key")
    client.script_error = None
    with pytest.raises(BackendUnavailableError):
        await limiter.allow_wait("key")


@pytest.mark.asyncio
async def test_fixed_window_wait_paths() -> None:
    client: Any = FakeRedis()
    limiter = AsyncRedisLuaFixedWindow(client, limit=2, period=10.0)
    assert await limiter.allow_wait("key") == 0.0
    client.hash_values = [b"100.0", b"2"]
    assert await limiter.allow_wait("key") == 10.0
    client.hash_values = [b"80.0", b"2"]  # old window is treated as empty
    assert await limiter.allow_wait("key") == 0.0
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("key", cost=3)


@pytest.mark.asyncio
async def test_sliding_log_wait_paths_and_cost_parsing() -> None:
    client: Any = FakeRedis()
    limiter = AsyncRedisLuaSlidingWindowLog(client, limit=3, period=10.0)
    assert await limiter.allow_wait("key") == 0.0
    client.sorted_entries = [(b"2:first", 95.0), ("1:second", 98.0)]
    assert await limiter.allow_wait("key", cost=2) == 5.0
    assert await limiter.allow_wait("key", cost=3) == 8.0
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("key", cost=4)


@pytest.mark.asyncio
async def test_token_bucket_wait_paths() -> None:
    client: Any = FakeRedis()
    limiter = AsyncRedisGcraTokenBucket(client, capacity=3, refill_rate=1.0)
    assert await limiter.allow_wait("key") == 0.0
    client.string_value = b"104.0"
    assert await limiter.allow_wait("key", cost=3) == 4.0
    fixed = AsyncRedisGcraTokenBucket(client, capacity=2, refill_rate=0.0)
    client.string_value = None
    assert await fixed.allow_wait("key", cost=2) == 0.0
    client.string_value = b"2"
    with pytest.raises(UnsatisfiableRequestError):
        await fixed.allow_wait("key")
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("key", cost=4)


@pytest.mark.asyncio
async def test_leaky_bucket_wait_paths() -> None:
    client: Any = FakeRedis()
    meter = AsyncRedisLuaLeakyBucketMeter(client, capacity=3, leak_rate=1.0)
    assert await meter.allow_wait("key") == 0.0
    client.hash_values = [b"3.0", b"100.0"]
    assert await meter.allow_wait("key") == 1.0
    frozen_meter = AsyncRedisLuaLeakyBucketMeter(client, capacity=2, leak_rate=0.0)
    with pytest.raises(UnsatisfiableRequestError):
        await frozen_meter.allow_wait("key")

    queue = AsyncRedisLuaLeakyBucketQueue(client, capacity=3, leak_rate=1.0)
    client.hash_values = [b"3", b"100.0"]
    assert await queue.allow_wait("key") == 1.0
    client.hash_values = [b"3", b"98.0"]
    assert await queue.allow_wait("key") == 0.0
    frozen_queue = AsyncRedisLuaLeakyBucketQueue(client, capacity=2, leak_rate=0.0)
    client.hash_values = [b"2", b"100.0"]
    with pytest.raises(UnsatisfiableRequestError):
        await frozen_queue.allow_wait("key")
    with pytest.raises(UnsatisfiableRequestError):
        await queue.allow_wait("key", cost=4)

