# tests/test_redis_lua_leaky_bucket.py
"""RedisLuaLeakyBucketMeter / RedisLuaLeakyBucketQueue against real
Redis. One file for both variants, matching this project's existing
leaky_bucket.py/test_leaky_bucket.py convention. Real elapsed time
(no FakeClock -- see rlimit.redis_lua_scripts's module docstring).

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from rlimit.algorithms.redis_lua_leaky_bucket import (
    RedisLuaLeakyBucketMeter,
    RedisLuaLeakyBucketQueue,
)
from rlimit.base import UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key() -> str:
    return f"lb-{uuid.uuid4().hex[:10]}"


# --- Meter ----------------------------------------------------------------


def test_meter_allows_up_to_capacity(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketMeter(redis_client, capacity=3, leak_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_meter_leaks_over_real_elapsed_time(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketMeter(redis_client, capacity=10, leak_rate=10.0)
    key = _fresh_key()
    limiter.allow(key, cost=10)
    assert limiter.remaining(key) == 0
    time.sleep(0.5)  # ~5 units should have leaked at 10/s
    remaining = limiter.remaining(key)
    assert 3 <= remaining <= 7  # real-time tolerance band, not exact


def test_meter_allow_wait_zero_leak_rate_raises(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketMeter(redis_client, capacity=1, leak_rate=0.0)
    key = _fresh_key()
    limiter.allow(key)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(key)


def test_meter_zero_cost_does_not_touch_storage(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketMeter(redis_client, capacity=5, leak_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


# --- Queue ------------------------------------------------------------


def test_queue_allows_up_to_capacity(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketQueue(redis_client, capacity=3, leak_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_queue_drains_whole_items_only(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketQueue(redis_client, capacity=5, leak_rate=5.0)  # 5/s
    key = _fresh_key()
    limiter.allow(key, cost=5)
    assert limiter.remaining(key) == 0
    time.sleep(0.45)  # ~2.25 items worth of time -> exactly 2 whole items drain
    assert limiter.remaining(key) == 2


def test_queue_allow_wait_raises_when_cost_exceeds_capacity(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaLeakyBucketQueue(redis_client, capacity=5, leak_rate=1.0)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(_fresh_key(), cost=6)


def test_queue_zero_cost_does_not_touch_storage(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketQueue(redis_client, capacity=5, leak_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


def test_queue_allow_wait_then_retry_succeeds(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketQueue(redis_client, capacity=5, leak_rate=2.0)
    key = _fresh_key()
    limiter.allow(key, cost=5)
    wait = limiter.allow_wait(key, cost=1)
    assert wait > 0.0
    time.sleep(wait + 0.05)
    assert limiter.allow(key, cost=1) is True
