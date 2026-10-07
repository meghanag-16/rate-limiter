# tests/test_redis_lua_ttl.py
"""TTL behavior tests for the Redis Lua Redis-native algorithms
-- review #19, flagged as entirely missing in the first pass. Covers,
per the review's explicit list:

  - a key written by allow() actually carries a TTL (not persistent)
  - waiting past that TTL makes the key disappear and the limiter
    behaves like a fresh key afterward
  - an explicit `ttl_seconds=` constructor argument is honored exactly
  - the FALLBACK_TTL_SECONDS path (leak_rate/refill_rate == 0, where
    the algorithm's own characteristic time can't be computed)
  - `ttl_seconds` input validation (review #22): zero, negative, NaN,
    and infinity are all rejected at construction time, whether the
    bad value came from an explicit `ttl_seconds=` argument or (in
    principle) a computed one.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring. Uses
real `time.sleep()` throughout -- no FakeClock is available for this
backend (see limivault.redis_lua_scripts's module docstring), so a TTL
is let genuinely expire by sleeping a real, short interval.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from limivault.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from limivault.algorithms.redis_lua_leaky_bucket import RedisLuaLeakyBucketMeter
from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from limivault.redis_lua_scripts import FALLBACK_TTL_SECONDS, TTL_BUFFER_SECONDS
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# --- A written key actually carries a TTL --------------------------------


def test_fixed_window_key_carries_a_ttl(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=5, period=10.0, ttl_seconds=3.0)
    key = _fresh_key("ttl-fw")
    limiter.allow(key)
    pttl_ms = redis_client.pttl(limiter._data_key(key))
    assert 0 < pttl_ms <= 3000


def test_gcra_key_carries_a_ttl(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(
        redis_client, capacity=5, refill_rate=1.0, ttl_seconds=3.0
    )
    key = _fresh_key("ttl-gcra")
    limiter.allow(key)
    pttl_ms = redis_client.pttl(limiter._data_key(key))
    assert 0 < pttl_ms <= 3000


# --- Key expires and the limiter behaves like fresh afterward -----------


def test_key_expires_and_limiter_resets_like_a_fresh_key(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=1, period=60.0, ttl_seconds=1.0)
    key = _fresh_key("ttl-expire")

    assert limiter.allow(key) is True
    assert limiter.allow(key) is False  # exhausted within the (60s) window

    time.sleep(1.3)  # past the 1s TTL, real elapsed time plus margin
    assert redis_client.exists(limiter._data_key(key)) == 0

    # The key is gone -- next allow() must behave exactly like a
    # never-seen key, not carry over the exhausted state.
    assert limiter.allow(key) is True


# --- Explicit ttl_seconds is honored exactly -----------------------------


def test_explicit_ttl_seconds_is_used_verbatim(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaLeakyBucketMeter(
        redis_client, capacity=5, leak_rate=1.0, ttl_seconds=42.0
    )
    assert limiter._ttl_seconds == 42.0
    key = _fresh_key("ttl-explicit")
    limiter.allow(key)
    pttl_ms = redis_client.pttl(limiter._data_key(key))
    assert 0 < pttl_ms <= 42000


# --- Automatic TTL computation (non-zero rate) ---------------------------


def test_automatic_ttl_for_fixed_window_is_period_times_two_plus_buffer(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=5, period=10.0)
    assert limiter._ttl_seconds == pytest.approx(10.0 * 2 + TTL_BUFFER_SECONDS)


def test_automatic_ttl_for_gcra_is_capacity_over_rate_plus_buffer(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=10, refill_rate=2.0)
    assert limiter._ttl_seconds == pytest.approx(10 / 2.0 + TTL_BUFFER_SECONDS)


# --- Fallback TTL when rate == 0 (characteristic time undefined) --------


def test_gcra_falls_back_to_fallback_ttl_when_refill_rate_is_zero(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=0.0)
    assert limiter._ttl_seconds == FALLBACK_TTL_SECONDS


def test_leaky_bucket_meter_falls_back_to_fallback_ttl_when_leak_rate_is_zero(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaLeakyBucketMeter(redis_client, capacity=5, leak_rate=0.0)
    assert limiter._ttl_seconds == FALLBACK_TTL_SECONDS


# --- ttl_seconds input validation (review #22) ---------------------------


def test_zero_ttl_seconds_raises(redis_client: redis_sync.Redis) -> None:
    with pytest.raises(ValueError):
        RedisLuaFixedWindow(redis_client, limit=5, period=10.0, ttl_seconds=0)


def test_negative_ttl_seconds_raises(redis_client: redis_sync.Redis) -> None:
    with pytest.raises(ValueError):
        RedisLuaFixedWindow(redis_client, limit=5, period=10.0, ttl_seconds=-5.0)


def test_nan_ttl_seconds_raises(redis_client: redis_sync.Redis) -> None:
    with pytest.raises(ValueError):
        RedisLuaFixedWindow(
            redis_client, limit=5, period=10.0, ttl_seconds=float("nan")
        )


def test_infinite_ttl_seconds_raises(redis_client: redis_sync.Redis) -> None:
    with pytest.raises(ValueError):
        RedisLuaFixedWindow(
            redis_client, limit=5, period=10.0, ttl_seconds=float("inf")
        )


def test_zero_ttl_seconds_raises_for_gcra_too(redis_client: redis_sync.Redis) -> None:
    with pytest.raises(ValueError):
        RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=1.0, ttl_seconds=0)
