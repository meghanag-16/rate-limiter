# tests/test_redis_lua_token_bucket.py
"""RedisGcraTokenBucket against real Redis. Covers basic burst/refill
behavior, the GCRA-specific refill_rate==0 fallback path (see
limivault.redis_lua_scripts's module docstring, GCRA gap note), and
allow_wait().

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key() -> str:
    return f"tb-{uuid.uuid4().hex[:10]}"


def test_bucket_starts_full(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=1.0)
    assert limiter.remaining(_fresh_key()) == 5


def test_allow_consumes_tokens(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.remaining(key) == 4


def test_allow_denied_when_empty(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=2, refill_rate=0.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_allow_wait_raises_when_cost_exceeds_capacity(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=1.0)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(_fresh_key(), cost=6)


def test_zero_cost_does_not_touch_storage_for_new_key(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=1.0)
    key = _fresh_key()
    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


# --- GCRA-specific: refill_rate == 0 fallback path (documented gap) ------


def test_refill_rate_zero_allows_exactly_capacity_then_denies_forever(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=3, refill_rate=0.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False
    time.sleep(0.5)  # confirms "forever", not just "not yet"
    assert limiter.allow(key) is False


def test_refill_rate_zero_allow_wait_raises_when_bucket_is_full(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=1, refill_rate=0.0)
    key = _fresh_key()
    limiter.allow(key)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(key)


# --- Real refill over real elapsed time (no FakeClock -- see this
# project's Redis Lua/GCRA module docstring for why) ---------------------------


def test_refill_over_real_elapsed_time(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=5, refill_rate=5.0)  # 5 tok/s
    key = _fresh_key()
    for _ in range(5):
        limiter.allow(key)
    assert limiter.allow(key) is False
    time.sleep(0.5)  # ~2.5 tokens should have refilled
    assert limiter.allow(key) is True


def test_allow_wait_then_retry_succeeds(redis_client: redis_sync.Redis) -> None:
    limiter = RedisGcraTokenBucket(redis_client, capacity=2, refill_rate=2.0)
    key = _fresh_key()
    limiter.allow(key, cost=2)  # drain fully
    wait = limiter.allow_wait(key, cost=1)
    assert wait > 0.0
    time.sleep(wait + 0.05)  # small buffer past the estimate
    assert limiter.allow(key, cost=1) is True
