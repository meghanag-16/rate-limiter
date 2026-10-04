# tests/test_redis_lua_cost_contract.py
"""Shared cost-handling contract tests (sync) for all six Redis Lua/GCRA
(Lua) Redis-native algorithms, mirroring test_cost_contract.py's
factory-table pattern exactly (see that file's docstring for the full
rationale: centralizing the shared contract instead of duplicating it
six times, with a contract violation showing up by name via the
`name` value threaded through each assertion).

Contract under test (identical to every RateLimiter in this project --
see base.py's module docstring):
  - cost must be an int (not bool, not float, including whole-number
    floats and NaN/infinity)
  - negative cost is denied by allow() / raises on allow_wait()
  - cost == 0 is always allowed and is a true no-op (touches no state)
  - cost == capacity/limit is allowed alone; cost == capacity+1 is
    denied by allow() / raises UnsatisfiableRequestError on
    allow_wait()

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring. Every
limiter below is constructed with NO `clock=` argument (see
rlimit.redis_lua_scripts's module docstring -- these classes use
Redis's own server clock, not client-side injection), and each gets
its own randomized key so tests never collide within the shared
Redis database `redis_client` provides.
"""

from __future__ import annotations

import uuid

import pytest
import redis as redis_sync

from rlimit.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from rlimit.algorithms.redis_lua_leaky_bucket import (
    RedisLuaLeakyBucketMeter,
    RedisLuaLeakyBucketQueue,
)
from rlimit.algorithms.redis_lua_sliding_window_counter import (
    RedisLuaSlidingWindowCounter,
)
from rlimit.algorithms.redis_lua_sliding_window_log import RedisLuaSlidingWindowLog
from rlimit.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from rlimit.base import RateLimiter, UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]

_CAPACITY = 5


def _fresh_key() -> str:
    return f"cost-contract-{uuid.uuid4().hex[:10]}"


def _make_limiters(client: redis_sync.Redis) -> list[tuple[str, RateLimiter]]:
    """One instance of each of the six sync Redis Lua/GCRA algorithms,
    capacity/limit=5 uniformly so the shared cost-boundary assertions
    below (cost==5 allowed, cost==6 denied) apply identically to every
    entry, matching test_cost_contract.py's `_make_limiters` pattern."""
    return [
        (
            "RedisLuaFixedWindow",
            RedisLuaFixedWindow(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "RedisGcraTokenBucket",
            RedisGcraTokenBucket(client, capacity=_CAPACITY, refill_rate=1.0),
        ),
        (
            "RedisLuaSlidingWindowLog",
            RedisLuaSlidingWindowLog(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "RedisLuaSlidingWindowCounter",
            RedisLuaSlidingWindowCounter(client, limit=_CAPACITY, period=60.0),
        ),
        (
            "RedisLuaLeakyBucketMeter",
            RedisLuaLeakyBucketMeter(client, capacity=_CAPACITY, leak_rate=1.0),
        ),
        (
            "RedisLuaLeakyBucketQueue",
            RedisLuaLeakyBucketQueue(client, capacity=_CAPACITY, leak_rate=1.0),
        ),
    ]


# --- Zero-cost is a true no-op, for every algorithm ---------------------


def test_zero_cost_returns_true_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=0) is True, name


def test_zero_cost_allow_wait_is_zero_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow_wait(_fresh_key(), cost=0) == 0.0, name


def test_zero_cost_does_not_prevent_a_later_full_cost_call(
    redis_client: redis_sync.Redis,
) -> None:
    """Indirect proof that cost==0 didn't touch state: a brand-new key
    hit with cost=0 first, then cost=capacity, must still admit the
    full capacity -- if the zero-cost call had written anything, this
    would come up short."""
    for name, limiter in _make_limiters(redis_client):
        key = _fresh_key()
        assert limiter.allow(key, cost=0) is True, name
        assert limiter.allow(key, cost=_CAPACITY) is True, name


# --- Cost type validation, for every algorithm ---------------------------


def test_float_cost_rejected_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=2.0) is False, name  # type: ignore[arg-type]


def test_bool_cost_rejected_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        # bool is a subtype of int -- no type: ignore needed, see
        # test_cost_contract.py's identical note.
        assert limiter.allow(_fresh_key(), cost=True) is False, name


def test_nan_cost_rejected_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=float("nan")) is False, name  # type: ignore[arg-type]


def test_infinite_cost_rejected_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=float("inf")) is False, name  # type: ignore[arg-type]


def test_float_cost_raises_value_error_on_allow_wait_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for _name, limiter in _make_limiters(redis_client):
        with pytest.raises(ValueError):
            limiter.allow_wait(_fresh_key(), cost=2.0)  # type: ignore[arg-type]


def test_bool_cost_raises_value_error_on_allow_wait_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for _name, limiter in _make_limiters(redis_client):
        with pytest.raises(ValueError):
            limiter.allow_wait(_fresh_key(), cost=True)


# --- Negative cost still denied/raises, for every algorithm --------------


def test_negative_cost_still_denied_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=-1) is False, name


def test_negative_cost_still_raises_on_allow_wait_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for _name, limiter in _make_limiters(redis_client):
        with pytest.raises(ValueError):
            limiter.allow_wait(_fresh_key(), cost=-1)


# --- Cost boundary: exactly at capacity/limit vs. one above ---------------


def test_cost_exactly_at_capacity_is_allowed_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=_CAPACITY) is True, name


def test_cost_one_above_capacity_denied_by_allow_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        assert limiter.allow(_fresh_key(), cost=_CAPACITY + 1) is False, name


def test_cost_one_above_capacity_raises_unsatisfiable_on_allow_wait_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for _name, limiter in _make_limiters(redis_client):
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait(_fresh_key(), cost=_CAPACITY + 1)


# --- cost=2 (an ordinary multi-unit request), for every algorithm --------


def test_cost_two_consumed_atomically_for_every_algorithm(
    redis_client: redis_sync.Redis,
) -> None:
    for name, limiter in _make_limiters(redis_client):
        key = _fresh_key()
        assert limiter.allow(key, cost=2) is True, name
        assert limiter.remaining(key) == _CAPACITY - 2, name


def test_all_six_sync_lua_algorithms_are_covered_by_this_file() -> None:
    """Guards against a silent gap the same way
    test_metrics_all_algorithms.py's equivalent count-pinning test
    does -- if a 7th Redis Lua/GCRA algorithm is ever added and this file's
    factory table isn't updated to match, this at least documents the
    expected count."""
    import redis as _r

    dummy = _r.Redis()  # never connected to -- only used to count factories
    try:
        assert len(_make_limiters(dummy)) == 6
    finally:
        dummy.close()
