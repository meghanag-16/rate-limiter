# tests/test_redis_lua_script_contract.py
"""Level 2 tests (review #20): verify the raw return shape of each of
the six Lua scripts directly via EVAL, independent of the Python
wrapper classes in limivault.algorithms.redis_lua_*.py.

WHY THIS FILE EXISTS, SEPARATELY FROM THE PER-ALGORITHM TEST FILES:
every wrapper class unpacks the script's return value positionally --
`result[0]` (allowed), `result[2]` (remaining) -- with no structural
check in between. If someone edits a script's return statement six
months from now (reorders the triple, adds a field, changes what
index 2 means for one algorithm but not another), the per-algorithm
tests could keep passing for the wrong reason (e.g. if a script
regression happens to still produce a truthy/falsy result[0]) while
the actual CONTRACT between script and wrapper has quietly broken.
These tests pin the exact four-element shape (v3: allowed,
cost_applied, quota, now) and exact values for known inputs, calling
`client.eval(...)` directly -- no RateLimiter class
involved at all -- so a script-level regression fails here specifically,
by name, rather than surfacing as a confusing failure somewhere in a
wrapper class's own test.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import uuid

import redis as redis_sync

from limivault.redis_lua_scripts import (
    FIXED_WINDOW,
    GCRA_TOKEN_BUCKET,
    LEAKY_BUCKET_METER,
    LEAKY_BUCKET_QUEUE,
    SLIDING_WINDOW_COUNTER,
    SLIDING_WINDOW_LOG,
)
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key(prefix: str) -> str:
    return f"script-contract-{prefix}-{uuid.uuid4().hex[:8]}"


def _eval(
    client: redis_sync.Redis,
    script: str,
    key: str,
    args: list[bytes | str | memoryview[int] | bytearray | int | float],
) -> list[int | float]:
    raw = client.eval(script, 1, key, *args)
    # redis-py returns a list of ints/bytes for a Lua table reply.
    # The first three elements (allowed, cost_applied, quota) are
    # always whole numbers -- normalize those to plain ints. The 4th
    # element (v3: the script's own server-time reading, `now`) is a
    # float and must NOT be truncated to an int, or every assertion
    # below comparing against a fresh wall-clock value would be
    # comparing apples to a rounded-down orange.
    return [int(x) for x in raw[:3]] + [float(raw[3])]


def _assert_now_is_plausible(now: float) -> None:
    """v3: the script's 4th return element is its own `TIME`-based
    `now` reading. Assert it's a real, current timestamp -- not 0.0
    (the old placeholder a regression could reintroduce) and within
    a generous window of the test process's own wall clock (same
    machine as the Redis container in this test suite, so a tight
    bound is meaningful, not flaky)."""
    import time

    assert now > 0.0
    assert abs(now - time.time()) < 10.0


# --- FIXED_WINDOW: ARGV = cost, mode, limit, period, ttl -----------------


def test_fixed_window_script_returns_four_element_shape(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("fw")
    result = _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])
    assert len(result) == 4
    _assert_now_is_plausible(result[3])


def test_fixed_window_script_admits_and_reports_remaining(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("fw")
    r1 = _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])
    assert r1[:3] == [1, 1, 2]
    r2 = _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])
    assert r2[:3] == [1, 1, 1]
    r3 = _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])
    assert r3[:3] == [1, 1, 0]
    r4 = _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])
    assert r4[:3] == [0, 0, 0]
    # `now` must never go backwards across successive calls to the
    # same key -- the whole point of the shared-server-clock design.
    assert r1[3] <= r2[3] <= r3[3] <= r4[3]


def test_fixed_window_script_peek_mode_does_not_mutate(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("fw-peek")
    _eval(redis_client, FIXED_WINDOW, key, [1, "0", 3, 60.0, 999])  # count=1
    peek1 = _eval(redis_client, FIXED_WINDOW, key, [1, "1", 3, 60.0, 999])
    peek2 = _eval(redis_client, FIXED_WINDOW, key, [1, "1", 3, 60.0, 999])
    assert peek1[:3] == peek2[:3] == [1, 1, 2]  # unchanged across repeated peeks


# --- GCRA_TOKEN_BUCKET: ARGV = cost, mode, capacity, refill_rate, ttl ----


def test_gcra_script_returns_four_element_shape(redis_client: redis_sync.Redis) -> None:
    key = _fresh_key("gcra")
    result = _eval(redis_client, GCRA_TOKEN_BUCKET, key, [1, "0", 5, 1.0, 999])
    assert len(result) == 4
    _assert_now_is_plausible(result[3])


def test_gcra_script_admits_and_reports_tokens_after(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("gcra")
    r1 = _eval(redis_client, GCRA_TOKEN_BUCKET, key, [1, "0", 5, 1.0, 999])
    assert r1[:3] == [1, 1, 4]


def test_gcra_script_denies_when_exhausted(redis_client: redis_sync.Redis) -> None:
    key = _fresh_key("gcra-deny")
    for _ in range(5):
        _eval(redis_client, GCRA_TOKEN_BUCKET, key, [1, "0", 5, 1.0, 999])
    result = _eval(redis_client, GCRA_TOKEN_BUCKET, key, [1, "0", 5, 1.0, 999])
    assert result[0] == 0
    assert result[1] == 0


# --- SLIDING_WINDOW_COUNTER: ARGV = cost, mode, limit, period, ttl -------


def test_sliding_window_counter_script_returns_four_element_shape(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("swc")
    result = _eval(redis_client, SLIDING_WINDOW_COUNTER, key, [1, "0", 5, 60.0, 999])
    assert len(result) == 4
    assert result[:3] == [1, 1, 4]
    _assert_now_is_plausible(result[3])


# --- LEAKY_BUCKET_METER: ARGV = cost, mode, capacity, leak_rate, ttl -----


def test_leaky_bucket_meter_script_returns_four_element_shape(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("lbm")
    result = _eval(redis_client, LEAKY_BUCKET_METER, key, [1, "0", 5, 1.0, 999])
    assert len(result) == 4
    assert result[:3] == [1, 1, 4]
    _assert_now_is_plausible(result[3])


# --- LEAKY_BUCKET_QUEUE: ARGV = cost, mode, capacity, leak_rate, ttl -----


def test_leaky_bucket_queue_script_returns_four_element_shape(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("lbq")
    result = _eval(redis_client, LEAKY_BUCKET_QUEUE, key, [1, "0", 5, 1.0, 999])
    assert len(result) == 4
    assert result[:3] == [1, 1, 4]
    _assert_now_is_plausible(result[3])


# --- SLIDING_WINDOW_LOG: ARGV = cost, mode, limit, period, ttl, suffix ---


def test_sliding_window_log_script_returns_four_element_shape(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("swl")
    result = _eval(redis_client, SLIDING_WINDOW_LOG, key, [1, "0", 5, 60.0, 999, "s1"])
    assert len(result) == 4
    assert result[:3] == [1, 1, 4]
    _assert_now_is_plausible(result[3])


def test_sliding_window_log_script_peek_mode_prunes_but_does_not_add(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("swl-peek")
    _eval(redis_client, SLIDING_WINDOW_LOG, key, [1, "0", 5, 60.0, 999, "s1"])
    peek = _eval(redis_client, SLIDING_WINDOW_LOG, key, [1, "1", 5, 60.0, 999, ""])
    assert peek[:3] == [1, 1, 4]
    members = redis_client.zrange(key, 0, -1)
    assert len(members) == 1  # peek did not add a second member


def test_sliding_window_log_script_denies_when_full(
    redis_client: redis_sync.Redis,
) -> None:
    key = _fresh_key("swl-full")
    for i in range(3):
        _eval(redis_client, SLIDING_WINDOW_LOG, key, [1, "0", 3, 60.0, 999, f"s{i}"])
    result = _eval(
        redis_client, SLIDING_WINDOW_LOG, key, [1, "0", 3, 60.0, 999, "s-extra"]
    )
    assert result[0] == 0
    assert result[1] == 0
