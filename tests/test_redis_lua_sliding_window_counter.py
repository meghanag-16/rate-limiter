# tests/test_redis_lua_sliding_window_counter.py
"""RedisLuaSlidingWindowCounter against real Redis. Covers basic
weighted-decay behavior across a real window rollover (real sleep, no
FakeClock -- see rlimit.redis_lua_scripts's module docstring) and
allow_wait().

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from rlimit.algorithms.redis_lua_sliding_window_counter import (
    RedisLuaSlidingWindowCounter,
)
from rlimit.base import UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key() -> str:
    return f"swc-{uuid.uuid4().hex[:10]}"


def _redis_time(redis_client: redis_sync.Redis) -> float:
    seconds, microseconds = redis_client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


def test_allows_up_to_limit_within_first_window(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowCounter(
        redis_client,
        limit=3,
        period=60.0,
    )
    key = _fresh_key()

    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_allow_wait_raises_when_cost_exceeds_limit(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowCounter(
        redis_client,
        limit=5,
        period=60.0,
    )

    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(_fresh_key(), cost=6)


def test_zero_cost_does_not_touch_storage_for_new_key(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowCounter(
        redis_client,
        limit=5,
        period=60.0,
    )
    key = _fresh_key()

    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


# --- Boundary: real window rollover, weighted decay --------------------


def test_weighted_count_decays_across_a_real_window_rollover(
    redis_client: redis_sync.Redis,
) -> None:
    """period=6.0: fill the first window completely, then, when the
    actual Redis-aligned window still has enough time remaining,
    confirm the quota is still exhausted.

    The test deliberately uses Redis TIME rather than deriving the
    remaining window time from the duration of the Python fill loop.
    The Lua implementation aligns windows to Redis's Unix epoch clock,
    so the Python loop duration alone cannot tell us how close we are
    to the next window boundary.

    Finally, wait well past a complete window and confirm that the
    previous window's contribution no longer blocks admission.
    """
    period = 6.0
    limiter = RedisLuaSlidingWindowCounter(
        redis_client,
        limit=10,
        period=period,
    )
    key = _fresh_key()

    # Capture the actual Redis-aligned window before filling.
    start_now = _redis_time(redis_client)
    start_window = (start_now // period) * period

    for _ in range(10):
        assert limiter.allow(key) is True

    assert limiter.allow(key) is False

    # Re-read Redis TIME after the fill. This tells us whether the
    # fill itself crossed the Redis-aligned window boundary.
    now = _redis_time(redis_client)
    current_window = (now // period) * period

    if current_window == start_window:
        elapsed_in_window = now - current_window
        remaining_in_window = period - elapsed_in_window

        # Keep a generous margin before the actual boundary so that
        # scheduler overshoot cannot accidentally move this assertion
        # into the next window.
        safety_buffer = 1.0

        if remaining_in_window > safety_buffer:
            sleep_for = min(
                remaining_in_window - safety_buffer,
                2.0,
            )

            time.sleep(sleep_for)

            # We intentionally remain inside the same Redis-aligned
            # window, so the full current-window count must still block.
            assert limiter.allow(key) is False

    # From wherever we are now, waiting a complete additional period
    # guarantees that the state above cannot remain the active current
    # or immediately previous window indefinitely.
    time.sleep(period + 2.0)

    assert limiter.allow(key) is True


def test_allow_wait_then_retry_succeeds(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowCounter(
        redis_client,
        limit=4,
        period=2.0,
    )
    key = _fresh_key()

    assert limiter.allow(key, cost=4) is True

    wait = limiter.allow_wait(key, cost=2)

    assert wait > 0.0

    time.sleep(wait + 0.05)

    assert limiter.allow(key, cost=2) is True