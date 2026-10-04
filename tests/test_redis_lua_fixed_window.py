# tests/test_redis_lua_fixed_window.py
"""RedisLuaFixedWindow against real Redis. Mirrors
test_fixed_window.py's structure/spirit for the in-memory
FixedWindow, adapted for this family's shared-Redis-server-clock
design (see rlimit.redis_lua_scripts's module docstring): there is no
FakeClock to
inject here, so window-boundary tests use real, short periods and real
`time.sleep()` instead of `clock.advance()`.

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from rlimit.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from rlimit.base import UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key() -> str:
    return f"fw-{uuid.uuid4().hex[:10]}"


def _wait_for_middle_of_window(client: redis_sync.Redis, period: float) -> None:
    """Wait until Redis's own clock is well inside a fixed window.

    The Lua script anchors windows to Redis TIME, so local ``time.time()``
    can be out of phase (for example, with a container clock). Targeting
    the middle of the window also leaves ample room for scheduler delays
    before the next boundary.
    """
    lower = period * 0.25
    upper = period * 0.75
    target = period * 0.5
    while True:
        seconds, microseconds = client.time()
        remainder = (float(seconds) + float(microseconds) / 1_000_000.0) % period
        if lower <= remainder <= upper:
            return
        time.sleep((target - remainder) % period)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_allows_up_to_limit_within_window(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_remaining_decreases_with_each_allow(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert limiter.remaining(key) == 3
    limiter.allow(key)
    assert limiter.remaining(key) == 2


def test_denied_call_does_not_consume_quota(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=1, period=60.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False
    assert limiter.remaining(key) == 0


def test_different_keys_have_independent_windows(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=1, period=60.0)
    a, b = _fresh_key(), _fresh_key()
    assert limiter.allow(a) is True
    assert limiter.allow(b) is True
    assert limiter.allow(a) is False
    assert limiter.allow(b) is False


def test_allow_wait_raises_when_cost_exceeds_limit(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=5, period=60.0)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(_fresh_key(), cost=6)


def test_zero_cost_does_not_touch_storage_for_new_key(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaFixedWindow(redis_client, limit=5, period=60.0)
    key = _fresh_key()
    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


# ---------------------------------------------------------------------------
# Boundary: real window rollover via real sleep (no FakeClock available
# for the Lua/GCRA classes -- see this file's module docstring). `period=2.0` kept
# short so the test suite doesn't take unnecessarily long, matching the
# spirit (not the letter) of the existing FakeClock-based boundary
# tests: assert exact admitted counts on either side of the rollover.
# ---------------------------------------------------------------------------


def test_window_rolls_over_after_real_period_elapses(
    redis_client: redis_sync.Redis,
) -> None:
    period = 2.0
    _wait_for_middle_of_window(redis_client, period)
    limiter = RedisLuaFixedWindow(redis_client, limit=1, period=period)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False
    time.sleep(2.2)  # real elapsed time past the 2s window, plus margin
    assert limiter.allow(key) is True


def test_window_does_not_roll_over_before_period_elapses(
    redis_client: redis_sync.Redis,
) -> None:
    # Use Redis's clock and a generous window so both calls stay far from
    # either boundary even if the test process is briefly descheduled.
    period = 10.0
    _wait_for_middle_of_window(redis_client, period)
    limiter = RedisLuaFixedWindow(redis_client, limit=1, period=period)
    key = _fresh_key()
    assert limiter.allow(key) is True
    time.sleep(0.3)  # well within the 2s window
    assert limiter.allow(key) is False
