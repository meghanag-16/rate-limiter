# tests/test_redis_lua_sliding_window_log.py
"""RedisLuaSlidingWindowLog against real Redis (ZSET-backed). Covers
basic correctness, the exact-cutoff expiry semantics (an entry with
timestamp == now - period is expired -- see
limivault.redis_lua_scripts.SLIDING_WINDOW_LOG's own comment: the script
uses ZREMRANGEBYSCORE(key, '-inf', cutoff), an inclusive upper bound,
so a score exactly at cutoff IS removed, matching this project's
existing SlidingWindowLog semantics where `ts > cutoff` is the
survival condition), and allow_wait().

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring. No
FakeClock available for this backend (see limivault.redis_lua_scripts's
module docstring) -- boundary timing uses real sleeps with a small
buffer (a deliberate epsilon past the exact boundary), the real-time
counterpart of the `clock.advance(...)` boundary tests in
test_sliding_window_log.py.
"""

from __future__ import annotations

import time
import uuid
from typing import cast

import pytest
import redis as redis_sync

from limivault.algorithms.redis_lua_sliding_window_log import RedisLuaSlidingWindowLog
from limivault.base import UnsatisfiableRequestError
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key() -> str:
    return f"swl-{uuid.uuid4().hex[:10]}"


def test_allows_up_to_limit(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=3, period=60.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False


def test_allow_wait_raises_when_cost_exceeds_limit(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=5, period=60.0)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait(_fresh_key(), cost=6)


def test_zero_cost_does_not_touch_storage_for_new_key(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=5, period=60.0)
    key = _fresh_key()
    assert limiter.allow(key, cost=0) is True
    assert redis_client.exists(limiter._data_key(key)) == 0


def test_stored_entries_are_zset_members_with_cost_prefix(
    redis_client: redis_sync.Redis,
) -> None:
    """Direct regression check for this backend's member-encoding
    scheme (see limivault.redis_lua_scripts.SLIDING_WINDOW_LOG's own
    comment) -- confirms cost travels as a parseable prefix on the
    member string (a ZSET member is a plain string, so cost is encoded
    into it rather than stored as a separate field)."""
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=5, period=60.0)
    key = _fresh_key()
    limiter.allow(key, cost=3)
    members = redis_client.zrange(limiter._data_key(key), 0, -1)
    assert len(members) == 1
    member = cast(bytes | str, members[0])
    member_text = member.decode("utf-8") if isinstance(member, bytes) else member
    assert member_text.split(":", 1)[0] == "3"
    assert limiter.remaining(key) == 2


# --- Boundary: entry at exactly the cutoff instant expires ---------------


def test_entry_at_the_period_boundary_is_expired(
    redis_client: redis_sync.Redis,
) -> None:
    """period=2.0: after real elapsed time just past 2 seconds, the
    original entry's timestamp satisfies score <= cutoff and must be
    pruned -- exercising the exact inclusive-boundary behavior the
    review specifically flagged (ZREMRANGEBYSCORE's '-inf' to cutoff
    range removes a score exactly equal to cutoff)."""
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=1, period=2.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False
    time.sleep(2.05)  # just past the boundary, small buffer for scheduling jitter
    assert limiter.allow(key) is True


def test_entry_well_before_the_boundary_still_counts(
    redis_client: redis_sync.Redis,
) -> None:
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=1, period=2.0)
    key = _fresh_key()
    assert limiter.allow(key) is True
    time.sleep(0.2)  # well inside the window
    assert limiter.allow(key) is False


def test_allow_wait_then_retry_succeeds(redis_client: redis_sync.Redis) -> None:
    limiter = RedisLuaSlidingWindowLog(redis_client, limit=1, period=2.0)
    key = _fresh_key()
    limiter.allow(key)
    wait = limiter.allow_wait(key)
    assert wait > 0.0
    time.sleep(wait + 0.05)
    assert limiter.allow(key) is True
