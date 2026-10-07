# tests/test_redis_lua_ergonomics.py
"""Ergonomics tests (block_until_allowed, wait(), rate_limit(),
KeyedLimiter) against Redis-backed Lua rate limiters.

Verifies that limivault.ergonomics works with Redis Lua/GCRA algorithms that
have no injected clock parameter -- the ergonomics layer's own timeout
clock (default time.monotonic) is fully independent of the limiter's
internal time source (Redis TIME).

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync

from limivault.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from limivault.base import UnsatisfiableRequestError
from limivault.ergonomics import (
    KeyedLimiter,
    RateLimitTimeoutError,
    block_until_allowed,
    rate_limit,
    wait,
)
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]


def _fresh_key(prefix: str = "erg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _redis_time(redis_client: redis_sync.Redis) -> float:
    seconds, microseconds = redis_client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


# ------------------------------------------------------------------
# block_until_allowed
# ------------------------------------------------------------------


class TestBlockUntilAllowed:
    def test_immediate_admission_when_under_limit(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=5,
            period=60.0,
            key_prefix="blka:",
        )
        key = _fresh_key("blk1")

        block_until_allowed(limiter, key, cost=1)

        assert limiter.remaining(key) == 4

    def test_blocks_then_admits_after_window_rollover(
        self, redis_client: redis_sync.Redis
    ) -> None:
        """FixedWindow limit=1, short period.

        The first call fills the current Redis-aligned window.
        block_until_allowed() must wait until the next Redis-aligned
        window before admitting the second request.

        The test deliberately does not assert a fixed minimum elapsed
        duration because the request may occur very close to the end of
        the 0.15s Redis-aligned window.
        """
        period = 0.15

        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=period,
            key_prefix="blkwait:",
        )
        key = _fresh_key("blk2")

        assert limiter.allow(key) is True

        # Redis TIME is the clock used by the limiter itself. Capture
        # the current window before calling block_until_allowed().
        before = _redis_time(redis_client)
        before_window = (before // period) * period

        t0 = time.monotonic()

        block_until_allowed(
            limiter,
            key,
            cost=1,
            timeout=2.0,
        )

        elapsed = time.monotonic() - t0

        # It must actually have blocked rather than immediately admitting.
        assert elapsed > 0.0

        # Confirm that Redis has moved into a new fixed-window interval.
        after = _redis_time(redis_client)
        after_window = (after // period) * period

        assert after_window > before_window

        # The second request was admitted in the new window, consuming
        # its quota.
        assert limiter.remaining(key) == 0

    def test_timeout_raises_rate_limit_timeout_error(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="blkto:",
        )
        key = _fresh_key("blk3")

        assert limiter.allow(key) is True

        with pytest.raises(RateLimitTimeoutError):
            block_until_allowed(
                limiter,
                key,
                cost=1,
                timeout=0.05,
            )

    def test_invalid_cost_raises_valueerror(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=5,
            refill_rate=1.0,
            key_prefix="blkinv:",
        )
        key = _fresh_key("blkinv")

        with pytest.raises(ValueError):
            block_until_allowed(
                limiter,
                key,
                cost=-1,
            )

    def test_unsatisfiable_request_error_propagates(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=5,
            refill_rate=0.0,
            key_prefix="blkuns:",
        )
        key = _fresh_key("blkuns")

        with pytest.raises(UnsatisfiableRequestError):
            block_until_allowed(
                limiter,
                key,
                cost=6,
            )

    def test_custom_clock_and_sleep_are_respected(
        self, redis_client: redis_sync.Redis
    ) -> None:
        """Caller-supplied clock/sleep are used for timeout, not the
        limiter's internal clock.
        """
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="blkcus:",
        )
        key = _fresh_key("blk4")

        limiter.allow(key)

        clock_time = [0.0]

        def fake_clock() -> float:
            clock_time[0] += 0.01
            return clock_time[0]

        # With period=60s and no Redis time advancement, allow() keeps
        # returning False. The loop checks fake_clock (which advances
        # 0.01 each iteration, with a no-op sleep) until timeout=0.1.
        with pytest.raises(RateLimitTimeoutError):
            block_until_allowed(
                limiter,
                key,
                cost=1,
                timeout=0.1,
                clock=fake_clock,
                sleep=lambda _: None,
            )

        assert clock_time[0] > 0.0


# ------------------------------------------------------------------
# wait() context manager
# ------------------------------------------------------------------


class TestWaitContextManager:
    def test_context_manager_admits_and_runs_block(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=5,
            period=60.0,
            key_prefix="wctx:",
        )
        key = _fresh_key("wctx")

        ran = False

        with wait(limiter, key, timeout=1.0):
            ran = True

        assert ran is True
        assert limiter.remaining(key) == 4

    def test_timeout_in_context_manager(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="wctxto:",
        )
        key = _fresh_key("wctx2")

        limiter.allow(key)

        with pytest.raises(RateLimitTimeoutError):
            with wait(limiter, key, timeout=0.05):
                pass


# ------------------------------------------------------------------
# KeyedLimiter
# ------------------------------------------------------------------


class TestKeyedLimiter:
    def test_global_key_shares_quota(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=2,
            period=60.0,
            key_prefix="klim:",
        )
        keyed = KeyedLimiter(limiter)

        keyed.block_until_allowed()
        keyed.block_until_allowed()

        assert limiter.remaining(keyed._GLOBAL_KEY) == 0

    def test_per_key_derivation(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="kpkey:",
        )
        keyed = KeyedLimiter(
            limiter,
            key_func=lambda user_id: f"user:{user_id}",
        )

        keyed.block_until_allowed("alice")

        assert keyed.key_for_call("alice") == "user:alice"

        # alice's quota is consumed; bob has a different key.
        assert limiter.allow("user:bob") is True

    def test_keyed_wait_context_manager(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=5,
            period=60.0,
            key_prefix="kwait:",
        )
        keyed = KeyedLimiter(
            limiter,
            key_func=lambda uid: uid,
        )

        ran = False

        with keyed.wait("u1", timeout=1.0):
            ran = True

        assert ran is True
        assert limiter.remaining("u1") == 4


# ------------------------------------------------------------------
# rate_limit() decorator
# ------------------------------------------------------------------


class TestRateLimitDecorator:
    def test_decorator_blocks_then_calls_function(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=5,
            refill_rate=100.0,
            key_prefix="rldec:",
        )
        call_count = [0]

        @rate_limit(limiter, timeout=2.0)
        def do_work() -> str:
            call_count[0] += 1
            return "ok"

        result = do_work()

        assert result == "ok"
        assert call_count[0] == 1

    def test_decorator_per_key(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="rlpk:",
        )
        results = []

        @rate_limit(
            limiter,
            key_func=lambda name, *a, **k: name,
            timeout=0.05,
        )
        def greet(name: str) -> str:
            results.append(name)
            return f"hello {name}"

        assert greet("alice") == "hello alice"
        assert greet("bob") == "hello bob"

        assert len(results) == 2


# ------------------------------------------------------------------
# Invalid / zero cost fast paths
# ------------------------------------------------------------------


class TestFastPaths:
    def test_cost_zero_does_not_block(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisLuaFixedWindow(
            redis_client,
            limit=1,
            period=60.0,
            key_prefix="fast0:",
        )
        key = _fresh_key("fast0")

        limiter.allow(key)

        # cost=0 returns True immediately without touching storage.
        block_until_allowed(
            limiter,
            key,
            cost=0,
        )

    def test_negative_cost_does_not_block(
        self, redis_client: redis_sync.Redis
    ) -> None:
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=5,
            refill_rate=1.0,
            key_prefix="fastneg:",
        )
        key = _fresh_key("fastneg")

        # cost=-1 is invalid; allow() returns False,
        # allow_wait() raises ValueError.
        with pytest.raises(ValueError):
            block_until_allowed(
                limiter,
                key,
                cost=-1,
            )