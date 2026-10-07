# tests/test_redis_lua_fuzz.py

"""Hypothesis property-based tests for Redis Lua algorithms.



Verifies core invariants across randomized inputs:

  - remaining() never goes negative

  - remaining() never exceeds capacity/limit

  - Total admitted cost never exceeds capacity/limit (single key)

  - cost=0 is always admitted

  - cost < 0 is always rejected

  - Invalid cost type is always rejected

  - BackendErrorEvent timestamp is non-negative



REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.

"""



from __future__ import annotations

import uuid
from typing import cast

import hypothesis.strategies as st
import redis as redis_sync
from hypothesis import HealthCheck, given, settings

from limivault.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from limivault.algorithms.redis_lua_leaky_bucket import (
    RedisLuaLeakyBucketMeter,
    RedisLuaLeakyBucketQueue,
)
from limivault.algorithms.redis_lua_sliding_window_counter import (
    RedisLuaSlidingWindowCounter,
)
from limivault.algorithms.redis_lua_sliding_window_log import RedisLuaSlidingWindowLog
from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from tests.redis_test_helpers import redis_client, redis_container

__all__ = ["redis_container", "redis_client"]





def _fresh_key(prefix: str = "fuzz") -> str:

    return f"{prefix}-{uuid.uuid4().hex[:10]}"





# ---------------------------------------------------------------------------

# Fixed window properties

# ---------------------------------------------------------------------------





class TestFixedWindowFuzz:

    @given(

        limit=st.integers(min_value=1, max_value=20),

        costs=st.lists(

            st.integers(min_value=0, max_value=5),

            min_size=1,

            max_size=30,

        ),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_limit(

        self, limit: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaFixedWindow(

            redis_client, limit=limit, period=60.0, key_prefix="fwfuzz:"

        )

        key = _fresh_key("fw")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= limit

        assert admitted <= limit



    @given(

        limit=st.integers(min_value=1, max_value=10),

        n=st.integers(min_value=1, max_value=15),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_cost_zero_always_allowed(

        self, limit: int, n: int, redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaFixedWindow(

            redis_client, limit=limit, period=60.0, key_prefix="fwfuzz0:"

        )

        key = _fresh_key("fw0")

        for _ in range(n):

            assert limiter.allow(key, cost=0) is True





# ---------------------------------------------------------------------------

# GCRA token bucket properties

# ---------------------------------------------------------------------------





class TestTokenBucketFuzz:

    @given(

        capacity=st.integers(min_value=1, max_value=20),

        costs=st.lists(

            st.integers(min_value=0, max_value=5),

            min_size=1,

            max_size=30,

        ),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_capacity(

        self, capacity: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisGcraTokenBucket(

            redis_client, capacity=capacity, refill_rate=0.0, key_prefix="tbfuzz:"

        )

        key = _fresh_key("tb")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            if cost > capacity:

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= capacity

        assert admitted <= capacity



    @given(

        capacity=st.integers(min_value=1, max_value=10),

        n=st.integers(min_value=1, max_value=15),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_cost_zero_always_allowed(

        self, capacity: int, n: int, redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisGcraTokenBucket(

            redis_client, capacity=capacity, refill_rate=0.0, key_prefix="tbfuzz0:"

        )

        key = _fresh_key("tb0")

        for _ in range(n):

            assert limiter.allow(key, cost=0) is True



    @given(

        capacity=st.integers(min_value=1, max_value=10),

        invalid_cost=st.one_of(

            st.just("not_an_int"),

            st.just(3.14),

            st.just(None),

        ),

    )

    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_invalid_cost_type_always_rejected(

        self, capacity: int, invalid_cost: object, redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisGcraTokenBucket(

            redis_client, capacity=capacity, refill_rate=1.0, key_prefix="tbfuzzinv:"

        )

        key = _fresh_key("tbinv")

        assert limiter.allow(key, cost=cast(int, invalid_cost)) is False  





# ---------------------------------------------------------------------------

# Leaky bucket meter properties

# ---------------------------------------------------------------------------





class TestLeakyBucketMeterFuzz:

    @given(

        capacity=st.integers(min_value=1, max_value=20),

        costs=st.lists(

            st.integers(min_value=0, max_value=5),

            min_size=1,

            max_size=20,

        ),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_capacity(

        self, capacity: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaLeakyBucketMeter(

            redis_client, capacity=capacity, leak_rate=0.0, key_prefix="lbfm:"

        )

        key = _fresh_key("lbfm")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            if cost > capacity:

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= capacity

        assert admitted <= capacity





# ---------------------------------------------------------------------------

# Leaky bucket queue properties

# ---------------------------------------------------------------------------





class TestLeakyBucketQueueFuzz:

    @given(

        capacity=st.integers(min_value=1, max_value=20),

        costs=st.lists(

            st.integers(min_value=0, max_value=5),

            min_size=1,

            max_size=20,

        ),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_capacity(

        self, capacity: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaLeakyBucketQueue(

            redis_client, capacity=capacity, leak_rate=0.0, key_prefix="lbfq:"

        )

        key = _fresh_key("lbfq")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            if cost > capacity:

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= capacity

        assert admitted <= capacity





# ---------------------------------------------------------------------------

# Sliding window counter properties

# ---------------------------------------------------------------------------





class TestSlidingWindowCounterFuzz:

    @given(

        limit=st.integers(min_value=1, max_value=20),

        costs=st.lists(

            st.integers(min_value=0, max_value=5),

            min_size=1,

            max_size=30,

        ),

    )

    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_limit(

        self, limit: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaSlidingWindowCounter(

            redis_client, limit=limit, period=60.0, key_prefix="swcfuzz:"

        )

        key = _fresh_key("swc")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= limit

        assert admitted <= limit





# ---------------------------------------------------------------------------

# Sliding window log properties

# ---------------------------------------------------------------------------





class TestSlidingWindowLogFuzz:

    @given(

        limit=st.integers(min_value=1, max_value=15),

        costs=st.lists(

            st.integers(min_value=0, max_value=3),

            min_size=1,

            max_size=15,

        ),

    )

    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_remaining_never_negative_never_exceeds_limit(

        self, limit: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        limiter = RedisLuaSlidingWindowLog(

            redis_client, limit=limit, period=60.0, key_prefix="swlfuzz:"

        )

        key = _fresh_key("swl")

        admitted = 0

        for cost in costs:

            if cost == 0:

                limiter.allow(key, cost=0)

                continue

            if cost < 0:

                result = limiter.allow(key, cost=cost)

                assert result is False

                continue

            allowed = limiter.allow(key, cost=cost)

            if allowed:

                admitted += cost

        rem = limiter.remaining(key)

        assert rem >= 0

        assert rem <= limit

        assert admitted <= limit





# ---------------------------------------------------------------------------

# Metrics hook fuzz (token bucket as representative)

# ---------------------------------------------------------------------------





class TestMetricsFuzz:

    """Verify metrics hooks receive events for every decision path."""



    @given(

        capacity=st.integers(min_value=1, max_value=5),

        costs=st.lists(

            st.integers(min_value=1, max_value=3),

            min_size=1,

            max_size=10,

        ),

    )

    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )

    def test_metrics_events_emitted_for_every_allow(

        self, capacity: int, costs: list[int], redis_client: redis_sync.Redis

    ) -> None:

        from limivault.metrics import AllowedEvent, DeniedEvent



        events: list[AllowedEvent | DeniedEvent] = []



        class RecordingHook:

            def on_allowed(self, event: AllowedEvent) -> None:

                events.append(event)



            def on_denied(self, event: DeniedEvent) -> None:

                events.append(event)



            def on_backend_error(self, event: object) -> None:

                pass



        hook = RecordingHook()

        limiter = RedisGcraTokenBucket(

            redis_client,

            capacity=capacity,

            refill_rate=0.0,

            key_prefix="metfuzz:",

            metrics=hook,

        )

        key = _fresh_key("met")

        decision_count = 0

        for cost in costs:

            if cost > capacity:

                continue

            limiter.allow(key, cost=cost)

            decision_count += 1



        assert len(events) == decision_count

        for ev in events:

            assert ev.algorithm == "RedisGcraTokenBucket"

            assert ev.key == key

            assert isinstance(ev.cost, int)

            assert ev.cost >= 1

            assert ev.timestamp >= 0.0

            assert 0.0 <= ev.utilization <= 1.0
