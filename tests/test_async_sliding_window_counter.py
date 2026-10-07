# tests/test_async_sliding_window_counter.py
"""Tests for AsyncSlidingWindowCounter (in-memory).

This file previously (mistakenly) held a near-copy of the Lua async
test (it imported AsyncRedisLuaSlidingWindowCounter and needed Docker),
so the in-memory AsyncSlidingWindowCounter had no dedicated
boundary-race / property / allow_wait coverage. The Lua content now
lives only in tests/test_async_redis_lua_sliding_window_counter.py.

Mirrors test_sliding_window_counter.py (sync) where semantics are
identical, and follows the conventions of the other async in-memory
test files: FakeClock defined locally, concurrency tests run against
YieldingAsyncStorage (tests/async_test_helpers.py) so coroutines
genuinely interleave instead of passing by construction.

Cost-type / negative-cost / zero-cost contract tests are NOT repeated
here -- test_async_cost_contract.py already covers them for this class.

NO DOCKER REQUIRED.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from limivault.algorithms.async_sliding_window_counter import AsyncSlidingWindowCounter
from limivault.base import UnsatisfiableRequestError
from limivault.storage import AsyncInMemoryStorage
from tests.async_test_helpers import YieldingAsyncStorage


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --- Constructor validation -------------------------------------------


def test_defaults_use_async_in_memory_storage() -> None:
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0)
    assert isinstance(limiter._storage, AsyncInMemoryStorage)


def test_accepts_injected_clock_and_storage() -> None:
    clock = FakeClock()
    storage = AsyncInMemoryStorage()
    limiter = AsyncSlidingWindowCounter(
        limit=5, period=10.0, storage=storage, clock=clock
    )
    assert limiter._storage is storage
    assert limiter._clock is clock


def test_zero_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=0, period=10.0)


def test_negative_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=-5, period=10.0)


def test_zero_period_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=5, period=0)


def test_negative_period_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=5, period=-10.0)


def test_nan_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=float("nan"), period=10.0)  # type: ignore[arg-type]


def test_infinite_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=float("inf"), period=10.0)  # type: ignore[arg-type]


def test_nan_period_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=5, period=float("nan"))


def test_infinite_period_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowCounter(limit=5, period=float("inf"))


# --- Basic correctness -------------------------------------------------


@pytest.mark.asyncio
async def test_allows_up_to_limit_within_first_window() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=3, period=10.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_full_window_then_new_window_fully_blocked_at_boundary() -> None:
    """At the instant a new window starts, prev_count = old count with
    overlap_fraction ~1.0, so a fully used previous window still fully
    blocks the new window at t=period."""
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    for _ in range(5):
        assert await limiter.allow("k") is True
    clock.advance(10.0)
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_weighted_count_decays_across_window_boundary() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=10, period=60.0, clock=clock)
    for _ in range(10):
        assert await limiter.allow("k") is True
    clock.advance(60.0)  # new window, prev_count=10, overlap=1.0
    assert await limiter.remaining("k") == 0
    clock.advance(30.0)  # halfway through new window, overlap=0.5
    assert await limiter.remaining("k") == 5


@pytest.mark.asyncio
async def test_denied_request_does_not_consume_quota() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=1, period=10.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False
    assert await limiter.remaining("k") == 0


@pytest.mark.asyncio
async def test_keys_are_independent() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=1, period=10.0, clock=clock)
    assert await limiter.allow("a") is True
    assert await limiter.allow("b") is True
    assert await limiter.allow("a") is False
    assert await limiter.allow("b") is False


@pytest.mark.asyncio
async def test_more_than_one_window_gap_resets_prev_count() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(25.0)  # more than two full periods
    assert await limiter.remaining("k") == 5


@pytest.mark.asyncio
async def test_cost_equal_to_limit_allowed_alone() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    assert await limiter.allow("k", cost=5) is True
    assert await limiter.allow("k", cost=1) is False


@pytest.mark.asyncio
async def test_cost_exceeding_limit_denied_by_allow() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    assert await limiter.allow("k", cost=6) is False


# --- allow_wait ---------------------------------------------------------


@pytest.mark.asyncio
async def test_allow_wait_zero_when_capacity_available() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    assert await limiter.allow_wait("k") == 0.0


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(limit=5, period=10.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


@pytest.mark.asyncio
async def test_wait_then_retry_deterministic_regression() -> None:
    """Same minimal scenario as the sync regression test: prev_count == 0
    for the very first window, where a naive "wait for next boundary"
    fallback would provide no actual decay."""
    clock = FakeClock()
    limiter = AsyncSlidingWindowCounter(
        limit=4, period=3.4760647670440266, clock=clock
    )
    await limiter.allow("k", cost=4)
    wait = await limiter.allow_wait("k", cost=2)
    clock.advance(wait)
    assert await limiter.allow("k", cost=2) is True


@given(
    limit=st.integers(min_value=1, max_value=20),
    period=st.floats(
        min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    warmup_costs=st.lists(
        st.integers(min_value=1, max_value=5), min_size=1, max_size=20
    ),
    cost=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_wait_then_retry_property(
    limit: int, period: float, warmup_costs: list[int], cost: int
) -> None:
    async def scenario() -> None:
        clock = FakeClock(start=0.0)
        limiter = AsyncSlidingWindowCounter(limit=limit, period=period, clock=clock)
        for c in warmup_costs:
            await limiter.allow("k", cost=min(c, limit))
            clock.advance(period / len(warmup_costs) / 3)
        capped = min(cost, limit)
        try:
            wait = await limiter.allow_wait("k", cost=capped)
        except UnsatisfiableRequestError:
            return
        if wait == 0.0:
            return
        clock.advance(wait)
        assert await limiter.allow("k", cost=capped) is True

    asyncio.run(scenario())


# --- Boundary race: frozen clock + forced real interleaving ------------


@pytest.mark.asyncio
async def test_boundary_race_burst_within_one_window_admits_exactly_limit() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowCounter(
        limit=10, period=60.0, storage=storage, clock=clock
    )
    results = await asyncio.gather(*[limiter.allow("k") for _ in range(25)])
    assert sum(results) == 10


@pytest.mark.asyncio
async def test_boundary_race_exactly_at_rollover_is_fully_blocked() -> None:
    """First burst exhausts window 1 (frozen clock). The clock is then
    stepped by exactly one period, landing on the rollover instant
    where prev_count == limit at overlap ~1.0, so the second burst must
    admit ZERO -- the most delicate moment in this algorithm, checked
    under forced coroutine interleaving."""
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowCounter(
        limit=10, period=60.0, storage=storage, clock=clock
    )

    first = await asyncio.gather(*[limiter.allow("k") for _ in range(25)])
    assert sum(first) == 10

    clock.advance(60.0)

    second = await asyncio.gather(*[limiter.allow("k") for _ in range(25)])
    assert sum(second) == 0


@pytest.mark.asyncio
async def test_boundary_race_halfway_through_new_window_admits_decayed_quota() -> None:
    """Half a period into the new window, overlap=0.5, so a fully used
    previous window (10) contributes 5 -- exactly 5 of a concurrent
    burst must be admitted."""
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowCounter(
        limit=10, period=60.0, storage=storage, clock=clock
    )
    for _ in range(10):
        await limiter.allow("k")

    clock.advance(90.0)  # 30s into the next window

    results = await asyncio.gather(*[limiter.allow("k") for _ in range(12)])
    assert sum(results) == 5


# --- Concurrent property test ------------------------------------------


async def _run_concurrent_allow(limit: int, num_requests: int) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowCounter(
        limit=limit, period=60.0, storage=storage, clock=clock
    )
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    limit=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50, deadline=None)
def test_property_allowed_never_exceeds_limit(limit: int, num_requests: int) -> None:
    results = asyncio.run(_run_concurrent_allow(limit, num_requests))
    assert sum(results) <= limit


# --- Clock read must happen while the per-key lock is held -------------


class TestClockReadAfterLock:
    """Async equivalent of clock-inside-the-lock regression
    check (see test_async_fixed_window.py's identically named class for
    the full rationale): inspects whether the per-key asyncio.Lock is
    held at the exact moment clock() is called."""

    @pytest.mark.asyncio
    async def test_clock_is_called_while_lock_is_held(self) -> None:
        storage = AsyncInMemoryStorage()
        key = "k"
        lock_states: list[bool] = []

        def recording_clock() -> float:
            lock_obj = storage._locks.get(key)
            lock_states.append(lock_obj.locked() if lock_obj is not None else False)
            return 0.0

        limiter = AsyncSlidingWindowCounter(
            limit=5, period=60.0, storage=storage, clock=recording_clock
        )
        await limiter.allow(key)
        await limiter.allow(key)

        assert len(lock_states) == 2
        assert all(lock_states), (
            "clock() was called while the per-key lock was NOT held -- "
            "the clock read moved outside `async with "
            "self._storage.lock(key):`, reintroducing TOCTOU "
            "race in async form."
        )
