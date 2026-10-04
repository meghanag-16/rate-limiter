# tests/test_async_fixed_window.py
"""Tests for AsyncFixedWindow.

FakeClock is defined locally in this file rather than imported from a
shared conftest, matching the project's established convention (see
summary: importing a shared FakeClock from conftest caused
sys.path collection errors on Windows).

Fixed from the first pass: import path was
`rlimit.async_fixed_window`, which does not exist -- the implementation
lives in `rlimit.algorithms.async_fixed_window`, matching the sync
package layout. Also: the original boundary-race and property tests
ran against plain AsyncInMemoryStorage, whose get()/set() contain no
real `await`, so asyncio.gather() rarely gave two coroutines a genuine
chance to interleave -- those tests passed even though they weren't
exercising the lock/clock-ordering the way they looked like they were.
They now run against YieldingAsyncStorage (tests/async_test_helpers.py),
which forces a real event-loop yield inside get()/set().
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlimit.algorithms.async_fixed_window import AsyncFixedWindow
from rlimit.base import UnsatisfiableRequestError
from rlimit.storage import AsyncInMemoryStorage
from tests.async_test_helpers import YieldingAsyncStorage


class FakeClock:
    """Manually-advanced clock for deterministic tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --- Basic unit tests -------------------------------------------------


@pytest.mark.asyncio
async def test_allow_within_limit() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=3, period=60.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True


@pytest.mark.asyncio
async def test_allow_denies_over_limit() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=2, period=60.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_window_resets_after_period() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False
    clock.advance(10.0)
    assert await limiter.allow("k") is True


@pytest.mark.asyncio
async def test_allow_wait_returns_zero_when_allowed() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=2, period=60.0, clock=clock)
    assert await limiter.allow_wait("k") == 0.0


@pytest.mark.asyncio
async def test_allow_wait_returns_time_to_next_window() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")
    wait = await limiter.allow_wait("k")
    assert wait == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


@pytest.mark.asyncio
async def test_remaining_reflects_consumed_quota() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    await limiter.allow("k", cost=2)
    assert await limiter.remaining("k") == 3


@pytest.mark.asyncio
async def test_negative_cost_denied_not_raised() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    assert await limiter.allow("k", cost=-1) is False


def test_nan_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncFixedWindow(limit=float("nan"), period=10.0)  # type: ignore[arg-type]


def test_infinite_period_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncFixedWindow(limit=5, period=float("inf"))


# --- Boundary race: frozen clock + forced real interleaving -----------


@pytest.mark.asyncio
async def test_boundary_race_exact_limit_admitted() -> None:
    """All coroutines in a burst share one frozen `now` and race on a
    storage backend that forces real event-loop yields inside get()/
    set(), so this genuinely exercises the per-key lock rather than
    just checking a count that never had a chance to be wrong."""
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncFixedWindow(limit=10, period=60.0, storage=storage, clock=clock)
    key = "boundary"

    results = await asyncio.gather(*[limiter.allow(key) for _ in range(25)])
    assert sum(results) == 10


@pytest.mark.asyncio
async def test_boundary_race_across_window_rollover() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncFixedWindow(limit=10, period=60.0, storage=storage, clock=clock)
    key = "boundary"

    first_burst = await asyncio.gather(*[limiter.allow(key) for _ in range(10)])
    assert sum(first_burst) == 10

    clock.advance(60.0)  # roll into the next window, deterministically

    second_burst = await asyncio.gather(*[limiter.allow(key) for _ in range(10)])
    assert sum(second_burst) == 10


# --- Concurrent property test: interleaving randomized by Hypothesis --


async def _run_concurrent_allow(
    limit: int, period: float, num_requests: int
) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncFixedWindow(limit=limit, period=period, storage=storage, clock=clock)
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    limit=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50)
def test_property_allowed_never_exceeds_limit(limit: int, num_requests: int) -> None:
    results = asyncio.run(_run_concurrent_allow(limit, 60.0, num_requests))
    assert sum(results) <= limit


# --- Deterministic clock-after-lock ordering test ----------------------


class TestClockReadAfterLock:
    """Deterministic regression test for clock-inside-the-lock ordering
    -- the async equivalent of the concern test_fixed_window.py's
    TestClockReadInsideLock guards against in the sync suite.

    Unlike a thread, a coroutine can only be suspended at an `await`
    point, so forcing a stale-vs-fresh interleaving the way the sync
    test does (a pausing clock + a real OS thread racing ahead) doesn't
    translate directly. Instead this takes a simpler, equally direct
    approach appropriate to asyncio: it inspects whether the per-key
    asyncio.Lock is actually held (`.locked()`) at the exact moment
    `clock()` is called. If `now = self._clock()` is read inside
    `async with self._storage.lock(key):` (correct), the lock is always
    held when clock() runs. If it were ever moved above that `async
    with` , the lock would
    not yet be held -- or would not yet even exist for the very first
    call -- at the moment clock() runs, and this test would fail.
    """

    @pytest.mark.asyncio
    async def test_clock_is_called_while_lock_is_held(self) -> None:
        storage = AsyncInMemoryStorage()
        key = "k"
        lock_states: list[bool] = []

        def recording_clock() -> float:
            lock_obj = storage._locks.get(key)
            lock_states.append(lock_obj.locked() if lock_obj is not None else False)
            return 0.0

        limiter = AsyncFixedWindow(
            limit=5, period=60.0, storage=storage, clock=recording_clock
        )
        await limiter.allow(key)
        await limiter.allow(key)

        assert len(lock_states) == 2
        assert all(lock_states), (
            "clock() was called while the per-key lock was NOT held -- "
            "the clock read has moved outside `async with "
            "self._storage.lock(key):`"
            "TOCTOU race in async form."
        )
