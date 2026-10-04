# tests/test_async_sliding_window_log.py
"""Tests for AsyncSlidingWindowLog. FakeClock defined
locally -- see test_async_fixed_window.py for why.

Fixed from the first pass: import path corrected to
`rlimit.algorithms.async_sliding_window_log`. Boundary-race and
property tests now run against YieldingAsyncStorage
(tests/async_test_helpers.py) instead of plain AsyncInMemoryStorage, so
they force real event-loop interleaving between coroutines instead of
passing by construction -- see async_test_helpers.py's docstring for
the full rationale.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlimit.algorithms.async_sliding_window_log import AsyncSlidingWindowLog
from rlimit.base import UnsatisfiableRequestError
from tests.async_test_helpers import YieldingAsyncStorage


class FakeClock:
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
    limiter = AsyncSlidingWindowLog(limit=3, period=60.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_entries_expire_out_of_window() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=1, period=10.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False
    clock.advance(10.01)
    assert await limiter.allow("k") is True


@pytest.mark.asyncio
async def test_sliding_no_boundary_burst() -> None:
    """Unlike fixed window, a trailing period-second span never admits
    more than `limit`, even straddling what would be a fixed-window
    boundary."""
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=2, period=10.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    clock.advance(9.0)
    # Still within the trailing 10s window of the first two requests.
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_allow_wait_zero_when_allowed() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=3, period=60.0, clock=clock)
    assert await limiter.allow_wait("k") == 0.0


@pytest.mark.asyncio
async def test_allow_wait_returns_time_until_oldest_expires() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")
    wait = await limiter.allow_wait("k")
    assert wait == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_limit() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=5, period=60.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


@pytest.mark.asyncio
async def test_remaining_reflects_active_entries() -> None:
    clock = FakeClock()
    limiter = AsyncSlidingWindowLog(limit=5, period=60.0, clock=clock)
    await limiter.allow("k", cost=2)
    assert await limiter.remaining("k") == 3


def test_nan_limit_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncSlidingWindowLog(limit=float("nan"), period=10.0)  # type: ignore[arg-type]


# --- Boundary race: frozen clock + forced real interleaving -----------


@pytest.mark.asyncio
async def test_boundary_race_at_expiry_instant() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowLog(
        limit=5, period=10.0, storage=storage, clock=clock
    )
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(10.01)  # all five entries expire, frozen for the burst

    results = await asyncio.gather(*[limiter.allow("k") for _ in range(8)])
    assert sum(results) == 5


# --- Concurrent property test -------------------------------------------


async def _run_concurrent_allow(limit: int, num_requests: int) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncSlidingWindowLog(
        limit=limit, period=60.0, storage=storage, clock=clock
    )
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    limit=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50)
def test_property_allowed_never_exceeds_limit(limit: int, num_requests: int) -> None:
    results = asyncio.run(_run_concurrent_allow(limit, num_requests))
    assert sum(results) <= limit
