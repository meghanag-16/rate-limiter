# tests/test_async_token_bucket.py
"""Tests for AsyncTokenBucket. FakeClock defined locally --
see test_async_fixed_window.py for why.

Fixed from the first pass: import path corrected to
`rlimit.algorithms.async_token_bucket`. Boundary-race and property
tests now run against YieldingAsyncStorage (tests/async_test_helpers.py)
instead of plain AsyncInMemoryStorage, so they force real event-loop
interleaving between coroutines instead of passing by construction --
see async_test_helpers.py's docstring for the full rationale.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlimit.algorithms.async_token_bucket import AsyncTokenBucket
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
async def test_bucket_starts_full() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    assert await limiter.remaining("k") == 5


@pytest.mark.asyncio
async def test_allow_consumes_tokens() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.remaining("k") == 4


@pytest.mark.asyncio
async def test_allow_denied_when_empty() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=2, refill_rate=0.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_refill_over_time() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    for _ in range(5):
        await limiter.allow("k")
    assert await limiter.allow("k") is False
    clock.advance(3.0)
    assert await limiter.remaining("k") == 3


@pytest.mark.asyncio
async def test_refill_capped_at_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    clock.advance(1000.0)
    assert await limiter.remaining("k") == 5


@pytest.mark.asyncio
async def test_allow_wait_zero_when_tokens_available() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    assert await limiter.allow_wait("k") == 0.0


@pytest.mark.asyncio
async def test_allow_wait_returns_seconds_to_refill() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=1, refill_rate=0.5, clock=clock)
    await limiter.allow("k")
    wait = await limiter.allow_wait("k")
    assert wait == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_allow_wait_raises_when_refill_rate_zero_and_insufficient() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=1, refill_rate=0.0, clock=clock)
    await limiter.allow("k")
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k")


@pytest.mark.asyncio
async def test_allow_wait_raises_when_cost_exceeds_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


def test_nan_capacity_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncTokenBucket(capacity=float("nan"), refill_rate=1.0)  # type: ignore[arg-type]


def test_negative_refill_rate_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncTokenBucket(capacity=5, refill_rate=-1.0)


# --- Boundary race: frozen clock + forced real interleaving -----------


@pytest.mark.asyncio
async def test_boundary_race_during_active_refill() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncTokenBucket(
        capacity=5, refill_rate=1.0, storage=storage, clock=clock
    )
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(3.0)  # exactly 3 tokens refilled, frozen for the burst

    results = await asyncio.gather(*[limiter.allow("k") for _ in range(10)])
    assert sum(results) == 3


# --- Concurrent property test -------------------------------------------


async def _run_concurrent_allow(
    capacity: int, refill_rate: float, num_requests: int
) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncTokenBucket(
        capacity=capacity, refill_rate=refill_rate, storage=storage, clock=clock
    )
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    capacity=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50)
def test_property_allowed_never_exceeds_capacity(
    capacity: int, num_requests: int
) -> None:
    # refill_rate=0.0 with a frozen clock means capacity is a hard,
    # unambiguous ceiling for the whole burst.
    results = asyncio.run(_run_concurrent_allow(capacity, 0.0, num_requests))
    assert sum(results) <= capacity
