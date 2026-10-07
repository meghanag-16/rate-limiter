# tests/test_async_leaky_bucket.py
"""Tests for AsyncLeakyBucketMeter and AsyncLeakyBucketQueue.
FakeClock defined locally -- see test_async_fixed_window.py for why.

Fixed from the first pass: import path corrected to
`limivault.algorithms.async_leaky_bucket`. Boundary-race and property
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

from limivault.algorithms.async_leaky_bucket import (
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from limivault.base import UnsatisfiableRequestError
from tests.async_test_helpers import YieldingAsyncStorage


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --- AsyncLeakyBucketMeter: basic unit tests ---------------------------


@pytest.mark.asyncio
async def test_meter_allow_within_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketMeter(capacity=5, leak_rate=1.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.remaining("k") == 4


@pytest.mark.asyncio
async def test_meter_allow_denied_when_full() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketMeter(capacity=2, leak_rate=0.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_meter_leaks_over_time() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketMeter(capacity=5, leak_rate=1.0, clock=clock)
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(3.0)
    assert await limiter.remaining("k") == 3


@pytest.mark.asyncio
async def test_meter_allow_wait_zero_leak_rate_raises() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketMeter(capacity=1, leak_rate=0.0, clock=clock)
    await limiter.allow("k")
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k")


@pytest.mark.asyncio
async def test_meter_allow_wait_raises_when_cost_exceeds_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketMeter(capacity=5, leak_rate=1.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


def test_meter_nan_leak_rate_rejected() -> None:
    # No type: ignore needed: `leak_rate` is typed as `float`, so
    # float("nan") is a valid argument type-wise -- rejected only at
    # runtime by _reject_non_finite(), not by mypy.
    with pytest.raises(ValueError):
        AsyncLeakyBucketMeter(capacity=5, leak_rate=float("nan"))


# --- AsyncLeakyBucketMeter: boundary race + property test --------------


@pytest.mark.asyncio
async def test_meter_boundary_race_at_leak_crossing() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncLeakyBucketMeter(
        capacity=5, leak_rate=1.0, storage=storage, clock=clock
    )
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(3.0)  # exactly 3 volume leaked, frozen for the burst

    results = await asyncio.gather(*[limiter.allow("k") for _ in range(10)])
    assert sum(results) == 3


async def _run_concurrent_meter_allow(capacity: int, num_requests: int) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncLeakyBucketMeter(
        capacity=capacity, leak_rate=0.0, storage=storage, clock=clock
    )
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    capacity=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50)
def test_meter_property_allowed_never_exceeds_capacity(
    capacity: int, num_requests: int
) -> None:
    results = asyncio.run(_run_concurrent_meter_allow(capacity, num_requests))
    assert sum(results) <= capacity


# --- AsyncLeakyBucketQueue: basic unit tests ----------------------------


@pytest.mark.asyncio
async def test_queue_allow_within_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketQueue(capacity=3, leak_rate=1.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.remaining("k") == 2


@pytest.mark.asyncio
async def test_queue_allow_denied_when_full() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketQueue(capacity=2, leak_rate=0.0, clock=clock)
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is True
    assert await limiter.allow("k") is False


@pytest.mark.asyncio
async def test_queue_drains_whole_items_only() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketQueue(capacity=5, leak_rate=1.0, clock=clock)
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(2.5)  # only 2 whole items drain, not 2.5
    assert await limiter.remaining("k") == 2


@pytest.mark.asyncio
async def test_queue_allow_wait_zero_leak_rate_raises() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketQueue(capacity=1, leak_rate=0.0, clock=clock)
    await limiter.allow("k")
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k")


@pytest.mark.asyncio
async def test_queue_allow_wait_raises_when_cost_exceeds_capacity() -> None:
    clock = FakeClock()
    limiter = AsyncLeakyBucketQueue(capacity=5, leak_rate=1.0, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        await limiter.allow_wait("k", cost=6)


# --- AsyncLeakyBucketQueue: boundary race + property test --------------


@pytest.mark.asyncio
async def test_queue_boundary_race_at_drain_instant() -> None:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncLeakyBucketQueue(
        capacity=5, leak_rate=1.0, storage=storage, clock=clock
    )
    for _ in range(5):
        await limiter.allow("k")
    clock.advance(3.0)  # exactly 3 whole items drain, frozen for the burst

    results = await asyncio.gather(*[limiter.allow("k") for _ in range(10)])
    assert sum(results) == 3


async def _run_concurrent_queue_allow(capacity: int, num_requests: int) -> list[bool]:
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncLeakyBucketQueue(
        capacity=capacity, leak_rate=0.0, storage=storage, clock=clock
    )
    return await asyncio.gather(*[limiter.allow("k") for _ in range(num_requests)])


@given(
    capacity=st.integers(min_value=1, max_value=20),
    num_requests=st.integers(min_value=0, max_value=40),
)
@settings(max_examples=50)
def test_queue_property_allowed_never_exceeds_capacity(
    capacity: int, num_requests: int
) -> None:
    results = asyncio.run(_run_concurrent_queue_allow(capacity, num_requests))
    assert sum(results) <= capacity
