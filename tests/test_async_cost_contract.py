# tests/test_async_cost_contract.py
"""Shared cost-handling contract tests (async), covering all five
async algorithms uniformly. Mirrors test_cost_contract.py's sync
version exactly -- see that file's docstring for the full rationale
for centralizing these instead of duplicating into each
test_async_<algorithm>.py file.
"""

from __future__ import annotations

import pytest

from limivault.algorithms.async_fixed_window import AsyncFixedWindow
from limivault.algorithms.async_leaky_bucket import (
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from limivault.algorithms.async_sliding_window_counter import AsyncSlidingWindowCounter
from limivault.algorithms.async_sliding_window_log import AsyncSlidingWindowLog
from limivault.algorithms.async_token_bucket import AsyncTokenBucket
from limivault.base import AsyncRateLimiter
from limivault.storage import AsyncInMemoryStorage


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_limiters(
    clock: FakeClock,
) -> list[tuple[str, AsyncRateLimiter, AsyncInMemoryStorage]]:
    """One instance of each async algorithm, each with its own fresh
    AsyncInMemoryStorage so state-touching assertions are reliable and
    isolated per algorithm."""
    limiters: list[tuple[str, AsyncRateLimiter, AsyncInMemoryStorage]] = []

    s1 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncFixedWindow",
            AsyncFixedWindow(limit=5, period=60.0, storage=s1, clock=clock),
            s1,
        )
    )

    s2 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncTokenBucket",
            AsyncTokenBucket(capacity=5, refill_rate=1.0, storage=s2, clock=clock),
            s2,
        )
    )

    s3 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncSlidingWindowLog",
            AsyncSlidingWindowLog(limit=5, period=60.0, storage=s3, clock=clock),
            s3,
        )
    )

    s4 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncSlidingWindowCounter",
            AsyncSlidingWindowCounter(limit=5, period=60.0, storage=s4, clock=clock),
            s4,
        )
    )

    s5 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncLeakyBucketMeter",
            AsyncLeakyBucketMeter(capacity=5, leak_rate=1.0, storage=s5, clock=clock),
            s5,
        )
    )

    s6 = AsyncInMemoryStorage()
    limiters.append(
        (
            "AsyncLeakyBucketQueue",
            AsyncLeakyBucketQueue(capacity=5, leak_rate=1.0, storage=s6, clock=clock),
            s6,
        )
    )

    return limiters


# --- Zero-cost is a true no-op, for every algorithm ---------------------


@pytest.mark.asyncio
async def test_zero_cost_returns_true_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, _ in _make_limiters(clock):
        assert await limiter.allow("k", cost=0) is True, name


@pytest.mark.asyncio
async def test_zero_cost_does_not_create_state_for_new_key() -> None:
    """The core bug this fixes: a zero-cost call on a never-seen key
    must not write any state at all, for any algorithm."""
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert await limiter.allow("brand-new-key", cost=0) is True, name
        assert await storage.get("brand-new-key") is None, name


@pytest.mark.asyncio
async def test_zero_cost_leaves_existing_state_unchanged() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        await limiter.allow("k", cost=1)  # establish some real state
        state_before = await storage.get("k")
        assert await limiter.allow("k", cost=0) is True, name
        assert await storage.get("k") == state_before, name


@pytest.mark.asyncio
async def test_zero_cost_allow_wait_is_zero_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, _ in _make_limiters(clock):
        assert await limiter.allow_wait("k", cost=0) == 0.0, name


@pytest.mark.asyncio
async def test_sliding_window_log_zero_cost_does_not_grow_the_log() -> None:
    """Direct regression test for the specific bug flagged in review,
    async side: repeated zero-cost calls used to append a
    (timestamp, 0) entry every time, growing the log indefinitely
    despite consuming no quota."""
    clock = FakeClock()
    storage = AsyncInMemoryStorage()
    limiter = AsyncSlidingWindowLog(
        limit=5, period=60.0, storage=storage, clock=clock
    )

    for _ in range(50):
        assert await limiter.allow("k", cost=0) is True

    assert await storage.get("k") is None  # never touched, not even once


# --- Cost type validation, for every algorithm ---------------------------


@pytest.mark.asyncio
async def test_float_cost_rejected_by_allow_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert await limiter.allow("k", cost=2.0) is False, name  # type: ignore[arg-type]
        assert await storage.get("k") is None, name


@pytest.mark.asyncio
async def test_bool_cost_rejected_by_allow_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        # No type: ignore needed: bool is a subtype of int, so
        # cost=True type-checks against `cost: int` even though the
        # runtime check correctly rejects it.
        assert await limiter.allow("k", cost=True) is False, name
        assert await storage.get("k") is None, name


@pytest.mark.asyncio
async def test_nan_cost_still_rejected_by_allow_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert await limiter.allow("k", cost=float("nan")) is False, name  # type: ignore[arg-type]
        assert await storage.get("k") is None, name


@pytest.mark.asyncio
async def test_float_cost_raises_value_error_on_allow_wait_for_every_algorithm() -> (
    None
):
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            await limiter.allow_wait("k", cost=2.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bool_cost_raises_value_error_on_allow_wait_for_every_algorithm() -> None:
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            # bool is a subtype of int -- no type: ignore needed, see
            # the equivalent allow() test above.
            await limiter.allow_wait("k", cost=True)


# --- Negative cost still denied/raises, for every algorithm --------------


@pytest.mark.asyncio
async def test_negative_cost_still_denied_by_allow_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert await limiter.allow("k", cost=-1) is False, name
        assert await storage.get("k") is None, name


@pytest.mark.asyncio
async def test_negative_cost_still_raises_on_allow_wait_for_every_algorithm() -> None:
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            await limiter.allow_wait("k", cost=-1)
