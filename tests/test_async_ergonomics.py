# tests/test_async_ergonomics.py
"""Tests for limivault.async_ergonomics (async).

FakeClock defined locally, per project convention -- see
test_async_fixed_window.py for why. Mirrors test_ergonomics.py's sync
suite test-for-test where the semantics are identical; the concurrency
test at the bottom uses YieldingAsyncStorage (tests/async_test_helpers.py)
instead of ThreadPoolExecutor, matching how the rest of the
async suite forces genuine event-loop interleaving rather than relying
on asyncio.gather over a storage backend with no real await in it.
"""

from __future__ import annotations

import asyncio

import pytest

from limivault.algorithms.async_fixed_window import AsyncFixedWindow
from limivault.async_ergonomics import (
    AsyncKeyedLimiter,
    RateLimitTimeoutError,
    async_block_until_allowed,
    async_rate_limit,
    async_wait,
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


# --- async_block_until_allowed(): basic behavior --------------------------


@pytest.mark.asyncio
async def test_returns_immediately_when_already_allowed() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=3, period=60.0, clock=clock)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    await async_block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)

    assert sleeps == []
    assert await limiter.remaining("k") == 2


@pytest.mark.asyncio
async def test_sleeps_and_retries_until_window_rolls_over() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    await async_block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)

    assert sleeps == [pytest.approx(10.0)]
    assert await limiter.remaining("k") == 0


@pytest.mark.asyncio
async def test_does_not_trust_a_single_wait_if_capacity_taken_in_between() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    call_count = {"n": 0}

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)
        call_count["n"] += 1
        if call_count["n"] == 1:
            await limiter.allow("k")  # another caller sneaks in

    await async_block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)

    assert call_count["n"] == 2


# --- async_block_until_allowed(): timeout ----------------------------------


@pytest.mark.asyncio
async def test_negative_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(ValueError):
        await async_block_until_allowed(
            limiter, "k", timeout=-1.0, clock=clock, sleep=fake_sleep
        )
    assert sleeps == []


@pytest.mark.asyncio
async def test_nan_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        await async_block_until_allowed(limiter, "k", timeout=float("nan"), clock=clock)


@pytest.mark.asyncio
async def test_infinite_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        await async_block_until_allowed(limiter, "k", timeout=float("inf"), clock=clock)


@pytest.mark.asyncio
async def test_timeout_raises_when_wait_exceeds_it() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    with pytest.raises(RateLimitTimeoutError):
        await async_block_until_allowed(
            limiter, "k", timeout=1.0, clock=clock, sleep=fake_sleep
        )

    assert sleeps == [] or sleeps[0] <= 1.0


@pytest.mark.asyncio
async def test_timeout_does_not_raise_when_wait_is_within_budget() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    await async_block_until_allowed(
        limiter, "k", timeout=30.0, clock=clock, sleep=fake_sleep
    )
    assert await limiter.remaining("k") == 0


@pytest.mark.asyncio
async def test_timeout_zero_raises_immediately_if_not_already_allowed() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(RateLimitTimeoutError):
        await async_block_until_allowed(
            limiter, "k", timeout=0.0, clock=clock, sleep=fake_sleep
        )
    assert sleeps == []


@pytest.mark.asyncio
async def test_timeout_zero_succeeds_if_already_allowed() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=3, period=10.0, clock=clock)

    async def fake_sleep(seconds: float) -> None:
        pass

    await async_block_until_allowed(
        limiter, "k", timeout=0.0, clock=clock, sleep=fake_sleep
    )
    assert await limiter.remaining("k") == 2


# --- async_block_until_allowed(): unsatisfiable / invalid cost ------------


@pytest.mark.asyncio
async def test_unsatisfiable_cost_raises_immediately_without_sleeping() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(UnsatisfiableRequestError):
        await async_block_until_allowed(
            limiter, "k", cost=100, clock=clock, sleep=fake_sleep
        )
    assert sleeps == []


@pytest.mark.asyncio
async def test_invalid_cost_type_raises_value_error_via_allow_wait() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        await async_block_until_allowed(
            limiter, "k", cost=2.5, clock=clock  # type: ignore[arg-type]
        )


# --- async_wait (context manager) ------------------------------------------


@pytest.mark.asyncio
async def test_async_wait_blocks_before_entering_and_runs_the_block() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    ran = {"value": False}

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    async with async_wait(limiter, "k", clock=clock, sleep=fake_sleep):
        ran["value"] = True

    assert ran["value"] is True


@pytest.mark.asyncio
async def test_async_wait_does_not_suppress_exceptions_from_the_block() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=3, period=10.0, clock=clock)

    async def fake_sleep(seconds: float) -> None:
        pass

    with pytest.raises(ValueError, match="boom"):
        async with async_wait(limiter, "k", clock=clock, sleep=fake_sleep):
            raise ValueError("boom")


@pytest.mark.asyncio
async def test_limiter_wait_method_is_equivalent_to_the_free_function() -> None:
    """AsyncRateLimiter.wait() (base.py) is a thin forwarding wrapper
    around async_ergonomics.async_wait -- verifies limiter.wait(key)
    actually blocks and runs the wrapped block end-to-end, the same
    way the free function does above."""
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")  # exhaust the window

    ran = {"value": False}

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    async with limiter.wait("k", clock=clock, sleep=fake_sleep):
        ran["value"] = True

    assert ran["value"] is True
    assert clock() == pytest.approx(10.0)  # had to wait for the next window


@pytest.mark.asyncio
async def test_async_wait_timeout_prevents_block_body_from_running() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("k")

    ran = {"value": False}

    async def fake_sleep(seconds: float) -> None:
        pass

    with pytest.raises(RateLimitTimeoutError):
        async with async_wait(limiter, "k", timeout=0.0, clock=clock, sleep=fake_sleep):
            ran["value"] = True

    assert ran["value"] is False


# --- AsyncKeyedLimiter -------------------------------------------------------


@pytest.mark.asyncio
async def test_keyed_limiter_defaults_to_a_single_global_key() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    keyed = AsyncKeyedLimiter(limiter)

    assert keyed.key_for_call("alice") == keyed.key_for_call("bob")


@pytest.mark.asyncio
async def test_keyed_limiter_uses_key_func_when_provided() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)
    keyed = AsyncKeyedLimiter(limiter, key_func=lambda user_id: user_id)

    assert keyed.key_for_call("alice") == "alice"
    assert keyed.key_for_call("bob") == "bob"
    assert keyed.key_for_call("alice") != keyed.key_for_call("bob")


# --- async_rate_limit decorator ----------------------------------------------


@pytest.mark.asyncio
async def test_decorator_blocks_until_allowed_then_calls_function() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("__global__")

    calls: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    @async_rate_limit(limiter, clock=clock, sleep=fake_sleep)
    async def do_thing(x: int) -> int:
        calls.append(x)
        return x * 2

    result = await do_thing(5)
    assert result == 10
    assert calls == [5]


@pytest.mark.asyncio
async def test_decorator_global_key_shares_limit_across_all_calls() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    calls: list[str] = []

    @async_rate_limit(limiter, clock=clock, sleep=fake_sleep)
    async def do_thing(user_id: str) -> None:
        calls.append(user_id)

    await do_thing("alice")
    await do_thing("bob")

    assert calls == ["alice", "bob"]
    assert clock() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_decorator_per_key_does_not_block_different_keys() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    calls: list[str] = []

    @async_rate_limit(
        limiter, key_func=lambda user_id: user_id, clock=clock, sleep=fake_sleep
    )
    async def do_thing(user_id: str) -> None:
        calls.append(user_id)

    await do_thing("alice")
    await do_thing("bob")

    assert calls == ["alice", "bob"]
    assert clock() == 0.0


@pytest.mark.asyncio
async def test_decorator_preserves_function_metadata() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=5, period=60.0, clock=clock)

    @async_rate_limit(limiter, clock=clock)
    async def do_thing(x: int) -> int:
        """A docstring."""
        return x

    assert do_thing.__name__ == "do_thing"
    assert do_thing.__doc__ == "A docstring."


@pytest.mark.asyncio
async def test_decorator_timeout_raises_and_does_not_call_function() -> None:
    clock = FakeClock()
    limiter = AsyncFixedWindow(limit=1, period=10.0, clock=clock)
    await limiter.allow("__global__")

    calls: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        pass

    @async_rate_limit(limiter, timeout=0.0, clock=clock, sleep=fake_sleep)
    async def do_thing(x: int) -> int:
        calls.append(x)
        return x

    with pytest.raises(RateLimitTimeoutError):
        await do_thing(1)
    assert calls == []


# --- Concurrency: forced real event-loop interleaving ----------------------


@pytest.mark.asyncio
async def test_two_coroutines_racing_single_capacity_key_both_eventually_admitted() -> (
    None
):
    """Uses YieldingAsyncStorage so the two racing coroutines actually
    interleave at the storage level (see async_test_helpers.py's
    docstring for why plain AsyncInMemoryStorage wouldn't genuinely
    exercise this). Proves async_block_until_allowed's retry loop does
    not double-admit and both callers eventually succeed.

    Same strengthened final-state assertions as the sync version's
    test_two_threads_racing_single_capacity_key_both_eventually_admitted:
    with limit=1 and the pre-race allow() already exhausting window
    [0, 10), the loser of the race can only be admitted in window
    [20, 30) -- so the clock must land on exactly 20.0 and the
    limiter's final window must show its slot fully consumed
    (remaining() == 0). Just checking both coroutines "eventually
    returned" would pass even for a broken implementation that sleeps
    once and proceeds without re-checking, under some interleavings.
    """
    clock = FakeClock()
    storage = YieldingAsyncStorage()
    limiter = AsyncFixedWindow(limit=1, period=10.0, storage=storage, clock=clock)
    await limiter.allow("k")  # exhaust the only slot in window [0, 10)

    admitted: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        target = clock() + seconds
        if target > clock():
            clock.advance(target - clock())

    async def worker(name: str) -> None:
        await async_block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)
        admitted.append(name)

    await asyncio.gather(worker("a"), worker("b"))

    assert sorted(admitted) == ["a", "b"]
    assert clock() == pytest.approx(20.0)
    assert await limiter.remaining("k") == 0
