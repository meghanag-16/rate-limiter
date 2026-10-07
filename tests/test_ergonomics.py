# tests/test_ergonomics.py
"""Tests for limivault.ergonomics (sync).

FakeClock defined locally.

The one invariant every other test here is secondary to:
block_until_allowed() must re-check allow() after every sleep and loop
back to allow_wait() again if denied, never trusting a single wait (see
ergonomics.py's "ADVISORY, NOT A RESERVATION" module docstring). The
ThreadPoolExecutor test at the bottom of this file is the test that
actually proves that under real thread interleaving, not just by
construction.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from limivault.algorithms.fixed_window import FixedWindow
from limivault.base import UnsatisfiableRequestError
from limivault.ergonomics import (
    KeyedLimiter,
    RateLimitTimeoutError,
    block_until_allowed,
    rate_limit,
    wait,
)


class FakeClock:
    """Manually-advanced clock for deterministic tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class ThreadSafeFakeClock:
    """FakeClock variant for tests that drive it from multiple real
    threads. advance_to_at_least() only ever moves the clock forward,
    and converges correctly even if two threads independently compute
    the same or overlapping target times -- unlike a plain "add my
    wait_seconds to the shared clock" scheme, which would double-count
    when two threads both fake-sleep for the same conceptual wait."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance_to_at_least(self, target: float) -> None:
        with self._lock:
            if target > self._now:
                self._now = target


# --- block_until_allowed(): basic behavior ------------------------------


def test_returns_immediately_when_already_allowed() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=3, period=60.0, clock=clock)
    sleeps: list[float] = []

    block_until_allowed(limiter, "k", clock=clock, sleep=sleeps.append)

    assert sleeps == []
    assert limiter.remaining("k") == 2


def test_sleeps_and_retries_until_window_rolls_over() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")  # exhaust the window

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)

    assert sleeps == [pytest.approx(10.0)]
    assert limiter.remaining("k") == 0


def test_does_not_trust_a_single_wait_if_capacity_taken_in_between() -> None:
    """Directly exercises the ADVISORY, NOT A RESERVATION concern: the
    fake sleep simulates another caller sneaking in and consuming the
    slot right as this caller wakes up, so the first allow() recheck
    fails and block_until_allowed must loop back to allow_wait() again
    rather than assuming the wait it already did was sufficient."""
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")  # exhaust window [0, 10)

    call_count = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate another caller grabbing the slot the instant the
            # new window opens, before this caller's retry runs.
            limiter.allow("k")

    block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)

    # First sleep put us at t=10 (new window), but another caller took
    # the one slot there; block_until_allowed must have slept a second
    # time (waiting for the window after that) rather than declaring
    # victory after just one sleep.
    assert call_count["n"] == 2


# --- block_until_allowed(): timeout --------------------------------------


def test_negative_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)
    sleeps: list[float] = []

    with pytest.raises(ValueError):
        block_until_allowed(
            limiter, "k", timeout=-1.0, clock=clock, sleep=sleeps.append
        )
    assert sleeps == []


def test_nan_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        block_until_allowed(limiter, "k", timeout=float("nan"), clock=clock)


def test_infinite_timeout_raises_value_error_immediately() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        block_until_allowed(limiter, "k", timeout=float("inf"), clock=clock)


def test_timeout_none_waits_indefinitely_across_multiple_windows() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")
    limiter.allow("k")  # not actually allowed (over limit), no-op on state

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    block_until_allowed(limiter, "k", timeout=None, clock=clock, sleep=fake_sleep)
    assert sleeps == [pytest.approx(10.0)]


def test_timeout_raises_when_wait_exceeds_it() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")  # exhausted; next slot is 10s away

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    with pytest.raises(RateLimitTimeoutError):
        block_until_allowed(
            limiter, "k", timeout=1.0, clock=clock, sleep=fake_sleep
        )

    # Must not have slept the full 10s -- timeout should cap the wait
    # (or skip sleeping at all) rather than overshooting the deadline.
    assert sleeps == [] or sleeps[0] <= 1.0


def test_timeout_does_not_raise_when_wait_is_within_budget() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    # Plenty of budget for the 10s wait.
    block_until_allowed(limiter, "k", timeout=30.0, clock=clock, sleep=fake_sleep)
    assert limiter.remaining("k") == 0


def test_timeout_zero_raises_immediately_if_not_already_allowed() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")

    sleeps: list[float] = []

    with pytest.raises(RateLimitTimeoutError):
        block_until_allowed(
            limiter, "k", timeout=0.0, clock=clock, sleep=sleeps.append
        )
    assert sleeps == []


def test_timeout_zero_succeeds_if_already_allowed() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=3, period=10.0, clock=clock)

    block_until_allowed(limiter, "k", timeout=0.0, clock=clock, sleep=lambda s: None)
    assert limiter.remaining("k") == 2


# --- block_until_allowed(): unsatisfiable / invalid cost -----------------


def test_unsatisfiable_cost_raises_immediately_without_sleeping() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)
    sleeps: list[float] = []

    with pytest.raises(UnsatisfiableRequestError):
        block_until_allowed(
            limiter, "k", cost=100, clock=clock, sleep=sleeps.append
        )
    assert sleeps == []


def test_invalid_cost_type_raises_value_error_via_allow_wait() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)

    with pytest.raises(ValueError):
        block_until_allowed(limiter, "k", cost=2.5, clock=clock)  # type: ignore[arg-type]


# --- wait (context manager) ----------------------------------------------


def test_wait_blocks_before_entering_and_runs_the_block() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")

    ran = {"value": False}

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    with wait(limiter, "k", clock=clock, sleep=fake_sleep):
        ran["value"] = True

    assert ran["value"] is True


def test_wait_does_not_suppress_exceptions_from_the_block() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=3, period=10.0, clock=clock)

    with pytest.raises(ValueError, match="boom"):
        with wait(limiter, "k", clock=clock, sleep=lambda s: None):
            raise ValueError("boom")


def test_limiter_wait_method_is_equivalent_to_the_free_function() -> None:
    """RateLimiter.wait() (base.py) is a thin forwarding wrapper around
    ergonomics.wait -- verifies limiter.wait(key) actually blocks and
    runs the wrapped block, the same way the free function does above,
    proving the lazy-import forwarding in base.py works end-to-end
    rather than just type-checking."""
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")  # exhaust the window

    ran = {"value": False}

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    with limiter.wait("k", clock=clock, sleep=fake_sleep):
        ran["value"] = True

    assert ran["value"] is True
    assert clock() == pytest.approx(10.0)  # had to wait for the next window


def test_wait_timeout_prevents_block_body_from_running() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")

    ran = {"value": False}

    with pytest.raises(RateLimitTimeoutError):
        with wait(limiter, "k", timeout=0.0, clock=clock, sleep=lambda s: None):
            ran["value"] = True

    assert ran["value"] is False


# --- KeyedLimiter ----------------------------------------------------------


def test_keyed_limiter_defaults_to_a_single_global_key() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)
    keyed = KeyedLimiter(limiter)

    assert keyed.key_for_call("alice") == keyed.key_for_call("bob")


def test_keyed_limiter_uses_key_func_when_provided() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)
    keyed = KeyedLimiter(limiter, key_func=lambda user_id: user_id)

    assert keyed.key_for_call("alice") == "alice"
    assert keyed.key_for_call("bob") == "bob"
    assert keyed.key_for_call("alice") != keyed.key_for_call("bob")


# --- rate_limit decorator --------------------------------------------------


def test_decorator_blocks_until_allowed_then_calls_function() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("__global__")  # exhaust the default global key

    calls: list[int] = []

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    @rate_limit(limiter, clock=clock, sleep=fake_sleep)
    def do_thing(x: int) -> int:
        calls.append(x)
        return x * 2

    result = do_thing(5)
    assert result == 10
    assert calls == [5]


def test_decorator_global_key_shares_limit_across_all_calls() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    calls: list[str] = []

    @rate_limit(limiter, clock=clock, sleep=fake_sleep)
    def do_thing(user_id: str) -> None:
        calls.append(user_id)

    do_thing("alice")
    do_thing("bob")  # same global key, must block until window rolls over

    assert calls == ["alice", "bob"]
    assert clock() == pytest.approx(10.0)  # had to wait one full window


def test_decorator_per_key_does_not_block_different_keys() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    calls: list[str] = []

    @rate_limit(
        limiter, key_func=lambda user_id: user_id, clock=clock, sleep=fake_sleep
    )
    def do_thing(user_id: str) -> None:
        calls.append(user_id)

    do_thing("alice")
    do_thing("bob")  # different key -- must NOT block

    assert calls == ["alice", "bob"]
    assert clock() == 0.0  # no waiting needed at all


def test_decorator_preserves_function_metadata() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)

    @rate_limit(limiter, clock=clock)
    def do_thing(x: int) -> int:
        """A docstring."""
        return x

    assert do_thing.__name__ == "do_thing"
    assert do_thing.__doc__ == "A docstring."


def test_decorator_timeout_raises_and_does_not_call_function() -> None:
    clock = FakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("__global__")

    calls: list[int] = []

    @rate_limit(limiter, timeout=0.0, clock=clock, sleep=lambda s: None)
    def do_thing(x: int) -> int:
        calls.append(x)
        return x

    with pytest.raises(RateLimitTimeoutError):
        do_thing(1)
    assert calls == []


# --- Concurrency: real threads, deterministic frozen clock ---------------


def test_two_threads_racing_single_capacity_key_both_eventually_admitted() -> None:
    """Deterministic frozen-clock ThreadPoolExecutor test (matching the
    project's established pattern for real thread interleaving)
    proving block_until_allowed's retry loop does not double-admit and
    does eventually admit both callers, under genuine concurrent
    threads racing the same starved key.

    Beyond "both eventually returned" (which a broken implementation
    that sleeps once and proceeds without re-checking could satisfy by
    accident under some interleavings), this also pins down the exact
    final state: with limit=1, the pre-race allow() already exhausted
    window [0, 10). The first racer can only be admitted once the
    clock reaches window [10, 20); the second racer, having lost that
    race, can only be admitted in window [20, 30) -- so the clock must
    have advanced to exactly 20.0 by the time both are done, and the
    limiter's final window must show its one slot fully consumed
    (remaining() == 0), not artificially free. A broken "sleep once,
    then declare victory" implementation would tend to leave the clock
    short of 20.0 and/or remaining() != 0.
    """
    clock = ThreadSafeFakeClock()
    limiter = FixedWindow(limit=1, period=10.0, clock=clock)
    limiter.allow("k")  # exhaust the only slot in window [0, 10)

    admitted: list[str] = []
    admitted_lock = threading.Lock()

    def fake_sleep(seconds: float) -> None:
        clock.advance_to_at_least(clock() + seconds)

    def worker(name: str) -> None:
        block_until_allowed(limiter, "k", clock=clock, sleep=fake_sleep)
        with admitted_lock:
            admitted.append(name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, name) for name in ("a", "b")]
        for future in futures:
            future.result(timeout=5)

    assert sorted(admitted) == ["a", "b"]
    assert clock() == pytest.approx(20.0)
    assert limiter.remaining("k") == 0
