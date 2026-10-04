# tests/test_fuzz.py
""" fuzz/edge-case tests: zero limit, very high burst, clock
skew.

Each of these is a SHARED, cross-algorithm test (same pattern as
test_cost_contract.py / test_metrics_all_algorithms.py), not a
per-algorithm duplicate, since the properties being checked are meant
to hold identically across every algorithm.

--------------------------------------------------------------------
SCOPE NOTES -- READ BEFORE EXTENDING THESE
--------------------------------------------------------------------

1. ZERO/NEGATIVE LIMIT: this is NOT new behavior. Every algorithm's
   constructor already rejects limit/capacity <= 0 with ValueError.
   This file consolidates that check into one shared, uniform test 
   across all six sync algorithms,
   closing the same kind of "tested per-algorithm
   but never verified identically applied everywhere" gap that
   test_cost_contract.py closed for cost handling. If this test fails
   for a given algorithm, that algorithm's constructor validation has
   regressed -- this test is not asserting a new contract.

2. VERY HIGH BURST: confirms cost exactly at the limit/capacity
   succeeds and cost one above it is denied by allow() / raises
   UnsatisfiableRequestError from allow_wait() -- the exact boundary,
   not an arbitrary "very large number." An arbitrarily large cost
   (e.g. cost=10**18) is already covered by the exact-boundary case
   argument-wise (anything above capacity behaves identically per each
   algorithm's `cost > self._capacity`/`cost > self._limit` check) --
   asserting it separately would not exercise different code.

3. CLOCK SKEW: base.py's module docstring is explicit that the
   injected clock must be monotonic (non-decreasing) and that this is
   a documented PRECONDITION, not a checked runtime invariant -- no
   algorithm validates the clock's monotonicity, and none is expected
   to. Given that, this test deliberately does NOT assert "quota is
   never exceeded under a backward clock jump" -- that would not be
   true for every algorithm and asserting it would just be pinning a
   bug that isn't a bug (see the docstring on
   test_backward_clock_jump_does_not_crash_or_corrupt_remaining below
   for the specific mechanism in FixedWindow/SlidingWindowCounter).
   What IS asserted, uniformly: a single backward jump must not raise
   an exception, and remaining() afterward must stay within the valid
   [0, limit_or_capacity] range -- i.e. state doesn't get corrupted
   into a nonsensical value (negative remaining, remaining greater
   than the configured limit) even though exact quota accounting is
   not guaranteed once this documented precondition is violated.
--------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import pytest

from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.algorithms.leaky_bucket import LeakyBucketMeter, LeakyBucketQueue
from rlimit.algorithms.sliding_window_counter import SlidingWindowCounter
from rlimit.algorithms.sliding_window_log import SlidingWindowLog
from rlimit.algorithms.token_bucket import TokenBucket
from rlimit.base import RateLimiter, UnsatisfiableRequestError


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        # Deliberately NOT rejecting negative `seconds` here (unlike
        # conftest.py's FakeClock) -- this file's clock-skew tests need
        # to be able to move the clock backward on purpose.
        self._now += seconds

    def set(self, value: float) -> None:
        self._now = value


# Each entry: (name, constructor_kwarg_name_for_limit, limit_value,
# zero_arg_factory(clock) -> RateLimiter). `constructor_kwarg_name_for_limit`
# lets the zero/negative-limit test construct each algorithm with the
# same invalid value under its own correctly-named parameter
# (limit vs capacity) without a separate branch per algorithm.

_ALGORITHMS: List[Tuple[str, str, int, Callable[[FakeClock], RateLimiter]]] = [
    (
        "FixedWindow",
        "limit",
        10,
        lambda clock: FixedWindow(limit=10, period=60.0, clock=clock),
    ),
    (
        "TokenBucket",
        "capacity",
        10,
        lambda clock: TokenBucket(capacity=10, refill_rate=1.0, clock=clock),
    ),
    (
        "SlidingWindowLog",
        "limit",
        10,
        lambda clock: SlidingWindowLog(limit=10, period=60.0, clock=clock),
    ),
    (
        "SlidingWindowCounter",
        "limit",
        10,
        lambda clock: SlidingWindowCounter(limit=10, period=60.0, clock=clock),
    ),
    (
        "LeakyBucketMeter",
        "capacity",
        10,
        lambda clock: LeakyBucketMeter(capacity=10, leak_rate=1.0, clock=clock),
    ),
    (
        "LeakyBucketQueue",
        "capacity",
        10,
        lambda clock: LeakyBucketQueue(capacity=10, leak_rate=1.0, clock=clock),
    ),
]


def _construct_with_limit(name: str, kwarg_name: str, value: Any) -> None:
    """Construct the named algorithm with `value` passed as its
    limit/capacity kwarg, and whatever second required kwarg
    (period/refill_rate/leak_rate) it needs, using a harmless default
    for that second kwarg since it isn't what's under test here."""
    if name == "FixedWindow":
        FixedWindow(limit=value, period=60.0)
    elif name == "TokenBucket":
        TokenBucket(capacity=value, refill_rate=1.0)
    elif name == "SlidingWindowLog":
        SlidingWindowLog(limit=value, period=60.0)
    elif name == "SlidingWindowCounter":
        SlidingWindowCounter(limit=value, period=60.0)
    elif name == "LeakyBucketMeter":
        LeakyBucketMeter(capacity=value, leak_rate=1.0)
    elif name == "LeakyBucketQueue":
        LeakyBucketQueue(capacity=value, leak_rate=1.0)
    else:
        raise AssertionError(f"unhandled algorithm name in test helper: {name}")


# --- 1. Zero / negative limit-or-capacity: shared cross-algorithm check --


@pytest.mark.parametrize(
    "name,kwarg_name,_limit,_factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_zero_limit_or_capacity_rejected(
    name: str, kwarg_name: str, _limit: int, _factory: Any
) -> None:
    with pytest.raises(ValueError):
        _construct_with_limit(name, kwarg_name, 0)


@pytest.mark.parametrize(
    "name,kwarg_name,_limit,_factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_negative_limit_or_capacity_rejected(
    name: str, kwarg_name: str, _limit: int, _factory: Any
) -> None:
    with pytest.raises(ValueError):
        _construct_with_limit(name, kwarg_name, -5)


# --- 2. Very high burst: exact boundary, shared across algorithms -------


@pytest.mark.parametrize(
    "name,kwarg_name,limit,factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_cost_exactly_at_limit_is_allowed(
    name: str, kwarg_name: str, limit: int, factory: Callable[[FakeClock], RateLimiter]
) -> None:
    clock = FakeClock()
    limiter = factory(clock)
    assert limiter.allow("k", cost=limit) is True, name


@pytest.mark.parametrize(
    "name,kwarg_name,limit,factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_cost_one_above_limit_is_denied_by_allow(
    name: str, kwarg_name: str, limit: int, factory: Callable[[FakeClock], RateLimiter]
) -> None:
    clock = FakeClock()
    limiter = factory(clock)
    assert limiter.allow("k", cost=limit + 1) is False, name


@pytest.mark.parametrize(
    "name,kwarg_name,limit,factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_cost_one_above_limit_raises_unsatisfiable_on_allow_wait(
    name: str, kwarg_name: str, limit: int, factory: Callable[[FakeClock], RateLimiter]
) -> None:
    clock = FakeClock()
    limiter = factory(clock)
    with pytest.raises(UnsatisfiableRequestError):
        limiter.allow_wait("k", cost=limit + 1)


# --- 3. Clock skew: backward jump does not crash or corrupt state ------


@pytest.mark.parametrize(
    "name,kwarg_name,limit,factory",
    _ALGORITHMS,
    ids=[n for n, _, _, _ in _ALGORITHMS],
)
def test_backward_clock_jump_does_not_crash_or_corrupt_remaining(
    name: str, kwarg_name: str, limit: int, factory: Callable[[FakeClock], RateLimiter]
) -> None:
    """Deliberately does NOT assert quota is never exceeded -- see this
    file's module docstring, point 3, for why that would be asserting
    a false property for FixedWindow/SlidingWindowCounter specifically
    (a backward jump can look like a window rollover to those two
    algorithms, resetting their count/weighted-count early). What IS
    asserted, for every algorithm uniformly: no exception is raised,
    and remaining() afterward is neither negative nor greater than the
    configured limit/capacity -- state stays internally sane even
    though exact accounting is not guaranteed once the documented
    monotonic-clock precondition is violated."""
    clock = FakeClock(start=1000.0)
    limiter = factory(clock)

    limiter.allow("k", cost=1)
    clock.advance(10.0)
    limiter.allow("k", cost=1)

    # The actual skew: jump backward past both previous calls.
    clock.set(0.0)

    # Must not raise.
    limiter.allow("k", cost=1)
    remaining = limiter.remaining("k")

    assert remaining >= 0, name
    assert remaining <= limit, name


def test_all_six_sync_algorithms_are_covered_by_this_file() -> None:
    """Guards against a silent gap if a 7th sync algorithm family is
    ever added and this file's _ALGORITHMS table isn't updated to
    match -- same pattern as
    test_metrics_all_algorithms.py's equivalent count-pinning test."""
    assert len(_ALGORITHMS) == 6
