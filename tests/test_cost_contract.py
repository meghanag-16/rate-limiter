# tests/test_cost_contract.py
"""Shared cost-handling contract tests (sync), covering all five sync
algorithms uniformly. See base.py's module docstring for the full
contract this enforces: cost must be an int (not bool, not float --
including whole-number floats), negative cost is denied/raises,
cost == 0 is always allowed and must not touch storage at all (not
even to create state for a brand-new key), consistent across every
algorithm rather than tested piecemeal per algorithm file.

Centralized here rather than duplicated into each existing
test_<algorithm>.py file, since the contract itself is now identical
across algorithms -- a shared/parametrized-style test is less to keep
in sync than five near-identical copies, and a contract violation in
any one algorithm shows up immediately by name in the failure output
via the `name` values threaded through each assertion below.
"""

from __future__ import annotations

import pytest

from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.algorithms.leaky_bucket import LeakyBucketMeter, LeakyBucketQueue
from rlimit.algorithms.sliding_window_counter import SlidingWindowCounter
from rlimit.algorithms.sliding_window_log import SlidingWindowLog
from rlimit.algorithms.token_bucket import TokenBucket
from rlimit.base import RateLimiter
from rlimit.storage import InMemoryStorage


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_limiters(
    clock: FakeClock,
) -> list[tuple[str, RateLimiter, InMemoryStorage]]:
    """One instance of each sync algorithm, each with its own fresh
    InMemoryStorage so state-touching assertions are reliable and
    isolated per algorithm."""
    limiters: list[tuple[str, RateLimiter, InMemoryStorage]] = []

    s1 = InMemoryStorage()
    limiters.append(
        ("FixedWindow", FixedWindow(limit=5, period=60.0, storage=s1, clock=clock), s1)
    )

    s2 = InMemoryStorage()
    limiters.append(
        (
            "TokenBucket",
            TokenBucket(capacity=5, refill_rate=1.0, storage=s2, clock=clock),
            s2,
        )
    )

    s3 = InMemoryStorage()
    limiters.append(
        (
            "SlidingWindowLog",
            SlidingWindowLog(limit=5, period=60.0, storage=s3, clock=clock),
            s3,
        )
    )

    s4 = InMemoryStorage()
    limiters.append(
        (
            "SlidingWindowCounter",
            SlidingWindowCounter(limit=5, period=60.0, storage=s4, clock=clock),
            s4,
        )
    )

    s5 = InMemoryStorage()
    limiters.append(
        (
            "LeakyBucketMeter",
            LeakyBucketMeter(capacity=5, leak_rate=1.0, storage=s5, clock=clock),
            s5,
        )
    )

    s6 = InMemoryStorage()
    limiters.append(
        (
            "LeakyBucketQueue",
            LeakyBucketQueue(capacity=5, leak_rate=1.0, storage=s6, clock=clock),
            s6,
        )
    )

    return limiters


# --- Zero-cost is a true no-op, for every algorithm ---------------------


def test_zero_cost_returns_true_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, _ in _make_limiters(clock):
        assert limiter.allow("k", cost=0) is True, name


def test_zero_cost_does_not_create_state_for_new_key() -> None:
    """The core bug this fixes: a zero-cost call on a never-seen key
    must not write any state at all, for any algorithm."""
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert limiter.allow("brand-new-key", cost=0) is True, name
        assert storage.get("brand-new-key") is None, name


def test_zero_cost_leaves_existing_state_unchanged() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        limiter.allow("k", cost=1)  # establish some real state
        state_before = storage.get("k")
        assert limiter.allow("k", cost=0) is True, name
        assert storage.get("k") == state_before, name


def test_zero_cost_allow_wait_is_zero_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, _ in _make_limiters(clock):
        assert limiter.allow_wait("k", cost=0) == 0.0, name


def test_sliding_window_log_zero_cost_does_not_grow_the_log() -> None:
    """Direct regression test for the specific bug flagged in review:
    repeated zero-cost calls used to append a (timestamp, 0) entry
    every time, growing the log indefinitely despite consuming no
    quota."""
    clock = FakeClock()
    storage = InMemoryStorage()
    limiter = SlidingWindowLog(limit=5, period=60.0, storage=storage, clock=clock)

    for _ in range(50):
        assert limiter.allow("k", cost=0) is True

    assert storage.get("k") is None  # never touched, not even once


# --- Cost type validation, for every algorithm ---------------------------


def test_float_cost_rejected_by_allow_for_every_algorithm() -> None:
    """Cost must be an int -- a finite, whole-number float like 2.0 is
    still rejected, not just NaN/infinity, since silently accepting it
    would let float state leak into what should be integer counters."""
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert limiter.allow("k", cost=2.0) is False, name  # type: ignore[arg-type]
        assert storage.get("k") is None, name


def test_bool_cost_rejected_by_allow_for_every_algorithm() -> None:
    """bool is technically an int subclass in Python; explicitly
    rejected anyway since `cost=True`/`cost=False` as a rate-limit cost
    is not a meaningful value a caller should be able to pass."""
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        # No type: ignore needed here: bool is a subtype of int in
        # Python's type system, so cost=True type-checks fine against
        # `cost: int` even though the runtime check correctly rejects
        # it. An ignore comment on this line is therefore unused (see
        # test_bool_cost_raises_value_error_on_allow_wait_for_every_algorithm
        # below for the same reasoning applied to allow_wait()).
        assert limiter.allow("k", cost=True) is False, name
        assert storage.get("k") is None, name


def test_nan_cost_still_rejected_by_allow_for_every_algorithm() -> None:
    """NaN is a float, so it's now caught by the int-type check rather
    than a separate math.isfinite() check -- still rejected either
    way, verified explicitly so the refactor didn't silently drop this
    case."""
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert limiter.allow("k", cost=float("nan")) is False, name  # type: ignore[arg-type]
        assert storage.get("k") is None, name


def test_float_cost_raises_value_error_on_allow_wait_for_every_algorithm() -> None:
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=2.0)  # type: ignore[arg-type]


def test_bool_cost_raises_value_error_on_allow_wait_for_every_algorithm() -> None:
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            # bool is a subtype of int, so no type: ignore is needed --
            # see the equivalent allow() test above for the full
            # rationale.
            limiter.allow_wait("k", cost=True)


# --- Negative cost still denied/raises, for every algorithm --------------


def test_negative_cost_still_denied_by_allow_for_every_algorithm() -> None:
    clock = FakeClock()
    for name, limiter, storage in _make_limiters(clock):
        assert limiter.allow("k", cost=-1) is False, name
        assert storage.get("k") is None, name


def test_negative_cost_still_raises_on_allow_wait_for_every_algorithm() -> None:
    clock = FakeClock()
    for _name, limiter, _ in _make_limiters(clock):
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=-1)
