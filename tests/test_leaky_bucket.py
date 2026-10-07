"""Unit tests for limivault.algorithms.leaky_bucket (LeakyBucketMeter and
LeakyBucketQueue).

Constructor validation: capacity > 0 and leak_rate >= 0 are now enforced
in __init__ for both classes (raises ValueError). leak_rate == 0 remains
valid (a bucket/queue that never drains) since it's an existing, tested
configuration. See TestMeterConstructor / TestQueueConstructor.

Negative cost: allow() rejects cost < 0 by returning False without
touching stored state; allow_wait() raises ValueError. This is a
consistency fix -- unlike the other algorithms, leaky bucket's
negative-cost handling was already self-limiting (the max(0.0, ...)
floor in the leak math clamps an over-negative internal value back to
empty rather than over-refunding), but it's still invalid input that's
now rejected at the API boundary rather than silently accepted.

FLAGGED (minor, non-blocking, unchanged by this fix): stress-testing the
"wait the returned amount, then retry" property against ~2000 random
scenarios per class found NO logic bugs, but did find that failures can
occur purely from floating-point rounding at the exact boundary
(volume/depth + cost comes out a few ULPs above capacity, e.g.
10.000000000000002 vs 10). This is a standard float-comparison issue,
not a design flaw in the leak math itself -- the property tests below
use a tiny epsilon on the wait time to avoid flaky failures from this.


this file already had ThreadPoolExecutor stress tests (TestConcurrency,
covering both Meter and Queue) and sequential Hypothesis property tests
(TestMeterProperties / TestQueueProperties). New additions:
TestConcurrentProperties (Hypothesis over randomized thread interleaving
for both variants), TestBoundaryRace (threads racing right at a
leak/drain-interval crossing), and TestMultiprocessing (real OS
processes via mp_workers.py, both variants).
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from limivault.algorithms.leaky_bucket import LeakyBucketMeter, LeakyBucketQueue
from limivault.base import UnsatisfiableRequestError
from limivault.storage import InMemoryStorage
from tests.mp_workers import hammer_leaky_bucket_meter, hammer_leaky_bucket_queue


class FakeClock:
    """Manually advanceable clock for deterministic tests. Defined
    locally (not imported from conftest) so this file has no dependency
    on pytest's import-mode/sys.path configuration -- matches the
    pattern already used in test_fixed_window.py / test_token_bucket.py.
    The `fake_clock` fixture in conftest.py wraps the same shape and is
    still used via normal fixture injection where convenient."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# Tiny buffer added on top of allow_wait()'s returned duration in the
# "wait then retry" property tests, to absorb float rounding at exact
# capacity boundaries (see module docstring). Not needed for the
# deterministic unit tests below, only the randomized property tests.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Constructor / defaults -- LeakyBucketMeter
# ---------------------------------------------------------------------------


class TestMeterConstructor:
    def test_defaults_use_in_memory_storage(self) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1)
        assert isinstance(limiter._storage, InMemoryStorage)

    def test_defaults_use_time_monotonic(self) -> None:
        import time

        limiter = LeakyBucketMeter(capacity=5, leak_rate=1)
        assert limiter._clock is time.monotonic

    def test_accepts_injected_clock_and_storage(self, fake_clock: FakeClock) -> None:
        storage = InMemoryStorage()
        limiter = LeakyBucketMeter(
            capacity=5, leak_rate=1, storage=storage, clock=fake_clock
        )
        assert limiter._storage is storage
        assert limiter._clock is fake_clock

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=-5, leak_rate=1)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=0, leak_rate=1)

    def test_negative_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=5, leak_rate=-1)

    def test_zero_leak_rate_is_still_valid(self) -> None:
        # A bucket that never drains is a legitimate configuration
        # (see test_zero_leak_rate_raises_on_allow_wait_when_full below).
        LeakyBucketMeter(capacity=5, leak_rate=0)

    # -----------------------------------------------------------------
    #  NaN/infinity validation. See test_fixed_window.py's
    # equivalent section for the full rationale.
    # -----------------------------------------------------------------

    def test_nan_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            # `capacity` is typed as int; float("nan") is intentionally
            # passed anyway to prove the runtime NaN guard catches it.
            LeakyBucketMeter(capacity=float("nan"), leak_rate=1)  # type: ignore[arg-type]

    def test_infinite_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=float("inf"), leak_rate=1)  # type: ignore[arg-type]

    def test_nan_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=5, leak_rate=float("nan"))

    def test_infinite_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketMeter(capacity=5, leak_rate=float("inf"))


# ---------------------------------------------------------------------------
# Basic correctness -- LeakyBucketMeter
# ---------------------------------------------------------------------------


class TestMeterBasicAllow:
    def test_allows_up_to_capacity(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=3, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False

    def test_denied_request_does_not_add_volume(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=1, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False
        assert limiter.remaining("k") == 0

    def test_volume_leaks_continuously(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=10, leak_rate=2, clock=fake_clock)
        limiter.allow("k", cost=10)
        assert limiter.remaining("k") == 0
        fake_clock.advance(1.0)  # leaks 2 units
        assert limiter.remaining("k") == 2
        fake_clock.advance(4.0)  # leaks 8 more, fully drained
        assert limiter.remaining("k") == 10

    def test_volume_never_leaks_below_zero(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=10, leak_rate=2, clock=fake_clock)
        limiter.allow("k", cost=2)
        fake_clock.advance(1000.0)
        assert limiter.remaining("k") == 10

    def test_keys_are_independent(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=1, leak_rate=1, clock=fake_clock)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False
        assert limiter.allow("b") is False


# ---------------------------------------------------------------------------
# Invalid / edge cost -- LeakyBucketMeter
# ---------------------------------------------------------------------------


class TestMeterInvalidCost:
    def test_cost_exceeding_capacity_denied_not_raised_by_allow(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k", cost=6) is False

    def test_cost_exceeding_capacity_raises_on_allow_wait(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait("k", cost=6)

    def test_zero_leak_rate_raises_on_allow_wait_when_full(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=0, clock=fake_clock)
        limiter.allow("k", cost=5)
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait("k", cost=1)

    def test_zero_cost_always_allowed(self, fake_clock: FakeClock) -> None:
        # Not explicitly documented; current code allows it.
        limiter = LeakyBucketMeter(capacity=1, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=1)
        assert limiter.allow("k", cost=0) is True

    def test_negative_cost_rejected_by_allow(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=1, leak_rate=0, clock=fake_clock)
        limiter.allow("k", cost=1)
        assert limiter.remaining("k") == 0
        assert limiter.allow("k", cost=-5) is False
        assert limiter.remaining("k") == 0  # state untouched by the rejected call

    def test_negative_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=-1)

    def test_nan_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=2)
        assert limiter.remaining("k") == 3
        assert limiter.allow("k", cost=float("nan")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 3  # untouched, not silently corrupted

    def test_infinite_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k", cost=float("inf")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 5

    def test_nan_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("nan"))  # type: ignore[arg-type]

    def test_infinite_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("inf"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# allow_wait -- LeakyBucketMeter
# ---------------------------------------------------------------------------


class TestMeterAllowWait:
    def test_zero_wait_when_capacity_available(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow_wait("k") == 0.0

    def test_wait_matches_leak_math(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketMeter(capacity=10, leak_rate=2, clock=fake_clock)
        limiter.allow("k", cost=10)
        wait = limiter.allow_wait("k", cost=4)
        assert wait == pytest.approx(2.0)  # need 4 units to leak out at 2/s
        fake_clock.advance(wait)
        assert limiter.allow("k", cost=4) is True


# ---------------------------------------------------------------------------
# Constructor / defaults -- LeakyBucketQueue
# ---------------------------------------------------------------------------


class TestQueueConstructor:
    def test_defaults_use_in_memory_storage(self) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1)
        assert isinstance(limiter._storage, InMemoryStorage)

    def test_defaults_use_time_monotonic(self) -> None:
        import time

        limiter = LeakyBucketQueue(capacity=5, leak_rate=1)
        assert limiter._clock is time.monotonic

    def test_accepts_injected_clock_and_storage(self, fake_clock: FakeClock) -> None:
        storage = InMemoryStorage()
        limiter = LeakyBucketQueue(
            capacity=5, leak_rate=1, storage=storage, clock=fake_clock
        )
        assert limiter._storage is storage
        assert limiter._clock is fake_clock

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=-5, leak_rate=1)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=0, leak_rate=1)

    def test_negative_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=5, leak_rate=-1)

    def test_zero_leak_rate_is_still_valid(self) -> None:
        # A queue that never drains is a legitimate configuration
        # (see test_zero_leak_rate_raises_on_allow_wait_when_full below).
        LeakyBucketQueue(capacity=5, leak_rate=0)

    # -----------------------------------------------------------------
    # NaN/infinity validation. See test_fixed_window.py's
    # equivalent section for the full rationale.
    # -----------------------------------------------------------------

    def test_nan_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            # `capacity` is typed as int; float("nan") is intentionally
            # passed anyway to prove the runtime NaN guard catches it.
            LeakyBucketQueue(capacity=float("nan"), leak_rate=1)  # type: ignore[arg-type]

    def test_infinite_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=float("inf"), leak_rate=1)  # type: ignore[arg-type]

    def test_nan_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=5, leak_rate=float("nan"))

    def test_infinite_leak_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            LeakyBucketQueue(capacity=5, leak_rate=float("inf"))


# ---------------------------------------------------------------------------
# Basic correctness -- LeakyBucketQueue
# ---------------------------------------------------------------------------


class TestQueueBasicAllow:
    def test_allows_up_to_capacity(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=3, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False

    def test_denied_request_does_not_add_depth(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=1, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False
        assert limiter.remaining("k") == 0

    def test_drains_one_whole_item_per_interval(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)  # 1/s
        limiter.allow("k", cost=5)
        assert limiter.remaining("k") == 0
        fake_clock.advance(0.999)
        assert limiter.remaining("k") == 0  # not a whole interval yet
        fake_clock.advance(0.002)
        assert limiter.remaining("k") == 1  # exactly one item drained

    def test_last_drain_advances_by_drained_amount_not_reset_to_now(
        self, fake_clock: FakeClock
    ) -> None:
        """Regression test for the flagged last_drain advancement
        strategy: fractional progress toward the next drain must be
        preserved across calls rather than discarded by resetting
        last_drain to `now` on every read."""
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=5)
        fake_clock.advance(1.5)  # 1 whole item drains; 0.5s of progress remains
        assert limiter.remaining("k") == 1
        fake_clock.advance(0.5)  # total progress toward 2nd item = 1.0s -> drains
        assert limiter.remaining("k") == 2

    def test_depth_never_drains_below_zero(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=2)
        fake_clock.advance(1000.0)
        assert limiter.remaining("k") == 5

    def test_keys_are_independent(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=1, leak_rate=1, clock=fake_clock)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False
        assert limiter.allow("b") is False


# ---------------------------------------------------------------------------
# Invalid / edge cost -- LeakyBucketQueue
# ---------------------------------------------------------------------------


class TestQueueInvalidCost:
    def test_cost_exceeding_capacity_denied_not_raised_by_allow(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k", cost=6) is False

    def test_cost_exceeding_capacity_raises_on_allow_wait(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait("k", cost=6)

    def test_zero_leak_rate_raises_on_allow_wait_when_full(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=0, clock=fake_clock)
        limiter.allow("k", cost=5)
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait("k", cost=1)

    def test_zero_cost_always_allowed(self, fake_clock: FakeClock) -> None:
        # Not explicitly documented; current code allows it.
        limiter = LeakyBucketQueue(capacity=1, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=1)
        assert limiter.allow("k", cost=0) is True

    def test_negative_cost_rejected_by_allow(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=1, leak_rate=0, clock=fake_clock)
        limiter.allow("k", cost=1)
        assert limiter.remaining("k") == 0
        assert limiter.allow("k", cost=-5) is False
        assert limiter.remaining("k") == 0  # state untouched by the rejected call

    def test_negative_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=-1)

    def test_nan_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=2)
        assert limiter.remaining("k") == 3
        assert limiter.allow("k", cost=float("nan")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 3  # untouched, not silently corrupted

    def test_infinite_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow("k", cost=float("inf")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 5

    def test_nan_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("nan"))  # type: ignore[arg-type]

    def test_infinite_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("inf"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# allow_wait -- LeakyBucketQueue
# ---------------------------------------------------------------------------


class TestQueueAllowWait:
    def test_zero_wait_when_capacity_available(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        assert limiter.allow_wait("k") == 0.0

    def test_wait_for_single_item(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=1, clock=fake_clock)
        limiter.allow("k", cost=5)
        wait = limiter.allow_wait("k", cost=1)
        assert wait == pytest.approx(1.0)
        fake_clock.advance(wait)
        assert limiter.allow("k", cost=1) is True

    def test_wait_for_multiple_items(self, fake_clock: FakeClock) -> None:
        limiter = LeakyBucketQueue(capacity=5, leak_rate=2, clock=fake_clock)  # 2/s
        limiter.allow("k", cost=5)
        wait = limiter.allow_wait("k", cost=3)
        assert wait == pytest.approx(1.5)
        fake_clock.advance(wait)
        assert limiter.allow("k", cost=3) is True


# ---------------------------------------------------------------------------
# Concurrency (ThreadPoolExecutor, both variants)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_meter_threaded_hammering_never_exceeds_capacity(self) -> None:
        import time as real_time

        capacity = 50
        limiter = LeakyBucketMeter(
            capacity=capacity, leak_rate=0, clock=real_time.monotonic
        )
        allowed = []
        lock = threading.Lock()

        def hit() -> None:
            if limiter.allow("shared-key"):
                with lock:
                    allowed.append(True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(hit) for _ in range(500)]
            for f in futures:
                f.result()

        assert len(allowed) == capacity

    def test_queue_threaded_hammering_never_exceeds_capacity(self) -> None:
        import time as real_time

        capacity = 50
        limiter = LeakyBucketQueue(
            capacity=capacity, leak_rate=0, clock=real_time.monotonic
        )
        allowed = []
        lock = threading.Lock()

        def hit() -> None:
            if limiter.allow("shared-key"):
                with lock:
                    allowed.append(True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(hit) for _ in range(500)]
            for f in futures:
                f.result()

        assert len(allowed) == capacity

    def test_queue_threaded_hammering_independent_keys(self) -> None:
        import time as real_time

        limiter = LeakyBucketQueue(capacity=10, leak_rate=0, clock=real_time.monotonic)
        results: dict[str, list[bool]] = {"a": [], "b": []}
        lock = threading.Lock()

        def hit(key: str) -> None:
            r = limiter.allow(key)
            with lock:
                results[key].append(r)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(hit, "a") for _ in range(50)] + [
                ex.submit(hit, "b") for _ in range(50)
            ]
            for f in futures:
                f.result()

        assert sum(results["a"]) == 10
        assert sum(results["b"]) == 10


# ---------------------------------------------------------------------------
# Property-based tests (sequential, both variants)
# ---------------------------------------------------------------------------


class TestMeterProperties:
    @given(
        capacity=st.integers(min_value=1, max_value=20),
        leak_rate=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        costs=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=50),
    )
    @settings(max_examples=100, deadline=None)
    def test_volume_never_exceeds_capacity(
        self, capacity: int, leak_rate: float, costs: list[int]
    ) -> None:
        clock = FakeClock(start=0.0)
        limiter = LeakyBucketMeter(capacity=capacity, leak_rate=leak_rate, clock=clock)
        for cost in costs:
            limiter.allow("k", cost=cost)
            assert limiter.remaining("k") >= 0
            clock.advance(1.0 / leak_rate / 3)

    @given(
        capacity=st.integers(min_value=1, max_value=20),
        leak_rate=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        warmup_costs=st.lists(
            st.integers(min_value=1, max_value=5), min_size=1, max_size=20
        ),
        cost=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=200, deadline=None)
    def test_wait_then_retry_property(
        self,
        capacity: int,
        leak_rate: float,
        warmup_costs: list[int],
        cost: int,
    ) -> None:
        clock = FakeClock(start=0.0)
        limiter = LeakyBucketMeter(capacity=capacity, leak_rate=leak_rate, clock=clock)
        for c in warmup_costs:
            limiter.allow("k", cost=min(c, capacity))
            clock.advance(1.0 / leak_rate / 3)
        cost = min(cost, capacity)
        try:
            wait = limiter.allow_wait("k", cost=cost)
        except UnsatisfiableRequestError:
            return
        if wait == 0.0:
            return
        # epsilon absorbs float rounding, see module docstring
        clock.advance(wait + _EPS)
        assert limiter.allow("k", cost=cost) is True


class TestQueueProperties:
    @given(
        capacity=st.integers(min_value=1, max_value=20),
        leak_rate=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        costs=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=50),
    )
    @settings(max_examples=100, deadline=None)
    def test_depth_never_exceeds_capacity(
        self, capacity: int, leak_rate: float, costs: list[int]
    ) -> None:
        clock = FakeClock(start=0.0)
        limiter = LeakyBucketQueue(capacity=capacity, leak_rate=leak_rate, clock=clock)
        for cost in costs:
            limiter.allow("k", cost=cost)
            assert limiter.remaining("k") >= 0
            clock.advance(1.0 / leak_rate / 3)

    @given(
        capacity=st.integers(min_value=1, max_value=20),
        leak_rate=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        warmup_costs=st.lists(
            st.integers(min_value=1, max_value=5), min_size=1, max_size=20
        ),
        cost=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=200, deadline=None)
    def test_wait_then_retry_property(
        self,
        capacity: int,
        leak_rate: float,
        warmup_costs: list[int],
        cost: int,
    ) -> None:
        clock = FakeClock(start=0.0)
        limiter = LeakyBucketQueue(capacity=capacity, leak_rate=leak_rate, clock=clock)
        for c in warmup_costs:
            limiter.allow("k", cost=min(c, capacity))
            clock.advance(1.0 / leak_rate / 3)
        cost = min(cost, capacity)
        try:
            wait = limiter.allow_wait("k", cost=cost)
        except UnsatisfiableRequestError:
            return
        if wait == 0.0:
            return
        clock.advance(wait + _EPS)
        assert limiter.allow("k", cost=cost) is True


# ---------------------------------------------------------------------------
# concurrency correctness (both variants)
# ---------------------------------------------------------------------------


class TestConcurrentProperties:
    """Hypothesis property test where the interleaving itself is
    randomized (real threads racing on a shared key), for both
    variants."""

    @given(
        capacity=st.integers(min_value=1, max_value=30),
        n_threads=st.integers(min_value=2, max_value=16),
        attempts_per_thread=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=40, deadline=None)
    def test_meter_never_exceeds_capacity_under_concurrent_interleaving(
        self, capacity: int, n_threads: int, attempts_per_thread: int
    ) -> None:
        import time as real_time

        limiter = LeakyBucketMeter(
            capacity=capacity, leak_rate=0, clock=real_time.monotonic
        )
        allowed_count = 0
        count_lock = threading.Lock()

        def hit() -> None:
            nonlocal allowed_count
            for _ in range(attempts_per_thread):
                if limiter.allow("k"):
                    with count_lock:
                        allowed_count += 1

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count <= capacity
        assert limiter.remaining("k") >= 0

    @given(
        capacity=st.integers(min_value=1, max_value=30),
        n_threads=st.integers(min_value=2, max_value=16),
        attempts_per_thread=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=40, deadline=None)
    def test_queue_never_exceeds_capacity_under_concurrent_interleaving(
        self, capacity: int, n_threads: int, attempts_per_thread: int
    ) -> None:
        import time as real_time

        limiter = LeakyBucketQueue(
            capacity=capacity, leak_rate=0, clock=real_time.monotonic
        )
        allowed_count = 0
        count_lock = threading.Lock()

        def hit() -> None:
            nonlocal allowed_count
            for _ in range(attempts_per_thread):
                if limiter.allow("k"):
                    with count_lock:
                        allowed_count += 1

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count <= capacity
        assert limiter.remaining("k") >= 0


class TestBoundaryRace:
    """Race condition tests targeting active leak/drain crossings. For
    the Queue variant in particular, this targets the discrete
    drain-interval boundary specifically -- the moment `_get_state`
    computes a whole item just finished draining -- which is the most
    delicate point in that algorithm's math (see leaky_bucket.py's
    `_get_state` docstring on last_drain advancement).

    CORRECTED (see PR discussion): the first version used the real wall
    clock with a threading.Barrier, then asserted `sum(results) <=
    capacity`. Same root issue as the other algorithms' TestBoundaryRace
    classes: real elapsed time during a burst genuinely leaks/drains
    additional capacity beyond a static starting point, so a bound that
    ignores elapsed leak/drain can legitimately be exceeded by correct
    behavior, not just by a bug. Fixed the same way: a shared FakeClock
    frozen for each burst (so leak/drain is exactly zero *during* the
    race), stepped by an exact, deterministic amount between bursts
    (`capacity / leak_rate`, computed once rather than approximated via
    real-time sleeps). Verified against 100 repeated trials with zero
    failures for both variants before being kept here.
    """

    def test_meter_concurrent_burst_against_full_bucket_admits_exactly_capacity(
        self,
    ) -> None:
        clock = FakeClock()
        capacity = 20
        leak_rate = 5.0
        limiter = LeakyBucketMeter(capacity=capacity, leak_rate=leak_rate, clock=clock)
        # Bucket starts empty (0 volume used), so all `capacity` units
        # are available for the whole frozen-clock race.

        n_threads = 40
        barrier = threading.Barrier(n_threads)
        lock = threading.Lock()
        results: list[bool] = []

        def hit() -> None:
            barrier.wait()
            r = limiter.allow("k")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == min(n_threads, capacity)

    def test_meter_concurrent_burst_after_exact_full_drain_admits_exactly_capacity(
        self,
    ) -> None:
        """First burst fills the bucket via a frozen-clock race. The
        clock is then advanced by exactly `capacity / leak_rate` -- the
        precise, deterministic time needed to fully drain -- before a
        second frozen-clock burst, which must again admit exactly
        `capacity`."""
        clock = FakeClock()
        capacity = 20
        leak_rate = 5.0
        limiter = LeakyBucketMeter(capacity=capacity, leak_rate=leak_rate, clock=clock)
        n_threads = 40
        lock = threading.Lock()

        def burst() -> list[bool]:
            barrier = threading.Barrier(n_threads)
            results: list[bool] = []

            def hit() -> None:
                barrier.wait()
                r = limiter.allow("k")
                with lock:
                    results.append(r)

            threads = [threading.Thread(target=hit) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return results

        first = burst()
        assert sum(first) == min(n_threads, capacity)

        clock.advance(capacity / leak_rate)  # exact, deterministic full drain

        second = burst()
        assert sum(second) == min(n_threads, capacity)

    def test_queue_concurrent_burst_against_empty_queue_admits_exactly_capacity(
        self,
    ) -> None:
        clock = FakeClock()
        capacity = 20
        leak_rate = 5.0
        limiter = LeakyBucketQueue(capacity=capacity, leak_rate=leak_rate, clock=clock)

        n_threads = 40
        barrier = threading.Barrier(n_threads)
        lock = threading.Lock()
        results: list[bool] = []

        def hit() -> None:
            barrier.wait()
            r = limiter.allow("k")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == min(n_threads, capacity)

    def test_queue_concurrent_burst_after_exact_full_drain_admits_exactly_capacity(
        self,
    ) -> None:
        """Same pattern as the Meter version above, but for the discrete
        drain math: `capacity / leak_rate` is exactly the elapsed time
        needed for `math.floor(elapsed * leak_rate)` to equal
        `capacity` whole items, so the second burst's queue is
        deterministically fully empty, not just approximately so."""
        clock = FakeClock()
        capacity = 20
        leak_rate = 5.0
        limiter = LeakyBucketQueue(capacity=capacity, leak_rate=leak_rate, clock=clock)
        n_threads = 40
        lock = threading.Lock()

        def burst() -> list[bool]:
            barrier = threading.Barrier(n_threads)
            results: list[bool] = []

            def hit() -> None:
                barrier.wait()
                r = limiter.allow("k")
                with lock:
                    results.append(r)

            threads = [threading.Thread(target=hit) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return results

        first = burst()
        assert sum(first) == min(n_threads, capacity)

        clock.advance(capacity / leak_rate)  # exact, deterministic full drain

        second = burst()
        assert sum(second) == min(n_threads, capacity)


class TestMultiprocessing:
    """Same invariant as TestConcurrency, but across real OS processes
    sharing one InMemoryStorage(multiprocess_safe=True) instance, for
    both variants."""

    def test_meter_multiprocess_hammering_never_exceeds_capacity(self) -> None:
        capacity = 40
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 5
        attempts_per_proc = 30

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(
                    hammer_leaky_bucket_meter,
                    storage,
                    capacity,
                    0.0,
                    "shared-key",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results = [f.result() for f in futures]

        assert sum(results) == capacity

    def test_queue_multiprocess_hammering_never_exceeds_capacity(self) -> None:
        capacity = 40
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 5
        attempts_per_proc = 30

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(
                    hammer_leaky_bucket_queue,
                    storage,
                    capacity,
                    0.0,
                    "shared-key",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results = [f.result() for f in futures]

        assert sum(results) == capacity
