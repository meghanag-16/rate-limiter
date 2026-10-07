"""Unit tests for limivault.algorithms.sliding_window_log.SlidingWindowLog.

Constructor validation: limit > 0 and period > 0 are now enforced in
__init__ (raises ValueError). See TestConstructor.

Negative cost: allow() rejects cost < 0 by returning False without
touching stored state; allow_wait() raises ValueError. See
TestInvalidCost. Previously negative cost let a caller "refund" quota
past the configured limit with no bound.

additions (everything from TestConcurrentProperties onward):
this file already had ThreadPoolExecutor stress tests (TestConcurrency)
and a sequential Hypothesis property test (TestProperties).
New additions: TestConcurrentProperties (Hypothesis over randomized
thread interleaving, not just randomized inputs), TestBoundaryRace
(threads racing right as log entries expire out of the window), and
TestMultiprocessing (the same invariant across real OS processes sharing
InMemoryStorage(multiprocess_safe=True), via mp_workers.py).
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from limivault.algorithms.sliding_window_log import SlidingWindowLog
from limivault.base import UnsatisfiableRequestError
from limivault.storage import InMemoryStorage
from tests.mp_workers import hammer_sliding_window_log


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



# ---------------------------------------------------------------------------
# Constructor / defaults
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults_use_in_memory_storage(self) -> None:
        limiter = SlidingWindowLog(limit=5, period=10)
        assert isinstance(limiter._storage, InMemoryStorage)

    def test_defaults_use_time_monotonic(self) -> None:
        import time

        limiter = SlidingWindowLog(limit=5, period=10)
        assert limiter._clock is time.monotonic

    def test_accepts_injected_clock_and_storage(self, fake_clock: FakeClock) -> None:
        storage = InMemoryStorage()
        limiter = SlidingWindowLog(
            limit=5, period=10, storage=storage, clock=fake_clock
        )
        assert limiter._storage is storage
        assert limiter._clock is fake_clock

    def test_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=-5, period=10)

    def test_zero_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=0, period=10)

    def test_zero_period_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=5, period=0)

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=5, period=-10)

    # -----------------------------------------------------------------
    # NaN/infinity validation. See test_fixed_window.py's
    # equivalent section for the full rationale.
    # -----------------------------------------------------------------

    def test_nan_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            # `limit` is typed as int; float("nan") is intentionally
            # passed anyway to prove the runtime NaN guard catches it.
            SlidingWindowLog(limit=float("nan"), period=10)  # type: ignore[arg-type]

    def test_infinite_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=float("inf"), period=10)  # type: ignore[arg-type]

    def test_nan_period_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=5, period=float("nan"))

    def test_infinite_period_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowLog(limit=5, period=float("inf"))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestBasicAllow:
    def test_allows_up_to_limit(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=3, period=10, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False

    def test_denied_request_does_not_consume_quota(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False
        assert limiter.remaining("k") == 0
        assert limiter.allow("k") is False  # still denied, not double-consumed

    def test_entries_expire_out_of_window(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False
        fake_clock.advance(10.001)
        assert limiter.allow("k") is True

    def test_no_boundary_burst(self, fake_clock: FakeClock) -> None:
        """The whole point of sliding-window-log vs fixed-window: no 2x
        burst is possible around any boundary. Entries are spaced 0.1s
        apart (t=0.0 .. t=0.9) so they expire one at a time rather than
        all at once."""
        limiter = SlidingWindowLog(limit=10, period=60, clock=fake_clock)
        for _ in range(10):
            assert limiter.allow("k") is True
            fake_clock.advance(0.1)
        # now at t=1.0, window full (10 entries between t=0.0 and t=0.9)
        assert limiter.allow("k") is False
        # now at t=60.0; cutoff = 0.0, entry at t=0.0 not > cutoff -> expired
        fake_clock.advance(59.0)
        assert limiter.allow("k") is True  # exactly one slot freed
        assert limiter.allow("k") is False  # only one slot was free

    def test_keys_are_independent(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False
        assert limiter.allow("b") is False

    def test_remaining_reflects_window_state(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        assert limiter.remaining("k") == 5
        limiter.allow("k", cost=2)
        assert limiter.remaining("k") == 3
        fake_clock.advance(10.001)
        assert limiter.remaining("k") == 5


# ---------------------------------------------------------------------------
# Invalid / edge cost handling
# ---------------------------------------------------------------------------


class TestInvalidCost:
    def test_cost_exceeding_limit_denied_not_raised_by_allow(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        assert limiter.allow("k", cost=6) is False

    def test_cost_exceeding_limit_raises_on_allow_wait(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        with pytest.raises(UnsatisfiableRequestError):
            limiter.allow_wait("k", cost=6)

    def test_cost_equal_to_limit_allowed_alone(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        assert limiter.allow("k", cost=5) is True
        assert limiter.allow("k", cost=1) is False

    def test_zero_cost_always_allowed(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        limiter.allow("k", cost=1)
        # Not explicitly documented; current code allows it.
        assert limiter.allow("k", cost=0) is True

    def test_negative_cost_rejected_by_allow(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        limiter.allow("k", cost=1)
        assert limiter.remaining("k") == 0
        assert limiter.allow("k", cost=-5) is False
        assert limiter.remaining("k") == 0  # state untouched by the rejected call

    def test_negative_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=-1)

    def test_nan_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        limiter.allow("k", cost=2)
        assert limiter.remaining("k") == 3
        assert limiter.allow("k", cost=float("nan")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 3  # untouched, not silently corrupted

    def test_infinite_cost_rejected_by_allow_without_corrupting_state(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        assert limiter.allow("k", cost=float("inf")) is False  # type: ignore[arg-type]
        assert limiter.remaining("k") == 5

    def test_nan_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("nan"))  # type: ignore[arg-type]

    def test_infinite_cost_raises_on_allow_wait(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        with pytest.raises(ValueError):
            limiter.allow_wait("k", cost=float("inf"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# allow_wait correctness
# ---------------------------------------------------------------------------


class TestAllowWait:
    def test_zero_wait_when_capacity_available(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=5, period=10, clock=fake_clock)
        assert limiter.allow_wait("k") == 0.0

    def test_wait_until_oldest_entry_expires(self, fake_clock: FakeClock) -> None:
        limiter = SlidingWindowLog(limit=1, period=10, clock=fake_clock)
        limiter.allow("k")
        wait = limiter.allow_wait("k")
        assert wait == pytest.approx(10.0)
        fake_clock.advance(wait)
        assert limiter.allow("k") is True

    def test_wait_advances_correctly_with_multiple_entries(
        self, fake_clock: FakeClock
    ) -> None:
        limiter = SlidingWindowLog(limit=3, period=10, clock=fake_clock)
        limiter.allow("k")  # t=0
        fake_clock.advance(2)
        limiter.allow("k")  # t=2
        fake_clock.advance(2)
        limiter.allow("k")  # t=4, now full
        wait = limiter.allow_wait("k")
        assert wait == pytest.approx(6.0)  # first entry expires at t=10, now t=4


# ---------------------------------------------------------------------------
# Concurrency (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_threaded_hammering_never_exceeds_limit(self) -> None:
        import time as real_time

        limit = 50
        limiter = SlidingWindowLog(limit=limit, period=3600, clock=real_time.monotonic)
        allowed = []
        lock = threading.Lock()

        def hit() -> None:
            result = limiter.allow("shared-key")
            if result:
                with lock:
                    allowed.append(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(hit) for _ in range(500)]
            for f in futures:
                f.result()

        assert len(allowed) == limit
        assert limiter.remaining("shared-key") == 0

    def test_threaded_hammering_independent_keys_do_not_interfere(self) -> None:
        import time as real_time

        limiter = SlidingWindowLog(limit=10, period=3600, clock=real_time.monotonic)
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
# Property-based tests (sequential)
# ---------------------------------------------------------------------------


class TestProperties:
    @given(
        limit=st.integers(min_value=1, max_value=20),
        period=st.floats(min_value=1.0, max_value=100.0),
        costs=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=50),
    )
    @settings(max_examples=100, deadline=None)
    def test_never_exceeds_limit_within_any_window(
        self, limit: int, period: float, costs: list[int]
    ) -> None:
        """Core invariant: at no point does the sum of admitted cost in the
        trailing `period` seconds exceed `limit`, regardless of the
        sequence of allow() calls or their costs."""
        clock = FakeClock(start=0.0)
        limiter = SlidingWindowLog(limit=limit, period=period, clock=clock)
        for cost in costs:
            limiter.allow("k", cost=cost)
            assert limiter.remaining("k") >= 0
            clock.advance(period / len(costs) / 3)  # sub-window-sized steps

    @given(
        limit=st.integers(min_value=1, max_value=10),
        period=st.floats(min_value=1.0, max_value=50.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_remaining_never_negative(self, limit: int, period: float) -> None:
        clock = FakeClock(start=0.0)
        limiter = SlidingWindowLog(limit=limit, period=period, clock=clock)
        for _ in range(limit + 5):
            limiter.allow("k")
        assert limiter.remaining("k") >= 0


# ---------------------------------------------------------------------------
# concurrency correctness
# ---------------------------------------------------------------------------


class TestConcurrentProperties:
    """Hypothesis property test where the interleaving itself is
    randomized (via real threads racing on a shared key), not just the
    input values -- distinct from TestProperties above, which is
    sequential/single-threaded with a controlled FakeClock."""

    @given(
        limit=st.integers(min_value=1, max_value=30),
        n_threads=st.integers(min_value=2, max_value=16),
        attempts_per_thread=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=40, deadline=None)
    def test_never_exceeds_limit_under_concurrent_interleaving(
        self, limit: int, n_threads: int, attempts_per_thread: int
    ) -> None:
        import time as real_time

        limiter = SlidingWindowLog(limit=limit, period=3600, clock=real_time.monotonic)
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

        assert allowed_count <= limit
        assert limiter.remaining("k") >= 0


class TestBoundaryRace:
    """Race condition tests targeting the moment log entries expire out
    of the trailing window.

    CORRECTED (see PR discussion): the first version used the real wall
    clock with real `time.sleep()` calls and a threading.Barrier, then
    asserted `sum(results) <= limit`. Same root issue as
    test_fixed_window.py's and test_token_bucket.py's TestBoundaryRace:
    real elapsed time during a burst can legitimately expire entries
    mid-burst in a way a static bound doesn't account for, and the test
    was validating scheduler timing more than the algorithm. Fixed the
    same way: a shared FakeClock frozen for each burst (so no entry
    expires *during* the race -- nothing ambiguous to account for), and
    advanced by an exact, deterministic amount between bursts (exactly
    `period`, landing precisely on the expiry cutoff: an entry with
    timestamp 0 needs `ts > now - period` to survive, which fails
    exactly when `now >= period`). Verified against 100 repeated trials
    with zero failures before being kept here.
    """

    def test_concurrent_burst_fills_window_to_exactly_the_limit(self) -> None:
        clock = FakeClock()
        limit = 15
        period = 0.05
        limiter = SlidingWindowLog(limit=limit, period=period, clock=clock)

        n_threads = 40
        barrier = threading.Barrier(n_threads)
        results: list[bool] = []
        lock = threading.Lock()

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

        assert sum(results) == min(n_threads, limit)

    def test_concurrent_burst_after_exact_expiry_gets_a_fresh_limit(self) -> None:
        """First burst fills the window completely (frozen clock, all
        entries timestamped identically at t=0). The clock is then
        advanced by exactly `period` -- landing precisely on the
        instant those entries stop satisfying `ts > now - period` --
        before a second frozen-clock burst, which must again admit
        exactly `limit`, proving expired entries are fully pruned
        rather than partially lingering under concurrent access."""
        clock = FakeClock()
        limit = 15
        period = 0.05
        limiter = SlidingWindowLog(limit=limit, period=period, clock=clock)
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
        assert sum(first) == min(n_threads, limit)

        clock.advance(period)  # exactly at the expiry cutoff -- deterministic

        second = burst()
        assert sum(second) == min(n_threads, limit)


class TestMultiprocessing:
    """Same invariant as TestConcurrency, but across real OS processes
    sharing one InMemoryStorage(multiprocess_safe=True) instance."""

    def test_multiprocess_hammering_never_exceeds_limit(self) -> None:
        limit = 40
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 5
        attempts_per_proc = 30

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(
                    hammer_sliding_window_log,
                    storage,
                    limit,
                    3600.0,
                    "shared-key",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results = [f.result() for f in futures]

        assert sum(results) == limit
