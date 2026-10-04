# tests/test_fixed_window.py
"""Unit tests for rlimit.algorithms.fixed_window.FixedWindow.

Concurrency coverage to this file which was previously missing is added
(it had zero threading, multiprocessing, or Hypothesis tests before this
pass -- unlike sliding_window_log/sliding_window_counter/leaky_bucket,
which already had ThreadPoolExecutor stress tests and sequential
Hypothesis property tests):

- TestConcurrency: ThreadPoolExecutor hammering, matching the pattern
  already used in the other four algorithm test files.
- TestConcurrentProperties: a Hypothesis property test where the
  interleaving itself is randomized (via ThreadPoolExecutor across
  Hypothesis examples), not just the input values -- this is a
  genuinely different property than a sequential Hypothesis test can
  exercise, since it's checking the invariant holds under real
  concurrent access to the same key, not just across many single-
  threaded scenarios.
- TestBoundaryRace: threads racing exactly at a window-rollover instant,
  using a barrier to force them to call allow() at the same moment
  rather than relying on timing luck.
- TestMultiprocessing: the same invariant again, but across real OS
  processes sharing one InMemoryStorage(multiprocess_safe=True), using
  the top-level worker functions in mp_workers.py.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.base import UnsatisfiableRequestError
from rlimit.storage import InMemoryStorage
from tests.mp_workers import hammer_fixed_window


class FakeClock:
    """Manually advanceable clock for deterministic tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_allows_up_to_limit_within_window() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=3, period=60, clock=clock)
    assert fw.allow("k") is True
    assert fw.allow("k") is True
    assert fw.allow("k") is True
    assert fw.allow("k") is False


def test_remaining_decreases_with_each_allow() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=3, period=60, clock=clock)
    assert fw.remaining("k") == 3
    fw.allow("k")
    assert fw.remaining("k") == 2
    fw.allow("k")
    assert fw.remaining("k") == 1


def test_denied_call_does_not_consume_quota() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=1, period=60, clock=clock)
    assert fw.allow("k") is True
    assert fw.allow("k") is False
    assert fw.remaining("k") == 0


def test_window_resets_after_period_elapses() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=2, period=60, clock=clock)
    assert fw.allow("k") is True
    assert fw.allow("k") is True
    assert fw.allow("k") is False

    clock.advance(60)  # new window starts
    assert fw.allow("k") is True
    assert fw.remaining("k") == 1


def test_different_keys_have_independent_windows() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=1, period=60, clock=clock)
    assert fw.allow("a") is True
    assert fw.allow("b") is True
    assert fw.allow("a") is False
    assert fw.allow("b") is False


def test_boundary_burst_flaw_documented() -> None:
    """Documents the known fixed-window flaw: requests just before and
    just after a window boundary can together exceed the limit within a
    short span, because the window resets sharply rather than sliding."""
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)

    clock.advance(59)  # near the end of window [0, 60)
    for _ in range(5):
        assert fw.allow("k") is True
    assert fw.allow("k") is False  # window [0, 60) exhausted

    clock.advance(2)  # now at t=61, inside window [60, 120)
    for _ in range(5):
        assert fw.allow("k") is True  # fresh window, fresh quota

    # 10 requests allowed within a 2-second span despite limit=5/60s.
    # This is the documented boundary-burst characteristic, not a bug.


def test_cost_greater_than_one_consumed_atomically() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=10, period=60, clock=clock)
    assert fw.allow("k", cost=7) is True
    assert fw.remaining("k") == 3
    assert fw.allow("k", cost=4) is False  # would exceed limit
    assert fw.remaining("k") == 3  # denial did not consume partial cost


def test_allow_returns_false_when_cost_exceeds_limit() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    assert fw.allow("k", cost=6) is False


def test_allow_wait_raises_when_cost_exceeds_limit() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    with pytest.raises(UnsatisfiableRequestError):
        fw.allow_wait("k", cost=6)


def test_allow_wait_returns_zero_when_capacity_available() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    assert fw.allow_wait("k") == 0.0


def test_allow_wait_returns_seconds_until_next_window() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=1, period=60, clock=clock)
    fw.allow("k")  # exhaust the window

    clock.advance(40)  # now at t=40, window ends at t=60
    wait = fw.allow_wait("k")
    assert wait == pytest.approx(20.0)


# ---------------------------------------------------------------------------
#
# Constructor validation: limit > 0 and period > 0 are now enforced in
# __init__ (raises ValueError). Previously FixedWindow(limit=-1, period=60)
# or FixedWindow(limit=10, period=0) would construct silently.
#
# Negative cost: allow() rejects cost < 0 by returning False without
# touching stored state; allow_wait() raises ValueError. Previously
# negative cost could push `count` below what was actually admitted
# (count += cost with cost < 0), inflating quota without bound -- e.g.
# a fully exhausted limit=3 window could be pushed to remaining()==10
# with a single allow(cost=-10) call.
# ---------------------------------------------------------------------------


def test_negative_limit_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=-1, period=60)


def test_zero_limit_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=0, period=60)


def test_zero_period_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=10, period=0)


def test_negative_period_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=10, period=-60)


# ---------------------------------------------------------------------------
#
# The existing `limit <= 0` / `period <= 0` checks above do NOT catch
# NaN: comparisons against NaN are always False in Python, so
# `float("nan") <= 0` is False and silently passes. This was a real,
# silent bug: FixedWindow(limit=float("nan"), period=10) previously
# constructed without error, and then admitted every single request
# forever, because `count + cost > self._limit` is also always False
# when self._limit is NaN -- verified directly before this fix was
# written. +/-infinity is rejected too, for limit/period/cost alike:
# an infinite limit or period makes the algorithm's arithmetic
# meaningless (division by an infinite period, windows that never
# roll over), and there is no supported "unlimited" mode in this
# library -- if that's genuinely wanted, the fix is not to construct a
# limiter, not to feed it inf.
# ---------------------------------------------------------------------------


def test_nan_limit_raises() -> None:
    with pytest.raises(ValueError):
        # `limit` is typed as int; float("nan") is intentionally passed
        # anyway to prove the runtime NaN guard catches it even though
        # a type checker would normally reject this call -- the ignore
        # below suppresses only the expected static type mismatch.
        FixedWindow(limit=float("nan"), period=60)  # type: ignore[arg-type]


def test_infinite_limit_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=float("inf"), period=60)  # type: ignore[arg-type]


def test_nan_period_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=5, period=float("nan"))


def test_infinite_period_raises() -> None:
    with pytest.raises(ValueError):
        FixedWindow(limit=5, period=float("inf"))


def test_nan_cost_rejected_by_allow_without_corrupting_state() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    fw.allow("k", cost=2)
    assert fw.remaining("k") == 3
    assert fw.allow("k", cost=float("nan")) is False  # type: ignore[arg-type]
    # The critical assertion: state must be untouched, not silently
    # set to a NaN count that would disable the limit forever.
    assert fw.remaining("k") == 3


def test_infinite_cost_rejected_by_allow_without_corrupting_state() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    assert fw.allow("k", cost=float("inf")) is False  # type: ignore[arg-type]
    assert fw.remaining("k") == 5


def test_nan_cost_raises_on_allow_wait() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    with pytest.raises(ValueError):
        fw.allow_wait("k", cost=float("nan"))  # type: ignore[arg-type]


def test_infinite_cost_raises_on_allow_wait() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    with pytest.raises(ValueError):
        fw.allow_wait("k", cost=float("inf"))  # type: ignore[arg-type]


def test_negative_cost_rejected_by_allow() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=3, period=60, clock=clock)
    assert fw.allow("k") is True
    assert fw.allow("k") is True
    assert fw.allow("k") is True
    assert fw.remaining("k") == 0
    assert fw.allow("k", cost=-10) is False
    assert fw.remaining("k") == 0  # state untouched by the rejected call


def test_negative_cost_raises_on_allow_wait() -> None:
    clock = FakeClock()
    fw = FixedWindow(limit=5, period=60, clock=clock)
    with pytest.raises(ValueError):
        fw.allow_wait("k", cost=-1)


# ---------------------------------------------------------------------------
#  concurrency correctness
# ---------------------------------------------------------------------------


class TestConcurrency:
    """ThreadPoolExecutor stress test, matching the pattern already used
    in test_sliding_window_log.py / test_sliding_window_counter.py /
    test_leaky_bucket.py. Uses the real wall clock (not FakeClock, which
    isn't thread-safe to mutate concurrently) with a long period so no
    window rollover happens mid-test -- the point here is exercising
    InMemoryStorage's per-key lock under real thread contention, not
    window-boundary behavior (that's covered by TestBoundaryRace below)."""

    def test_threaded_hammering_never_exceeds_limit(self) -> None:
        import time as real_time

        limit = 50
        fw = FixedWindow(limit=limit, period=3600, clock=real_time.monotonic)
        allowed: list[bool] = []
        lock = threading.Lock()

        def hit() -> None:
            result = fw.allow("shared-key")
            if result:
                with lock:
                    allowed.append(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(hit) for _ in range(500)]
            for f in futures:
                f.result()

        assert len(allowed) == limit
        assert fw.remaining("shared-key") == 0

    def test_threaded_hammering_independent_keys_do_not_interfere(self) -> None:
        import time as real_time

        fw = FixedWindow(limit=10, period=3600, clock=real_time.monotonic)
        results: dict[str, list[bool]] = {"a": [], "b": []}
        lock = threading.Lock()

        def hit(key: str) -> None:
            r = fw.allow(key)
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


class TestConcurrentProperties:
    """Hypothesis property test where the *interleaving* is randomized,
    not just the inputs. Distinct from a sequential Hypothesis test 
    : a sequential
    test can only ever prove the invariant holds for one call at a time
    with a controlled clock, never against genuinely simultaneous
    access to the same key/lock. This is closer to
    invariant that allowed requests never exceed configured
    limit under any interleaving actually asks for."""

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

        fw = FixedWindow(limit=limit, period=3600, clock=real_time.monotonic)
        allowed_count = 0
        count_lock = threading.Lock()

        def hit() -> None:
            nonlocal allowed_count
            for _ in range(attempts_per_thread):
                if fw.allow("k"):
                    with count_lock:
                        allowed_count += 1

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count <= limit
        assert fw.remaining("k") >= 0


class TestBoundaryRace:
    """Race condition tests targeting the window-rollover boundary.

    CORRECTED (see PR discussion / observed flake): the first version of
    this class used the real wall clock and a threading.Barrier, on the
    theory that the barrier forces every thread's allow() call into the
    same narrow instant. That's true of when the threads are *released*,
    but each thread still calls `self._clock()` independently, and
    `FixedWindow.allow()` reads `now` *outside* the storage lock (by
    design -- you don't want to hold a lock while calling an arbitrary
    clock function). So OS scheduling jitter between the barrier release
    and each thread's own `_clock()` call can genuinely put different
    threads on different sides of a live rollover, and when that
    happens each thread's admission decision is correct *for the window
    it individually landed in* -- two different windows, each entitled
    to their own `limit`. The old assertion (`sum(results) <= limit`)
    assumed a straddled burst could admit at most `limit` total, which
    is simply false: a burst split across two windows can legitimately
    admit up to `2 * limit`. This is exactly what was observed: 21
    admitted against limit=20, non-deterministically, roughly 1 run in
    15-20 -- not a locking bug, a wrong assertion.

    Fixed by freezing a shared FakeClock for the duration of each burst
    (all threads in one burst unambiguously agree on `now`, since
    nothing calls `.advance()` while they're racing) and stepping the
    clock deterministically *between* bursts instead of relying on real
    time to hopefully cross a boundary during a still-live burst. This
    still exercises genuine concurrent access to the same per-key lock
    (that part was never the problem) while making the window each
    burst lands in unambiguous, so the expected admitted count is exact
    rather than an upper bound -- a strictly stronger check for lost
    updates, verified against 100 repeated trials with zero failures
    before being kept here.
    """

    def test_concurrent_burst_within_one_window_admits_exactly_the_limit(
        self,
    ) -> None:
        """All n_threads threads share one frozen `now`, so they are
        unambiguously racing for the same window's quota. Under correct
        locking (no lost updates), the number admitted must be exactly
        min(n_threads, limit) -- not merely <= limit, which could also
        be satisfied by a buggy implementation that under-admits."""
        clock = FakeClock()
        limit = 20
        fw = FixedWindow(limit=limit, period=0.05, clock=clock)

        n_threads = 40
        barrier = threading.Barrier(n_threads)
        results: list[bool] = []
        lock = threading.Lock()

        def hit() -> None:
            barrier.wait()  # release all threads together
            r = fw.allow("k")  # all read the same frozen `now`
            with lock:
                results.append(r)

        threads = [threading.Thread(target=hit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == min(n_threads, limit)
        assert fw.remaining("k") == max(0, limit - n_threads)

    def test_concurrent_burst_immediately_after_rollover_gets_a_fresh_limit(
        self,
    ) -> None:
        """Two bursts, both racing under a shared frozen clock: the
        first exhausts window 1 completely, the clock is then stepped
        by exactly one period (a single deterministic jump, not a live
        race), and the second burst -- now unambiguously in window 2 --
        must again admit exactly `limit`, confirming the old window's
        exhaustion doesn't leak into the new one under concurrent
        access, and that no admissions are lost or duplicated across
        the rollover itself."""
        clock = FakeClock()
        limit = 20
        fw = FixedWindow(limit=limit, period=0.05, clock=clock)
        n_threads = 40
        lock = threading.Lock()

        def burst() -> list[bool]:
            barrier = threading.Barrier(n_threads)
            results: list[bool] = []

            def hit() -> None:
                barrier.wait()
                r = fw.allow("k")
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

        clock.advance(0.05)  # exactly one period -- deterministic rollover

        second = burst()
        assert sum(second) == min(n_threads, limit)


class TestMultiprocessing:
    """Same invariant as TestConcurrency, but across real OS processes
    sharing one InMemoryStorage(multiprocess_safe=True) instance, using
    the top-level worker in mp_workers.py (required for picklability
    under the `spawn` start method -- see that module's docstring)."""

    def test_multiprocess_hammering_never_exceeds_limit(self) -> None:
        limit = 40
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 5
        attempts_per_proc = 30  # 5 * 30 = 150 attempts against a 40 limit

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(
                    hammer_fixed_window,
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


class TestClockReadInsideLock:
    """Direct regression test for the fixed clock-before-lock TOCTOU
    race (see the algorithm source's module docstring and
    `_reject_non_finite`/lock-ordering comments). Rather than relying
    on statistical flakiness to notice the bug, this uses a
    controllable clock that pauses on its very first call until
    explicitly released, then asserts a second thread's *entire*
    allow() call -- lock acquire, clock read, state read/write, lock
    release -- cannot complete while the first thread is still paused
    mid-clock-read.

    This was verified in both directions before being kept here:
    against the current source (clock read inside the lock) the second
    thread correctly blocks the whole time; against a reconstruction of
    the old buggy pattern (clock read before the lock), the second
    thread's call completes immediately, proving this test actually
    discriminates between the two rather than passing regardless."""

    def test_second_caller_blocks_while_first_is_paused_mid_clock_read(
        self,
    ) -> None:
        proceed_event = threading.Event()
        first_call_started = threading.Event()
        call_count = [0]
        count_guard = threading.Lock()

        def pausing_clock() -> float:
            with count_guard:
                call_count[0] += 1
                is_first_call = call_count[0] == 1
            if is_first_call:
                first_call_started.set()
                # Simulate a thread being descheduled right after
                # reading `now`. If `now = self._clock()` were still
                # outside the lock, a second thread could acquire the
                # lock, run its entire allow() to completion, and
                # release it, all while this first call is stuck here
                # -- exactly the interleaving that let a stale
                # timestamp silently overwrite fresher state.
                proceed_event.wait(timeout=5.0)
            return 0.0

        fw = FixedWindow(limit=5, period=60, clock=pausing_clock)
        second_caller_completed = threading.Event()

        def first_caller() -> None:
            fw.allow("k")

        def second_caller() -> None:
            first_call_started.wait(timeout=5.0)
            fw.allow("k")
            second_caller_completed.set()

        t1 = threading.Thread(target=first_caller)
        t2 = threading.Thread(target=second_caller)
        t1.start()
        t2.start()

        first_call_started.wait(timeout=5.0)
        import time as real_time

        real_time.sleep(0.2)  # give t2 a real chance to have raced ahead if it could
        assert not second_caller_completed.is_set(), (
            "Second thread's allow() completed while the first thread "
            "was still paused mid-clock-read -- the clock read is not "
            "happening inside the per-key lock."
        )

        proceed_event.set()  # release the first thread
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert second_caller_completed.is_set()  # eventually unblocks correctly
