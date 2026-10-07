# tests/test_storage_lock_cleanup.py
"""Tests for InMemoryStorage / AsyncInMemoryStorage
opportunistic idle lock-sweep (see storage.py's module docstring for
the full design rationale -- these tests exist specifically to pin
down the DOCUMENTED, DELIBERATE defaults and their edges, not just the
happy path of "does the dict eventually shrink").

Uses small, explicit lock_sweep_interval / lock_idle_seconds values
(NOT the documented production defaults of 1000 / 300.0) so these
tests run fast and deterministically -- see
test_documented_default_values_are_the_stated_constants below for the
one test that pins the actual shipped defaults themselves, guarding
against them silently drifting from what the docstring claims.

A separate `clock` is injected into InMemoryStorage/AsyncInMemoryStorage
itself (independent of any algorithm-level clock -- see storage.py's
module docstring on this deliberate separation) so idle-time advancement
is fully deterministic in these tests, never real wall-clock time.
"""

from __future__ import annotations

import asyncio
import threading
from typing import cast

import pytest

from limivault.storage import (
    _DEFAULT_LOCK_IDLE_SECONDS,
    _DEFAULT_LOCK_SWEEP_INTERVAL,
    AsyncInMemoryStorage,
    InMemoryStorage,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _locks(storage: InMemoryStorage) -> dict[str, threading.Lock]:
    """Type-narrowing helper for tests: `InMemoryStorage._locks` is
    typed `dict[str, threading.Lock] | None` because it's `None` in
    multiprocess_safe mode (see storage.py). Every test in this file
    that inspects `_locks` directly only ever does so against a
    default-mode (multiprocess_safe=False) instance, where it is
    always a real dict -- this helper asserts that precondition once
    so call sites don't each need their own `assert ... is not None`
    under `mypy --strict`."""
    assert storage._locks is not None
    return storage._locks


# --- Documented defaults are pinned, not silently drifted -------------


def test_documented_default_values_are_the_stated_constants() -> None:
    """Guards storage.py's module docstring claim: sweep interval 1000
    calls, idle threshold 300.0 seconds. If either constant changes,
    this test forces the docstring to be updated in the same change."""
    assert _DEFAULT_LOCK_SWEEP_INTERVAL == 1000
    assert _DEFAULT_LOCK_IDLE_SECONDS == 300.0


def test_default_constructor_uses_the_documented_defaults() -> None:
    storage = InMemoryStorage()
    assert storage._lock_sweep_interval == _DEFAULT_LOCK_SWEEP_INTERVAL
    assert storage._lock_idle_seconds == _DEFAULT_LOCK_IDLE_SECONDS


# --- Constructor validation ---------------------------------------------


def test_zero_sweep_interval_raises() -> None:
    with pytest.raises(ValueError):
        InMemoryStorage(lock_sweep_interval=0)


def test_negative_sweep_interval_raises() -> None:
    with pytest.raises(ValueError):
        InMemoryStorage(lock_sweep_interval=-5)


def test_zero_idle_seconds_raises() -> None:
    with pytest.raises(ValueError):
        InMemoryStorage(lock_idle_seconds=0)


def test_negative_idle_seconds_raises() -> None:
    with pytest.raises(ValueError):
        InMemoryStorage(lock_idle_seconds=-1.0)


def test_async_zero_sweep_interval_raises() -> None:
    with pytest.raises(ValueError):
        AsyncInMemoryStorage(lock_sweep_interval=0)


def test_async_zero_idle_seconds_raises() -> None:
    with pytest.raises(ValueError):
        AsyncInMemoryStorage(lock_idle_seconds=0)


# --- Sync sweep behavior, edges around the configured thresholds -------


class TestSyncSweepEdges:
    def test_sweep_does_not_run_before_the_nth_call(self) -> None:
        """With sweep_interval=5, the first 4 calls must not trigger a
        sweep at all -- verified indirectly: an idle-eligible lock
        created on call 1 must still be present after calls 2-4, even
        though it's already past the idle threshold by then."""
        clock = FakeClock()
        storage = InMemoryStorage(
            lock_sweep_interval=5, lock_idle_seconds=10.0, clock=clock
        )
        storage.lock("stale-key")  # call 1 of 5
        clock.advance(20.0)  # already idle-eligible

        for _ in range(3):  # calls 2, 3, 4 -- still below the interval
            storage.lock("other-key")

        assert "stale-key" in _locks(storage)  # not yet swept

    def test_sweep_runs_on_the_nth_call_and_evicts_eligible_locks(self) -> None:
        clock = FakeClock()
        storage = InMemoryStorage(
            lock_sweep_interval=5, lock_idle_seconds=10.0, clock=clock
        )
        storage.lock("stale-key")  # call 1
        clock.advance(20.0)  # now well past the 10s idle threshold

        for _ in range(4):  # calls 2, 3, 4, 5 -- the 5th triggers a sweep
            storage.lock("other-key")

        assert "stale-key" not in _locks(storage)

    def test_lock_idle_but_under_threshold_survives_a_sweep(self) -> None:
        clock = FakeClock()
        storage = InMemoryStorage(
            lock_sweep_interval=3, lock_idle_seconds=10.0, clock=clock
        )
        storage.lock("fresh-key")  # call 1
        clock.advance(5.0)  # idle, but under the 10s threshold

        for _ in range(2):  # calls 2, 3 -- triggers a sweep on call 3
            storage.lock("other-key")

        assert "fresh-key" in _locks(storage)

    def test_held_lock_is_never_evicted_even_when_idle_timestamp_is_stale(
        self,
    ) -> None:
        """Core safety property: a lock currently held by another
        caller must survive a sweep regardless of how long ago it was
        last looked up, since the sweep's non-blocking acquire()
        attempt fails while it's held (see storage.py's ELIGIBILITY
        section)."""
        clock = FakeClock()
        storage = InMemoryStorage(
            lock_sweep_interval=2, lock_idle_seconds=1.0, clock=clock
        )
        # storage.lock() returns AbstractContextManager[bool] by the
        # StorageBackend interface; InMemoryStorage's default mode
        # concretely always returns a real threading.Lock, which is
        # what the direct .acquire()/.release() calls below need --
        # cast() asserts that known-concrete type for the test without
        # widening StorageBackend's public return type.
        held_lock = cast(threading.Lock, storage.lock("held-key"))  # call 1
        held_lock.acquire()
        try:
            clock.advance(100.0)  # far past the idle threshold
            storage.lock("other-key")  # call 2 -- triggers a sweep

            locks = _locks(storage)
            assert "held-key" in locks
            # And it's still the SAME lock object -- a second caller
            # for this key must contend with the one actually held,
            # not silently get a fresh, uncontended lock.
            assert locks["held-key"] is held_lock
        finally:
            held_lock.release()

    def test_unrelated_keys_are_not_evicted_by_a_sweep(self) -> None:
        """Sweeping must only ever remove idle+unlocked entries -- a
        fresh, actively-used key created moments before a sweep must
        never disappear."""
        clock = FakeClock()
        storage = InMemoryStorage(
            lock_sweep_interval=3, lock_idle_seconds=10.0, clock=clock
        )
        storage.lock("old-idle-key")
        clock.advance(20.0)
        storage.lock("brand-new-key")  # call 2, freshly stamped
        storage.lock("trigger-key")  # call 3 -- triggers the sweep

        locks = _locks(storage)
        assert "old-idle-key" not in locks
        assert "brand-new-key" in locks

    def test_multiprocess_safe_mode_is_unaffected_by_sweep_settings(self) -> None:
        """multiprocess_safe=True uses the fixed lock pool,
        which has no leak by construction -- sweep params are accepted
        but have no effect in this mode (see storage.py's module
        docstring)."""
        storage = InMemoryStorage(
            multiprocess_safe=True, lock_sweep_interval=1, lock_idle_seconds=0.001
        )
        lock1 = storage.lock("a")
        lock2 = storage.lock("a")
        assert lock1 is lock2  # same pool slot, never evicted/recreated


# --- Async sweep behavior, mirroring the sync edges above --------------


class TestAsyncSweepEdges:
    @pytest.mark.asyncio
    async def test_sweep_runs_on_the_nth_call_and_evicts_eligible_locks(
        self,
    ) -> None:
        clock = FakeClock()
        storage = AsyncInMemoryStorage(
            lock_sweep_interval=5, lock_idle_seconds=10.0, clock=clock
        )
        async with storage.lock("stale-key"):  # call 1
            pass
        clock.advance(20.0)

        for _ in range(4):
            async with storage.lock("other-key"):
                pass

        assert "stale-key" not in storage._locks

    @pytest.mark.asyncio
    async def test_lock_idle_but_under_threshold_survives_a_sweep(self) -> None:
        clock = FakeClock()
        storage = AsyncInMemoryStorage(
            lock_sweep_interval=3, lock_idle_seconds=10.0, clock=clock
        )
        async with storage.lock("fresh-key"):
            pass
        clock.advance(5.0)

        for _ in range(2):
            async with storage.lock("other-key"):
                pass

        assert "fresh-key" in storage._locks

    @pytest.mark.asyncio
    async def test_held_lock_is_never_evicted_while_a_coroutine_holds_it(
        self,
    ) -> None:
        clock = FakeClock()
        storage = AsyncInMemoryStorage(
            lock_sweep_interval=2, lock_idle_seconds=1.0, clock=clock
        )
        held_event = asyncio.Event()
        release_event = asyncio.Event()

        async def holder() -> None:
            async with storage.lock("held-key"):
                held_event.set()
                await release_event.wait()

        task = asyncio.create_task(holder())
        await held_event.wait()

        clock.advance(100.0)
        async with storage.lock("other-key"):  # triggers a sweep
            pass

        assert "held-key" in storage._locks

        release_event.set()
        await task
