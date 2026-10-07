# tests/test_storage.py
"""Unit tests for limivault.storage.InMemoryStorage.

Added multiprocess_safe=True to InMemoryStorage (see the module
docstring in storage.py for the full design rationale: a fixed pool of
pre-created Manager locks, indexed by a process-stable hash, because a
Manager object itself can't be pickled to a worker process and so new
per-key locks can't be minted on demand once workers exist). The tests
below exercise that mode specifically: that it still behaves correctly
in-process (round trip, independent keys), that it actually enforces
mutual exclusion *across real OS processes* (not just across threads),
and that the default (multiprocess_safe=False) behavior is completely
unchanged .
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ProcessPoolExecutor

from limivault.storage import InMemoryStorage
from tests.mp_workers import hammer_fixed_window


def test_get_missing_key_returns_none() -> None:
    s = InMemoryStorage()
    assert s.get("missing") is None


def test_set_then_get_round_trip() -> None:
    s = InMemoryStorage()
    s.set("user1", {"tokens": 5})
    assert s.get("user1") == {"tokens": 5}


def test_set_overwrites_existing_state() -> None:
    s = InMemoryStorage()
    s.set("user1", {"tokens": 5})
    s.set("user1", {"tokens": 2})
    assert s.get("user1") == {"tokens": 2}


def test_different_keys_are_independent() -> None:
    s = InMemoryStorage()
    s.set("a", {"tokens": 1})
    s.set("b", {"tokens": 2})
    assert s.get("a") == {"tokens": 1}
    assert s.get("b") == {"tokens": 2}


def test_lock_is_reentrant_safe_context_manager() -> None:
    """The lock() context manager can be entered and exited via `with`."""
    s = InMemoryStorage()
    with s.lock("user1"):
        pass  # if this doesn't raise, basic protocol works


def test_lock_same_key_returns_same_lock_object() -> None:
    """Repeated calls to lock() for the same key reuse one underlying lock."""
    s = InMemoryStorage()
    lock1 = s.lock("user1")
    lock2 = s.lock("user1")
    assert lock1 is lock2


def test_lock_blocks_same_key_across_threads() -> None:
    """Two threads locking the SAME key must serialize (~0.4s total)."""
    s = InMemoryStorage()
    results: list[str] = []

    def hold() -> None:
        with s.lock("shared"):
            time.sleep(0.2)
            results.append("done")

    start = time.time()
    t1 = threading.Thread(target=hold)
    t2 = threading.Thread(target=hold)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start

    assert len(results) == 2
    assert elapsed >= 0.35  # should be ~0.4s, not ~0.2s


def test_lock_does_not_block_different_keys() -> None:
    """Two threads locking DIFFERENT keys must run in parallel (~0.2s total)."""
    s = InMemoryStorage()
    results: list[str] = []

    def hold(key: str) -> None:
        with s.lock(key):
            time.sleep(0.2)
            results.append(key)

    start = time.time()
    t1 = threading.Thread(target=hold, args=("a",))
    t2 = threading.Thread(target=hold, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start

    assert set(results) == {"a", "b"}
    assert elapsed < 0.35  # should be ~0.2s, not ~0.4s


def test_lock_concurrent_creation_same_key_no_duplicate_locks() -> None:
    """Many threads racing to create a lock for the same new key must
    all end up with the identical Lock object (tests the _locks_guard)."""
    s = InMemoryStorage()
    seen: list[int] = []
    barrier = threading.Barrier(10)

    def grab() -> None:
        barrier.wait()
        seen.append(id(s.lock("new_key")))

    threads = [threading.Thread(target=grab) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen)) == 1  # all threads got the same lock object


# ---------------------------------------------------------------------------
# multiprocess_safe=True
# ---------------------------------------------------------------------------


class TestMultiprocessSafeDefaults:
    def test_default_constructor_is_not_multiprocess_safe(self) -> None:
        # Unchanged default -- existing single-process/threaded callers
        # get exactly the backend, no IPC overhead.
        s = InMemoryStorage()
        assert s._multiprocess_safe is False

    def test_multiprocess_safe_flag_is_opt_in(self) -> None:
        s = InMemoryStorage(multiprocess_safe=True)
        assert s._multiprocess_safe is True

    def test_zero_or_negative_pool_size_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            InMemoryStorage(multiprocess_safe=True, mp_lock_pool_size=0)
        with pytest.raises(ValueError):
            InMemoryStorage(multiprocess_safe=True, mp_lock_pool_size=-3)


class TestMultiprocessSafeInProcessBehavior:
    """multiprocess_safe=True must still behave correctly used from a
    single process -- these mirror the plain-mode tests above but
    against a Manager-backed instance."""

    def test_set_then_get_round_trip(self) -> None:
        s = InMemoryStorage(multiprocess_safe=True)
        s.set("user1", {"tokens": 5})
        assert s.get("user1") == {"tokens": 5}

    def test_different_keys_are_independent(self) -> None:
        s = InMemoryStorage(multiprocess_safe=True)
        s.set("a", {"tokens": 1})
        s.set("b", {"tokens": 2})
        assert s.get("a") == {"tokens": 1}
        assert s.get("b") == {"tokens": 2}

    def test_lock_is_usable_as_context_manager(self) -> None:
        s = InMemoryStorage(multiprocess_safe=True)
        with s.lock("user1"):
            pass


class TestMultiprocessSafePickling:
    """The specific failure mode this backend exists to avoid: pickling
    the whole InMemoryStorage instance (as happens implicitly whenever
    it's passed as an argument to a ProcessPoolExecutor worker) must not
    try to pickle the underlying Manager object."""

    def test_instance_is_picklable(self) -> None:
        import pickle

        s = InMemoryStorage(multiprocess_safe=True)
        s.set("k", {"count": 1})
        blob = pickle.dumps(s)
        restored: InMemoryStorage = pickle.loads(blob)
        # The restored copy still talks to the SAME manager-backed dict,
        # since DictProxy reconnects to the manager's server process --
        # it isn't a disconnected local copy.
        assert restored.get("k") == {"count": 1}

    def test_plain_mode_instance_is_not_picklable(self) -> None:
        # Plain mode (multiprocess_safe=False) holds real threading.Lock
        # objects in _locks/_locks_guard, which were never picklable and
        # shouldn't be -- that mode is intentionally process-local.
        #  __getstate__ only strips `_manager` (the thing
        # that's unpicklable *and* unneeded after construction in
        # multiprocess_safe mode); it doesn't and shouldn't paper over
        # plain mode's threading.Lock objects, since pretending those
        # could safely cross a process boundary would be actively wrong
        # -- a lock unpickled in another process is a distinct lock that
        # provides no real mutual exclusion with the original.
        import pickle

        import pytest

        s = InMemoryStorage()
        s.set("k", {"count": 1})
        with pytest.raises(TypeError):
            pickle.dumps(s)


class TestMultiprocessSafeCrossProcess:
    """The real point of this mode: correctness under genuine OS-process
    concurrency, not just threads. Uses ProcessPoolExecutor (the same
    tool the algorithm-level multiprocessing stress tests use)
    and the FixedWindow algorithm as a simple, well-understood vehicle
    for hammering the storage layer -- the assertion is about the
    storage's locking, not about FixedWindow itself (that algorithm is
    covered on its own terms in test_fixed_window.py).
    """

    def test_shared_storage_enforces_exact_limit_across_processes(self) -> None:
        limit = 60
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 6
        attempts_per_proc = 40  # 6 * 40 = 240 total attempts against a 60 limit

        with ProcessPoolExecutor(max_workers=n_procs) as ex:
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

    def test_independent_keys_do_not_interfere_across_processes(self) -> None:
        limit = 20
        storage = InMemoryStorage(multiprocess_safe=True)
        n_procs = 4
        attempts_per_proc = 10  # 4 * 10 = 40 attempts per key against a 20 limit

        with ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures_a = [
                ex.submit(
                    hammer_fixed_window,
                    storage,
                    limit,
                    3600.0,
                    "key-a",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            futures_b = [
                ex.submit(
                    hammer_fixed_window,
                    storage,
                    limit,
                    3600.0,
                    "key-b",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results_a = [f.result() for f in futures_a]
            results_b = [f.result() for f in futures_b]

        assert sum(results_a) == limit
        assert sum(results_b) == limit


class TestMultiprocessLockPoolCollisions:
    """Correction: multiprocess_safe mode's fixed
    lock pool (see storage.py's module docstring) means two different
    keys CAN map to the same pool slot -- documented internally as a
    tradeoff, but never actually exercised by a test that forces a real
    collision. The other cross-process tests above use the default
    pool size (64) with a couple of arbitrary key names, which may or
    may not have ever actually collided in any given run.

    `mp_lock_pool_size=1` forces every key onto the single lock --
    the worst case, not a probabilistic one -- so these tests verify
    correctness under a *guaranteed* collision rather than an
    incidental one.

    Scope note: these tests are about correctness, not performance.
    With pool_size=1 there genuinely is no independence between keys'
    locks (every key serializes against every other key) -- that's the
    documented tradeoff of multiprocess_safe mode, and these tests are
    not claiming per-key lock independence holds under collision, only
    that shared-slot access still produces correct admitted counts.
    """

    def test_single_key_stays_correct_under_forced_single_slot(self) -> None:
        limit = 30
        storage = InMemoryStorage(multiprocess_safe=True, mp_lock_pool_size=1)
        n_procs = 4
        attempts_per_proc = 20  # 4 * 20 = 80 attempts against a 30 limit

        with ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(
                    hammer_fixed_window,
                    storage,
                    limit,
                    3600.0,
                    "key-a",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results = [f.result() for f in futures]

        assert sum(results) == limit

    def test_two_different_keys_stay_independent_despite_sharing_one_slot(
        self,
    ) -> None:
        """key-a and key-b are FORCED to share the same lock slot
        (pool_size=1), but each key's own quota accounting must still
        be fully correct and independent of the other -- the shared
        lock affects contention/throughput only, never correctness."""
        limit = 20
        storage = InMemoryStorage(multiprocess_safe=True, mp_lock_pool_size=1)
        n_procs = 4
        attempts_per_proc = 10  # 4 * 10 = 40 attempts per key against a 20 limit

        with ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures_a = [
                ex.submit(
                    hammer_fixed_window,
                    storage,
                    limit,
                    3600.0,
                    "key-a",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            futures_b = [
                ex.submit(
                    hammer_fixed_window,
                    storage,
                    limit,
                    3600.0,
                    "key-b",
                    attempts_per_proc,
                )
                for _ in range(n_procs)
            ]
            results_a = [f.result() for f in futures_a]
            results_b = [f.result() for f in futures_b]

        assert sum(results_a) == limit
        assert sum(results_b) == limit
