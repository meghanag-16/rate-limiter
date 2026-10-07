"""Top-level worker functions for multiprocessing stress tests.

Why this module has to exist separately (not just defined inline in each
test function): `multiprocessing`'s `spawn` start method -- the default on
Windows, and on macOS since Python 3.8 -- re-imports the target module in
each child process and pickles every argument sent to the worker. Pickling
a function only works if it's a plain top-level function reachable by
`module.qualname` import, not a closure or a function nested inside a test.
A worker defined inside a `def test_...():` body will fail to pickle under
`spawn` with `AttributeError: Can't get local object ...` (this fails
differently, and often silently, under `fork`, which is why it's easy to
miss if you only test on Linux). So every worker used with
ProcessPoolExecutor lives here, at module level.

Each worker builds its OWN limiter instance inside the child process,
wrapping the SAME `InMemoryStorage(multiprocess_safe=True)` object passed
in from the parent (see limivault.storage module docstring for how that
storage instance survives being pickled across the process boundary).
`leak_rate=0` / `refill_rate=0` are used for the two algorithms that decay
over time, matching the existing single-process ThreadPoolExecutor stress
tests in test_leaky_bucket.py -- the point of these tests is exercising
storage-level cross-process locking, not clock-driven refill/leak
behavior, so removing time-decay from the equation keeps the expected
final count exact and deterministic.
"""

from __future__ import annotations

import time

from limivault.storage import StorageBackend


def hammer_fixed_window(
    storage: StorageBackend, limit: int, period: float, key: str, attempts: int
) -> int:
    from limivault.algorithms.fixed_window import FixedWindow

    limiter = FixedWindow(
        limit=limit, period=period, storage=storage, clock=time.monotonic
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))


def hammer_token_bucket(
    storage: StorageBackend,
    capacity: int,
    refill_rate: float,
    key: str,
    attempts: int,
) -> int:
    from limivault.algorithms.token_bucket import TokenBucket

    limiter = TokenBucket(
        capacity=capacity,
        refill_rate=refill_rate,
        storage=storage,
        clock=time.monotonic,
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))


def hammer_sliding_window_log(
    storage: StorageBackend, limit: int, period: float, key: str, attempts: int
) -> int:
    from limivault.algorithms.sliding_window_log import SlidingWindowLog

    limiter = SlidingWindowLog(
        limit=limit, period=period, storage=storage, clock=time.monotonic
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))


def hammer_sliding_window_counter(
    storage: StorageBackend, limit: int, period: float, key: str, attempts: int
) -> int:
    from limivault.algorithms.sliding_window_counter import SlidingWindowCounter

    limiter = SlidingWindowCounter(
        limit=limit, period=period, storage=storage, clock=time.monotonic
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))


def hammer_leaky_bucket_meter(
    storage: StorageBackend,
    capacity: int,
    leak_rate: float,
    key: str,
    attempts: int,
) -> int:
    from limivault.algorithms.leaky_bucket import LeakyBucketMeter

    limiter = LeakyBucketMeter(
        capacity=capacity,
        leak_rate=leak_rate,
        storage=storage,
        clock=time.monotonic,
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))


def hammer_leaky_bucket_queue(
    storage: StorageBackend,
    capacity: int,
    leak_rate: float,
    key: str,
    attempts: int,
) -> int:
    from limivault.algorithms.leaky_bucket import LeakyBucketQueue

    limiter = LeakyBucketQueue(
        capacity=capacity,
        leak_rate=leak_rate,
        storage=storage,
        clock=time.monotonic,
    )
    return sum(1 for _ in range(attempts) if limiter.allow(key))
