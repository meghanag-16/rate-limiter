# benchmarks/bench_common.py
"""Shared factory table for all 12 concrete limivault algorithm classes
(6 sync + 6 async), reused by the earlier per-call benchmark prototype
(pytest-benchmark timing) and memory_footprint.py (tracemalloc-based
memory measurement).

DESIGN NOTE: every factory uses a deliberately huge limit/capacity
(10**9) so that allow() calls are essentially always admitted during a
benchmark run. This is a conscious choice, not an oversight: the cost
of a single allow() call is dominated by the lock acquire + storage
get/set + arithmetic, and that cost is very similar whether the call
is ultimately allowed or denied (both paths do the same read-compute-
write work under the lock) -- see each algorithm file's allow()
implementation. Using a huge limit avoids the benchmark's throughput
number being an artifact of how often the deny early-exit happens to
trigger, and keeps every algorithm's benchmark representative of
"steady, normal traffic" rather than "mostly-denied traffic."

This mirrors the same factory-table pattern already used in
tests/test_metrics_all_algorithms.py and tests/test_cost_contract.py
(one small factory per concrete class, looped over rather than
duplicated file-by-file).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Tuple

from limivault.logging import configure_logging

# MUST run before any `limivault.algorithms.*` import below. Each
# algorithm module calls `get_logger(__name__).bind(...)` at its own
# import time (see e.g. fixed_window.py's `_log = get_logger(__name__)`
# at module level), and structlog's bind() resolves/caches the active
# processor chain at bind-time, not at each later log-call time --
# verified directly: a logger bound before configure_logging() runs
# keeps emitting DEBUG lines even after a later configure_logging()
# call, because that later call only affects loggers bound after it.
# So this must be the first limivault import in this file, ahead of the
# algorithm imports below, or DEBUG decision
# logging would flood stdout during throughput/memory benchmark runs.
configure_logging(level=logging.WARNING)

from limivault.algorithms.async_fixed_window import AsyncFixedWindow  # noqa: E402
from limivault.algorithms.async_leaky_bucket import (  # noqa: E402
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from limivault.algorithms.async_sliding_window_counter import (  # noqa: E402
    AsyncSlidingWindowCounter,
)
from limivault.algorithms.async_sliding_window_log import (  # noqa: E402
    AsyncSlidingWindowLog,
)
from limivault.algorithms.async_token_bucket import AsyncTokenBucket  # noqa: E402
from limivault.algorithms.fixed_window import FixedWindow  # noqa: E402
from limivault.algorithms.leaky_bucket import (  # noqa: E402
    LeakyBucketMeter,
    LeakyBucketQueue,
)
from limivault.algorithms.sliding_window_counter import SlidingWindowCounter  # noqa: E402
from limivault.algorithms.sliding_window_log import SlidingWindowLog  # noqa: E402
from limivault.algorithms.token_bucket import TokenBucket  # noqa: E402
from limivault.storage import AsyncInMemoryStorage, InMemoryStorage  # noqa: E402

_HUGE = 10**9

# Each entry: (display_name, is_async, zero-arg factory returning a
# fresh limiter with a FRESH InMemoryStorage/AsyncInMemoryStorage --
# fresh storage per limiter matters for the memory-footprint script,
# which needs to measure one limiter's storage growth in isolation.

SYNC_FACTORIES: List[Tuple[str, Callable[[], Any]]] = [
    (
        "FixedWindow",
        lambda: FixedWindow(
            limit=_HUGE, period=3600.0, storage=InMemoryStorage(), clock=time.monotonic
        ),
    ),
    (
        "TokenBucket",
        lambda: TokenBucket(
            capacity=_HUGE,
            refill_rate=1.0,
            storage=InMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "SlidingWindowLog",
        lambda: SlidingWindowLog(
            limit=_HUGE, period=3600.0, storage=InMemoryStorage(), clock=time.monotonic
        ),
    ),
    (
        "SlidingWindowCounter",
        lambda: SlidingWindowCounter(
            limit=_HUGE, period=3600.0, storage=InMemoryStorage(), clock=time.monotonic
        ),
    ),
    (
        "LeakyBucketMeter",
        lambda: LeakyBucketMeter(
            capacity=_HUGE,
            leak_rate=1.0,
            storage=InMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "LeakyBucketQueue",
        lambda: LeakyBucketQueue(
            capacity=_HUGE,
            leak_rate=1.0,
            storage=InMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
]

ASYNC_FACTORIES: List[Tuple[str, Callable[[], Any]]] = [
    (
        "AsyncFixedWindow",
        lambda: AsyncFixedWindow(
            limit=_HUGE,
            period=3600.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "AsyncTokenBucket",
        lambda: AsyncTokenBucket(
            capacity=_HUGE,
            refill_rate=1.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "AsyncSlidingWindowLog",
        lambda: AsyncSlidingWindowLog(
            limit=_HUGE,
            period=3600.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "AsyncSlidingWindowCounter",
        lambda: AsyncSlidingWindowCounter(
            limit=_HUGE,
            period=3600.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "AsyncLeakyBucketMeter",
        lambda: AsyncLeakyBucketMeter(
            capacity=_HUGE,
            leak_rate=1.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
    (
        "AsyncLeakyBucketQueue",
        lambda: AsyncLeakyBucketQueue(
            capacity=_HUGE,
            leak_rate=1.0,
            storage=AsyncInMemoryStorage(),
            clock=time.monotonic,
        ),
    ),
]


def all_factory_names() -> List[str]:
    return [n for n, _ in SYNC_FACTORIES] + [n for n, _ in ASYNC_FACTORIES]
