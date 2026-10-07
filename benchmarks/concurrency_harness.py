# benchmarks/concurrency_harness.py
"""Core engine for the async concurrency benchmark (replaces the
earlier per-call pytest-benchmark harness, which only measured a single
`await limiter.allow()` in isolation and said nothing about throughput,
tail latency, or behavior under real concurrency. This harness follows
the design discussion that replaced that prototype.

This module has no CLI and no __main__ -- it's imported by
benchmark_async_concurrency.py, which owns argument parsing, the
matrix sweep, and reporting. Kept separate so the actual measurement
logic (task orchestration, latency collection, percentile math) is
independently testable/reusable without dragging in argparse.

--------------------------------------------------------------------
DESIGN DECISIONS
--------------------------------------------------------------------

1. Per-task latency collection, no shared counters. Each concurrent
   task accumulates its OWN list of latencies and its OWN allowed/
   denied/error counts locally, returning them for the caller to merge
   via asyncio.gather()'s return values. This avoids any question of
   whether incrementing a shared int or appending to a shared list
   across coroutines needs a lock -- it doesn't need one this way, by
   construction, rather than by reasoning about asyncio's single-
   threaded scheduling guarantees at each mutation site.

2. Timing uses time.perf_counter_ns(), never the limiter's injected
   `clock`. The limiter's clock (real time.monotonic in this harness,
   or a SimulationClock elsewhere in this project) drives the
   ALGORITHM's admit/deny math; it has nothing to do with how long an
   `await limiter.allow()` call actually took on the wall clock. Wall-
   clock benchmark timing and the algorithm's own notion of "now" are
   two separate concerns and are never the same clock here -- see this
   project's simulation.py module docstring for the same
   distinction made in the other direction (SimulationClock exists
   FOR the algorithm, not for timing it).

3. Key distribution is precomputed and deterministic, not randomized
   per call. For a matrix cell with `concurrency` tasks each issuing
   `requests_per_task` requests against `num_keys` distinct keys, the
   full flattened sequence of `concurrency * requests_per_task` keys
   is built once as `key_index % num_keys` (round-robin), then split
   into `concurrency` contiguous chunks, one per task. `num_keys=1`
   naturally produces the hot-key/high-contention workload (every task
   hits the same key); `num_keys=100` or `num_keys=10000` naturally
   produces the many-key workload (state creation, storage growth,
   per-key lock creation). This is deterministic specifically so a
   given (concurrency, num_keys, requests_per_task) combination always
   exercises the same key-access pattern across repetitions and across
   algorithms -- repetitions vary in wall-clock outcome, not in which
   keys were hit.

4. Warm-up runs against a SEPARATE key namespace, on the SAME limiter/
   storage/connection. The point of warm-up (per the design this
   responds to) is to absorb one-time costs -- Python bytecode
   compilation on first call, Redis connection/pool establishment --
   NOT to pre-consume the quota of the keys that are about to be
   measured. Running warm-up traffic against the actual measured keys
   would silently change the allowed/denied counts reported for the
   real measurement (a warmed-up bucket already has less headroom than
   a fresh one), which would misrepresent what happened during the
   timed run. Warm-up traffic uses key prefix "__warmup__" on the same
   limiter, so it still exercises the same storage backend and (for
   Redis) the same live connection, without touching any key the
   measured run will later use.

5. A fresh limiter + fresh storage per matrix cell (not per
   repetition-within-a-cell reused across repetitions). This is the
   caller's (benchmark_async_concurrency.py's) responsibility, not
   enforced here -- run_matrix_cell() takes an already-constructed
   limiter and does not know or care how many times its caller intends
   to reuse it. Reusing one limiter across repetitions of the SAME
   cell would carry leftover consumed quota from repetition N into
   repetition N+1, corrupting the allowed/denied counts of every
   repetition after the first -- this is a correctness requirement on
   the caller, documented here because it's easy to miss.

6. Percentiles use linear interpolation between the two nearest ranks
   (the same method NumPy's default `percentile` uses), not
   nearest-rank truncation. Implemented directly against a plain
   sorted list (see `_percentile` below) rather than pulling in NumPy
   as a dependency for one function -- this project has consistently
   avoided adding a dependency for something a few lines of stdlib
   arithmetic covers (see limivault.metrics's module docstring, point 3,
   declining to import prometheus_client for the same reason).
--------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from limivault.base import AsyncRateLimiter

__all__ = [
    "MatrixCellResult",
    "run_matrix_cell",
    "percentile_us",
]


def percentile_us(sorted_latencies_ns: List[int], pct: float) -> float:
    """Linear-interpolation percentile of `sorted_latencies_ns` (which
    MUST already be sorted ascending), returned in microseconds.
    `pct` is in [0, 100]. Returns 0.0 for an empty input (a matrix
    cell where every call errored before a latency could be recorded)
    rather than raising, since a benchmark run reporting "0" for a
    degenerate empty-latency cell is more useful than crashing the
    whole matrix sweep partway through.
    """
    n = len(sorted_latencies_ns)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_latencies_ns[0] / 1000.0
    rank = (pct / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_latencies_ns[int(rank)] / 1000.0
    lower_val = sorted_latencies_ns[int(lower)]
    upper_val = sorted_latencies_ns[int(upper)]
    frac = rank - lower
    interpolated_ns = lower_val + (upper_val - lower_val) * frac
    return interpolated_ns / 1000.0


@dataclass
class MatrixCellResult:
    """One measured (algorithm, backend, concurrency, num_keys,
    requests_per_task, repetition) run. See
    benchmark_async_concurrency.py for how these are assembled into
    the full matrix and aggregated across repetitions."""

    algorithm: str
    backend: str
    concurrency: int
    num_keys: int
    requests_per_task: int
    repetition: int
    cost: int

    total_requests: int
    allowed: int
    denied: int
    errors: int

    wall_seconds: float
    throughput_per_sec: float

    latency_mean_us: float
    latency_min_us: float
    latency_p50_us: float
    latency_p95_us: float
    latency_p99_us: float
    latency_max_us: float


def _build_key_chunks(
    concurrency: int, requests_per_task: int, num_keys: int, key_prefix: str
) -> List[List[str]]:
    """Precompute the full deterministic key sequence (see this
    module's docstring, point 3) and split it into `concurrency`
    contiguous per-task chunks of `requests_per_task` keys each."""
    total = concurrency * requests_per_task
    flat = [f"{key_prefix}{i % num_keys}" for i in range(total)]
    return [
        flat[t * requests_per_task : (t + 1) * requests_per_task]
        for t in range(concurrency)
    ]


async def _task_worker(
    limiter: "AsyncRateLimiter", keys: List[str], cost: int, record_latency: bool
) -> Tuple[List[int], int, int, int]:
    """Issue one `allow()` call per key in `keys`, in order, on this
    one task/coroutine. Returns (latencies_ns, allowed_count,
    denied_count, error_count) -- all local to this task, merged by
    the caller after asyncio.gather() (see this module's docstring,
    point 1)."""
    latencies_ns: List[int] = []
    allowed = 0
    denied = 0
    errors = 0
    for key in keys:
        start_ns = time.perf_counter_ns() if record_latency else 0
        try:
            result = await limiter.allow(key, cost=cost)
        except Exception:
            errors += 1
            continue
        if record_latency:
            latencies_ns.append(time.perf_counter_ns() - start_ns)
        if result:
            allowed += 1
        else:
            denied += 1
    return latencies_ns, allowed, denied, errors


async def run_warmup(
    limiter: "AsyncRateLimiter",
    concurrency: int,
    warmup_requests_per_task: int,
    cost: int,
) -> None:
    """Fire `concurrency` tasks x `warmup_requests_per_task` requests
    each against a dedicated "__warmup__" key namespace (see this
    module's docstring, point 4). Results are discarded -- this exists
    purely to absorb one-time costs before the timed run. A no-op if
    `warmup_requests_per_task` is 0."""
    if warmup_requests_per_task <= 0 or concurrency <= 0:
        return
    chunks = _build_key_chunks(
        concurrency, warmup_requests_per_task, num_keys=1, key_prefix="__warmup__:"
    )
    await asyncio.gather(
        *[
            _task_worker(limiter, chunk, cost, record_latency=False)
            for chunk in chunks
        ]
    )


async def run_matrix_cell(
    limiter: "AsyncRateLimiter",
    *,
    algorithm: str,
    backend: str,
    concurrency: int,
    num_keys: int,
    requests_per_task: int,
    repetition: int,
    cost: int,
    key_prefix: str = "req:",
) -> MatrixCellResult:
    """Run ONE matrix cell's measured phase (no warm-up -- call
    run_warmup() separately first if wanted, per this module's
    docstring point 4) against `limiter`, and return the aggregated
    result.

    `limiter` must be freshly constructed for this cell (see this
    module's docstring, point 5) -- this function does not validate
    that, since it has no way to know the limiter's history.
    """
    chunks = _build_key_chunks(concurrency, requests_per_task, num_keys, key_prefix)

    wall_start = time.perf_counter()
    task_results = await asyncio.gather(
        *[
            _task_worker(limiter, chunk, cost, record_latency=True)
            for chunk in chunks
        ]
    )
    wall_seconds = time.perf_counter() - wall_start

    all_latencies_ns: List[int] = []
    total_allowed = 0
    total_denied = 0
    total_errors = 0
    for latencies_ns, allowed, denied, errors in task_results:
        all_latencies_ns.extend(latencies_ns)
        total_allowed += allowed
        total_denied += denied
        total_errors += errors

    total_requests = concurrency * requests_per_task
    all_latencies_ns.sort()

    mean_us = (
        (sum(all_latencies_ns) / len(all_latencies_ns)) / 1000.0
        if all_latencies_ns
        else 0.0
    )
    min_us = all_latencies_ns[0] / 1000.0 if all_latencies_ns else 0.0
    max_us = all_latencies_ns[-1] / 1000.0 if all_latencies_ns else 0.0

    return MatrixCellResult(
        algorithm=algorithm,
        backend=backend,
        concurrency=concurrency,
        num_keys=num_keys,
        requests_per_task=requests_per_task,
        repetition=repetition,
        cost=cost,
        total_requests=total_requests,
        allowed=total_allowed,
        denied=total_denied,
        errors=total_errors,
        wall_seconds=wall_seconds,
        throughput_per_sec=(total_requests / wall_seconds) if wall_seconds > 0 else 0.0,
        latency_mean_us=mean_us,
        latency_min_us=min_us,
        latency_p50_us=percentile_us(all_latencies_ns, 50),
        latency_p95_us=percentile_us(all_latencies_ns, 95),
        latency_p99_us=percentile_us(all_latencies_ns, 99),
        latency_max_us=max_us,
    )
