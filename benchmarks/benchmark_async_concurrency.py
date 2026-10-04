# benchmarks/benchmark_async_concurrency.py
"""async concurrency benchmark. Replaces the earlier
per-call pytest-benchmark harness (the earlier per-call benchmark prototype,
removed) with a full matrix sweep over algorithm x backend x
concurrency x key-distribution, reporting throughput, latency
percentiles, and allowed/denied counts per cell.

Run with:
    python -m benchmarks.benchmark_async_concurrency

By default this runs the FULL matrix (all 6 async algorithms against
the in-memory backend, every configured concurrency level and key
count) but with CONSERVATIVE per-cell size
(--requests-per-task 100, --repetitions 3) so a first run completes in
a reasonable time. Every dimension is a CLI flag -- nothing is
hardcoded -- so a larger, closer-to-literal run (concurrency up to
1000, 1000 requests/task, 5+ repetitions) is one command away:

    python -m benchmarks.benchmark_async_concurrency \\
        --requests-per-task 1000 --repetitions 5

BACKENDS: only the in-memory backend (AsyncInMemoryStorage) is
benchmarked by this script, so it needs no Redis and no Docker. The
Redis-native Lua/GCRA limiters are benchmarked separately by
benchmarks/lua_benchmark_async_concurrency.py, which reuses the same
measurement engine (benchmarks/concurrency_harness.py).

Every matrix cell gets a fresh limiter + fresh storage, so no cell can
see leftover quota consumption from a previous cell or repetition --
see concurrency_harness.py's module docstring, points 4-5, for the full
rationale.

Output: every individual repetition's row is written to
--output-csv (if given) and to stdout; a second, smaller table
prints the MEDIAN across repetitions for each (algorithm, backend,
concurrency, num_keys) cell, since a single run's numbers are noise
-- see the design discussion this responds to, point 10 ("repeat each
benchmark... report median throughput, p50/p95/p99 latency").

--------------------------------------------------------------------
CELL TIMEOUT
--------------------------------------------------------------------
--cell-timeout-seconds (default 60s) bounds how long any single (cell,
repetition) can run before being reported as a timeout and skipped, so
a pathological cell cannot silently hang the entire matrix with zero
visibility. Raising it does not make a slow cell faster -- it only
delays finding out that the cell is slow. Pass 0 to disable the limit.
--------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import statistics
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from rlimit.logging import configure_logging

# MUST run before any rlimit.algorithms.* import -- see
# benchmarks/bench_common.py's identical comment for the full
# rationale (structlog binds its module-level logger at import time,
# not at first log call, so configuring afterward is too late).
configure_logging(level=logging.WARNING)

from benchmarks.concurrency_harness import (  # noqa: E402
    MatrixCellResult,
    run_matrix_cell,
    run_warmup,
)
from rlimit.algorithms.async_fixed_window import AsyncFixedWindow  # noqa: E402
from rlimit.algorithms.async_leaky_bucket import (  # noqa: E402
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from rlimit.algorithms.async_sliding_window_counter import (  # noqa: E402
    AsyncSlidingWindowCounter,
)
from rlimit.algorithms.async_sliding_window_log import (  # noqa: E402
    AsyncSlidingWindowLog,
)
from rlimit.algorithms.async_token_bucket import AsyncTokenBucket  # noqa: E402
from rlimit.base import AsyncRateLimiter  # noqa: E402
from rlimit.storage import AsyncInMemoryStorage, AsyncStorageBackend  # noqa: E402

_DEFAULT_CONCURRENCY = [1, 10, 100, 500, 1000]
_DEFAULT_NUM_KEYS = [1, 100, 10000]
_DEFAULT_ALGORITHMS = [
    "AsyncFixedWindow",
    "AsyncTokenBucket",
    "AsyncSlidingWindowLog",
    "AsyncSlidingWindowCounter",
    "AsyncLeakyBucketMeter",
    "AsyncLeakyBucketQueue",
]
_DEFAULT_BACKENDS = ["memory"]

# Deliberately huge by default -- see this file's own --limit/--rate
# help text and bench_common.py's identical rationale: a huge quota
# keeps the default matrix representative of "steady, mostly-allowed
# traffic" rather than "mostly-denied traffic," and keeps the
# allowed/denied counts this script reports meaningful as a sanity
# check (denied should stay at/near 0 for the default matrix; a
# nonzero denied count under the default huge limit would itself be a
# signal something is wrong). Pass --limit/--rate explicitly for a
# deliberately tight-limit run.
_DEFAULT_LIMIT = 10**9
_DEFAULT_RATE = 1.0

_CSV_FIELDNAMES = [
    "algorithm",
    "backend",
    "concurrency",
    "num_keys",
    "requests_per_task",
    "repetition",
    "cost",
    "total_requests",
    "allowed",
    "denied",
    "errors",
    "wall_seconds",
    "throughput_per_sec",
    "latency_mean_us",
    "latency_min_us",
    "latency_p50_us",
    "latency_p95_us",
    "latency_p99_us",
    "latency_max_us",
]


# --- Algorithm construction, parameterized by storage + limit/rate ------

_ALGO_BUILDERS: Dict[
    str, Callable[[AsyncStorageBackend, int, float], AsyncRateLimiter]
] = {
    "AsyncFixedWindow": lambda storage, limit, rate: AsyncFixedWindow(
        limit=limit, period=rate, storage=storage, clock=time.monotonic
    ),
    "AsyncTokenBucket": lambda storage, limit, rate: AsyncTokenBucket(
        capacity=limit, refill_rate=rate, storage=storage, clock=time.monotonic
    ),
    "AsyncSlidingWindowLog": lambda storage, limit, rate: AsyncSlidingWindowLog(
        limit=limit, period=rate, storage=storage, clock=time.monotonic
    ),
    "AsyncSlidingWindowCounter": lambda storage, limit, rate: AsyncSlidingWindowCounter(
        limit=limit, period=rate, storage=storage, clock=time.monotonic
    ),
    "AsyncLeakyBucketMeter": lambda storage, limit, rate: AsyncLeakyBucketMeter(
        capacity=limit, leak_rate=rate, storage=storage, clock=time.monotonic
    ),
    "AsyncLeakyBucketQueue": lambda storage, limit, rate: AsyncLeakyBucketQueue(
        capacity=limit, leak_rate=rate, storage=storage, clock=time.monotonic
    ),
}
# NOTE on --limit/--rate mapping: for FixedWindow/SlidingWindowLog/
# SlidingWindowCounter, `limit` maps to the constructor's `limit`
# kwarg and `rate` maps to `period` (seconds). For the three bucket
# algorithms, `limit` maps to `capacity` and `rate` maps to
# `refill_rate`/`leak_rate` (units/second). This mirrors the same
# limit-vs-capacity naming split already present throughout this
# project's algorithm files -- this script uses one generic pair of
# flag names across all six rather than algorithm-specific flags,
# since the underlying "how much quota" / "how fast it replenishes"
# concept is the same across all six even though the constructor
# kwarg names differ.


def _make_memory_storage() -> AsyncStorageBackend:
    return AsyncInMemoryStorage()


# --- Matrix sweep ---------------------------------------------------------


async def _run_one_repetition(
    *,
    algorithm: str,
    backend: str,
    concurrency: int,
    num_keys: int,
    requests_per_task: int,
    repetition: int,
    cost: int,
    limit: int,
    rate: float,
    warmup_requests_per_task: int,
) -> MatrixCellResult:
    """Fresh storage + fresh limiter for this ONE repetition (see
    concurrency_harness.py's module docstring, point 5, for why reuse
    across repetitions would corrupt the counts). Split out from
    _run_cell_repetitions so the whole warmup+measurement pair can be
    wrapped in a single asyncio.wait_for() timeout below -- warmup
    alone can be the slow part, so the timeout has to cover both, not
    just the measured phase."""
    storage = _make_memory_storage()
    limiter = _ALGO_BUILDERS[algorithm](storage, limit, rate)

    await run_warmup(limiter, concurrency, warmup_requests_per_task, cost)
    return await run_matrix_cell(
        limiter,
        algorithm=algorithm,
        backend=backend,
        concurrency=concurrency,
        num_keys=num_keys,
        requests_per_task=requests_per_task,
        repetition=repetition,
        cost=cost,
    )


async def _run_cell_repetitions(
    *,
    algorithm: str,
    backend: str,
    concurrency: int,
    num_keys: int,
    requests_per_task: int,
    repetitions: int,
    cost: int,
    limit: int,
    rate: float,
    warmup_requests_per_task: int,
    cell_timeout_seconds: float,
) -> List[MatrixCellResult]:
    results: List[MatrixCellResult] = []
    for rep in range(repetitions):
        coro = _run_one_repetition(
            algorithm=algorithm,
            backend=backend,
            concurrency=concurrency,
            num_keys=num_keys,
            requests_per_task=requests_per_task,
            repetition=rep,
            cost=cost,
            limit=limit,
            rate=rate,
            warmup_requests_per_task=warmup_requests_per_task,
        )
        try:
            if cell_timeout_seconds > 0:
                result = await asyncio.wait_for(coro, timeout=cell_timeout_seconds)
            else:
                result = await coro
        except asyncio.TimeoutError:
            # Report it clearly and skip the REST of this cell's
            # repetitions (not just this one) -- a timeout strongly
            # implies the whole (algorithm, backend, concurrency,
            # num_keys) configuration is the problem, not one unlucky
            # repetition, so retrying it immediately would almost
            # certainly just time out again and waste the same amount
            # of time a second time.
            print(
                f"  TIMEOUT after {cell_timeout_seconds}s: {algorithm} "
                f"backend={backend} concurrency={concurrency} "
                f"num_keys={num_keys} repetition={rep}. Skipping the "
                f"remaining repetitions of this cell and moving on.",
                file=sys.stderr,
            )
            break
        results.append(result)
    return results


async def run_full_matrix(args: argparse.Namespace) -> List[MatrixCellResult]:
    all_results: List[MatrixCellResult] = []

    backends = list(args.backends)

    total_cells = (
        len(args.algorithms)
        * len(backends)
        * len(args.concurrency)
        * len(args.num_keys)
    )
    cell_index = 0

    for algorithm in args.algorithms:
        for backend in backends:
            for concurrency in args.concurrency:
                for num_keys in args.num_keys:
                    cell_index += 1
                    print(
                        f"[{cell_index}/{total_cells}] {algorithm} backend={backend} "
                        f"concurrency={concurrency} num_keys={num_keys} "
                        f"requests_per_task={args.requests_per_task} "
                        f"repetitions={args.repetitions}",
                        file=sys.stderr,
                    )
                    cell_results = await _run_cell_repetitions(
                        algorithm=algorithm,
                        backend=backend,
                        concurrency=concurrency,
                        num_keys=num_keys,
                        requests_per_task=args.requests_per_task,
                        repetitions=args.repetitions,
                        cost=args.cost,
                        limit=args.limit,
                        rate=args.rate,
                        warmup_requests_per_task=args.warmup_requests_per_task,
                        cell_timeout_seconds=args.cell_timeout_seconds,
                    )
                    all_results.extend(cell_results)

    return all_results


# --- Reporting -------------------------------------------------------------


def _write_csv(results: List[MatrixCellResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def _print_raw_table(results: List[MatrixCellResult]) -> None:
    header = (
        f"{'algorithm':<26} {'backend':<7} {'conc':>5} {'keys':>6} {'rep':>4} "
        f"{'total':>7} {'allowed':>8} {'denied':>7} {'errors':>7} "
        f"{'thr/s':>10} {'p50us':>9} {'p95us':>9} {'p99us':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.algorithm:<26} {r.backend:<7} {r.concurrency:>5} {r.num_keys:>6} "
            f"{r.repetition:>4} {r.total_requests:>7} {r.allowed:>8} {r.denied:>7} "
            f"{r.errors:>7} {r.throughput_per_sec:>10.1f} {r.latency_p50_us:>9.1f} "
            f"{r.latency_p95_us:>9.1f} {r.latency_p99_us:>10.1f}"
        )


def _aggregate_by_cell(
    results: List[MatrixCellResult],
) -> List[Tuple[Tuple[str, str, int, int], Dict[str, Any]]]:
    """Group by (algorithm, backend, concurrency, num_keys) across
    repetitions, and compute the median of each numeric metric -- see
    this file's module docstring on why a single run's number isn't
    trusted on its own."""
    groups: Dict[Tuple[str, str, int, int], List[MatrixCellResult]] = {}
    for r in results:
        key = (r.algorithm, r.backend, r.concurrency, r.num_keys)
        groups.setdefault(key, []).append(r)

    aggregated: List[Tuple[Tuple[str, str, int, int], Dict[str, Any]]] = []
    for key, reps in groups.items():
        aggregated.append(
            (
                key,
                {
                    "throughput_per_sec": statistics.median(
                        r.throughput_per_sec for r in reps
                    ),
                    "latency_p50_us": statistics.median(r.latency_p50_us for r in reps),
                    "latency_p95_us": statistics.median(r.latency_p95_us for r in reps),
                    "latency_p99_us": statistics.median(r.latency_p99_us for r in reps),
                    "allowed_min": min(r.allowed for r in reps),
                    "allowed_max": max(r.allowed for r in reps),
                    "denied_min": min(r.denied for r in reps),
                    "denied_max": max(r.denied for r in reps),
                    "errors_total": sum(r.errors for r in reps),
                    "repetitions": len(reps),
                },
            )
        )
    return aggregated


def _print_aggregate_table(
    aggregated: List[Tuple[Tuple[str, str, int, int], Dict[str, Any]]]
) -> None:
    header = (
        f"{'algorithm':<26} {'backend':<7} {'conc':>5} {'keys':>6} {'reps':>5} "
        f"{'median thr/s':>13} {'median p50us':>13} {'median p95us':>13} "
        f"{'median p99us':>13} {'allowed(min-max)':>18} "
        f"{'denied(min-max)':>17} {'errs':>5}"
    )
    print(header)
    print("-" * len(header))
    for (algorithm, backend, concurrency, num_keys), agg in aggregated:
        allowed_range = f"{agg['allowed_min']}-{agg['allowed_max']}"
        denied_range = f"{agg['denied_min']}-{agg['denied_max']}"
        print(
            f"{algorithm:<26} {backend:<7} {concurrency:>5} {num_keys:>6} "
            f"{agg['repetitions']:>5} {agg['throughput_per_sec']:>13.1f} "
            f"{agg['latency_p50_us']:>13.1f} {agg['latency_p95_us']:>13.1f} "
            f"{agg['latency_p99_us']:>13.1f} {allowed_range:>18} {denied_range:>17} "
            f"{agg['errors_total']:>5}"
        )


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-matrix async concurrency benchmark for rlimit."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=_DEFAULT_ALGORITHMS,
        choices=_DEFAULT_ALGORITHMS,
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=_DEFAULT_BACKENDS,
        choices=["memory"],
        help="Storage backends to benchmark. Only the in-memory backend is "
        "available here; the Redis/Lua limiters are benchmarked by "
        "benchmarks/lua_benchmark_async_concurrency.py.",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=_DEFAULT_CONCURRENCY
    )
    parser.add_argument("--num-keys", type=int, nargs="+", default=_DEFAULT_NUM_KEYS)
    parser.add_argument(
        "--requests-per-task",
        type=int,
        default=100,
        help="Requests per concurrent task. The reference design's "
        "literal suggestion is ~1000; default is 100 to keep a first "
        "run's total runtime reasonable -- raise this explicitly for "
        "a larger run.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per cell, aggregated via median. The "
        "reference design suggests 5-10; default is 3 to keep a first "
        "run's total runtime reasonable.",
    )
    parser.add_argument("--cost", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help="Maps to `limit` (window/log/counter algorithms) or "
        "`capacity` (bucket algorithms). Deliberately huge by default "
        "-- see this file's module docstring.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=_DEFAULT_RATE,
        help="Maps to `period` in seconds (window/log/counter) or "
        "`refill_rate`/`leak_rate` per second (bucket algorithms).",
    )
    parser.add_argument("--warmup-requests-per-task", type=int, default=20)
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=60.0,
        help="Max wall time per (cell, repetition) before it's reported as a "
        "timeout and the rest of that cell's repetitions are skipped, so one "
        "pathological cell can't silently hang the whole matrix. Pass 0 to "
        "disable (wait indefinitely). See this file's module docstring, "
        "'CELL TIMEOUT'.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to write raw per-repetition rows as CSV",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = asyncio.run(run_full_matrix(args))

    if not results:
        return 1

    print()
    print("=== Raw per-repetition results ===")
    _print_raw_table(results)

    print()
    print("=== Median across repetitions, per cell ===")
    _print_aggregate_table(_aggregate_by_cell(results))

    if args.output_csv:
        _write_csv(results, args.output_csv)
        print(f"\nWrote {len(results)} raw rows to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
