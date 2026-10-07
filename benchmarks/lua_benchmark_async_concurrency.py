# benchmarks/lua_benchmark_async_concurrency.py
"""Concurrency/throughput benchmark for the 6 async Redis Lua/GCRA
Redis-native limiters -- the benchmarks/ integration flagged as
missing in review.

Mirrors benchmarks/benchmark_async_concurrency.py's matrix-sweep
shape (algorithm x concurrency x num_keys, repeated and aggregated by
median, with per-cell latency percentiles) but scoped down to what
actually varies for this family:

  - No "backend" dimension. The in-memory async algorithms take a
    swappable storage backend (see benchmark_async_concurrency.py);
    every Lua/GCRA class in this module IS its own backend (talks
    straight to redis.asyncio.Redis) -- there is nothing else to swap
    in, so that whole axis of that matrix doesn't apply here.
  - REUSES the existing measurement engine, not a second
    reimplementation: `run_warmup`, `run_matrix_cell`, `MatrixCellResult`,
    and `percentile_us` are imported from the project's own
    benchmarks/concurrency_harness.py unchanged. Every Redis Lua/GCRA async
    class implements the same AsyncRateLimiter ABC
    (allow(key, cost) -> bool) as every in-memory async class, so the
    harness's task-worker/latency-collection logic (which only ever
    calls `await limiter.allow(key, cost=cost)`) works against these
    classes with zero modification -- confirmed by inspection of
    concurrency_harness.py's `_task_worker`, which has no
    in-memory-specific assumption anywhere in it.

Run with:
    python -m benchmarks.lua_benchmark_async_concurrency
    python -m benchmarks.lua_benchmark_async_concurrency \\
        --concurrency 1 10 100 --requests-per-task 500 --repetitions 5

REQUIRES A REACHABLE REDIS -- unlike benchmark_async_concurrency.py,
which runs entirely in memory, there is no in-memory fallback here:
every class in this file's scope IS Redis-backed by definition. Start
one with:
    docker compose -f docker-compose.redis.yml up -d
If Redis isn't reachable at --redis-host/--redis-port, this script
prints a clear message and exits 1 rather than failing deep inside the
first matrix cell.

FRESH CLIENT + FRESH KEY_PREFIX PER REPETITION (not just per cell):
each (algorithm, concurrency, num_keys) cell's EVERY repetition gets
its own freshly connected redis.asyncio.Redis client and its own
randomized key_prefix, so no run can see another run's leftover keys,
following concurrency_harness.py's point 5 (fresh limiter/storage per
cell, stated there as the CALLER's responsibility).
Reusing one key_prefix across repetitions of the same cell would let
repetition N+1 inherit repetition N's already-consumed quota on every
key, corrupting every repetition after the first -- the exact bug
concurrency_harness.py's docstring warns against, restated here
because this script owns that responsibility for the Redis Lua/GCRA family.

NO --limit/--rate DENIAL-RATE CONCERN, SAME AS THE ORIGINAL SCRIPT:
this script defaults every factory to the same deliberately huge
limit/capacity as lua_bench_common.py's SYNC_FACTORIES/ASYNC_FACTORIES
(10**9) -- see that module's docstring for why. There is currently no
CLI flag to override this to a tight limit for this script specifically
(unlike the original benchmark_async_concurrency.py's --limit/--rate);
if a deliberately-denying run is wanted, edit lua_bench_common.py's
_HUGE constant directly. Flagged as a known, narrower feature set
versus the original script, not an oversight -- the tight-limit case
was judged lower priority than getting a working matrix shipped at all.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import statistics
import sys
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from limivault.logging import configure_logging

# Same ordering requirement as the original benchmark scripts (see
# bench_common.py's identical comment) -- structlog binds its
# module-level logger at import time, so this must run before any
# limivault.algorithms.* import below.
configure_logging(level=logging.WARNING)

from benchmarks.concurrency_harness import (  # noqa: E402
    MatrixCellResult,
    run_matrix_cell,
    run_warmup,
)
from benchmarks.lua_bench_common import ASYNC_FACTORIES  # noqa: E402
from limivault.base import AsyncRateLimiter  # noqa: E402

_DEFAULT_CONCURRENCY = [1, 10, 100, 500, 1000]
_DEFAULT_NUM_KEYS = [1, 100, 10000]
_DEFAULT_ALGORITHMS = [name for name, _ in ASYNC_FACTORIES]

_CSV_FIELDNAMES = [
    "algorithm",
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

_FACTORY_BY_NAME: Dict[str, Callable[[Any, str], AsyncRateLimiter]] = dict(
    ASYNC_FACTORIES
)


async def _check_redis_reachable(host: str, port: int) -> Optional[str]:
    try:
        import redis.asyncio as redis_async
    except ImportError:
        return (
            "The `redis` package is not installed. Install it before "
            "running this script."
        )
    client = redis_async.Redis(host=host, port=port, socket_connect_timeout=2.0)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001 -- any connection failure gets
        # the same clear guidance, not a redis-library-specific traceback.
        return (
            f"Could not reach Redis at {host}:{port} ({exc!r}). Start it "
            "with `docker compose -f docker-compose.redis.yml up -d`."
        )
    finally:
        await client.aclose()
    return None


async def _run_one_repetition(
    *,
    algorithm: str,
    concurrency: int,
    num_keys: int,
    requests_per_task: int,
    repetition: int,
    cost: int,
    warmup_requests_per_task: int,
    redis_host: str,
    redis_port: int,
) -> MatrixCellResult:
    """Fresh client + fresh limiter + fresh randomized key_prefix for
    this ONE repetition -- see this module's docstring on why reuse
    across repetitions would corrupt the counts."""
    import redis.asyncio as redis_async

    client = redis_async.Redis(host=redis_host, port=redis_port)
    prefix = f"luabench:{uuid.uuid4().hex[:10]}:"
    limiter = _FACTORY_BY_NAME[algorithm](client, prefix)

    try:
        await run_warmup(limiter, concurrency, warmup_requests_per_task, cost)
        result = await run_matrix_cell(
            limiter,
            algorithm=algorithm,
            backend="redis-lua",
            concurrency=concurrency,
            num_keys=num_keys,
            requests_per_task=requests_per_task,
            repetition=repetition,
            cost=cost,
        )
    finally:
        await client.aclose()
    return result


async def _run_cell_repetitions(
    *,
    algorithm: str,
    concurrency: int,
    num_keys: int,
    requests_per_task: int,
    repetitions: int,
    cost: int,
    warmup_requests_per_task: int,
    redis_host: str,
    redis_port: int,
    cell_timeout_seconds: float,
) -> List[MatrixCellResult]:
    results: List[MatrixCellResult] = []
    for rep in range(repetitions):
        coro = _run_one_repetition(
            algorithm=algorithm,
            concurrency=concurrency,
            num_keys=num_keys,
            requests_per_task=requests_per_task,
            repetition=rep,
            cost=cost,
            warmup_requests_per_task=warmup_requests_per_task,
            redis_host=redis_host,
            redis_port=redis_port,
        )
        try:
            if cell_timeout_seconds > 0:
                result = await asyncio.wait_for(coro, timeout=cell_timeout_seconds)
            else:
                result = await coro
        except asyncio.TimeoutError:
            print(
                f"  TIMEOUT after {cell_timeout_seconds}s: {algorithm} "
                f"concurrency={concurrency} num_keys={num_keys} repetition={rep}. "
                f"Skipping the remaining repetitions of this cell and moving on.",
                file=sys.stderr,
            )
            break
        results.append(result)
    return results


async def run_full_matrix(args: argparse.Namespace) -> List[MatrixCellResult]:
    problem = await _check_redis_reachable(args.redis_host, args.redis_port)
    if problem is not None:
        print(f"Cannot run: {problem}", file=sys.stderr)
        return []

    all_results: List[MatrixCellResult] = []
    total_cells = len(args.algorithms) * len(args.concurrency) * len(args.num_keys)
    cell_index = 0

    for algorithm in args.algorithms:
        for concurrency in args.concurrency:
            for num_keys in args.num_keys:
                cell_index += 1
                print(
                    f"[{cell_index}/{total_cells}] {algorithm} "
                    f"concurrency={concurrency} num_keys={num_keys} "
                    f"requests_per_task={args.requests_per_task} "
                    f"repetitions={args.repetitions}",
                    file=sys.stderr,
                )
                cell_results = await _run_cell_repetitions(
                    algorithm=algorithm,
                    concurrency=concurrency,
                    num_keys=num_keys,
                    requests_per_task=args.requests_per_task,
                    repetitions=args.repetitions,
                    cost=args.cost,
                    warmup_requests_per_task=args.warmup_requests_per_task,
                    redis_host=args.redis_host,
                    redis_port=args.redis_port,
                    cell_timeout_seconds=args.cell_timeout_seconds,
                )
                all_results.extend(cell_results)

    return all_results


def _write_csv(results: List[MatrixCellResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row.pop("backend", None)  # not a meaningful axis here; see module docstring
            writer.writerow(row)


def _print_raw_table(results: List[MatrixCellResult]) -> None:
    header = (
        f"{'algorithm':<32} {'conc':>5} {'keys':>6} {'rep':>4} "
        f"{'total':>7} {'allowed':>8} {'denied':>7} {'errors':>7} "
        f"{'thr/s':>10} {'p50us':>9} {'p95us':>9} {'p99us':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.algorithm:<32} {r.concurrency:>5} {r.num_keys:>6} "
            f"{r.repetition:>4} {r.total_requests:>7} {r.allowed:>8} {r.denied:>7} "
            f"{r.errors:>7} {r.throughput_per_sec:>10.1f} {r.latency_p50_us:>9.1f} "
            f"{r.latency_p95_us:>9.1f} {r.latency_p99_us:>10.1f}"
        )


def _aggregate_by_cell(
    results: List[MatrixCellResult],
) -> List[Tuple[Tuple[str, int, int], Dict[str, Any]]]:
    groups: Dict[Tuple[str, int, int], List[MatrixCellResult]] = {}
    for r in results:
        key = (r.algorithm, r.concurrency, r.num_keys)
        groups.setdefault(key, []).append(r)

    aggregated: List[Tuple[Tuple[str, int, int], Dict[str, Any]]] = []
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
    aggregated: List[Tuple[Tuple[str, int, int], Dict[str, Any]]]
) -> None:
    header = (
        f"{'algorithm':<32} {'conc':>5} {'keys':>6} {'reps':>5} "
        f"{'median thr/s':>13} {'median p50us':>13} {'median p95us':>13} "
        f"{'median p99us':>13} {'allowed(min-max)':>18} "
        f"{'denied(min-max)':>17} {'errs':>5}"
    )
    print(header)
    print("-" * len(header))
    for (algorithm, concurrency, num_keys), agg in aggregated:
        allowed_range = f"{agg['allowed_min']}-{agg['allowed_max']}"
        denied_range = f"{agg['denied_min']}-{agg['denied_max']}"
        print(
            f"{algorithm:<32} {concurrency:>5} {num_keys:>6} "
            f"{agg['repetitions']:>5} {agg['throughput_per_sec']:>13.1f} "
            f"{agg['latency_p50_us']:>13.1f} {agg['latency_p95_us']:>13.1f} "
            f"{agg['latency_p99_us']:>13.1f} {allowed_range:>18} {denied_range:>17} "
            f"{agg['errors_total']:>5}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Async concurrency benchmark for limivault's Redis Lua/GCRA "
        "(Lua/GCRA) Redis-native limiters."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=_DEFAULT_ALGORITHMS,
        choices=_DEFAULT_ALGORITHMS,
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=_DEFAULT_CONCURRENCY
    )
    parser.add_argument("--num-keys", type=int, nargs="+", default=_DEFAULT_NUM_KEYS)
    parser.add_argument("--requests-per-task", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cost", type=int, default=1)
    parser.add_argument("--warmup-requests-per-task", type=int, default=20)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=60.0,
        help="Max wall time per (cell, repetition) before it's reported as a "
        "timeout and the rest of that cell's repetitions are skipped. Pass 0 "
        "to disable.",
    )
    parser.add_argument("--output-csv", default=None)
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
