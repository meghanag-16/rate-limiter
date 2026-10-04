# benchmarks/memory_footprint.py
"""steady-state memory measurement: how much memory does each
algorithm's InMemoryStorage/AsyncInMemoryStorage accumulate after N
unique keys have each been used exactly once?

This is a standalone script, not a pytest test -- memory measurement
via tracemalloc doesn't fit pytest-benchmark's timing-oriented model,
and turning "how much memory did this use" into pass/fail assertions
would require picking an arbitrary threshold with no real basis. Run
directly:

    python -m benchmarks.memory_footprint
    python -m benchmarks.memory_footprint --n-keys 1000 5000 20000

--------------------------------------------------------------------
METHODOLOGY -- READ BEFORE TRUSTING THE NUMBERS
--------------------------------------------------------------------
For each algorithm and each N in --n-keys:
  1. tracemalloc.start()
  2. Read current traced memory (baseline).
  3. Construct a fresh limiter + fresh storage.
  4. Call allow() once per unique key, for N unique keys (async
     algorithms: run_until_complete() per call, same as
     the earlier per-call benchmark prototype -- see that file's docstring for why).
  5. Read current traced memory again; delta = after - before.
  6. tracemalloc.stop().

The reported "bytes/key" is (delta / N). This is a DELTA over the
whole traced process during that block, not a precise
sys.getsizeof()-style measurement of the storage object alone --
Python's own allocator overhead, dict resizing, and any incidental
allocation inside allow()'s call path (e.g. the small tuples/dicts
each algorithm's `_get_state`/`_get_state_with_diagnostics` helpers
build transiently) are all included, not just the state that persists
in `storage._data`/`storage._locks`. This is a reasonable proxy for
"how much does memory grow as key cardinality grows" -- the actual
question lock-cleanup design notes and this benchmark both
care about -- but it is NOT a claim about the exact byte size of one
stored state dict. Flagged explicitly rather than presented as more
precise than it is.

Each (algorithm, N) run uses a fresh process-level tracemalloc
start/stop cycle and a fresh limiter/storage instance, so results
across different N values for the same algorithm are independent
measurements, not a continuation of the same growing dataset.
--------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import asyncio
import tracemalloc
from typing import Any, Callable, List, Tuple

# Importing bench_common triggers its own configure_logging(level=
# WARNING) call BEFORE it imports any rlimit.algorithms.* module -- see
# that file's comment for why the ordering matters (structlog's bind()
# resolves logging config at bind-time, not at each log call). Without
# that ordering, per-call DEBUG decision logging would flood
# stdout and bury the results table below.
from benchmarks.bench_common import ASYNC_FACTORIES, SYNC_FACTORIES

_DEFAULT_N_KEYS = [100, 1000, 5000]


def _measure_sync(factory: Callable[[], Any], n_keys: int) -> int:
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()

    limiter = factory()
    for i in range(n_keys):
        limiter.allow(f"mem-key-{i}")

    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return after - before


def _measure_async(factory: Callable[[], Any], n_keys: int) -> int:
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()

    limiter = factory()
    loop = asyncio.new_event_loop()
    try:
        for i in range(n_keys):
            loop.run_until_complete(limiter.allow(f"mem-key-{i}"))
    finally:
        loop.close()

    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return after - before


def run(n_keys_list: List[int]) -> List[Tuple[str, int, int, float]]:
    """Returns a list of (algorithm_name, n_keys, delta_bytes,
    bytes_per_key) rows."""
    rows: List[Tuple[str, int, int, float]] = []

    for name, factory in SYNC_FACTORIES:
        for n_keys in n_keys_list:
            delta = _measure_sync(factory, n_keys)
            rows.append((name, n_keys, delta, delta / n_keys if n_keys else 0.0))

    for name, factory in ASYNC_FACTORIES:
        for n_keys in n_keys_list:
            delta = _measure_async(factory, n_keys)
            rows.append((name, n_keys, delta, delta / n_keys if n_keys else 0.0))

    return rows


def _print_table(rows: List[Tuple[str, int, int, float]]) -> None:
    header = f"{'algorithm':<28} {'n_keys':>8} {'delta_bytes':>14} {'bytes/key':>12}"
    print(header)
    print("-" * len(header))
    for name, n_keys, delta, per_key in rows:
        print(f"{name:<28} {n_keys:>8} {delta:>14} {per_key:>12.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure steady-state memory growth per unique key, "
        "for all 12 rlimit algorithm classes."
    )
    parser.add_argument(
        "--n-keys",
        type=int,
        nargs="+",
        default=_DEFAULT_N_KEYS,
        help=f"Key counts to measure at (default: {_DEFAULT_N_KEYS})",
    )
    args = parser.parse_args()

    rows = run(args.n_keys)
    _print_table(rows)


if __name__ == "__main__":
    main()
