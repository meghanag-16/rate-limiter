# src/limivault/cli.py
"""Command-line interface for limivault.

Commands:
    limivault demo           -- fire a scripted burst of requests at one
                             in-memory algorithm and print
                             ALLOWED/DENIED per request, with a live
                             remaining() count.
    limivault benchmark      -- forwards to `python -m benchmarks.
                             benchmark_async_concurrency`, the
                             in-memory algorithm x concurrency x key-
                             distribution matrix benchmark. Pass flags
                             after `--`, e.g.
                             `limivault benchmark -- --concurrency 1 10`.
    limivault memory         -- runs benchmarks/memory_footprint.py's
                             main().
    limivault lua-demo       -- the `demo` equivalent for the Redis
                             Lua/GCRA limiters; needs a reachable
                             Redis (--redis-host/--redis-port).
    limivault lua-benchmark  -- forwards to `python -m benchmarks.
                             lua_benchmark_async_concurrency`, the
                             matrix benchmark for the Lua/GCRA
                             limiters. Flags after `--` are passed
                             through untouched.

The two `lua-*` subcommands are implemented in limivault/lua_cli.py and
registered here, so there is a single `limivault` entry point.

The benchmark and memory commands forward to scripts in the repository's
benchmarks/ package. They are registered only when that directory is
available (in a source checkout or editable install), so a wheel install
does not advertise commands that it cannot run.

`demo` has no such limitation -- it only depends on the installed
limivault package itself, so it works from any pip install. `lua-demo`
likewise only needs the installed package, plus a reachable Redis.
"""

from __future__ import annotations

import argparse
import logging
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Callable

from limivault.logging import configure_logging

# MUST run before any `limivault.algorithms.*` import below (same
# ordering requirement as benchmarks/bench_common.py -- see that
# file's comment for the full explanation: structlog's bind() resolves
# the active config at bind-time, and each algorithm module binds its
# module-level logger at import time). Without this ordering, `limivault
# demo` would print a DEBUG line per allow() call in addition to its
# own ALLOWED/DENIED line. `benchmark`, `memory` and `lua-benchmark`
# run as separate subprocesses and configure this themselves.
configure_logging(level=logging.WARNING)

from limivault.algorithms.fixed_window import FixedWindow  # noqa: E402
from limivault.algorithms.leaky_bucket import (  # noqa: E402
    LeakyBucketMeter,
    LeakyBucketQueue,
)
from limivault.algorithms.sliding_window_counter import SlidingWindowCounter  # noqa: E402
from limivault.algorithms.sliding_window_log import SlidingWindowLog  # noqa: E402
from limivault.algorithms.token_bucket import TokenBucket  # noqa: E402
from limivault.base import RateLimiter  # noqa: E402
from limivault.lua_cli import (  # noqa: E402
    _add_lua_demo_args,
    _cmd_lua_benchmark,
    _cmd_lua_demo,
)

_DEMO_ALGORITHMS: dict[str, Callable[[float, float], RateLimiter]] = {
    "fixed_window": lambda p1, p2: FixedWindow(limit=int(p1), period=p2),
    "token_bucket": lambda p1, p2: TokenBucket(capacity=int(p1), refill_rate=p2),
    "sliding_window_log": lambda p1, p2: SlidingWindowLog(limit=int(p1), period=p2),
    "sliding_window_counter": lambda p1, p2: SlidingWindowCounter(
        limit=int(p1), period=p2
    ),
    "leaky_bucket_meter": lambda p1, p2: LeakyBucketMeter(
        capacity=int(p1), leak_rate=p2
    ),
    "leaky_bucket_queue": lambda p1, p2: LeakyBucketQueue(
        capacity=int(p1), leak_rate=p2
    ),
}


def _find_repo_root() -> Path:
    """Best-effort search for a directory containing benchmarks/,
    starting from this file's location and walking up. Returns the
    found path, or the current working directory if nothing is found
    (the caller checks existence afterward either way)."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "benchmarks").is_dir():
            return candidate
    return Path.cwd()


def _cmd_demo(args: argparse.Namespace) -> int:
    factory = _DEMO_ALGORITHMS[args.algorithm]
    limiter = factory(args.param1, args.param2)

    print(
        f"Demo: {args.algorithm} (param1={args.param1}, param2={args.param2}), "
        f"{args.requests} requests, {args.interval}s apart, cost={args.cost}"
    )
    for i in range(args.requests):
        allowed = limiter.allow(args.key, cost=args.cost)
        remaining = limiter.remaining(args.key)
        status = "ALLOWED" if allowed else "DENIED "
        print(f"  request {i + 1:>3}: {status}  remaining={remaining}")
        if args.interval > 0 and i < args.requests - 1:
            time.sleep(args.interval)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    root = _find_repo_root()
    bench_dir = root / "benchmarks"
    if not bench_dir.is_dir():
        print(
            "Could not find the benchmarks/ directory. `limivault benchmark` only "
            "works from a source checkout / editable install that still has "
            "benchmarks/ on disk -- see cli.py's module docstring.",
            file=sys.stderr,
        )
        return 1

    # argparse.REMAINDER captures a leading "--" literally rather than
    # treating it as the usual "end of this parser's own options"
    # separator (a well-known argparse.REMAINDER quirk) -- strip
    # exactly one leading "--" so `limivault benchmark -- --backends
    # memory` forwards `--backends memory`, not `-- --backends
    # memory`, to the inner script's own parser. Verified directly:
    # without this, the inner argparse call fails with "unrecognized
    # arguments: --".
    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    py_args = [sys.executable, "-m", "benchmarks.benchmark_async_concurrency"]
    py_args += passthrough
    result = subprocess.run(py_args, cwd=str(root))  # nosec
    return result.returncode


def _cmd_memory(args: argparse.Namespace) -> int:
    root = _find_repo_root()
    bench_dir = root / "benchmarks"
    if not bench_dir.is_dir():
        print(
            "Could not find the benchmarks/ directory. `limivault memory` only "
            "works from a source checkout / editable install that still has "
            "benchmarks/ on disk -- see cli.py's module docstring.",
            file=sys.stderr,
        )
        return 1

    py_args = [sys.executable, "-m", "benchmarks.memory_footprint"]
    if args.n_keys:
        py_args += ["--n-keys", *[str(n) for n in args.n_keys]]
    result = subprocess.run(py_args, cwd=str(root))  # nosec
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limivault", description="limivault command-line tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a scripted demo of one algorithm")
    demo.add_argument(
        "--algorithm",
        choices=sorted(_DEMO_ALGORITHMS.keys()),
        required=True,
    )
    demo.add_argument(
        "--param1",
        type=float,
        required=True,
        help="limit (window/log/counter algorithms) or capacity (bucket algorithms)",
    )
    demo.add_argument(
        "--param2",
        type=float,
        required=True,
        help="period in seconds (window/log/counter) or refill/leak rate per "
        "second (bucket algorithms)",
    )
    demo.add_argument("--key", default="demo-key")
    demo.add_argument("--cost", type=int, default=1)
    demo.add_argument("--requests", type=int, default=20)
    demo.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="real seconds to sleep between requests (0 = no delay)",
    )
    demo.set_defaults(func=_cmd_demo)

    repo_root = _find_repo_root()
    has_benchmarks = (repo_root / "benchmarks").is_dir()
    if has_benchmarks:
        benchmark = subparsers.add_parser(
            "benchmark",
            help="Run the in-memory async concurrency benchmark matrix "
            "(benchmarks/benchmark_async_concurrency.py)",
        )
        benchmark.add_argument(
            "passthrough",
            nargs=argparse.REMAINDER,
            help="Flags forwarded as-is to benchmark_async_concurrency.py, "
            "e.g. `limivault benchmark -- --concurrency 1 10 100`. "
            "Run `limivault benchmark -- --help` to see all available flags.",
        )
        benchmark.set_defaults(func=_cmd_benchmark)

        memory = subparsers.add_parser(
            "memory", help="Run the steady-state memory footprint measurement"
        )
        memory.add_argument(
            "--n-keys",
            type=int,
            nargs="+",
            default=None,
            help="key counts to measure at "
            "(default: benchmarks/memory_footprint.py's own default)",
        )
        memory.set_defaults(func=_cmd_memory)

    lua_demo = subparsers.add_parser(
        "lua-demo",
        help="Run a scripted demo of one Redis Lua/GCRA algorithm "
        "against a real Redis",
    )
    _add_lua_demo_args(lua_demo)
    lua_demo.set_defaults(func=_cmd_lua_demo)

    if has_benchmarks:
        lua_benchmark = subparsers.add_parser(
            "lua-benchmark",
            help="Run the async concurrency benchmark matrix for the Redis "
            "Lua/GCRA algorithms "
            "(benchmarks/lua_benchmark_async_concurrency.py)",
        )
        lua_benchmark.add_argument(
            "passthrough",
            nargs=argparse.REMAINDER,
            help="Flags forwarded as-is to lua_benchmark_async_concurrency.py, "
            "e.g. `limivault lua-benchmark -- --concurrency 1 10 100`. Run "
            "`limivault lua-benchmark -- --help` to see all available flags.",
        )
        lua_benchmark.set_defaults(func=_cmd_lua_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
