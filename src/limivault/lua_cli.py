# src/limivault/lua_cli.py
"""Command-line handlers for the Redis Lua/GCRA limiters.

This module holds the implementation of the two `limivault` subcommands
that target the Lua/GCRA family. They are registered on the single
`limivault` console script in limivault/cli.py, which imports
`_add_lua_demo_args`, `_cmd_lua_demo` and `_cmd_lua_benchmark` from
here; this module has no entry point of its own.

    limivault lua-demo        -- fire a scripted burst of allow() calls at
                              one Lua/GCRA algorithm against a real
                              Redis, printing ALLOWED/DENIED per
                              request with a live remaining() count.
                              The analogue of `limivault demo`.
    limivault lua-benchmark   -- forwards to `python -m benchmarks.
                              lua_benchmark_async_concurrency`, the
                              same "thin wrapper, not a second
                              implementation" pattern `limivault
                              benchmark` uses for the in-memory matrix
                              script. Flags after `--` are passed
                              through untouched.

NO `memory` SUBCOMMAND: see benchmarks/lua_bench_common.py's module
docstring for why a Python-process memory-footprint-per-key benchmark
does not make sense for this family (state lives in Redis, not this
process) -- there is nothing analogous to add here.

`lua-demo` needs a real, reachable Redis (--redis-host/--redis-port),
unlike `limivault demo`, which only needs the installed package -- the
Lua/GCRA classes have no in-memory mode by design (see
limivault.redis_lua_scripts's module docstring's "STANDALONE,
COEXISTING" section).
"""

from __future__ import annotations

import argparse
import logging
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Callable, Dict

from limivault.logging import configure_logging

# Same ordering requirement as the main cli.py -- see that file's
# identical comment (structlog binds its module logger at import
# time, so this must run before any limivault.algorithms.* import).
configure_logging(level=logging.WARNING)

import redis  # noqa: E402

from limivault.algorithms.redis_lua_fixed_window import (  # noqa: E402
    RedisLuaFixedWindow,
)
from limivault.algorithms.redis_lua_leaky_bucket import (  # noqa: E402
    RedisLuaLeakyBucketMeter,
    RedisLuaLeakyBucketQueue,
)
from limivault.algorithms.redis_lua_sliding_window_counter import (  # noqa: E402
    RedisLuaSlidingWindowCounter,
)
from limivault.algorithms.redis_lua_sliding_window_log import (  # noqa: E402
    RedisLuaSlidingWindowLog,
)
from limivault.algorithms.redis_lua_token_bucket import (  # noqa: E402
    RedisGcraTokenBucket,
)
from limivault.base import RateLimiter  # noqa: E402

_DEMO_ALGORITHMS: Dict[str, Callable[["redis.Redis", float, float], RateLimiter]] = {
    "fixed_window": lambda client, p1, p2: RedisLuaFixedWindow(
        client, limit=int(p1), period=p2
    ),
    "gcra_token_bucket": lambda client, p1, p2: RedisGcraTokenBucket(
        client, capacity=int(p1), refill_rate=p2
    ),
    "sliding_window_log": lambda client, p1, p2: RedisLuaSlidingWindowLog(
        client, limit=int(p1), period=p2
    ),
    "sliding_window_counter": lambda client, p1, p2: RedisLuaSlidingWindowCounter(
        client, limit=int(p1), period=p2
    ),
    "leaky_bucket_meter": lambda client, p1, p2: RedisLuaLeakyBucketMeter(
        client, capacity=int(p1), leak_rate=p2
    ),
    "leaky_bucket_queue": lambda client, p1, p2: RedisLuaLeakyBucketQueue(
        client, capacity=int(p1), leak_rate=p2
    ),
}


def _add_lua_demo_args(demo: argparse.ArgumentParser) -> None:
    demo.add_argument(
        "--algorithm", choices=sorted(_DEMO_ALGORITHMS.keys()), required=True
    )
    demo.add_argument(
        "--param1",
        type=float,
        required=True,
        help="limit (window/log/counter algorithms) or capacity "
        "(GCRA/bucket algorithms)",
    )
    demo.add_argument(
        "--param2",
        type=float,
        required=True,
        help="period in seconds (window/log/counter) or refill/leak rate per "
        "second (GCRA/bucket algorithms)",
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
    demo.add_argument("--redis-host", default="localhost")
    demo.add_argument("--redis-port", type=int, default=6379)


def _cmd_lua_demo(args: argparse.Namespace) -> int:
    import time

    client = redis.Redis(host=args.redis_host, port=args.redis_port)
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 -- one clear message for any failure
        print(
            f"Could not reach Redis at {args.redis_host}:{args.redis_port} "
            f"({exc!r}). Start it with "
            "`docker compose -f docker-compose.redis.yml up -d`.",
            file=sys.stderr,
        )
        return 1

    factory = _DEMO_ALGORITHMS[args.algorithm]
    limiter = factory(client, args.param1, args.param2)

    print(
        f"Lua demo: {args.algorithm} (param1={args.param1}, param2={args.param2}), "
        f"{args.requests} requests, {args.interval}s apart, cost={args.cost}, "
        f"redis={args.redis_host}:{args.redis_port}"
    )
    for i in range(args.requests):
        allowed = limiter.allow(args.key, cost=args.cost)
        remaining = limiter.remaining(args.key)
        status = "ALLOWED" if allowed else "DENIED "
        print(f"  request {i + 1:>3}: {status}  remaining={remaining}")
        if args.interval > 0 and i < args.requests - 1:
            time.sleep(args.interval)
    return 0


def _find_repo_root() -> Path:
    """Same best-effort search as the main cli.py's identical helper
    -- see that file's docstring for the full rationale (only works
    from a source checkout / editable install that still has
    benchmarks/ on disk next to the installed package)."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "benchmarks").is_dir():
            return candidate
    return Path.cwd()


def _cmd_lua_benchmark(args: argparse.Namespace) -> int:
    root = _find_repo_root()
    bench_dir = root / "benchmarks"
    if not bench_dir.is_dir():
        print(
            "Could not find the benchmarks/ directory. `limivault lua-benchmark` "
            "only works from a source checkout / editable install that still "
            "has benchmarks/ on disk -- see limivault/cli.py's module docstring.",
            file=sys.stderr,
        )
        return 1

    # Same argparse.REMAINDER "-- gets captured literally" quirk as
    # cli.py's `_cmd_benchmark` -- see that function's comment.
    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    py_args = [sys.executable, "-m", "benchmarks.lua_benchmark_async_concurrency"]
    py_args += passthrough
    result = subprocess.run(py_args, cwd=str(root))  # nosec
    return result.returncode
