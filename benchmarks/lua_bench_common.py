# benchmarks/lua_bench_common.py
"""Factory table for the 12 concrete Redis-native Lua/GCRA
limiter classes (6 sync + 6 async), mirroring the shape of the
existing project's benchmarks/bench_common.py for the in-memory
algorithms -- same "one small factory per concrete class, looped
over" pattern already used there and in
tests/test_metrics_all_algorithms.py / tests/test_cost_contract.py.

WHY THIS FILE IS SEPARATE FROM bench_common.py, NOT MERGED INTO IT:
The Lua/GCRA classes take a `redis.Redis` / `redis.asyncio.Redis` CLIENT
as their first constructor argument, not a `storage=` /
`clock=time.monotonic` pair -- their whole shape is different (see
redis_lua_scripts.py's module docstring: no client-side clock
injection at all for this family). Threading an `if is_lua: ...`
branch through the existing bench_common.py factories would make that
file harder to read for no benefit; a second, parallel factory table
scoped to just this family is more transparent.

WHY THERE IS NO `lua_memory_footprint.py` COMPANION (unlike
memory_footprint.py for the in-memory algorithms): that script
measures Python-process memory growth as unique key cardinality grows,
because InMemoryStorage genuinely accumulates one dict entry (+ one
lock, for the default backend) per key in the CLIENT process. The
Lua/GCRA classes' state lives entirely in REDIS -- every algorithm's
`_data_key()` result is a Redis key, not a Python object -- so
allow()'ing N unique keys does NOT grow this process's own memory in any
way proportional to N; the
only per-instance Python-side state is the constant handful of
attributes set in `__init__` (client reference, limit/capacity,
period/rate, the registered Script object) plus whatever redis-py's
own connection pool retains, none of which is what
tracemalloc-per-N-keys is designed to detect. A "memory footprint"
benchmark for this family would therefore either report ~0 growth
(true, but not informative) or measure Redis server-side memory
instead (a legitimate but entirely different tool -- `redis-cli
--bigkeys` / `MEMORY USAGE <key>` / `INFO memory`, not tracemalloc).
Flagged here explicitly rather than silently omitted.

Every factory uses a deliberately huge limit/capacity (10**9), same
rationale as bench_common.py's identical choice: keeps allow() calls
essentially always admitted, so the benchmark measures the steady-state
cost of one round trip + Lua execution, not how often the deny
early-exit happens to trigger.

Each factory takes the already-connected client and a `key_prefix`
(REQUIRED, not defaulted) so the caller can give every matrix cell /
repetition its own randomized namespace -- see
lua_benchmark_async_concurrency.py's module docstring, point 3, for
why reusing one key_prefix across repetitions would corrupt counts
the same way reusing one limiter/storage would in the existing
concurrency_harness.py (point 5 there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from limivault.algorithms.async_redis_lua_fixed_window import AsyncRedisLuaFixedWindow
from limivault.algorithms.async_redis_lua_leaky_bucket import (
    AsyncRedisLuaLeakyBucketMeter,
    AsyncRedisLuaLeakyBucketQueue,
)
from limivault.algorithms.async_redis_lua_sliding_window_counter import (
    AsyncRedisLuaSlidingWindowCounter,
)
from limivault.algorithms.async_redis_lua_sliding_window_log import (
    AsyncRedisLuaSlidingWindowLog,
)
from limivault.algorithms.async_redis_lua_token_bucket import AsyncRedisGcraTokenBucket
from limivault.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from limivault.algorithms.redis_lua_leaky_bucket import (
    RedisLuaLeakyBucketMeter,
    RedisLuaLeakyBucketQueue,
)
from limivault.algorithms.redis_lua_sliding_window_counter import (
    RedisLuaSlidingWindowCounter,
)
from limivault.algorithms.redis_lua_sliding_window_log import RedisLuaSlidingWindowLog
from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket

if TYPE_CHECKING:
    import redis
    import redis.asyncio

_HUGE = 10**9
_RATE = 1.0  # period (seconds) or refill/leak rate per second, huge-limit-relative

# Each entry: (display_name, factory(client, key_prefix) -> limiter).
# `client` for SYNC_FACTORIES is a redis.Redis; for ASYNC_FACTORIES a
# redis.asyncio.Redis. `key_prefix` is required from the caller (see
# module docstring) rather than defaulted to DEFAULT_KEY_PREFIX, so
# every benchmark run/repetition gets an isolated namespace.

SYNC_FACTORIES: List[Tuple[str, Callable[["redis.Redis", str], Any]]] = [
    (
        "RedisLuaFixedWindow",
        lambda client, prefix: RedisLuaFixedWindow(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "RedisGcraTokenBucket",
        lambda client, prefix: RedisGcraTokenBucket(
            client, capacity=_HUGE, refill_rate=_RATE, key_prefix=prefix
        ),
    ),
    (
        "RedisLuaSlidingWindowLog",
        lambda client, prefix: RedisLuaSlidingWindowLog(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "RedisLuaSlidingWindowCounter",
        lambda client, prefix: RedisLuaSlidingWindowCounter(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "RedisLuaLeakyBucketMeter",
        lambda client, prefix: RedisLuaLeakyBucketMeter(
            client, capacity=_HUGE, leak_rate=_RATE, key_prefix=prefix
        ),
    ),
    (
        "RedisLuaLeakyBucketQueue",
        lambda client, prefix: RedisLuaLeakyBucketQueue(
            client, capacity=_HUGE, leak_rate=_RATE, key_prefix=prefix
        ),
    ),
]

ASYNC_FACTORIES: List[Tuple[str, Callable[["redis.asyncio.Redis", str], Any]]] = [
    (
        "AsyncRedisLuaFixedWindow",
        lambda client, prefix: AsyncRedisLuaFixedWindow(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "AsyncRedisGcraTokenBucket",
        lambda client, prefix: AsyncRedisGcraTokenBucket(
            client, capacity=_HUGE, refill_rate=_RATE, key_prefix=prefix
        ),
    ),
    (
        "AsyncRedisLuaSlidingWindowLog",
        lambda client, prefix: AsyncRedisLuaSlidingWindowLog(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "AsyncRedisLuaSlidingWindowCounter",
        lambda client, prefix: AsyncRedisLuaSlidingWindowCounter(
            client, limit=_HUGE, period=3600.0, key_prefix=prefix
        ),
    ),
    (
        "AsyncRedisLuaLeakyBucketMeter",
        lambda client, prefix: AsyncRedisLuaLeakyBucketMeter(
            client, capacity=_HUGE, leak_rate=_RATE, key_prefix=prefix
        ),
    ),
    (
        "AsyncRedisLuaLeakyBucketQueue",
        lambda client, prefix: AsyncRedisLuaLeakyBucketQueue(
            client, capacity=_HUGE, leak_rate=_RATE, key_prefix=prefix
        ),
    ),
]


def all_factory_names() -> List[str]:
    return [n for n, _ in SYNC_FACTORIES] + [n for n, _ in ASYNC_FACTORIES]
