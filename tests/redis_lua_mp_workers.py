# tests/redis_lua_mp_workers.py
"""Top-level worker functions for the Redis Lua/GCRA multiprocessing
tests. These must be plain module-level functions, not closures, so
that `spawn`-based multiprocessing (the default on Windows/macOS) can
pickle them. Each worker takes plain host/port connection parameters
and builds a fresh client + limiter inside its own child process,
never a pre-built client passed in from the parent (a `redis.Redis`
client holds live socket connections that don't survive a
pickle/unpickle round trip the way a picklable Manager proxy does).

v2 note: none of these limiters take a `clock=` argument (see
rlimit.redis_lua_scripts's module docstring), so no worker below
passes one.
"""

from __future__ import annotations

import redis


def hammer_lua_fixed_window(
    host: str, port: int, limit: int, period: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow

    client = redis.Redis(host=host, port=port)
    limiter = RedisLuaFixedWindow(client, limit=limit, period=period)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count


def hammer_lua_gcra_token_bucket(
    host: str, port: int, capacity: int, refill_rate: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket

    client = redis.Redis(host=host, port=port)
    limiter = RedisGcraTokenBucket(client, capacity=capacity, refill_rate=refill_rate)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count


def hammer_lua_sliding_window_log(
    host: str, port: int, limit: int, period: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_sliding_window_log import RedisLuaSlidingWindowLog

    client = redis.Redis(host=host, port=port)
    limiter = RedisLuaSlidingWindowLog(client, limit=limit, period=period)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count


def hammer_lua_sliding_window_counter(
    host: str, port: int, limit: int, period: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_sliding_window_counter import (
        RedisLuaSlidingWindowCounter,
    )

    client = redis.Redis(host=host, port=port)
    limiter = RedisLuaSlidingWindowCounter(client, limit=limit, period=period)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count


def hammer_lua_leaky_bucket_meter(
    host: str, port: int, capacity: int, leak_rate: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_leaky_bucket import RedisLuaLeakyBucketMeter

    client = redis.Redis(host=host, port=port)
    limiter = RedisLuaLeakyBucketMeter(client, capacity=capacity, leak_rate=leak_rate)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count


def hammer_lua_leaky_bucket_queue(
    host: str, port: int, capacity: int, leak_rate: float, key: str, attempts: int
) -> int:
    from rlimit.algorithms.redis_lua_leaky_bucket import RedisLuaLeakyBucketQueue

    client = redis.Redis(host=host, port=port)
    limiter = RedisLuaLeakyBucketQueue(client, capacity=capacity, leak_rate=leak_rate)
    count = sum(1 for _ in range(attempts) if limiter.allow(key))
    client.close()
    return count
