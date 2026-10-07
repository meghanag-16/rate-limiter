# src/limivault/algorithms/__init__.py
"""All rate-limiter algorithm classes, importable from one place:

    from limivault.algorithms import FixedWindow, RedisGcraTokenBucket

The classes are exported lazily (PEP 562 module `__getattr__`): a name
is only imported from its submodule the first time it is accessed.
That is deliberate. Each algorithm module binds its structlog logger at
import time, and a logger bound before `configure_logging()` runs keeps
the old log level (see limivault/cli.py and the benchmarks/ scripts, which
call `configure_logging(level=WARNING)` before importing any algorithm).
If this package imported every algorithm eagerly, merely importing any
`limivault.algorithms.<module>` would bind all of them first and defeat
that ordering.

Families:
  - in-memory, sync:     FixedWindow, TokenBucket, SlidingWindowLog,
                         SlidingWindowCounter, LeakyBucketMeter,
                         LeakyBucketQueue
  - in-memory, async:    the same six with an `Async` prefix
  - Redis Lua/GCRA, sync:  RedisLuaFixedWindow, RedisGcraTokenBucket,
                         RedisLuaSlidingWindowLog,
                         RedisLuaSlidingWindowCounter,
                         RedisLuaLeakyBucketMeter,
                         RedisLuaLeakyBucketQueue
  - Redis Lua/GCRA, async: the same six with an `Async` prefix
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limivault.algorithms.async_fixed_window import AsyncFixedWindow
    from limivault.algorithms.async_leaky_bucket import (
        AsyncLeakyBucketMeter,
        AsyncLeakyBucketQueue,
    )
    from limivault.algorithms.async_redis_lua_fixed_window import (
        AsyncRedisLuaFixedWindow,
    )
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
    from limivault.algorithms.async_redis_lua_token_bucket import (
        AsyncRedisGcraTokenBucket,
    )
    from limivault.algorithms.async_sliding_window_counter import (
        AsyncSlidingWindowCounter,
    )
    from limivault.algorithms.async_sliding_window_log import AsyncSlidingWindowLog
    from limivault.algorithms.async_token_bucket import AsyncTokenBucket
    from limivault.algorithms.fixed_window import FixedWindow
    from limivault.algorithms.leaky_bucket import (
        LeakyBucketMeter,
        LeakyBucketQueue,
    )
    from limivault.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
    from limivault.algorithms.redis_lua_leaky_bucket import (
        RedisLuaLeakyBucketMeter,
        RedisLuaLeakyBucketQueue,
    )
    from limivault.algorithms.redis_lua_sliding_window_counter import (
        RedisLuaSlidingWindowCounter,
    )
    from limivault.algorithms.redis_lua_sliding_window_log import (
        RedisLuaSlidingWindowLog,
    )
    from limivault.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
    from limivault.algorithms.sliding_window_counter import SlidingWindowCounter
    from limivault.algorithms.sliding_window_log import SlidingWindowLog
    from limivault.algorithms.token_bucket import TokenBucket

# name -> submodule (relative to this package) that defines it.
_EXPORTS: dict[str, str] = {
    "FixedWindow": "fixed_window",
    "TokenBucket": "token_bucket",
    "SlidingWindowLog": "sliding_window_log",
    "SlidingWindowCounter": "sliding_window_counter",
    "LeakyBucketMeter": "leaky_bucket",
    "LeakyBucketQueue": "leaky_bucket",
    "AsyncFixedWindow": "async_fixed_window",
    "AsyncTokenBucket": "async_token_bucket",
    "AsyncSlidingWindowLog": "async_sliding_window_log",
    "AsyncSlidingWindowCounter": "async_sliding_window_counter",
    "AsyncLeakyBucketMeter": "async_leaky_bucket",
    "AsyncLeakyBucketQueue": "async_leaky_bucket",
    "RedisLuaFixedWindow": "redis_lua_fixed_window",
    "RedisGcraTokenBucket": "redis_lua_token_bucket",
    "RedisLuaSlidingWindowLog": "redis_lua_sliding_window_log",
    "RedisLuaSlidingWindowCounter": "redis_lua_sliding_window_counter",
    "RedisLuaLeakyBucketMeter": "redis_lua_leaky_bucket",
    "RedisLuaLeakyBucketQueue": "redis_lua_leaky_bucket",
    "AsyncRedisLuaFixedWindow": "async_redis_lua_fixed_window",
    "AsyncRedisGcraTokenBucket": "async_redis_lua_token_bucket",
    "AsyncRedisLuaSlidingWindowLog": "async_redis_lua_sliding_window_log",
    "AsyncRedisLuaSlidingWindowCounter": "async_redis_lua_sliding_window_counter",
    "AsyncRedisLuaLeakyBucketMeter": "async_redis_lua_leaky_bucket",
    "AsyncRedisLuaLeakyBucketQueue": "async_redis_lua_leaky_bucket",
}

__all__ = [
    "FixedWindow",
    "TokenBucket",
    "SlidingWindowLog",
    "SlidingWindowCounter",
    "LeakyBucketMeter",
    "LeakyBucketQueue",
    "AsyncFixedWindow",
    "AsyncTokenBucket",
    "AsyncSlidingWindowLog",
    "AsyncSlidingWindowCounter",
    "AsyncLeakyBucketMeter",
    "AsyncLeakyBucketQueue",
    "RedisLuaFixedWindow",
    "RedisGcraTokenBucket",
    "RedisLuaSlidingWindowLog",
    "RedisLuaSlidingWindowCounter",
    "RedisLuaLeakyBucketMeter",
    "RedisLuaLeakyBucketQueue",
    "AsyncRedisLuaFixedWindow",
    "AsyncRedisGcraTokenBucket",
    "AsyncRedisLuaSlidingWindowLog",
    "AsyncRedisLuaSlidingWindowCounter",
    "AsyncRedisLuaLeakyBucketMeter",
    "AsyncRedisLuaLeakyBucketQueue",
]


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
