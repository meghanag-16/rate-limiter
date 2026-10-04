"""Public API for the rlimit package.

Algorithm classes are exported lazily so importing rlimit does not
eagerly import every algorithm module.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlimit.algorithms.async_fixed_window import (
        AsyncFixedWindow as AsyncFixedWindow,
    )
    from rlimit.algorithms.async_leaky_bucket import (
        AsyncLeakyBucketMeter as AsyncLeakyBucketMeter,
    )
    from rlimit.algorithms.async_leaky_bucket import (
        AsyncLeakyBucketQueue as AsyncLeakyBucketQueue,
    )
    from rlimit.algorithms.async_redis_lua_fixed_window import (
        AsyncRedisLuaFixedWindow as AsyncRedisLuaFixedWindow,
    )
    from rlimit.algorithms.async_redis_lua_leaky_bucket import (
        AsyncRedisLuaLeakyBucketMeter as AsyncRedisLuaLeakyBucketMeter,
    )
    from rlimit.algorithms.async_redis_lua_leaky_bucket import (
        AsyncRedisLuaLeakyBucketQueue as AsyncRedisLuaLeakyBucketQueue,
    )
    from rlimit.algorithms.async_redis_lua_sliding_window_counter import (
        AsyncRedisLuaSlidingWindowCounter as AsyncRedisLuaSlidingWindowCounter,
    )
    from rlimit.algorithms.async_redis_lua_sliding_window_log import (
        AsyncRedisLuaSlidingWindowLog as AsyncRedisLuaSlidingWindowLog,
    )
    from rlimit.algorithms.async_redis_lua_token_bucket import (
        AsyncRedisGcraTokenBucket as AsyncRedisGcraTokenBucket,
    )
    from rlimit.algorithms.async_sliding_window_counter import (
        AsyncSlidingWindowCounter as AsyncSlidingWindowCounter,
    )
    from rlimit.algorithms.async_sliding_window_log import (
        AsyncSlidingWindowLog as AsyncSlidingWindowLog,
    )
    from rlimit.algorithms.async_token_bucket import (
        AsyncTokenBucket as AsyncTokenBucket,
    )
    from rlimit.algorithms.fixed_window import FixedWindow as FixedWindow
    from rlimit.algorithms.leaky_bucket import (
        LeakyBucketMeter as LeakyBucketMeter,
    )
    from rlimit.algorithms.leaky_bucket import (
        LeakyBucketQueue as LeakyBucketQueue,
    )
    from rlimit.algorithms.redis_lua_fixed_window import (
        RedisLuaFixedWindow as RedisLuaFixedWindow,
    )
    from rlimit.algorithms.redis_lua_leaky_bucket import (
        RedisLuaLeakyBucketMeter as RedisLuaLeakyBucketMeter,
    )
    from rlimit.algorithms.redis_lua_leaky_bucket import (
        RedisLuaLeakyBucketQueue as RedisLuaLeakyBucketQueue,
    )
    from rlimit.algorithms.redis_lua_sliding_window_counter import (
        RedisLuaSlidingWindowCounter as RedisLuaSlidingWindowCounter,
    )
    from rlimit.algorithms.redis_lua_sliding_window_log import (
        RedisLuaSlidingWindowLog as RedisLuaSlidingWindowLog,
    )
    from rlimit.algorithms.redis_lua_token_bucket import (
        RedisGcraTokenBucket as RedisGcraTokenBucket,
    )
    from rlimit.algorithms.sliding_window_counter import (
        SlidingWindowCounter as SlidingWindowCounter,
    )
    from rlimit.algorithms.sliding_window_log import (
        SlidingWindowLog as SlidingWindowLog,
    )
    from rlimit.algorithms.token_bucket import TokenBucket as TokenBucket
    from rlimit.async_ergonomics import (
        AsyncKeyedLimiter as AsyncKeyedLimiter,
    )
    from rlimit.async_ergonomics import (
        async_block_until_allowed as async_block_until_allowed,
    )
    from rlimit.async_ergonomics import (
        async_rate_limit as async_rate_limit,
    )
    from rlimit.async_ergonomics import (
        async_wait as async_wait,
    )
    from rlimit.base import (
        AsyncRateLimiter as AsyncRateLimiter,
    )
    from rlimit.base import (
        RateLimiter as RateLimiter,
    )
    from rlimit.base import (
        UnsatisfiableRequestError as UnsatisfiableRequestError,
    )
    from rlimit.ergonomics import (
        KeyedLimiter as KeyedLimiter,
    )
    from rlimit.ergonomics import (
        RateLimitTimeoutError as RateLimitTimeoutError,
    )
    from rlimit.ergonomics import (
        block_until_allowed as block_until_allowed,
    )
    from rlimit.ergonomics import (
        rate_limit as rate_limit,
    )
    from rlimit.ergonomics import (
        wait as wait,
    )
    from rlimit.exceptions import BackendUnavailableError as BackendUnavailableError
    from rlimit.metrics import (
        AllowedEvent as AllowedEvent,
    )
    from rlimit.metrics import (
        BackendErrorEvent as BackendErrorEvent,
    )
    from rlimit.metrics import (
        DeniedEvent as DeniedEvent,
    )
    from rlimit.metrics import (
        MetricsHook as MetricsHook,
    )
    from rlimit.metrics import (
        NoOpMetricsHook as NoOpMetricsHook,
    )
    from rlimit.rlimit_lua_simulation import (
        LuaSimulationRecorder as LuaSimulationRecorder,
    )
    from rlimit.rlimit_lua_simulation import (
        async_run_lua_simulation as async_run_lua_simulation,
    )
    from rlimit.rlimit_lua_simulation import (
        run_lua_simulation as run_lua_simulation,
    )
    from rlimit.simulation import (
        SimulationClock as SimulationClock,
    )
    from rlimit.simulation import (
        SimulationRecorder as SimulationRecorder,
    )
    from rlimit.simulation import (
        TrafficStep as TrafficStep,
    )
    from rlimit.simulation import (
        async_run_simulation as async_run_simulation,
    )
    from rlimit.simulation import (
        bursty_traffic as bursty_traffic,
    )
    from rlimit.simulation import (
        constant_rate_traffic as constant_rate_traffic,
    )
    from rlimit.simulation import (
        random_traffic as random_traffic,
    )
    from rlimit.simulation import (
        run_simulation as run_simulation,
    )


# Public name -> submodule relative to rlimit.
_EXPORTS: dict[str, str] = {
    "RateLimiter": "base",
    "AsyncRateLimiter": "base",
    "UnsatisfiableRequestError": "base",
    "BackendUnavailableError": "exceptions",
    "AllowedEvent": "metrics",
    "DeniedEvent": "metrics",
    "BackendErrorEvent": "metrics",
    "MetricsHook": "metrics",
    "NoOpMetricsHook": "metrics",
    "RateLimitTimeoutError": "ergonomics",
    "KeyedLimiter": "ergonomics",
    "block_until_allowed": "ergonomics",
    "rate_limit": "ergonomics",
    "wait": "ergonomics",
    "AsyncKeyedLimiter": "async_ergonomics",
    "async_block_until_allowed": "async_ergonomics",
    "async_rate_limit": "async_ergonomics",
    "async_wait": "async_ergonomics",
    "SimulationClock": "simulation",
    "SimulationRecorder": "simulation",
    "TrafficStep": "simulation",
    "run_simulation": "simulation",
    "async_run_simulation": "simulation",
    "constant_rate_traffic": "simulation",
    "bursty_traffic": "simulation",
    "random_traffic": "simulation",
    "LuaSimulationRecorder": "rlimit_lua_simulation",
    "run_lua_simulation": "rlimit_lua_simulation",
    "async_run_lua_simulation": "rlimit_lua_simulation",
    "FixedWindow": "algorithms.fixed_window",
    "TokenBucket": "algorithms.token_bucket",
    "SlidingWindowLog": "algorithms.sliding_window_log",
    "SlidingWindowCounter": "algorithms.sliding_window_counter",
    "LeakyBucketMeter": "algorithms.leaky_bucket",
    "LeakyBucketQueue": "algorithms.leaky_bucket",
    "AsyncFixedWindow": "algorithms.async_fixed_window",
    "AsyncTokenBucket": "algorithms.async_token_bucket",
    "AsyncSlidingWindowLog": "algorithms.async_sliding_window_log",
    "AsyncSlidingWindowCounter": "algorithms.async_sliding_window_counter",
    "AsyncLeakyBucketMeter": "algorithms.async_leaky_bucket",
    "AsyncLeakyBucketQueue": "algorithms.async_leaky_bucket",
    "RedisLuaFixedWindow": "algorithms.redis_lua_fixed_window",
    "RedisGcraTokenBucket": "algorithms.redis_lua_token_bucket",
    "RedisLuaSlidingWindowLog": "algorithms.redis_lua_sliding_window_log",
    "RedisLuaSlidingWindowCounter": "algorithms.redis_lua_sliding_window_counter",
    "RedisLuaLeakyBucketMeter": "algorithms.redis_lua_leaky_bucket",
    "RedisLuaLeakyBucketQueue": "algorithms.redis_lua_leaky_bucket",
    "AsyncRedisLuaFixedWindow": "algorithms.async_redis_lua_fixed_window",
    "AsyncRedisGcraTokenBucket": "algorithms.async_redis_lua_token_bucket",
    "AsyncRedisLuaSlidingWindowLog": "algorithms.async_redis_lua_sliding_window_log",
    "AsyncRedisLuaSlidingWindowCounter": (
        "algorithms.async_redis_lua_sliding_window_counter"
    ),
    "AsyncRedisLuaLeakyBucketMeter": "algorithms.async_redis_lua_leaky_bucket",
    "AsyncRedisLuaLeakyBucketQueue": "algorithms.async_redis_lua_leaky_bucket",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public algorithm classes lazily."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    value = getattr(
        importlib.import_module(f"{__name__}.{submodule}"),
        name,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in dir(rlimit)."""
    return sorted(set(globals()) | set(__all__))
