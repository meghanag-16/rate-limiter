# src/rlimit/algorithms/async_redis_lua_leaky_bucket.py
"""AsyncRedisLuaLeakyBucketMeter / AsyncRedisLuaLeakyBucketQueue --
async mirrors of redis_lua_leaky_bucket.py's two classes. See that
file's docstring and rlimit.redis_lua_scripts's module docstring for
the full design rationale. NO TESTS SHIP INLINE WITH THIS FILE -- see
tests/test_async_redis_lua_*.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import redis.exceptions

from rlimit.algorithms._redis_lua_base import (
    emit_decision,
    emit_error,
    is_valid_cost_type,
    log_decision,
    reject_non_finite,
    validate_ttl_seconds,
)
from rlimit.base import AsyncRateLimiter, UnsatisfiableRequestError
from rlimit.exceptions import BackendUnavailableError
from rlimit.logging import get_logger
from rlimit.metrics import MetricsHook, default_metrics_hook
from rlimit.redis_lua_scripts import (
    DEFAULT_KEY_PREFIX,
    FALLBACK_TTL_SECONDS,
    LEAKY_BUCKET_METER,
    LEAKY_BUCKET_QUEUE,
    TTL_BUFFER_SECONDS,
)

if TYPE_CHECKING:
    import redis.asyncio

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_log = get_logger(__name__)


async def _server_time(client: "redis.asyncio.Redis") -> float:
    seconds, microseconds = await client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


def _compute_ttl(capacity: int, leak_rate: float, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    if leak_rate > 0:
        return (capacity / leak_rate) + TTL_BUFFER_SECONDS
    return FALLBACK_TTL_SECONDS


class AsyncRedisLuaLeakyBucketMeter(AsyncRateLimiter):
    """Async, Redis-native, Lua-atomic leaky bucket (meter variant)."""

    _ALGORITHM_NAME = "AsyncRedisLuaLeakyBucketMeter"

    def __init__(
        self,
        client: "redis.asyncio.Redis",
        capacity: int,
        leak_rate: float,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: float | None = None,
        metrics: MetricsHook | None = None,
    ) -> None:
        reject_non_finite(capacity, "capacity")
        reject_non_finite(leak_rate, "leak_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if leak_rate < 0:
            raise ValueError(f"leak_rate must be non-negative, got {leak_rate}")
        self._client = client
        self._capacity = capacity
        self._leak_rate = leak_rate
        self._key_prefix = key_prefix
        self._ttl_seconds = _compute_ttl(capacity, leak_rate, ttl_seconds)
        validate_ttl_seconds(self._ttl_seconds)
        self._script = client.register_script(LEAKY_BUCKET_METER)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}lb_meter:{key}"

    async def allow(self, key: str, cost: int = 1) -> bool:
        if not is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[cost, "0", self._capacity, self._leak_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            try:
                error_ts = await _server_time(self._client)
            except Exception:
                error_ts = 0.0
            emit_error(
                self._metrics,
                algorithm=self._ALGORITHM_NAME,
                key=key,
                cost=cost,
                error=exc,
                timestamp=error_ts,
            )
            raise BackendUnavailableError(
                f"Redis backend unavailable while running script for key {key!r}"
            ) from exc
        allowed = bool(int(result[0]))
        remaining = int(result[2])
        now = float(result[3])
        utilization = (self._capacity - remaining) / self._capacity
        snapshot = {"remaining": remaining, "capacity": self._capacity}
        log_decision(
            _log,
            "async_redis_lua_leaky_bucket_meter_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            remaining=remaining,
            capacity=self._capacity,
        )
        emit_decision(
            self._metrics,
            algorithm=self._ALGORITHM_NAME,
            key=key,
            cost=cost,
            allowed=allowed,
            state=snapshot,
            utilization=utilization,
            timestamp=now,
        )
        return allowed

    async def allow_wait(self, key: str, cost: int = 1) -> float:
        if not is_valid_cost_type(cost):
            raise ValueError(
                f"cost must be an int, got {type(cost).__name__}: {cost!r}"
            )
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")
        if cost == 0:
            return 0.0
        if cost > self._capacity:
            raise UnsatisfiableRequestError(
                f"cost {cost} exceeds leaky bucket capacity {self._capacity}; "
                "can never be satisfied regardless of wait time"
            )
        try:
            raw = await self._client.hmget(self._data_key(key), "volume", "last_leak")
            now = await _server_time(self._client)
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        volume = float(raw[0]) if raw[0] is not None else 0.0
        last_leak = float(raw[1]) if raw[1] is not None else now
        elapsed = max(0.0, now - last_leak)
        volume = max(0.0, volume - elapsed * self._leak_rate)
        if volume + cost <= self._capacity:
            return 0.0
        excess = (volume + cost) - self._capacity
        if self._leak_rate <= 0:
            raise UnsatisfiableRequestError(
                f"leak_rate is {self._leak_rate}; bucket will never "
                "drain enough to satisfy this request"
            )
        return excess / self._leak_rate

    async def remaining(self, key: str) -> int:
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._capacity, self._leak_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])


class AsyncRedisLuaLeakyBucketQueue(AsyncRateLimiter):
    """Async, Redis-native, Lua-atomic leaky bucket (queue variant)."""

    _ALGORITHM_NAME = "AsyncRedisLuaLeakyBucketQueue"

    def __init__(
        self,
        client: "redis.asyncio.Redis",
        capacity: int,
        leak_rate: float,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: float | None = None,
        metrics: MetricsHook | None = None,
    ) -> None:
        reject_non_finite(capacity, "capacity")
        reject_non_finite(leak_rate, "leak_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if leak_rate < 0:
            raise ValueError(f"leak_rate must be non-negative, got {leak_rate}")
        self._client = client
        self._capacity = capacity
        self._leak_rate = leak_rate
        self._key_prefix = key_prefix
        self._ttl_seconds = _compute_ttl(capacity, leak_rate, ttl_seconds)
        validate_ttl_seconds(self._ttl_seconds)
        self._script = client.register_script(LEAKY_BUCKET_QUEUE)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}lb_queue:{key}"

    async def allow(self, key: str, cost: int = 1) -> bool:
        if not is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[cost, "0", self._capacity, self._leak_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            try:
                error_ts = await _server_time(self._client)
            except Exception:
                error_ts = 0.0
            emit_error(
                self._metrics,
                algorithm=self._ALGORITHM_NAME,
                key=key,
                cost=cost,
                error=exc,
                timestamp=error_ts,
            )
            raise BackendUnavailableError(
                f"Redis backend unavailable while running script for key {key!r}"
            ) from exc
        allowed = bool(int(result[0]))
        remaining = int(result[2])
        now = float(result[3])
        utilization = (self._capacity - remaining) / self._capacity
        snapshot = {"remaining": remaining, "capacity": self._capacity}
        log_decision(
            _log,
            "async_redis_lua_leaky_bucket_queue_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            remaining=remaining,
            capacity=self._capacity,
        )
        emit_decision(
            self._metrics,
            algorithm=self._ALGORITHM_NAME,
            key=key,
            cost=cost,
            allowed=allowed,
            state=snapshot,
            utilization=utilization,
            timestamp=now,
        )
        return allowed

    async def allow_wait(self, key: str, cost: int = 1) -> float:
        if not is_valid_cost_type(cost):
            raise ValueError(
                f"cost must be an int, got {type(cost).__name__}: {cost!r}"
            )
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")
        if cost == 0:
            return 0.0
        if cost > self._capacity:
            raise UnsatisfiableRequestError(
                f"cost {cost} exceeds leaky bucket queue capacity "
                f"{self._capacity}; can never be satisfied regardless of "
                "wait time"
            )
        try:
            raw = await self._client.hmget(self._data_key(key), "depth", "last_drain")
            now = await _server_time(self._client)
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        depth = int(float(raw[0])) if raw[0] is not None else 0
        last_drain = float(raw[1]) if raw[1] is not None else now
        elapsed = max(0.0, now - last_drain)
        drained = math.floor(elapsed * self._leak_rate) if self._leak_rate > 0 else 0
        depth = max(0, depth - drained)
        if drained > 0 and self._leak_rate > 0:
            last_drain = last_drain + drained / self._leak_rate

        if depth + cost <= self._capacity:
            return 0.0
        if self._leak_rate <= 0:
            raise UnsatisfiableRequestError(
                f"leak_rate is {self._leak_rate}; queue will never "
                "drain enough to satisfy this request"
            )
        items_to_drain = (depth + cost) - self._capacity
        next_drain_at = last_drain + (1 / self._leak_rate)
        wait_for_first = max(0.0, next_drain_at - now)
        remaining_items = items_to_drain - 1
        wait_for_rest = max(0, remaining_items) / self._leak_rate
        return wait_for_first + wait_for_rest

    async def remaining(self, key: str) -> int:
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._capacity, self._leak_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])
