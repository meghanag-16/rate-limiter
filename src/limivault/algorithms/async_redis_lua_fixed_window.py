# src/limivault/algorithms/async_redis_lua_fixed_window.py
"""AsyncRedisLuaFixedWindow -- async mirror of
redis_lua_fixed_window.RedisLuaFixedWindow. See that file's docstring
and limivault.redis_lua_scripts's module docstring for the full design
rationale. NO TESTS SHIP INLINE WITH THIS FILE -- see
tests/test_async_redis_lua_*.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import redis.exceptions

from limivault.algorithms._redis_lua_base import (
    emit_decision,
    emit_error,
    is_valid_cost_type,
    log_decision,
    reject_non_finite,
    validate_ttl_seconds,
)
from limivault.base import AsyncRateLimiter, UnsatisfiableRequestError
from limivault.exceptions import BackendUnavailableError
from limivault.logging import get_logger
from limivault.metrics import MetricsHook, default_metrics_hook
from limivault.redis_lua_scripts import (
    DEFAULT_KEY_PREFIX,
    FIXED_WINDOW,
    TTL_BUFFER_SECONDS,
)

if TYPE_CHECKING:
    import redis.asyncio

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_ALGORITHM_NAME = "AsyncRedisLuaFixedWindow"
_log = get_logger(__name__)


async def _server_time(client: "redis.asyncio.Redis") -> float:
    seconds, microseconds = await client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


class AsyncRedisLuaFixedWindow(AsyncRateLimiter):
    """Async, Redis-native, Lua-atomic fixed window counter."""

    def __init__(
        self,
        client: "redis.asyncio.Redis",
        limit: int,
        period: float,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: float | None = None,
        metrics: MetricsHook | None = None,
    ) -> None:
        reject_non_finite(limit, "limit")
        reject_non_finite(period, "period")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._client = client
        self._limit = limit
        self._period = period
        self._key_prefix = key_prefix
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else period * 2 + TTL_BUFFER_SECONDS
        )
        validate_ttl_seconds(self._ttl_seconds)
        self._script = client.register_script(FIXED_WINDOW)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}fixed_window:{key}"

    def _window_start(self, now: float) -> float:
        return (now // self._period) * self._period

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
                args=[cost, "0", self._limit, self._period, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            try:
                error_ts = await _server_time(self._client)
            except Exception:
                error_ts = 0.0
            emit_error(
                self._metrics,
                algorithm=_ALGORITHM_NAME,
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
        utilization = (self._limit - remaining) / self._limit
        snapshot = {"remaining": remaining, "limit": self._limit}
        log_decision(
            _log,
            "async_redis_lua_fixed_window_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            remaining=remaining,
            limit=self._limit,
        )
        emit_decision(
            self._metrics,
            algorithm=_ALGORITHM_NAME,
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
        if cost > self._limit:
            raise UnsatisfiableRequestError(
                f"cost {cost} exceeds fixed window limit {self._limit}; "
                "can never be satisfied regardless of wait time"
            )
        try:
            raw = await self._client.hmget(self._data_key(key), "window_start", "count")
            now = await _server_time(self._client)
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        window_start = self._window_start(now)
        stored_window = float(raw[0]) if raw[0] is not None else None
        count = int(float(raw[1])) if raw[1] is not None else 0
        if stored_window is None or stored_window != window_start:
            count = 0
        if count + cost <= self._limit:
            return 0.0
        next_window_start = window_start + self._period
        return max(0.0, next_window_start - now)

    async def remaining(self, key: str) -> int:
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._limit, self._period, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])
