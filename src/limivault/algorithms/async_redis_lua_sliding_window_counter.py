# src/limivault/algorithms/async_redis_lua_sliding_window_counter.py
"""AsyncRedisLuaSlidingWindowCounter -- async mirror of
redis_lua_sliding_window_counter.RedisLuaSlidingWindowCounter. See
that file's docstring and limivault.redis_lua_scripts's module docstring
for the full design rationale. NO TESTS SHIP INLINE WITH THIS FILE --
see tests/test_async_redis_lua_*.py.
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
    SLIDING_WINDOW_COUNTER,
    TTL_BUFFER_SECONDS,
)

if TYPE_CHECKING:
    import redis.asyncio

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_WAIT_EPSILON = 1e-9
_ALGORITHM_NAME = "AsyncRedisLuaSlidingWindowCounter"
_log = get_logger(__name__)


async def _server_time(client: "redis.asyncio.Redis") -> float:
    seconds, microseconds = await client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


class AsyncRedisLuaSlidingWindowCounter(AsyncRateLimiter):
    """Async, Redis-native, Lua-atomic sliding window counter."""

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
        self._script = client.register_script(SLIDING_WINDOW_COUNTER)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}sw_counter:{key}"

    def _window_start(self, now: float) -> float:
        return (now // self._period) * self._period

    async def _get_raw_state(self, key: str, now: float) -> tuple[float, int, int]:
        window_start = self._window_start(now)
        try:
            raw = await self._client.hmget(
                self._data_key(key), "window_start", "count", "prev_count"
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        stored_window = float(raw[0]) if raw[0] is not None else None
        count = int(float(raw[1])) if raw[1] is not None else 0
        prev_count = int(float(raw[2])) if raw[2] is not None else 0
        if stored_window is None:
            return window_start, 0, 0
        if stored_window == window_start:
            return window_start, count, prev_count
        if stored_window == window_start - self._period:
            return window_start, 0, count
        return window_start, 0, 0

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
            "async_redis_lua_sliding_window_counter_decision",
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
                f"cost {cost} exceeds sliding window counter limit "
                f"{self._limit}; can never be satisfied regardless of wait time"
            )
        try:
            now = await _server_time(self._client)
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        window_start, count, prev_count = await self._get_raw_state(key, now)
        elapsed_now = now - window_start
        overlap = max(0.0, (self._period - elapsed_now) / self._period)
        weighted = count + prev_count * overlap
        if weighted + cost <= self._limit:
            return 0.0

        target = self._limit - cost
        remaining_in_window = self._period - elapsed_now

        if prev_count > 0:
            decay_needed = weighted - target
            elapsed_needed = decay_needed * self._period / prev_count
            if elapsed_needed <= remaining_in_window:
                return max(0.0, elapsed_needed) + _WAIT_EPSILON

        if count <= 0:
            return max(0.0, remaining_in_window) + _WAIT_EPSILON
        extra_elapsed = self._period * (1 - target / count)
        extra_elapsed = max(0.0, extra_elapsed)
        return remaining_in_window + extra_elapsed + _WAIT_EPSILON

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
