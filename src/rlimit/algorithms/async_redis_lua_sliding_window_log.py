# src/rlimit/algorithms/async_redis_lua_sliding_window_log.py
"""AsyncRedisLuaSlidingWindowLog -- async mirror of
redis_lua_sliding_window_log.RedisLuaSlidingWindowLog. See that file's
docstring and rlimit.redis_lua_scripts's module docstring ("KNOWN,
DOCUMENTED GAP -- SlidingWindowLog remains O(n) per call") for the
full design rationale and the explicit statement that this is a
constant-factor/round-trip improvement, not an asymptotic fix.

NO TESTS SHIP INLINE WITH THIS FILE -- see tests/test_async_redis_lua_*.py.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

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
    SLIDING_WINDOW_LOG,
    TTL_BUFFER_SECONDS,
)

if TYPE_CHECKING:
    import redis.asyncio

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_ALGORITHM_NAME = "AsyncRedisLuaSlidingWindowLog"
_log = get_logger(__name__)


async def _server_time(client: "redis.asyncio.Redis") -> float:
    seconds, microseconds = await client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


def _parse_cost(member: bytes | str) -> int:
    text = member.decode("utf-8") if isinstance(member, bytes) else member
    return int(text.split(":", 1)[0])


class AsyncRedisLuaSlidingWindowLog(AsyncRateLimiter):
    """Async, Redis-native, Lua-atomic sliding window log (ZSET-backed)."""

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
        self._script = client.register_script(SLIDING_WINDOW_LOG)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}sw_log:{key}"

    async def allow(self, key: str, cost: int = 1) -> bool:
        if not is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True
        suffix = uuid.uuid4().hex
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[cost, "0", self._limit, self._period, self._ttl_seconds, suffix],
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
            "async_redis_lua_sliding_window_log_decision",
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
                f"cost {cost} exceeds sliding window log limit {self._limit}; "
                "can never be satisfied regardless of wait time"
            )
        try:
            now = await _server_time(self._client)
            cutoff = now - self._period
            await self._client.zremrangebyscore(self._data_key(key), "-inf", cutoff)
            raw_entries = await self._client.zrange(
                self._data_key(key), 0, -1, withscores=True
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc

        typed_entries = cast(list[tuple[bytes | str, float]], raw_entries)
        entries = [(ts, _parse_cost(member)) for member, ts in typed_entries]
        current_total = sum(c for _, c in entries)
        if current_total + cost <= self._limit:
            return 0.0

        sorted_entries = sorted(entries, key=lambda e: e[0])
        running_total = current_total
        for ts, entry_cost in sorted_entries:
            running_total -= entry_cost
            if running_total + cost <= self._limit:
                return max(0.0, (ts + self._period) - now)
        return max(0.0, (sorted_entries[-1][0] + self._period) - now)

    async def remaining(self, key: str) -> int:
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._limit, self._period, self._ttl_seconds, ""],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])
