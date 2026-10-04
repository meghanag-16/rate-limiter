# src/rlimit/algorithms/async_redis_lua_token_bucket.py
"""AsyncRedisGcraTokenBucket -- async mirror of
redis_lua_token_bucket.RedisGcraTokenBucket. See that file's docstring
and rlimit.redis_lua_scripts's module docstring for the full design
rationale (identical here, just async), including v2's shared-Redis-
server-clock change (no client-side `clock=` parameter on this class
either).

NO TESTS SHIP INLINE WITH THIS FILE -- see tests/test_async_redis_lua_*.py.
"""

from __future__ import annotations

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
    GCRA_TOKEN_BUCKET,
    TTL_BUFFER_SECONDS,
)

if TYPE_CHECKING:
    import redis.asyncio

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_ALGORITHM_NAME = "AsyncRedisGcraTokenBucket"
_log = get_logger(__name__)


async def _server_time(client: "redis.asyncio.Redis") -> float:
    seconds, microseconds = await client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


class AsyncRedisGcraTokenBucket(AsyncRateLimiter):
    """Async, Redis-native, GCRA-based token bucket."""

    def __init__(
        self,
        client: "redis.asyncio.Redis",
        capacity: int,
        refill_rate: float,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        ttl_seconds: float | None = None,
        metrics: MetricsHook | None = None,
    ) -> None:
        reject_non_finite(capacity, "capacity")
        reject_non_finite(refill_rate, "refill_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if refill_rate < 0:
            raise ValueError(f"refill_rate must be non-negative, got {refill_rate}")
        self._client = client
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._key_prefix = key_prefix
        if ttl_seconds is not None:
            self._ttl_seconds = ttl_seconds
        elif refill_rate > 0:
            self._ttl_seconds = (capacity / refill_rate) + TTL_BUFFER_SECONDS
        else:
            self._ttl_seconds = FALLBACK_TTL_SECONDS
        validate_ttl_seconds(self._ttl_seconds)
        self._script = client.register_script(GCRA_TOKEN_BUCKET)
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _data_key(self, key: str) -> str:
        return f"{self._key_prefix}gcra_tb:{key}"

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
                args=[cost, "0", self._capacity, self._refill_rate, self._ttl_seconds],
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
                f"Redis backend unavailable while running GCRA script for key {key!r}"
            ) from exc
        allowed = bool(int(result[0]))
        remaining = int(result[2])
        now = float(result[3])
        utilization = (self._capacity - remaining) / self._capacity
        snapshot = {"remaining": remaining, "capacity": self._capacity}
        log_decision(
            _log,
            "async_redis_lua_token_bucket_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            remaining=remaining,
            capacity=self._capacity,
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
        if cost > self._capacity:
            raise UnsatisfiableRequestError(
                f"cost {cost} exceeds token bucket capacity {self._capacity}; "
                "can never be satisfied regardless of wait time"
            )
        try:
            raw = await self._client.get(self._data_key(key))
            now = await _server_time(self._client)
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc

        if self._refill_rate <= 0:
            used = float(raw) if raw is not None else 0.0
            if used + cost <= self._capacity:
                return 0.0
            raise UnsatisfiableRequestError(
                f"refill_rate is {self._refill_rate}; bucket will never "
                "accumulate enough tokens to satisfy this request"
            )

        emission_interval = 1.0 / self._refill_rate
        burst_offset = self._capacity * emission_interval
        tat = float(raw) if raw is not None else now
        if tat < now:
            tat = now
        increment = cost * emission_interval
        new_tat = tat + increment
        allow_at = new_tat - burst_offset
        return max(0.0, allow_at - now)

    async def remaining(self, key: str) -> int:
        try:
            result = await self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._capacity, self._refill_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])
