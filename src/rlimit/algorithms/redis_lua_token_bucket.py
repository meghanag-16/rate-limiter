# src/rlimit/algorithms/redis_lua_token_bucket.py
"""RedisGcraTokenBucket -- Redis-native Lua/GCRA token
bucket. See rlimit.redis_lua_scripts's module docstring for the full
design rationale, including the v2 CHANGELOG (shared Redis-server
clock, HSET, ttl_seconds validation).

STANDALONE BY DESIGN: this class does NOT implement StorageBackend and
is NOT a drop-in swap for TokenBucket -- it talks to a `redis.Redis`
client directly, per the explicit decision that the Lua/GCRA family
should be a set of standalone Redis-native classes living alongside
the in-memory algorithm classes, not a new StorageBackend plugged
into them. It still implements the same RateLimiter ABC
(allow/allow_wait/remaining), so it's usable anywhere a RateLimiter is
expected.

NO CLIENT-SIDE CLOCK INJECTION (v2 change): unlike every other
RateLimiter in this project, this class has no `clock=` constructor
parameter. See rlimit.redis_lua_scripts's module docstring, CHANGELOG
item 1, for why: `now` is fetched via Redis's own `TIME` command, once
inside the atomic script for allow()/remaining(), and once via a plain
`TIME` call for allow_wait()'s Python-side estimate -- always the same
single server clock, regardless of how many machines or processes are
calling this limiter. This is a deliberate, documented departure from
this project's usual clock-injection testing convention, made
necessary by genuine multi-machine correctness (see the module
docstring's CONSEQUENCE FOR TESTING note).

GCRA (Generic Cell Rate Algorithm) is mathematically equivalent to a
token bucket: instead of storing a token count, it stores a single
"theoretical arrival time" (TAT). O(1) state, O(1) work per call, one
Redis key, no list/log of any kind.

KNOWN, DOCUMENTED GAP -- refill_rate == 0: GCRA's emission interval is
`1 / refill_rate`, undefined at refill_rate == 0. The Lua script
special-cases this by falling back to a plain used-cost counter capped
at capacity rather than forcing that case through GCRA's TAT formula.

allow_wait() IS NOT ATOMIC -- see rlimit.redis_lua_scripts's module
docstring for the full "ADVISORY, NOT A RESERVATION" restatement.
Concretely for this class: the stored TAT is read via a plain GET
(not the script), and any other caller anywhere can call allow() and
move that TAT between this read and whatever the caller does with the
returned wait duration.

NO TESTS SHIP INLINE WITH THIS FILE -- see tests/test_redis_lua_*.py
and tests/test_async_redis_lua_*.py for the dedicated Redis Lua/GCRA suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import redis
import redis.exceptions

from rlimit.algorithms._redis_lua_base import (
    emit_decision,
    emit_error,
    is_valid_cost_type,
    log_decision,
    reject_non_finite,
    validate_ttl_seconds,
)
from rlimit.base import RateLimiter, UnsatisfiableRequestError
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
    pass

_CATCHABLE_BACKEND_ERRORS = (redis.exceptions.RedisError, OSError)
_ALGORITHM_NAME = "RedisGcraTokenBucket"
_log = get_logger(__name__)


def _server_time(client: "redis.Redis") -> float:
    """Fetch the current time from the Redis server itself (not the
    local process clock) -- see module docstring's "NO CLIENT-SIDE
    CLOCK INJECTION" section for why. redis-py's `.time()` returns
    (seconds: int, microseconds: int)."""
    seconds, microseconds = client.time()
    return float(seconds) + float(microseconds) / 1_000_000.0


class RedisGcraTokenBucket(RateLimiter):
    """Sync, Redis-native, GCRA-based token bucket. See module
    docstring for the full design rationale."""

    def __init__(
        self,
        client: "redis.Redis",
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

    def allow(self, key: str, cost: int = 1) -> bool:
        if not is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True
        try:
            result = self._script(
                keys=[self._data_key(key)],
                args=[cost, "0", self._capacity, self._refill_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            try:
                error_ts = _server_time(self._client)
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
            "redis_lua_token_bucket_decision",
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

    def allow_wait(self, key: str, cost: int = 1) -> float:
        """ADVISORY, NOT A RESERVATION -- see this file's module
        docstring and rlimit.redis_lua_scripts's module docstring for
        the full contract. This reads the stored TAT with a plain GET
        (no atomicity with any later allow() call, by any caller,
        anywhere) and estimates the wait against the Redis server's
        current time (fetched via a separate TIME call, the same
        clock the atomic script itself uses)."""
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
            raw = self._client.get(self._data_key(key))
            now = _server_time(self._client)
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

    def remaining(self, key: str) -> int:
        try:
            result = self._script(
                keys=[self._data_key(key)],
                args=[0, "1", self._capacity, self._refill_rate, self._ttl_seconds],
            )
        except _CATCHABLE_BACKEND_ERRORS as exc:
            raise BackendUnavailableError(
                f"Redis backend unavailable while reading key {key!r}"
            ) from exc
        return int(result[2])
