# src/limivault/algorithms/token_bucket.py
"""Token bucket rate limiter.

additions: see fixed_window.py's module docstring for the full
rationale (metrics hook, BackendUnavailableError handling, DEBUG
logging) -- identical pattern applied here, not repeated in full.

simulation/visualization-only addition: AllowedEvent/
DeniedEvent now also carry `utilization` (here:
`(self._capacity - new_tokens) / self._capacity` -- how much of the
bucket's capacity is currently consumed, comparable across algorithms,
see limivault.metrics module docstring's section) and `timestamp`
(the `now` value already read inside the lock). BackendErrorEvent
carries `timestamp` via a fresh `self._clock()` call in the
except-handler for the same reason documented in fixed_window.py.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from limivault.base import RateLimiter, UnsatisfiableRequestError
from limivault.exceptions import BackendUnavailableError
from limivault.logging import get_logger
from limivault.metrics import (
    AllowedEvent,
    BackendErrorEvent,
    DeniedEvent,
    MetricsHook,
    default_metrics_hook,
    emit_allowed,
    emit_backend_error,
    emit_denied,
)
from limivault.storage import InMemoryStorage, StorageBackend

_log = get_logger(__name__)

_ALGORITHM_NAME = "TokenBucket"


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class TokenBucket(RateLimiter):
    """Token bucket. `capacity` is the max/initial tokens; `refill_rate`
    is tokens added per second, applied continuously."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        storage: StorageBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: MetricsHook | None = None,
    ) -> None:
        _reject_non_finite(capacity, "capacity")
        _reject_non_finite(refill_rate, "refill_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if refill_rate < 0:
            raise ValueError(
                f"refill_rate must be non-negative, got {refill_rate}"
            )
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._storage = storage if storage is not None else InMemoryStorage()
        self._clock = clock
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _get_state(self, key: str, now: float) -> tuple[float, float]:
        state = self._storage.get(key)
        if state is None:
            return float(self._capacity), now
        tokens: float = state["tokens"]
        last_refill: float = state["last_refill"]
        elapsed = max(0.0, now - last_refill)
        refilled = min(self._capacity, tokens + elapsed * self._refill_rate)
        return refilled, now

    def _get_state_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[float, float, float, float, float]:
        """Like `_get_state`, but also reports the pre-refill token
        count, elapsed time since the last refill, and the raw
        (uncapped) refill amount computed from `elapsed * refill_rate`
        -- the actual "transition" a debug reader needs to see whether
        a refill happened, how much, and whether it was clamped at
        capacity. Used only by allow()'s debug-logging path, kept
        separate from `_get_state` for the same reason documented in
        fixed_window.py's equivalent helper (zero regression risk to
        allow_wait()/remaining()).

        Returns (tokens_after_refill, last_refill, tokens_before,
        elapsed, raw_refill_amount).
        """
        state = self._storage.get(key)
        if state is None:
            return float(self._capacity), now, float(self._capacity), 0.0, 0.0
        tokens_before: float = state["tokens"]
        last_refill: float = state["last_refill"]
        elapsed = max(0.0, now - last_refill)
        raw_refill_amount = elapsed * self._refill_rate
        tokens_after_refill = min(self._capacity, tokens_before + raw_refill_amount)
        return tokens_after_refill, now, tokens_before, elapsed, raw_refill_amount

    def allow(self, key: str, cost: int = 1) -> bool:
        if not _is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True

        try:
            with self._storage.lock(key):
                now = self._clock()
                (
                    tokens,
                    last_refill,
                    tokens_before,
                    elapsed,
                    raw_refill_amount,
                ) = self._get_state_with_diagnostics(key, now)
                if tokens < cost:
                    self._storage.set(
                        key, {"tokens": tokens, "last_refill": last_refill}
                    )
                    allowed = False
                    new_tokens = tokens
                else:
                    new_tokens = tokens - cost
                    self._storage.set(
                        key, {"tokens": new_tokens, "last_refill": last_refill}
                    )
                    allowed = True
        except BackendUnavailableError as exc:
            emit_backend_error(
                self._metrics,
                BackendErrorEvent(
                    algorithm=_ALGORITHM_NAME,
                    key=key,
                    cost=cost,
                    error=exc,
                    timestamp=self._clock(),
                ),
            )
            raise

        # Transition-level debug logging: tokens_before is the stored
        # value prior to this call's refill; `refilled` is the raw
        # (pre-capacity-clamp) amount that elapsed*refill_rate would
        # add -- comparing it against (tokens - tokens_before) shows
        # whether the refill was clamped at capacity. `cost` explains
        # the remaining delta down to the final `tokens` value.
        _log.debug(
            "token_bucket_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            tokens_before=tokens_before,
            elapsed=elapsed,
            refilled=raw_refill_amount,
            tokens_after_refill=tokens,
            tokens=new_tokens,
            capacity=self._capacity,
        )
        snapshot = {"tokens": new_tokens, "capacity": self._capacity}
        #  normalized utilization -- fraction of capacity
        # currently consumed, i.e. the inverse of "how many tokens are
        # left". See limivault.metrics module docstring's section.
        utilization = (self._capacity - new_tokens) / self._capacity
        if allowed:
            emit_allowed(
                self._metrics,
                AllowedEvent(
                    algorithm=_ALGORITHM_NAME,
                    key=key,
                    cost=cost,
                    state=snapshot,
                    utilization=utilization,
                    timestamp=now,
                ),
            )
        else:
            emit_denied(
                self._metrics,
                DeniedEvent(
                    algorithm=_ALGORITHM_NAME,
                    key=key,
                    cost=cost,
                    state=snapshot,
                    utilization=utilization,
                    timestamp=now,
                ),
            )
        return allowed

    def allow_wait(self, key: str, cost: int = 1) -> float:
        if not _is_valid_cost_type(cost):
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
        with self._storage.lock(key):
            now = self._clock()
            tokens, _ = self._get_state(key, now)
            if tokens >= cost:
                return 0.0
            tokens_needed = cost - tokens
            if self._refill_rate <= 0:
                raise UnsatisfiableRequestError(
                    f"refill_rate is {self._refill_rate}; bucket will "
                    "never accumulate enough tokens to satisfy this request"
                )
            return tokens_needed / self._refill_rate

    def remaining(self, key: str) -> int:
        with self._storage.lock(key):
            now = self._clock()
            tokens, _ = self._get_state(key, now)
            return int(tokens)
