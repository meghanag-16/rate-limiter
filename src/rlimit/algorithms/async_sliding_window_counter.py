# src/rlimit/algorithms/async_sliding_window_counter.py
"""Async sliding window counter rate limiter. Direct async mirror of
sliding_window_counter.py including metrics/backend-error/
debug additions and utilization/timestamp event fields --
see fixed_window.py's module docstring for the full rationale
(identical pattern, not repeated per file)."""

from __future__ import annotations

import math
import time
from typing import Callable

from rlimit.base import AsyncRateLimiter, UnsatisfiableRequestError
from rlimit.exceptions import BackendUnavailableError
from rlimit.logging import get_logger
from rlimit.metrics import (
    AllowedEvent,
    BackendErrorEvent,
    DeniedEvent,
    MetricsHook,
    default_metrics_hook,
    emit_allowed,
    emit_backend_error,
    emit_denied,
)
from rlimit.storage import AsyncInMemoryStorage, AsyncStorageBackend

_log = get_logger(__name__)

_ALGORITHM_NAME = "AsyncSlidingWindowCounter"

_WAIT_EPSILON = 1e-9


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class AsyncSlidingWindowCounter(AsyncRateLimiter):
    """Async sliding window counter (weighted approximation)."""

    def __init__(
        self,
        limit: int,
        period: float,
        storage: AsyncStorageBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: MetricsHook | None = None,
    ) -> None:
        _reject_non_finite(limit, "limit")
        _reject_non_finite(period, "period")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._limit = limit
        self._period = period
        self._storage = storage if storage is not None else AsyncInMemoryStorage()
        self._clock = clock
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _window_start(self, now: float) -> float:
        return (now // self._period) * self._period

    async def _get_state(self, key: str, now: float) -> tuple[float, int, int]:
        state = await self._storage.get(key)
        current_window = self._window_start(now)
        if state is None:
            return current_window, 0, 0
        stored_window: float = state["window_start"]
        if stored_window == current_window:
            return current_window, state["count"], state["prev_count"]
        if stored_window == current_window - self._period:
            return current_window, 0, state["count"]
        return current_window, 0, 0

    def _weighted_count(
        self, window_start: float, count: int, prev_count: int, now: float
    ) -> float:
        elapsed = now - window_start
        overlap_fraction = max(0.0, (self._period - elapsed) / self._period)
        return count + prev_count * overlap_fraction

    async def allow(self, key: str, cost: int = 1) -> bool:
        if not _is_valid_cost_type(cost):
            return False
        if cost < 0:
            return False
        if cost == 0:
            return True

        try:
            async with self._storage.lock(key):
                now = self._clock()
                window_start, count, prev_count = await self._get_state(key, now)
                weighted_before = self._weighted_count(
                    window_start, count, prev_count, now
                )
                weighted = weighted_before
                if weighted + cost > self._limit:
                    await self._storage.set(
                        key,
                        {
                            "window_start": window_start,
                            "count": count,
                            "prev_count": prev_count,
                        },
                    )
                    allowed = False
                    final_weighted = weighted
                else:
                    await self._storage.set(
                        key,
                        {
                            "window_start": window_start,
                            "count": count + cost,
                            "prev_count": prev_count,
                        },
                    )
                    allowed = True
                    final_weighted = weighted + cost
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

        _log.debug(
            "async_sliding_window_counter_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            count=count,
            prev_count=prev_count,
            weighted_before=weighted_before,
            weighted=final_weighted,
            limit=self._limit,
        )
        snapshot = {"weighted": final_weighted, "limit": self._limit}
        utilization = final_weighted / self._limit
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

    async def allow_wait(self, key: str, cost: int = 1) -> float:
        if not _is_valid_cost_type(cost):
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
        async with self._storage.lock(key):
            now = self._clock()
            window_start, count, prev_count = await self._get_state(key, now)
            weighted = self._weighted_count(window_start, count, prev_count, now)
            if weighted + cost <= self._limit:
                return 0.0

            target = self._limit - cost
            elapsed_now = now - window_start
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
        async with self._storage.lock(key):
            now = self._clock()
            window_start, count, prev_count = await self._get_state(key, now)
            weighted = self._weighted_count(window_start, count, prev_count, now)
            return int(self._limit - weighted)
