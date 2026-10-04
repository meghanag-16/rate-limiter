# src/rlimit/algorithms/async_fixed_window.py
"""Async fixed window counter rate limiter. Direct async mirror of
fixed_window.py's FixedWindow, including metrics/backend-
error/debug-logging additions and utilization/timestamp
event fields -- see that file's module docstring for the full
rationale (identical here, not repeated). The metrics hook stays a
plain sync MetricsHook (see rlimit.metrics point 4
on why no async hook variant was added): calling it from this
async algorithm does NOT yield the event loop, so a slow hook
implementation blocks other coroutines the same way a slow injected
clock would -- this is exactly why rlimit.simulation's recorder
buffers in memory rather than doing I/O per call.
"""

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

_ALGORITHM_NAME = "AsyncFixedWindow"


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class AsyncFixedWindow(AsyncRateLimiter):
    """Async fixed window counter."""

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

    async def _get_state(self, key: str, now: float) -> tuple[float, int]:
        state = await self._storage.get(key)
        current_window = self._window_start(now)
        if state is None or state["window_start"] != current_window:
            return current_window, 0
        count: int = state["count"]
        return current_window, count

    async def _get_state_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[float, int, bool]:
        """Async mirror of FixedWindow's diagnostic helper -- see that
        file's docstring for the full rationale (identical here)."""
        state = await self._storage.get(key)
        current_window = self._window_start(now)
        if state is None:
            return current_window, 0, True
        rolled_over = state["window_start"] != current_window
        if rolled_over:
            return current_window, 0, True
        count: int = state["count"]
        return current_window, count, False

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
                window_start, count, rolled_over = (
                    await self._get_state_with_diagnostics(key, now)
                )
                if count + cost > self._limit:
                    await self._storage.set(
                        key, {"window_start": window_start, "count": count}
                    )
                    allowed = False
                    new_count = count
                else:
                    new_count = count + cost
                    await self._storage.set(
                        key, {"window_start": window_start, "count": new_count}
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

        _log.debug(
            "async_fixed_window_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            window_start=window_start,
            rolled_over=rolled_over,
            count_before=count,
            count=new_count,
            limit=self._limit,
        )
        snapshot = {
            "window_start": window_start,
            "count": new_count,
            "limit": self._limit,
        }
        utilization = new_count / self._limit
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
                f"cost {cost} exceeds fixed window limit {self._limit}; "
                "can never be satisfied regardless of wait time"
            )
        async with self._storage.lock(key):
            now = self._clock()
            window_start, count = await self._get_state(key, now)
            if count + cost <= self._limit:
                return 0.0
            next_window_start = window_start + self._period
            return max(0.0, next_window_start - now)

    async def remaining(self, key: str) -> int:
        async with self._storage.lock(key):
            now = self._clock()
            _, count = await self._get_state(key, now)
            return self._limit - count
