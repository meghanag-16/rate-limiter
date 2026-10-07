# src/limivault/algorithms/async_sliding_window_log.py
"""Async sliding window log rate limiter. Direct async mirror of
sliding_window_log.py including metrics/backend-error/debug
additions and utilization/timestamp event fields -- see
fixed_window.py's module docstring for the full rationale (identical
pattern, not repeated per file)."""

from __future__ import annotations

import math
import time
from typing import Callable

from limivault.base import AsyncRateLimiter, UnsatisfiableRequestError
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
from limivault.storage import AsyncInMemoryStorage, AsyncStorageBackend

_log = get_logger(__name__)

_ALGORITHM_NAME = "AsyncSlidingWindowLog"


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class AsyncSlidingWindowLog(AsyncRateLimiter):
    """Async sliding window log."""

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

    async def _get_log(self, key: str, now: float) -> list[tuple[float, int]]:
        state = await self._storage.get(key)
        if state is None:
            return []
        entries: list[tuple[float, int]] = state["entries"]
        cutoff = now - self._period
        return [(ts, c) for ts, c in entries if ts > cutoff]

    async def _get_log_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[list[tuple[float, int]], int, int]:
        """Async mirror of SlidingWindowLog's diagnostic helper -- see
        that file's docstring for the full rationale (identical here)."""
        state = await self._storage.get(key)
        if state is None:
            return [], 0, 0
        entries: list[tuple[float, int]] = state["entries"]
        cutoff = now - self._period
        kept = [(ts, c) for ts, c in entries if ts > cutoff]
        return kept, len(entries), len(entries) - len(kept)

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
                entries, entries_before, expired_count = (
                    await self._get_log_with_diagnostics(key, now)
                )
                current_total = sum(c for _, c in entries)
                if current_total + cost > self._limit:
                    await self._storage.set(key, {"entries": entries})
                    allowed = False
                    final_total = current_total
                    final_count = len(entries)
                else:
                    entries.append((now, cost))
                    await self._storage.set(key, {"entries": entries})
                    allowed = True
                    final_total = current_total + cost
                    final_count = len(entries)
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
            "async_sliding_window_log_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            entries_before=entries_before,
            expired=expired_count,
            entries_count=final_count,
            total_cost_before=current_total,
            total_cost=final_total,
            limit=self._limit,
        )
        snapshot = {
            "entries_count": final_count,
            "total_cost": final_total,
            "limit": self._limit,
        }
        utilization = final_total / self._limit
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
                f"cost {cost} exceeds sliding window log limit {self._limit}; "
                "can never be satisfied regardless of wait time"
            )
        async with self._storage.lock(key):
            now = self._clock()
            entries = await self._get_log(key, now)
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
        async with self._storage.lock(key):
            now = self._clock()
            entries = await self._get_log(key, now)
            return self._limit - sum(c for _, c in entries)
