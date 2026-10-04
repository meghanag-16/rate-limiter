# src/rlimit/algorithms/sliding_window_counter.py
"""Sliding window counter rate limiter. See other sources for the
full two-segment allow_wait() derivation and the clock-inside-
lock fix -- unchanged, not repeated here.

additions: see fixed_window.py's module docstring for the full
rationale (metrics hook, BackendUnavailableError handling, DEBUG
logging) -- identical pattern applied here.

simulation/visualization-only addition: AllowedEvent/
DeniedEvent now also carry `utilization` (here:
`final_weighted / self._limit`, the post-decision decayed weighted
count relative to the configured limit -- comparable across
algorithms, see rlimit.metrics module docstring's section) and
`timestamp` (the `now` value already read inside the lock).
BackendErrorEvent's `timestamp` uses a fresh `self._clock()` call in
the except-handler, per the reason documented in fixed_window.py.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from rlimit.base import RateLimiter, UnsatisfiableRequestError
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
from rlimit.storage import InMemoryStorage, StorageBackend

_log = get_logger(__name__)

_ALGORITHM_NAME = "SlidingWindowCounter"

_WAIT_EPSILON = 1e-9


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class SlidingWindowCounter(RateLimiter):
    """Sliding window counter (weighted approximation)."""

    def __init__(
        self,
        limit: int,
        period: float,
        storage: StorageBackend | None = None,
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
        self._storage = storage if storage is not None else InMemoryStorage()
        self._clock = clock
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _window_start(self, now: float) -> float:
        return (now // self._period) * self._period

    def _get_state(self, key: str, now: float) -> tuple[float, int, int]:
        state = self._storage.get(key)
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
                window_start, count, prev_count = self._get_state(key, now)
                weighted_before = self._weighted_count(
                    window_start, count, prev_count, now
                )
                weighted = weighted_before
                if weighted + cost > self._limit:
                    self._storage.set(
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
                    self._storage.set(
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

        # Transition-level debug logging: weighted_before is the
        # decayed weighted count (count + prev_count * overlap
        # fraction, see the algorithm's module docstring) at the
        # moment of decision, BEFORE this call's cost is applied.
        # count/prev_count are the two raw stored components that
        # weighted_before was computed from, so a reader can see the
        # decay math's inputs, not just its output.
        _log.debug(
            "sliding_window_counter_decision",
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

    def allow_wait(self, key: str, cost: int = 1) -> float:
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
        with self._storage.lock(key):
            now = self._clock()
            window_start, count, prev_count = self._get_state(key, now)
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

    def remaining(self, key: str) -> int:
        with self._storage.lock(key):
            now = self._clock()
            window_start, count, prev_count = self._get_state(key, now)
            weighted = self._weighted_count(window_start, count, prev_count, now)
            return int(self._limit - weighted)
