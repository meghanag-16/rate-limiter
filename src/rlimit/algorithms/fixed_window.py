# src/rlimit/algorithms/fixed_window.py
"""Fixed window counter rate limiter.

Divides time into fixed-size windows and counts requests within the
current window. Known boundary-burst flaw documented (not fixed) as in
prior -- unchanged, not repeated here.

`now = self._clock()` read inside the per-key lock (TOCTOU
fix, unchanged, not repeated here).

Additions (see rlimit.metrics / rlimit.exceptions module
docstrings for the full design rationale -- not repeated per-algorithm
file):

- `metrics: MetricsHook | None = None` constructor argument. Defaults
  to a no-op hook via `default_metrics_hook()` so `self._metrics` is
  always callable. allow() emits `AllowedEvent`/`DeniedEvent` for the
  real rate-limiting decision path only -- NOT for invalid-cost
  rejection and NOT for the cost == 0 no-op (both remain exactly as
  inert as before; see rlimit.metrics docstring point 8 for why this
  scope boundary is deliberate). Events are emitted AFTER the per-key
  lock is released, to keep the lock's critical section short.
  -- metrics hooks must still be fast (see point 4 of
  rlimit.metrics), but this avoids the hook call itself contributing
  to lock hold time either way.
- Storage access inside allow() is wrapped in a
  `except BackendUnavailableError` clause that emits a
  `BackendErrorEvent` and then re-raises the exception unchanged. Only
  allow() gets this treatment in this file, not allow_wait() -- see
  rlimit.metrics docstring point 9.
- One DEBUG-level structlog line per allow() decision (window
  rollover / count), emitted via the existing `rlimit.logging` setup.
  Uses values already computed for the algorithm's own logic (window
  start, count) rather than computing anything extra just for the log
  line, so there is no additional cost when DEBUG logging is disabled
  beyond structlog's own early level-filtering.

simulation/visualization-only addition:

- `AllowedEvent`/`DeniedEvent` now also carry `utilization` (here:
  `new_count / self._limit`, the post-decision count relative to the
  configured limit -- comparable across algorithms, see rlimit.metrics
  module docstring's section) and `timestamp` (the `now` value
  already read inside the lock). `BackendErrorEvent` carries
  `timestamp` via a fresh `self._clock()` call in the except-handler,
  since `now` may not be bound yet if the failure happened acquiring
  the lock itself.
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

_ALGORITHM_NAME = "FixedWindow"


def _reject_non_finite(value: float, name: str) -> None:
    """Raise ValueError if `value` is NaN or +/-infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    """True if `cost` is an actual int -- not bool, not a float.
    """
    return isinstance(cost, int) and not isinstance(cost, bool)


class FixedWindow(RateLimiter):
    """Fixed window counter.

    Allows up to `limit` units of cost per `period` seconds, per key.
    """

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

    def _get_state(self, key: str, now: float) -> tuple[float, int]:
        """Return (window_start, count) for `key`, resetting if the
        stored window has rolled over relative to `now`."""
        state = self._storage.get(key)
        current_window = self._window_start(now)
        if state is None or state["window_start"] != current_window:
            return current_window, 0
        count: int = state["count"]
        return current_window, count

    def _get_state_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[float, int, bool]:
        """Like `_get_state`, but also reports whether this call
        observed a window rollover (the stored window differs from the
        current one, including "no prior state at all"). Used only by
        allow()'s debug-logging path -- kept as a separate method
        rather than changing `_get_state`'s return shape, so
        `allow_wait()`/`remaining()` (which call `_get_state` directly)
        are untouched by this addition and carry zero regression risk
        from a debug-logging enhancement."""
        state = self._storage.get(key)
        current_window = self._window_start(now)
        if state is None:
            return current_window, 0, True
        rolled_over = state["window_start"] != current_window
        if rolled_over:
            return current_window, 0, True
        count: int = state["count"]
        return current_window, count, False

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
                window_start, count, rolled_over = self._get_state_with_diagnostics(
                    key, now
                )
                if count + cost > self._limit:
                    self._storage.set(
                        key, {"window_start": window_start, "count": count}
                    )
                    allowed = False
                    new_count = count
                else:
                    new_count = count + cost
                    self._storage.set(
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

        # Transition-level debug logging :
        # count_before is the count observed BEFORE this request's
        # cost was applied (0 if a window rollover was observed this
        # call); count is the count AFTER. `cost` is included so a
        # reader can see why count changed by exactly that amount
        # without cross-referencing a separate metrics event.
        _log.debug(
            "fixed_window_decision",
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
        #  normalized cross-algorithm utilization -- see
        # rlimit.metrics module docstring's section for the
        # per-algorithm formula table.
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
                f"cost {cost} exceeds fixed window limit {self._limit}; "
                "can never be satisfied regardless of wait time"
            )
        with self._storage.lock(key):
            now = self._clock()
            window_start, count = self._get_state(key, now)
            if count + cost <= self._limit:
                return 0.0
            next_window_start = window_start + self._period
            return max(0.0, next_window_start - now)

    def remaining(self, key: str) -> int:
        with self._storage.lock(key):
            now = self._clock()
            _, count = self._get_state(key, now)
            return self._limit - count
