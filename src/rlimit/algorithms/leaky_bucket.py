# src/rlimit/algorithms/leaky_bucket.py
"""Leaky bucket rate limiters (Meter + Queue variants). 

additions: see fixed_window.py's module docstring for the full
rationale (metrics hook, BackendUnavailableError handling, DEBUG
logging) -- identical pattern applied to both classes here.

simulation/visualization-only addition: AllowedEvent/
DeniedEvent now also carry `utilization` and `timestamp` -- for
LeakyBucketMeter, `utilization = final_volume / self._capacity`; for
LeakyBucketQueue, `utilization = final_depth / self._capacity`. Both
are "how full is the bucket right now" relative to capacity, comparable
across algorithms -- see rlimit.metrics module docstring's 
section. `timestamp` is the `now` value already read inside the lock;
BackendErrorEvent's `timestamp` uses a fresh `self._clock()` call in
the except-handler for the reason documented in fixed_window.py.
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


def _reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


class LeakyBucketMeter(RateLimiter):
    """Leaky bucket (meter variant, continuous volume)."""

    _ALGORITHM_NAME = "LeakyBucketMeter"

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
        storage: StorageBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: MetricsHook | None = None,
    ) -> None:
        _reject_non_finite(capacity, "capacity")
        _reject_non_finite(leak_rate, "leak_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if leak_rate < 0:
            raise ValueError(f"leak_rate must be non-negative, got {leak_rate}")
        self._capacity = capacity
        self._leak_rate = leak_rate
        self._storage = storage if storage is not None else InMemoryStorage()
        self._clock = clock
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _get_state(self, key: str, now: float) -> tuple[float, float]:
        state = self._storage.get(key)
        if state is None:
            return 0.0, now
        volume: float = state["volume"]
        last_leak: float = state["last_leak"]
        elapsed = max(0.0, now - last_leak)
        leaked = max(0.0, volume - elapsed * self._leak_rate)
        return leaked, now

    def _get_state_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[float, float, float, float, float]:
        """Like `_get_state`, but also reports pre-leak volume, elapsed
        time, and the raw (unclamped) leak amount -- see TokenBucket's
        equivalent helper for the full rationale (identical here, just
        draining instead of refilling). Returns (volume_after_leak,
        last_leak, volume_before, elapsed, leak_amount)."""
        state = self._storage.get(key)
        if state is None:
            return 0.0, now, 0.0, 0.0, 0.0
        volume_before: float = state["volume"]
        last_leak: float = state["last_leak"]
        elapsed = max(0.0, now - last_leak)
        leak_amount = elapsed * self._leak_rate
        volume_after_leak = max(0.0, volume_before - leak_amount)
        return volume_after_leak, now, volume_before, elapsed, leak_amount

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
                    volume,
                    last_leak,
                    volume_before,
                    elapsed,
                    leak_amount,
                ) = self._get_state_with_diagnostics(key, now)
                if volume + cost > self._capacity:
                    self._storage.set(
                        key, {"volume": volume, "last_leak": last_leak}
                    )
                    allowed = False
                    final_volume = volume
                else:
                    final_volume = volume + cost
                    self._storage.set(
                        key, {"volume": final_volume, "last_leak": last_leak}
                    )
                    allowed = True
        except BackendUnavailableError as exc:
            emit_backend_error(
                self._metrics,
                BackendErrorEvent(
                    algorithm=self._ALGORITHM_NAME,
                    key=key,
                    cost=cost,
                    error=exc,
                    timestamp=self._clock(),
                ),
            )
            raise

        _log.debug(
            "leaky_bucket_meter_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            volume_before=volume_before,
            elapsed=elapsed,
            leaked=leak_amount,
            volume_after_leak=volume,
            volume=final_volume,
            capacity=self._capacity,
        )
        snapshot = {"volume": final_volume, "capacity": self._capacity}
        utilization = final_volume / self._capacity
        if allowed:
            emit_allowed(
                self._metrics,
                AllowedEvent(
                    algorithm=self._ALGORITHM_NAME,
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
                    algorithm=self._ALGORITHM_NAME,
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
                f"cost {cost} exceeds leaky bucket capacity {self._capacity}; "
                "can never be satisfied regardless of wait time"
            )
        with self._storage.lock(key):
            now = self._clock()
            volume, _ = self._get_state(key, now)
            if volume + cost <= self._capacity:
                return 0.0
            excess = (volume + cost) - self._capacity
            if self._leak_rate <= 0:
                raise UnsatisfiableRequestError(
                    f"leak_rate is {self._leak_rate}; bucket will never "
                    "drain enough to satisfy this request"
                )
            return excess / self._leak_rate

    def remaining(self, key: str) -> int:
        with self._storage.lock(key):
            now = self._clock()
            volume, _ = self._get_state(key, now)
            return int(self._capacity - volume)


class LeakyBucketQueue(RateLimiter):
    """Leaky bucket (queue variant, discrete whole-item drains)."""

    _ALGORITHM_NAME = "LeakyBucketQueue"

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
        storage: StorageBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
        metrics: MetricsHook | None = None,
    ) -> None:
        _reject_non_finite(capacity, "capacity")
        _reject_non_finite(leak_rate, "leak_rate")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if leak_rate < 0:
            raise ValueError(f"leak_rate must be non-negative, got {leak_rate}")
        self._capacity = capacity
        self._leak_rate = leak_rate
        self._storage = storage if storage is not None else InMemoryStorage()
        self._clock = clock
        self._metrics = metrics if metrics is not None else default_metrics_hook()

    def _get_state(self, key: str, now: float) -> tuple[int, float]:
        state = self._storage.get(key)
        if state is None:
            return 0, now
        depth: int = state["depth"]
        last_drain: float = state["last_drain"]
        elapsed = max(0.0, now - last_drain)
        drained = math.floor(elapsed * self._leak_rate) if self._leak_rate > 0 else 0
        new_depth = max(0, depth - drained)
        if drained > 0 and self._leak_rate > 0:
            last_drain = last_drain + drained / self._leak_rate
        return new_depth, last_drain

    def _get_state_with_diagnostics(
        self, key: str, now: float
    ) -> tuple[int, float, int, float, int]:
        """Like `_get_state`, but also reports pre-drain depth, elapsed
        time, and the number of whole items actually drained -- see
        TokenBucket's equivalent helper for the full rationale.
        Returns (depth_after_drain, last_drain, depth_before, elapsed,
        drained_count)."""
        state = self._storage.get(key)
        if state is None:
            return 0, now, 0, 0.0, 0
        depth_before: int = state["depth"]
        last_drain: float = state["last_drain"]
        elapsed = max(0.0, now - last_drain)
        drained_count = (
            math.floor(elapsed * self._leak_rate) if self._leak_rate > 0 else 0
        )
        depth_after_drain = max(0, depth_before - drained_count)
        if drained_count > 0 and self._leak_rate > 0:
            last_drain = last_drain + drained_count / self._leak_rate
        return depth_after_drain, last_drain, depth_before, elapsed, drained_count

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
                    depth,
                    last_drain,
                    depth_before,
                    elapsed,
                    drained_count,
                ) = self._get_state_with_diagnostics(key, now)
                if depth + cost > self._capacity:
                    self._storage.set(
                        key, {"depth": depth, "last_drain": last_drain}
                    )
                    allowed = False
                    final_depth = depth
                else:
                    final_depth = depth + cost
                    self._storage.set(
                        key, {"depth": final_depth, "last_drain": last_drain}
                    )
                    allowed = True
        except BackendUnavailableError as exc:
            emit_backend_error(
                self._metrics,
                BackendErrorEvent(
                    algorithm=self._ALGORITHM_NAME,
                    key=key,
                    cost=cost,
                    error=exc,
                    timestamp=self._clock(),
                ),
            )
            raise

        _log.debug(
            "leaky_bucket_queue_decision",
            key=key,
            allowed=allowed,
            cost=cost,
            depth_before=depth_before,
            elapsed=elapsed,
            drained=drained_count,
            depth_after_drain=depth,
            depth=final_depth,
            capacity=self._capacity,
        )
        snapshot = {"depth": final_depth, "capacity": self._capacity}
        utilization = final_depth / self._capacity
        if allowed:
            emit_allowed(
                self._metrics,
                AllowedEvent(
                    algorithm=self._ALGORITHM_NAME,
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
                    algorithm=self._ALGORITHM_NAME,
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
                f"cost {cost} exceeds leaky bucket queue capacity "
                f"{self._capacity}; can never be satisfied regardless of "
                "wait time"
            )
        with self._storage.lock(key):
            now = self._clock()
            depth, last_drain = self._get_state(key, now)
            if depth + cost <= self._capacity:
                return 0.0
            if self._leak_rate <= 0:
                raise UnsatisfiableRequestError(
                    f"leak_rate is {self._leak_rate}; queue will never "
                    "drain enough to satisfy this request"
                )
            items_to_drain = (depth + cost) - self._capacity
            next_drain_at = last_drain + (1 / self._leak_rate)
            wait_for_first = max(0.0, next_drain_at - now)
            remaining_items = items_to_drain - 1
            wait_for_rest = max(0, remaining_items) / self._leak_rate
            return wait_for_first + wait_for_rest

    def remaining(self, key: str) -> int:
        with self._storage.lock(key):
            now = self._clock()
            depth, _ = self._get_state(key, now)
            return self._capacity - depth
