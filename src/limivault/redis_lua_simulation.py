# src/limivault/redis_lua_simulation.py
"""Real-time simulation driver for Redis-native Lua rate limiters.

Unlike simulation.py's run_simulation() which uses an injected
SimulationClock for fully deterministic, reproducible replay, this
driver is designed for limiters whose time source is the Redis server
itself (e.g. all Redis Lua algorithms) and therefore CANNOT use an
injected clock.

HOW IT WORKS:
  - Steps' `advance_by` values are executed as REAL time.sleep() calls
    (injectable via the `sleep=` parameter for testing).
  - The recorded timestamp for each event comes from the algorithm's
    own `_server_time()` call (Redis TIME), which moves at wall-clock
    speed.
  - No clock-identity assertion — Lua limiters have no `_clock`
    attribute.

TRADEOFF:
  This run is NOT deterministic. Redis TIME moves at real wall-clock
  speed; under system load or slow CI, the gap between advance_by and
  actual elapsed time can vary. The output CSV is a real-time
  utilization trace (useful for dashboard validation, load-shape
  visualization, and latency analysis), NOT a deterministic replay
  suitable for bit-identical regression tests.

  For deterministic simulation, use simulation.py's run_simulation()
  with an in-memory algorithm + SimulationClock instead.

USAGE:
    from limivault.redis_lua_simulation import (
        run_lua_simulation, LuaSimulationRecorder,
        constant_rate_traffic,
    )

    recorder = LuaSimulationRecorder()
    limiter = RedisGcraTokenBucket(
        client, capacity=10, refill_rate=1.0, metrics=recorder,
    )
    steps = constant_rate_traffic("user:1", num_requests=100, interval=0.5)
    run_lua_simulation(limiter, steps, recorder)
    recorder.flush_to_csv("trace.csv")

The TrafficStep dataclass, constant_rate_traffic, bursty_traffic, and
random_traffic helpers are re-exported from limivault.simulation here for
convenience, so callers only need to import this one module.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from limivault.metrics import AllowedEvent, BackendErrorEvent, DeniedEvent

if TYPE_CHECKING:

    from limivault.base import AsyncRateLimiter, RateLimiter

# Re-export traffic helpers from simulation.py — callers import from
# this one module and get everything they need.
from limivault.simulation import (  # noqa: F401
    TrafficStep,
    bursty_traffic,
    constant_rate_traffic,
    random_traffic,
)

__all__ = [
    "LuaSimulationRecorder",
    "TrafficStep",
    "run_lua_simulation",
    "async_run_lua_simulation",
    "constant_rate_traffic",
    "bursty_traffic",
    "random_traffic",
]

_CSV_FIELDNAMES = [
    "timestamp",
    "algorithm",
    "key",
    "cost",
    "decision",
    "utilization",
    "raw_state",
]


class LuaSimulationRecorder:
    """A MetricsHook that buffers allowed/denied/backend_error events
    as in-memory rows, identical in schema to
    simulation.SimulationRecorder but kept separate to avoid coupling
    the Lua simulation driver to the workspace SimulationRecorder's
    internals. Same CSV output shape: timestamp, algorithm, key, cost,
    decision, utilization, raw_state."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def on_allowed(self, event: AllowedEvent) -> None:
        self._rows.append(self._row_from_decision_event(event, decision="allowed"))

    def on_denied(self, event: DeniedEvent) -> None:
        self._rows.append(self._row_from_decision_event(event, decision="denied"))

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        self._rows.append(
            {
                "timestamp": event.timestamp,
                "algorithm": event.algorithm,
                "key": event.key,
                "cost": event.cost,
                "decision": "backend_error",
                "utilization": "",
                "raw_state": json.dumps({"error": str(event.error)}),
            }
        )

    def _row_from_decision_event(
        self, event: Union[AllowedEvent, DeniedEvent], decision: str
    ) -> dict[str, Any]:
        return {
            "timestamp": event.timestamp,
            "algorithm": event.algorithm,
            "key": event.key,
            "cost": event.cost,
            "decision": decision,
            "utilization": event.utilization,
            "raw_state": json.dumps(dict(event.state)),
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def flush_to_csv(self, path: Union[str, Path]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)


def run_lua_simulation(
    limiter: "RateLimiter",
    steps: list[TrafficStep],
    recorder: LuaSimulationRecorder | None = None,
    sleep: Any = None,
) -> None:
    """Drive `steps` against a SYNC Redis-backed limiter using real
    time.

    `sleep` defaults to `time.sleep` but can be overridden (e.g. with
    a lambda that records elapsed time, or a mock for testing). Each
    step's `advance_by` is passed to sleep before allow() is called.

    `recorder` is optional — if the limiter was constructed with
    `metrics=recorder` already, just pass None here (the driver does
    not reassign private attributes; see simulation.py's point 3 for
    the design rationale).

    IMPORTANT: This driver does NOT advance an injected clock. Steps'
    advance_by values are real sleeps. The limiter's time source is
    Redis TIME, moving at wall-clock speed. See module docstring for
    the nondeterminism tradeoff.
    """
    import time as _time

    if sleep is None:
        sleep = _time.sleep

    for step in steps:
        if step.advance_by:
            sleep(step.advance_by)
        limiter.allow(step.key, cost=step.cost)


async def async_run_lua_simulation(
    limiter: "AsyncRateLimiter",
    steps: list[TrafficStep],
    recorder: LuaSimulationRecorder | None = None,
    sleep: Any = None,
) -> None:
    """Async mirror of run_lua_simulation — drives steps against an
    ASYNC Redis-backed limiter, awaiting each allow() call. Uses
    real-time sleeps (asyncio.sleep by default) between steps.
    """
    import asyncio as _asyncio

    if sleep is None:
        sleep = _asyncio.sleep

    for step in steps:
        if step.advance_by:
            await sleep(step.advance_by)
        await limiter.allow(step.key, cost=step.cost)
