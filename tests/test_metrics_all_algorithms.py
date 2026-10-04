# tests/test_metrics_all_algorithms.py
"""Metrics coverage across ALL 12 concrete algorithm classes (6 sync +
6 async), not just the FixedWindow/TokenBucket/LeakyBucketMeter spot
checks in test_metrics.py.

Flagged in review as a real gap: "Core behavior tested, but not every
implementation equivalently." test_metrics.py's spot checks proved the
wiring pattern works at all; this file proves it was actually APPLIED
identically to every algorithm file, the same way test_cost_contract.py
and test_async_cost_contract.py already do for the cost-handling
contract (looping over every algorithm with a shared assertion, rather
than hoping each file was edited the same way by inspection alone).

This intentionally checks the SAME two properties for every algorithm:
  1. An allowed decision fires exactly one AllowedEvent, with the
     correct `algorithm` name and `key`, and a non-empty `state`.
  2. A denied decision (once capacity is exhausted) fires exactly one
     DeniedEvent, same fields.
It does NOT re-verify each algorithm's exact state-snapshot field names
-- that's what test_metrics.py's targeted per-algorithm assertions and
test_debug_logging.py's field-name assertions already cover. This file
is about breadth (every class wired up) rather than depth (exact
per-algorithm field shapes), matching the actual gap that was flagged.
"""

from __future__ import annotations

from typing import Any

import pytest

from rlimit.algorithms.async_fixed_window import AsyncFixedWindow
from rlimit.algorithms.async_leaky_bucket import (
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from rlimit.algorithms.async_sliding_window_counter import AsyncSlidingWindowCounter
from rlimit.algorithms.async_sliding_window_log import AsyncSlidingWindowLog
from rlimit.algorithms.async_token_bucket import AsyncTokenBucket
from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.algorithms.leaky_bucket import LeakyBucketMeter, LeakyBucketQueue
from rlimit.algorithms.sliding_window_counter import SlidingWindowCounter
from rlimit.algorithms.sliding_window_log import SlidingWindowLog
from rlimit.algorithms.token_bucket import TokenBucket
from rlimit.metrics import AllowedEvent, BackendErrorEvent, DeniedEvent


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now


class RecordingHook:
    def __init__(self) -> None:
        self.allowed: list[AllowedEvent] = []
        self.denied: list[DeniedEvent] = []
        self.backend_errors: list[BackendErrorEvent] = []

    def on_allowed(self, event: AllowedEvent) -> None:
        self.allowed.append(event)

    def on_denied(self, event: DeniedEvent) -> None:
        self.denied.append(event)

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        self.backend_errors.append(event)


# Each entry: (display name, zero-arg factory that takes (clock, hook)
# and returns a fresh, capacity-1 limiter instance -- capacity/limit=1
# everywhere so "one allow() then a second allow() is denied" is a
# uniform way to exercise both AllowedEvent and DeniedEvent across
# every algorithm regardless of its specific constructor shape.

_SYNC_FACTORIES: list[tuple[str, Any]] = [
    ("FixedWindow", lambda clock: FixedWindow(limit=1, period=60.0, clock=clock)),
    (
        "TokenBucket",
        lambda clock: TokenBucket(capacity=1, refill_rate=0.0, clock=clock),
    ),
    (
        "SlidingWindowLog",
        lambda clock: SlidingWindowLog(limit=1, period=60.0, clock=clock),
    ),
    (
        "SlidingWindowCounter",
        lambda clock: SlidingWindowCounter(limit=1, period=60.0, clock=clock),
    ),
    (
        "LeakyBucketMeter",
        lambda clock: LeakyBucketMeter(capacity=1, leak_rate=0.0, clock=clock),
    ),
    (
        "LeakyBucketQueue",
        lambda clock: LeakyBucketQueue(capacity=1, leak_rate=0.0, clock=clock),
    ),
]

_ASYNC_FACTORIES: list[tuple[str, Any]] = [
    (
        "AsyncFixedWindow",
        lambda clock: AsyncFixedWindow(limit=1, period=60.0, clock=clock),
    ),
    (
        "AsyncTokenBucket",
        lambda clock: AsyncTokenBucket(capacity=1, refill_rate=0.0, clock=clock),
    ),
    (
        "AsyncSlidingWindowLog",
        lambda clock: AsyncSlidingWindowLog(limit=1, period=60.0, clock=clock),
    ),
    (
        "AsyncSlidingWindowCounter",
        lambda clock: AsyncSlidingWindowCounter(limit=1, period=60.0, clock=clock),
    ),
    (
        "AsyncLeakyBucketMeter",
        lambda clock: AsyncLeakyBucketMeter(capacity=1, leak_rate=0.0, clock=clock),
    ),
    (
        "AsyncLeakyBucketQueue",
        lambda clock: AsyncLeakyBucketQueue(capacity=1, leak_rate=0.0, clock=clock),
    ),
]


class TestSyncAlgorithmsEmitAllowedAndDenied:
    @pytest.mark.parametrize(
        "name,factory", _SYNC_FACTORIES, ids=[n for n, _ in _SYNC_FACTORIES]
    )
    def test_allowed_then_denied_emit_the_matching_events(
        self, name: str, factory: Any
    ) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = factory(clock)
        limiter._metrics = hook  # attach after construction: every
        # constructor already accepted metrics=... in __init__, but
        # reassigning here keeps the factory table above uniform and
        # decoupled from remembering to thread `metrics=hook` through
        # every lambda; both are equivalent since __init__ just stores
        # it on self._metrics.

        assert limiter.allow("k") is True
        assert len(hook.allowed) == 1, name
        assert hook.allowed[0].algorithm == name, name
        assert hook.allowed[0].key == "k", name
        assert hook.allowed[0].state != {}, name

        assert limiter.allow("k") is False
        assert len(hook.denied) == 1, name
        assert hook.denied[0].algorithm == name, name
        assert hook.denied[0].key == "k", name
        assert hook.denied[0].state != {}, name

        assert hook.backend_errors == [], name


class TestAsyncAlgorithmsEmitAllowedAndDenied:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name,factory", _ASYNC_FACTORIES, ids=[n for n, _ in _ASYNC_FACTORIES]
    )
    async def test_allowed_then_denied_emit_the_matching_events(
        self, name: str, factory: Any
    ) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = factory(clock)
        limiter._metrics = hook

        assert await limiter.allow("k") is True
        assert len(hook.allowed) == 1, name
        assert hook.allowed[0].algorithm == name, name
        assert hook.allowed[0].key == "k", name
        assert hook.allowed[0].state != {}, name

        assert await limiter.allow("k") is False
        assert len(hook.denied) == 1, name
        assert hook.denied[0].algorithm == name, name
        assert hook.denied[0].key == "k", name
        assert hook.denied[0].state != {}, name

        assert hook.backend_errors == [], name


# --- Confirm the full class roster matches what's actually shipped ----


def test_all_twelve_concrete_classes_are_covered_by_this_file() -> None:
    """Guard against silent gaps: if a 13th algorithm class is ever
    added and someone forgets to add it to the factory tables above,
    this at least documents and pins the expected count so the gap is
    visible in a diff, even though it can't detect a missing entry by
    name automatically."""
    assert len(_SYNC_FACTORIES) == 6
    assert len(_ASYNC_FACTORIES) == 6
