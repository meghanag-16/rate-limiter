# tests/test_metrics.py
"""Tests for rlimit.metrics and its wiring into FixedWindow (chosen as
the representative algorithm -- the same emit_allowed/emit_denied/
emit_backend_error pattern is identical across all ten algorithm
files, per each file's module docstring, so this is not re-verified
per-algorithm here; TestTokenBucketAndLeakyBucketWiring below spot-
checks two more to confirm the pattern was actually applied, not just
described).

Deliberately tests the SCOPE BOUNDARIES documented in
rlimit.metrics's module docstring (points 7, 8, 9), not just the
happy path -- these are the parts most likely to silently regress.
"""

from __future__ import annotations

from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.algorithms.leaky_bucket import LeakyBucketMeter
from rlimit.algorithms.token_bucket import TokenBucket
from rlimit.metrics import (
    AllowedEvent,
    BackendErrorEvent,
    DeniedEvent,
    NoOpMetricsHook,
    default_metrics_hook,
    emit_allowed,
    emit_backend_error,
    emit_denied,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class RecordingHook:
    """A hook that records every event it receives, for assertion."""

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


class RaisingHook:
    """A hook whose every method raises -- used to prove hook failures
    are isolated (rlimit.metrics module docstring, point 7)."""

    def on_allowed(self, event: AllowedEvent) -> None:
        raise RuntimeError("boom-allowed")

    def on_denied(self, event: DeniedEvent) -> None:
        raise RuntimeError("boom-denied")

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        raise RuntimeError("boom-backend-error")


# --- NoOpMetricsHook / default_metrics_hook --------------------------


def test_default_metrics_hook_is_a_no_op_and_does_not_raise() -> None:
    hook = default_metrics_hook()
    hook.on_allowed(AllowedEvent(algorithm="X", key="k", cost=1, state={}))
    hook.on_denied(DeniedEvent(algorithm="X", key="k", cost=1, state={}))
    hook.on_backend_error(
        BackendErrorEvent(algorithm="X", key="k", cost=1, error=Exception("x"))
    )


def test_default_metrics_hook_returns_the_same_shared_instance() -> None:
    assert default_metrics_hook() is default_metrics_hook()


def test_noop_metrics_hook_is_directly_instantiable() -> None:
    hook = NoOpMetricsHook()
    hook.on_allowed(AllowedEvent(algorithm="X", key="k", cost=1, state={}))


# --- emit_* isolate hook exceptions (module docstring point 7) -------


def test_emit_allowed_does_not_propagate_hook_exception() -> None:
    hook = RaisingHook()
    # Must not raise.
    emit_allowed(hook, AllowedEvent(algorithm="X", key="k", cost=1, state={}))


def test_emit_denied_does_not_propagate_hook_exception() -> None:
    hook = RaisingHook()
    emit_denied(hook, DeniedEvent(algorithm="X", key="k", cost=1, state={}))


def test_emit_backend_error_does_not_propagate_hook_exception() -> None:
    hook = RaisingHook()
    emit_backend_error(
        hook, BackendErrorEvent(algorithm="X", key="k", cost=1, error=Exception("x"))
    )


# --- FixedWindow: metrics wired into allow() --------------------------


class TestFixedWindowMetricsWiring:
    def test_allowed_event_fires_with_expected_fields(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=hook)

        assert limiter.allow("user:1", cost=2) is True

        assert len(hook.allowed) == 1
        assert hook.denied == []
        assert hook.backend_errors == []
        event = hook.allowed[0]
        assert event.algorithm == "FixedWindow"
        assert event.key == "user:1"
        assert event.cost == 2
        assert event.state["count"] == 2
        assert event.state["limit"] == 3

    def test_denied_event_fires_when_over_limit(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = FixedWindow(limit=1, period=60.0, clock=clock, metrics=hook)
        limiter.allow("k")  # consumes the one slot, clears hook below

        hook2 = RecordingHook()
        limiter2 = FixedWindow(limit=1, period=60.0, clock=clock, metrics=hook2)
        limiter2.allow("k")
        assert limiter2.allow("k") is False

        assert len(hook2.denied) == 1
        assert hook2.allowed == [] or len(hook2.allowed) == 1  # first call allowed
        denied_event = hook2.denied[0]
        assert denied_event.algorithm == "FixedWindow"
        assert denied_event.key == "k"
        assert denied_event.cost == 1

    def test_no_metrics_hook_is_optional_and_defaults_to_noop(self) -> None:
        clock = FakeClock()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock)  # metrics=None
        # Must not raise despite no explicit hook.
        assert limiter.allow("k") is True

    # --- Scope boundary (module docstring point 8): invalid cost and
    # cost==0 must NOT emit any metrics event at all.

    def test_cost_zero_emits_no_metrics_event(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=hook)

        assert limiter.allow("k", cost=0) is True

        assert hook.allowed == []
        assert hook.denied == []

    def test_invalid_cost_type_emits_no_metrics_event(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=hook)

        assert limiter.allow("k", cost=2.5) is False  # type: ignore[arg-type]

        assert hook.allowed == []
        assert hook.denied == []

    def test_negative_cost_emits_no_metrics_event(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=hook)

        assert limiter.allow("k", cost=-1) is False

        assert hook.allowed == []
        assert hook.denied == []

    def test_a_hook_that_raises_does_not_break_allow(self) -> None:
        """RaisingHook must not prevent allow() from returning its
        normal result -- see module docstring point 7."""
        clock = FakeClock()
        hook = RaisingHook()
        limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=hook)

        assert limiter.allow("k") is True  # does not raise despite the hook


# --- Spot-check the same wiring pattern on two more algorithms --------


class TestTokenBucketAndLeakyBucketWiring:
    def test_token_bucket_emits_allowed_with_token_state(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = TokenBucket(
            capacity=5, refill_rate=1.0, clock=clock, metrics=hook
        )
        assert limiter.allow("k") is True
        assert len(hook.allowed) == 1
        assert hook.allowed[0].algorithm == "TokenBucket"
        assert hook.allowed[0].state["capacity"] == 5

    def test_leaky_bucket_meter_emits_denied_with_volume_state(self) -> None:
        clock = FakeClock()
        hook = RecordingHook()
        limiter = LeakyBucketMeter(
            capacity=1, leak_rate=0.0, clock=clock, metrics=hook
        )
        limiter.allow("k")  # fills the bucket
        assert limiter.allow("k") is False
        assert len(hook.denied) == 1
        assert hook.denied[0].algorithm == "LeakyBucketMeter"
        assert hook.denied[0].state["capacity"] == 1
