# tests/test_backend_error_wiring.py
"""FixedWindow's BackendUnavailableError handling and metrics wiring,
tested against a failing in-memory storage stub.

These tests check FixedWindow's own behaviour when its storage fails
(emit a backend_error event, re-raise, never let a raising hook mask
the original error, resume normal metrics after recovery). Nothing
about them is specific to Redis, so they run against a plain
StorageBackend that fails on demand. No Redis and no Docker are
needed.

Scope boundary: allow_wait() raises
BackendUnavailableError on a backend failure but does NOT additionally
emit a backend_error metrics event (rlimit.metrics module docstring,
point 9).

FakeClock and RecordingHook are defined locally rather than in a
conftest, matching this project's convention for test doubles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.exceptions import BackendUnavailableError
from rlimit.metrics import AllowedEvent, BackendErrorEvent, DeniedEvent
from rlimit.storage import StorageBackend


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


class _FlakyStorage(StorageBackend):
    """Dict-backed StorageBackend whose get()/set() raise
    BackendUnavailableError while the matching name ("get" / "set") is
    in `fail_on`. Clearing `fail_on` makes it behave like a healthy
    backend again. lock() always succeeds."""

    def __init__(self, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        if "get" in self.fail_on:
            raise BackendUnavailableError("simulated get failure")
        return self._store.get(key)

    def set(self, key: str, state: dict[str, Any]) -> None:
        if "set" in self.fail_on:
            raise BackendUnavailableError("simulated set failure")
        self._store[key] = state

    @contextmanager
    def lock(self, key: str) -> Iterator[bool]:
        yield True


class TestFixedWindowBackendErrorWiring:
    def test_allow_emits_backend_error_event_and_reraises(self) -> None:
        clock = FakeClock()
        storage = _FlakyStorage(fail_on={"get"})
        hook = RecordingHook()
        limiter = FixedWindow(
            limit=5, period=60.0, storage=storage, clock=clock, metrics=hook
        )

        with pytest.raises(BackendUnavailableError):
            limiter.allow("k")

        assert len(hook.backend_errors) == 1
        assert hook.allowed == []
        assert hook.denied == []
        event = hook.backend_errors[0]
        assert event.algorithm == "FixedWindow"
        assert event.key == "k"
        assert isinstance(event.error, BackendUnavailableError)

    def test_a_raising_hook_does_not_mask_the_backend_error(self) -> None:
        """The metrics hook itself failing must not change or swallow
        the original BackendUnavailableError -- see rlimit.metrics
        module docstring point 7."""

        class RaisingHook:
            def on_allowed(self, event: AllowedEvent) -> None:
                raise RuntimeError("boom")

            def on_denied(self, event: DeniedEvent) -> None:
                raise RuntimeError("boom")

            def on_backend_error(self, event: BackendErrorEvent) -> None:
                raise RuntimeError("boom")

        clock = FakeClock()
        storage = _FlakyStorage(fail_on={"set"})
        limiter = FixedWindow(
            limit=5,
            period=60.0,
            storage=storage,
            clock=clock,
            metrics=RaisingHook(),
        )

        # Must still surface BackendUnavailableError, not the hook's
        # RuntimeError, and must not hang or raise anything else.
        with pytest.raises(BackendUnavailableError):
            limiter.allow("k")

    def test_allow_wait_raises_backend_unavailable_without_a_metrics_event(
        self,
    ) -> None:
        """Deliberate scope boundary (rlimit.metrics module
        docstring point 9): allow_wait() still raises
        BackendUnavailableError on a backend failure, but does NOT
        additionally emit a backend_error metrics event ."""
        clock = FakeClock()
        storage = _FlakyStorage(fail_on={"get"})
        hook = RecordingHook()
        limiter = FixedWindow(
            limit=5, period=60.0, storage=storage, clock=clock, metrics=hook
        )

        with pytest.raises(BackendUnavailableError):
            limiter.allow_wait("k")

        assert hook.backend_errors == []  # scope boundary, not a bug

    def test_recovering_backend_after_a_failure_resumes_normal_metrics(
        self,
    ) -> None:
        """A transient failure followed by a healthy call must resume
        normal allowed/denied metrics -- the failure state isn't
        sticky anywhere in the algorithm or hook plumbing."""
        clock = FakeClock()
        storage = _FlakyStorage(fail_on={"get"})
        hook = RecordingHook()
        limiter = FixedWindow(
            limit=5, period=60.0, storage=storage, clock=clock, metrics=hook
        )

        with pytest.raises(BackendUnavailableError):
            limiter.allow("k")
        assert len(hook.backend_errors) == 1

        storage.fail_on.clear()  # backend "recovers"
        assert limiter.allow("k") is True
        assert len(hook.allowed) == 1
        assert len(hook.backend_errors) == 1  # unchanged, not double-counted
