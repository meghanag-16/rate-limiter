# tests/test_base.py
from __future__ import annotations

import pytest

from limivault.base import RateLimiter


def test_rate_limiter_is_abstract() -> None:
    """RateLimiter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        RateLimiter()  # type: ignore[abstract]


def test_rate_limiter_abstract_methods() -> None:
    """The three required methods are all declared abstract."""
    assert RateLimiter.__abstractmethods__ == frozenset(
        {"allow", "allow_wait", "remaining"}
    )


def test_concrete_subclass_must_implement_all_methods() -> None:
    """A subclass missing any abstract method still can't be instantiated."""

    class IncompleteLimiter(RateLimiter):
        def allow(self, key: str, cost: int = 1) -> bool:
            return True

        def remaining(self, key: str) -> int:
            return 0

        # allow_wait intentionally not implemented

    with pytest.raises(TypeError):
        IncompleteLimiter()  # type: ignore[abstract]


def test_concrete_subclass_with_all_methods_instantiates() -> None:
    """A subclass implementing all three methods works normally."""

    class DummyLimiter(RateLimiter):
        def allow(self, key: str, cost: int = 1) -> bool:
            return True

        def allow_wait(self, key: str, cost: int = 1) -> float:
            return 0.0

        def remaining(self, key: str) -> int:
            return 5

    limiter = DummyLimiter()
    assert limiter.allow("k") is True
    assert limiter.allow_wait("k") == 0.0
    assert limiter.remaining("k") == 5