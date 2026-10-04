# tests/test_async_base.py
"""Tests for rlimit.base.AsyncRateLimiter.

Mirrors test_base.py's four tests for the sync RateLimiter ABC exactly,
just with async method implementations and await where needed.
"""

from __future__ import annotations

import pytest

from rlimit.base import AsyncRateLimiter


def test_async_rate_limiter_is_abstract() -> None:
    """AsyncRateLimiter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AsyncRateLimiter()  # type: ignore[abstract]


def test_async_rate_limiter_abstract_methods() -> None:
    """The three required methods are all declared abstract."""
    assert AsyncRateLimiter.__abstractmethods__ == frozenset(
        {"allow", "allow_wait", "remaining"}
    )


def test_concrete_async_subclass_must_implement_all_methods() -> None:
    """A subclass missing any abstract method still can't be instantiated."""

    class IncompleteAsyncLimiter(AsyncRateLimiter):
        async def allow(self, key: str, cost: int = 1) -> bool:
            return True

        async def remaining(self, key: str) -> int:
            return 0

        # allow_wait intentionally not implemented

    with pytest.raises(TypeError):
        IncompleteAsyncLimiter()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_concrete_async_subclass_with_all_methods_instantiates() -> None:
    """A subclass implementing all three methods works normally."""

    class DummyAsyncLimiter(AsyncRateLimiter):
        async def allow(self, key: str, cost: int = 1) -> bool:
            return True

        async def allow_wait(self, key: str, cost: int = 1) -> float:
            return 0.0

        async def remaining(self, key: str) -> int:
            return 5

    limiter = DummyAsyncLimiter()
    assert await limiter.allow("k") is True
    assert await limiter.allow_wait("k") == 0.0
    assert await limiter.remaining("k") == 5
