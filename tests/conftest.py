"""Shared pytest fixtures.

NOTE (flagged): the actual FakeClock implementation used was not
included in the files provided for this session, so this is a minimal
reimplementation based on the documented pattern ("manually advanceable
FakeClock via clock injection"). If your real fixture lives in a different
module or has a different API, swap this out and the tests below should
still work as long as it exposes __call__() -> float and advance(seconds).
"""

from typing import Any

import pytest

pytest_plugins = ["tests.redis_test_helpers"]


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start=0.0)
_DOCKER_FIXTURES = {
    "redis_container",
    "redis_connection_params",
    "redis_client",
    "async_redis_client",
}


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Label tests that request the shared Redis fixtures as Docker tests."""
    docker_marker = pytest.mark.docker
    for item in items:
        if _DOCKER_FIXTURES.intersection(item.fixturenames):
            item.add_marker(docker_marker)
