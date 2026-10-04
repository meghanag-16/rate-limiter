# tests/async_test_helpers.py
"""Shared async test helpers.

Kept as a plain top-level-importable module rather than a conftest.py
fixture, matching how `tests/mp_workers.py` is already imported
explicitly (`from tests.mp_workers import ...`) for the sync
multiprocessing tests -- consistent with this project's established
avoidance of conftest-based fixture sharing for cross-file test
infrastructure.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncContextManager

from rlimit.storage import AsyncInMemoryStorage, AsyncStorageBackend


class YieldingAsyncStorage(AsyncStorageBackend):
    """Wraps AsyncInMemoryStorage but forces a real event-loop yield
    (`await asyncio.sleep(0)`) inside both get() and set().

    Why this exists: AsyncInMemoryStorage's get()/set() are `async def`
    but contain no actual `await` in their bodies, so calling them does
    NOT hand control back to the event loop. Under `asyncio.gather()`,
    that means one coroutine's entire critical section (clock read ->
    get -> set) can run to completion before the next coroutine gets a
    chance to run at all -- verified directly: concurrency tests built
    on plain AsyncInMemoryStorage pass even when there is no genuine
    opportunity for two coroutines' critical sections to interleave in
    the first place, so they don't actually exercise the per-key lock
    or the clock-inside-the-lock ordering the way the sync
    ThreadPoolExecutor-based tests exercise real thread preemption.

    Using this wrapper as a limiter's storage forces real interleaving
    opportunities at the same points a real I/O-backed backend (e.g. a
    custom AsyncStorageBackend that talks to a network service, which
    would have genuine network latency here) would have them.
    Concurrency invariant tests built on top of this wrapper are
    therefore actually proving the lock and clock-ordering hold under
    interleaving, not just checking a final
    count that happened to come out right because nothing ever
    interleaved.
    """

    def __init__(self) -> None:
        self._inner = AsyncInMemoryStorage()

    async def get(self, key: str) -> dict[str, Any] | None:
        await asyncio.sleep(0)
        return await self._inner.get(key)

    async def set(self, key: str, state: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        await self._inner.set(key, state)

    def lock(self, key: str) -> AsyncContextManager[bool]:
        return self._inner.lock(key)
