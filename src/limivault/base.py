# src/limivault/base.py
"""Abstract interfaces for limivault.

Contains both the sync interface and the async interface.
The async interface mirrors the sync one method-for-method
and shares the same contract, spelled out in full below.

--------------------------------------------------------------------
CONTRACT (applies identically to RateLimiter and AsyncRateLimiter)
--------------------------------------------------------------------

cost type:
    - `cost` must be an actual `int`. This is enforced at runtime, not
      just declared in the type annotation -- Python does not stop a
      caller from passing `cost=0.5` or `cost=2.0` where an `int` is
      annotated, and doing so used to be silently accepted throughout
      every algorithm. That let float values leak into what should be
      integer counter state (e.g. a stored `count` becoming `2.0`
      instead of `2`), which is inconsistent with `remaining() -> int`
      and was never a supported input.
    - `bool` is rejected too, even though `bool` is technically an
      `int` subclass in Python. `cost=True`/`cost=False` is not a
      meaningful rate-limiting cost and is treated the same as any
      other wrong-type value below.
    - A wrong-type cost is treated exactly like an invalid value: see
      "invalid cost" under allow()/allow_wait() below. There is no
      separate "TypeError" channel -- wrong type and invalid value
      (negative) share the same failure mode per method, to keep the
      two failure surfaces (allow() never raises; allow_wait() raises
      for invalid input) each internally consistent rather than having
      wrong-type sometimes raise something allow() doesn't otherwise
      raise.

allow(key, cost=1):
    - Never raises. If `cost` exceeds the limiter's maximum possible
      capacity, returns False rather than raising.
    - `cost` that is not a valid int (wrong type, or a valid int that
      is negative) is denied: returns False without touching stored
      state at all -- not even to read it.
    - cost == 0 is always allowed (returns True) and is handled by an
      early return before the per-key lock is even acquired: it does
      not read storage, does not write storage, and does not create
      state for a key that has never been seen before. This was a
      real bug -- every algorithm's allow() used to
      still run its full read/lock/write path for cost == 0, and
      SlidingWindowLog specifically appended a (timestamp, 0) log
      entry on every zero-cost call, so repeated zero-cost calls grew
      the log indefinitely despite consuming no quota. Fixed by making
      the zero-cost case a true no-op, not merely "consumes nothing
      but still performs I/O."

allow_wait(key, cost=1):
    - Returns the number of seconds to wait until `allow(key, cost)`
      would return True. Returns 0.0 if the request would already be
      allowed (this includes cost == 0, which always returns 0.0 via
      the same early-return-before-any-storage-access pattern as
      allow()).
    - Raises UnsatisfiableRequestError if `cost` exceeds the limiter's
      maximum possible capacity -- no amount of waiting would help.
    - Raises ValueError if `cost` is not a valid int (wrong type, or a
      valid int that is negative). This is a deliberate asymmetry with
      allow(): allow() never raises by contract, so invalid cost is
      silently denied there, but allow_wait() already raises for the
      "can never be satisfied" case, so raising ValueError for
      "invalid input" too keeps both failure modes in the same channel
      (exceptions) rather than splitting them across a boolean return
      and an exception depending on which method you called.
    - ADVISORY, NOT A RESERVATION: the returned wait is a point-in-time
      calculation made while the per-key lock was held, but the lock
      is released the instant allow_wait() returns. Nothing about
      calling allow_wait() reserves capacity or prevents another
      concurrent caller (another thread, coroutine, process, or simply
      another caller on the same key) from consuming that capacity
      before you act on the returned wait time. A caller that needs an
      actual guarantee must loop and re-check, not trust a single
      wait-then-retry:

          while not <await >allow(key, cost):
              wait = <await >allow_wait(key, cost)
              <await asyncio.sleep(wait) / time.sleep(wait)>
              # loop back and re-check allow() -- do NOT assume the
              # single wait was sufficient, since another caller may
              # have consumed the capacity you were waiting for.

      This check-then-act gap is inherent to any limiter that doesn't
      support reserve/commit semantics. The blocking context
      manager / decorator interface must implement the retry loop
      above, not a single sleep-then-proceed. A real reservation API
      (if ever added) would be a new method, not a strengthened
      allow_wait() contract, since allow_wait() intentionally does not
      hold the lock across the wait -- holding a per-key lock across a
      real sleep would serialize every waiting caller on that key
      through the sleep itself, which is a different (and generally
      worse) tradeoff than the current advisory design.

remaining(key):
    - Never consumes quota. Purely observational.

Clock:
    - The `clock: Callable[[], float]` parameter accepted by every
      concrete implementation must be monotonic (non-decreasing).
      time.monotonic (the default) satisfies this. A clock that can
      go backwards -- e.g. raw wall-clock time subject to NTP
      adjustment -- is not supported: window/refill/leak arithmetic
      throughout every algorithm assumes `now` never decreases between
      calls, and a backwards jump can produce negative elapsed-time
      values that corrupt state in the same "silent, no exception"
      way the NaN/infinity bug did. This was never enforced
      at runtime (no validation call rejects a non-monotonic clock)
      and isn't planned to be -- it's a documented precondition on
      what you pass as `clock`, not a checked invariant.

Thread / coroutine / process safety:
    - Sync RateLimiter + the default InMemoryStorage
      (multiprocess_safe=False): safe across threads within one
      process. NOT safe across processes -- construct with
      multiprocess_safe=True for that (see storage.py).
    - Async AsyncRateLimiter + AsyncInMemoryStorage: safe across
      coroutines sharing one event loop in one process. NOT safe
      across OS threads, NOT safe across multiple event loops (even
      within one process), and NOT safe across processes. See
      AsyncInMemoryStorage's docstring in storage.py for the full
      reasoning and for what multi-process async coordination
      requires instead (the Redis-native async Lua/GCRA limiters, see
      limivault.redis_lua_scripts).
--------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    # Only needed for type hints on the wait() convenience methods
    # below; kept under TYPE_CHECKING (rather than imported at module
    # level) purely to keep this module's runtime import list minimal
    # -- these are stdlib types with no circularity concern either way.
    from contextlib import AbstractAsyncContextManager, AbstractContextManager


class UnsatisfiableRequestError(Exception):
    """Raised when a requested cost can never be satisfied, regardless of
    how long the caller waits (e.g. cost exceeds the limiter's maximum
    possible capacity)."""

    ...


class RateLimiter(ABC):
    """Abstract interface all sync rate limiting algorithms implement.

    See this module's docstring for the full contract (cost type and
    handling, clock requirements, thread safety, and allow_wait()'s
    advisory semantics) -- not repeated per-method here to avoid the
    contract drifting out of sync between methods and classes.
    """

    @abstractmethod
    def allow(self, key: str, cost: int = 1) -> bool:
        """Attempt to consume `cost` units of quota for `key`.

        Returns True if allowed (quota consumed), False if denied
        (quota unaffected, including for an invalid `cost`). Never
        raises. See module docstring for the full cost-handling
        contract (type validation, zero cost, negative cost,
        over-capacity cost).
        """
        ...

    @abstractmethod
    def allow_wait(self, key: str, cost: int = 1) -> float:
        """Return the number of seconds to wait until `allow(key, cost)`
        would return True. Returns 0.0 if already allowed.

        Raises UnsatisfiableRequestError if `cost` exceeds capacity;
        raises ValueError if `cost` is not a valid int (wrong type or
        negative). ADVISORY ONLY -- does not reserve capacity. See
        module docstring's "ADVISORY, NOT A RESERVATION" section
        before building any wait-then-retry logic on top of this
        method.
        """
        ...

    @abstractmethod
    def remaining(self, key: str) -> int:
        """Return the remaining quota for `key` without consuming any."""
        ...

    def wait(
        self,
        key: str,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "AbstractContextManager[Any]":
        """Convenience method: `limiter.wait(key)` is equivalent to
        `limivault.ergonomics.wait(limiter, key)`.

        Returns a context manager that blocks until `allow(key, cost)`
        succeeds before running the wrapped block:

            with limiter.wait("user:123"):
                do_the_rate_limited_thing()

        See ergonomics.wait's docstring for the full retry-loop and
        "ADVISORY, NOT A RESERVATION" semantics behind it -- this
        method does not reimplement any of that, it only forwards to
        the free function.

        Implemented via a lazy import *inside* this method rather than
        at module level: ergonomics.py imports RateLimiter from this
        module (base.py), so importing ergonomics at base.py's module
        level would create a circular import. The free function
        `limivault.ergonomics.wait(limiter, key, ...)` remains the
        canonical implementation; this method is a thin, optional
        convenience wrapper over it, not a second implementation.
        """
        from limivault.ergonomics import wait as _wait

        return _wait(self, key, cost=cost, timeout=timeout, clock=clock, sleep=sleep)


class AsyncRateLimiter(ABC):
    """Abstract interface all async rate limiting algorithms implement.

    Mirrors RateLimiter exactly -- same contract (see module docstring),
    same method semantics -- just with coroutine methods so callers
    under asyncio don't block the event loop while waiting on a lock.
    The clock parameter accepted by concrete implementations stays a
    plain sync Callable[[], float]; there is no separate "async clock"
    concept.
    """

    @abstractmethod
    async def allow(self, key: str, cost: int = 1) -> bool:
        """Attempt to consume `cost` units of quota for `key`.

        Returns True if allowed (quota consumed), False if denied
        (quota unaffected, including for an invalid `cost`). Never
        raises. See base.py's module docstring for the full
        cost-handling contract.
        """
        ...

    @abstractmethod
    async def allow_wait(self, key: str, cost: int = 1) -> float:
        """Return the number of seconds to wait until `allow(key, cost)`
        would return True. Returns 0.0 if already allowed.

        Raises UnsatisfiableRequestError if `cost` exceeds capacity;
        raises ValueError if `cost` is not a valid int (wrong type or
        negative). ADVISORY ONLY -- does not reserve capacity. Another
        coroutine can consume the capacity you're waiting for between
        this call returning and your retry. See base.py's module
        docstring's "ADVISORY, NOT A RESERVATION" section -- this is
        the exact concern blocking interface must implement
        a real retry loop around, not a single sleep-then-proceed.
        """
        ...

    @abstractmethod
    async def remaining(self, key: str) -> int:
        """Return the remaining quota for `key` without consuming any."""
        ...

    def wait(
        self,
        key: str,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> "AbstractAsyncContextManager[Any]":
        """Convenience method: `limiter.wait(key)` is equivalent to
        `limivault.async_ergonomics.async_wait(limiter, key)`.

        Usage:
            async with limiter.wait("user:123"):
                await do_the_rate_limited_thing()

        See async_ergonomics.async_wait's docstring for the full
        retry-loop and "ADVISORY, NOT A RESERVATION" semantics.
        Implemented via a lazy import inside this method for the same
        circular-import reason as RateLimiter.wait() above (mirrored
        here: async_ergonomics.py imports AsyncRateLimiter from this
        module). The free function
        `limivault.async_ergonomics.async_wait(limiter, key, ...)` remains
        the canonical implementation.
        """
        from limivault.async_ergonomics import async_wait as _async_wait

        return _async_wait(
            self, key, cost=cost, timeout=timeout, clock=clock, sleep=sleep
        )
