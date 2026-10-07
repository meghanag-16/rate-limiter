# src/limivault/async_ergonomics.py
"""Ergonomics layer (async).

Direct async mirror of ergonomics.py -- same three pieces
(async_block_until_allowed, async_wait, async_rate_limit /
AsyncKeyedLimiter), same "ADVISORY, NOT A RESERVATION" retry-loop
rationale (see ergonomics.py's module docstring, not repeated here in
full).

RateLimitTimeoutError is reused from ergonomics.py rather than defining
a second, separately-catchable exception type -- callers using both the
sync and async ergonomics layers in the same codebase (e.g. a sync CLI
tool and an async web service sharing rate-limit config) can catch one
exception type regardless of which side timed out.

`sleep` defaults to asyncio.sleep and must be an async callable
(awaiting it must not block the event loop). Passing a sync
time.sleep-like callable here would stall every other coroutine on the
loop for the duration of the wait, not just the caller -- this is the
async-specific hazard that has no sync-side equivalent, so it's called
out explicitly rather than assumed obvious.
"""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

from limivault.base import AsyncRateLimiter
from limivault.ergonomics import RateLimitTimeoutError, _reject_invalid_timeout

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

__all__ = [
    "RateLimitTimeoutError",
    "async_block_until_allowed",
    "async_wait",
    "AsyncKeyedLimiter",
    "async_rate_limit",
]


async def async_block_until_allowed(
    limiter: AsyncRateLimiter,
    key: str,
    cost: int = 1,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Async mirror of ergonomics.block_until_allowed. See that
    function's docstring for the full retry-loop rationale (identical
    here) -- the same ADVISORY, NOT A RESERVATION concern applies to
    AsyncRateLimiter.allow_wait().

    `timeout=None` (default) waits indefinitely; otherwise raises
    RateLimitTimeoutError once `timeout` seconds (per `clock`) have
    elapsed without success. Propagates UnsatisfiableRequestError as-is
    if the underlying limiter's allow_wait() raises it.

    Raises ValueError immediately if `timeout` is negative or
    non-finite (NaN/infinity) -- see ergonomics._reject_invalid_timeout()
    (reused here rather than duplicated) -- before any allow()/
    allow_wait() call is made.
    """
    _reject_invalid_timeout(timeout)

    if await limiter.allow(key, cost):
        return

    deadline = None if timeout is None else clock() + timeout

    while True:
        # Propagates UnsatisfiableRequestError untouched if raised here.
        wait_seconds = await limiter.allow_wait(key, cost)

        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                raise RateLimitTimeoutError(
                    f"timed out after {timeout}s waiting for key {key!r} "
                    f"(cost={cost})"
                )
            wait_seconds = min(wait_seconds, remaining)

        # Always await sleep, even when wait_seconds == 0 -- this is
        # the async-specific sharpest version of the concern: skipping
        # the sleep on a 0-wait meant a denied allow() ->
        # allow_wait()==0 -> denied allow() sequence could spin with no
        # `await` checkpoint at all, which doesn't just waste CPU (as
        # in the sync case) but actively starves every other coroutine
        # on the event loop, since nothing ever yields control back to
        # the loop. await sleep(0) guarantees a genuine yield point on
        # every iteration.
        await sleep(wait_seconds)

        if await limiter.allow(key, cost):
            return

        if deadline is not None and clock() >= deadline:
            raise RateLimitTimeoutError(
                f"timed out after {timeout}s waiting for key {key!r} "
                f"(cost={cost})"
            )
        # Otherwise loop back and ask allow_wait() again -- see
        # ergonomics.py's module docstring for why a single wait is
        # never trusted.


class async_wait:
    """Async context manager mirror of ergonomics.wait. See that
    class's docstring for the full rationale (identical here).

    Usage:
        async with async_wait(limiter, "user:123"):
            await do_the_rate_limited_thing()
    """

    def __init__(
        self,
        limiter: AsyncRateLimiter,
        key: str,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limiter = limiter
        self._key = key
        self._cost = cost
        self._timeout = timeout
        self._clock = clock
        self._sleep = sleep

    async def __aenter__(self) -> "async_wait":
        await async_block_until_allowed(
            self._limiter,
            self._key,
            cost=self._cost,
            timeout=self._timeout,
            clock=self._clock,
            sleep=self._sleep,
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class AsyncKeyedLimiter:
    """Async mirror of ergonomics.KeyedLimiter -- derives a rate-limit
    key from call arguments and, via block_until_allowed()/wait()
    below, is directly usable on its own (not just as an internal
    helper for the async_rate_limit() decorator). See
    ergonomics.KeyedLimiter's docstring for the full rationale
    (identical here, just async).

    `key_func` stays a plain *sync* callable -- deriving a string key
    from call arguments is not I/O and has no reason to be a coroutine;
    making it async would force every caller to `await` a trivial
    lookup for no benefit.
    """

    _GLOBAL_KEY = "__global__"

    def __init__(
        self,
        limiter: AsyncRateLimiter,
        key_func: Callable[..., str] | None = None,
    ) -> None:
        self.limiter = limiter
        self._key_func = key_func

    def key_for_call(self, *args: Any, **kwargs: Any) -> str:
        if self._key_func is None:
            return self._GLOBAL_KEY
        return self._key_func(*args, **kwargs)

    async def block_until_allowed(
        self,
        *args: Any,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **kwargs: Any,
    ) -> None:
        """Derive the key from (*args, **kwargs) via key_for_call(), then
        block until `self.limiter.allow(key, cost)` succeeds.

        Usable directly, with no decorator required:

            keyed = AsyncKeyedLimiter(limiter, key_func=lambda user_id: user_id)
            await keyed.block_until_allowed("alice")

        Delegates entirely to the module-level async_block_until_allowed()
        -- see that function's docstring for the full retry-loop,
        timeout, and "ADVISORY, NOT A RESERVATION" semantics.
        """
        call_key = self.key_for_call(*args, **kwargs)
        await async_block_until_allowed(
            self.limiter,
            call_key,
            cost=cost,
            timeout=timeout,
            clock=clock,
            sleep=sleep,
        )

    def wait(
        self,
        *args: Any,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **kwargs: Any,
    ) -> "async_wait":
        """Derive the key from (*args, **kwargs) via key_for_call(), then
        return an async context manager that blocks until admitted:

            keyed = AsyncKeyedLimiter(limiter, key_func=lambda user_id: user_id)
            async with keyed.wait("alice"):
                await do_the_rate_limited_thing()

        Delegates entirely to the module-level async_wait context
        manager class -- see its docstring for the full semantics.
        """
        call_key = self.key_for_call(*args, **kwargs)
        return async_wait(
            self.limiter,
            call_key,
            cost=cost,
            timeout=timeout,
            clock=clock,
            sleep=sleep,
        )


def async_rate_limit(
    limiter: AsyncRateLimiter,
    cost: int = 1,
    key_func: Callable[..., str] | None = None,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[[F], F]:
    """Async mirror of ergonomics.rate_limit. Decorates an `async def`
    function; the wrapped function is awaited after the retry loop
    admits the call.

    Global limit:
        @async_rate_limit(my_async_limiter, cost=1)
        async def do_thing():
            ...

    Per-key limit:
        @async_rate_limit(my_async_limiter, key_func=lambda user_id, *a, **k: user_id)
        async def do_thing(user_id, ...):
            ...

    Built directly on AsyncKeyedLimiter.block_until_allowed() -- this
    function is sugar over that one method, not a separate
    implementation of the key-then-block sequence. If you want per-key
    blocking without wrapping a whole coroutine function, use
    AsyncKeyedLimiter directly instead of reaching for this decorator.
    """
    keyed = AsyncKeyedLimiter(limiter, key_func)

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await keyed.block_until_allowed(
                *args,
                cost=cost,
                timeout=timeout,
                clock=clock,
                sleep=sleep,
                **kwargs,
            )
            return await func(*args, **kwargs)

        # cast, not `# type: ignore[return-value]` -- see
        # ergonomics.py's rate_limit() for why: an ignore here would
        # leak Any into every function decorated with @async_rate_limit,
        # cascading into spurious "untyped decorator" errors at each
        # call site.
        return cast(F, wrapper)

    return decorator