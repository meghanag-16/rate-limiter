# src/limivault/ergonomics.py
"""Ergonomics layer (sync).

Everything here is built on top of the existing RateLimiter contract
(allow()/allow_wait(), see base.py) -- nothing in this file talks to
storage or a clock's internal state directly, and nothing here is a new
rate-limiting algorithm. 

This module provides three main components:

1. block_until_allowed() -- the actual retry loop. Everything else in
   this file is sugar over this one function, so there is exactly one
   place the retry logic lives.
2. wait -- a context manager: `with wait(limiter, key): ...`
3. rate_limit() -- a decorator: `@rate_limit(limiter)` / with key_func
   for per-key limiting, via KeyedLimiter.

ADVISORY, NOT A RESERVATION (read before changing block_until_allowed):
base.py is explicit that allow_wait()'s returned duration is a
point-in-time calculation made while the per-key lock was held, but the
lock is released the instant allow_wait() returns -- nothing reserves
that capacity. A caller that sleeps for the returned duration and then
blindly calls allow() once, assuming that single wait was sufficient,
has a real bug: another caller (another thread, or another call to this
same blocking helper) can consume the capacity being waited on before
the sleeping caller wakes up. block_until_allowed() below always
re-checks allow() after sleeping and loops back to allow_wait() again
if denied -- it does not trust a single wait. This was the thing
that is one invariant every test in
this file's test suite must specifically verify (a starved key with two
concurrent callers must not double-admit).

Timeout: measured against `clock`, not wall-clock time directly, so
tests can drive it deterministically with a FakeClock the same way
every other timed test in this project already does. If you pass a
FakeClock to the limiter's own `clock=` constructor argument, pass the
*same* FakeClock instance here too -- block_until_allowed() has no way
to see the limiter's internal clock and does not assume it matches.

Sleep: also injectable (`sleep: Callable[[float], None]`, defaults to
time.sleep), for the same reason -- tests should never actually sleep
in real time.
"""

from __future__ import annotations

import math
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from limivault.base import RateLimiter

F = TypeVar("F", bound=Callable[..., Any])


class RateLimitTimeoutError(TimeoutError):
    """Raised by block_until_allowed() (and anything built on it) when
    `timeout` is not None and that many seconds elapse without the
    request becoming allowed.

    Inherits from the standard library's TimeoutError (not plain
    Exception) so callers can catch rate-limit timeouts using Python's
    ordinary timeout-handling idiom -- `except TimeoutError:` -- the
    same way they would for socket, asyncio, or concurrent.futures
    timeouts, without needing to know this project's specific
    exception type in advance.

    Distinct from UnsatisfiableRequestError (base.py): that one means
    "no amount of waiting would ever help" and is raised immediately,
    with no sleeping at all, by the underlying limiter's allow_wait().
    RateLimitTimeoutError means "this specific call gave up after
    waiting `timeout` seconds" -- the request might well have succeeded
    with more patience.
    """

    ...


def _reject_invalid_timeout(timeout: float | None) -> None:
    """Raise ValueError if `timeout` is negative or non-finite (NaN or
    +/-infinity).

    `timeout=None` (wait indefinitely) is always valid and returns
    immediately. A negative timeout has no sensible meaning (you can't
    wait a negative amount of time) and previously fell through to
    `deadline = clock() + timeout`, silently producing a deadline in
    the past and behaving like `timeout=0` by accident rather than by
    the caller's explicit choice. NaN silently passes ordinary `< 0`
    comparisons (they're always False against NaN, the same footgun
    documented for limit/period/cost validation throughout the
    algorithm files), and +/-infinity is rejected for the same reason
    those are rejected everywhere else in this project: there's no
    supported "unlimited" spelling other than `timeout=None` itself.
    """
    if timeout is None:
        return
    if not math.isfinite(timeout):
        raise ValueError(f"timeout must be finite, got {timeout}")
    if timeout < 0:
        raise ValueError(f"timeout must be non-negative, got {timeout}")


def block_until_allowed(
    limiter: RateLimiter,
    key: str,
    cost: int = 1,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block the calling thread until `limiter.allow(key, cost)` returns
    True.

    `timeout=None` (default) waits indefinitely. Otherwise raises
    RateLimitTimeoutError once `timeout` seconds (per `clock`) have
    elapsed without success.

    Propagates UnsatisfiableRequestError as-is if the underlying
    limiter's allow_wait() raises it (cost exceeds the limiter's
    maximum possible capacity) -- that case can never succeed no matter
    how long this function waits, so it fails fast rather than sleeping
    first.

    Raises ValueError immediately if `timeout` is negative or
    non-finite (NaN/infinity) -- see _reject_invalid_timeout() -- before
    any allow()/allow_wait() call is made.

    See this module's docstring for the "ADVISORY, NOT A RESERVATION"
    rationale for why this loops (re-checking allow() after every
    sleep) instead of sleeping once and assuming success.
    """
    _reject_invalid_timeout(timeout)

    if limiter.allow(key, cost):
        return

    deadline = None if timeout is None else clock() + timeout

    while True:
        # Propagates UnsatisfiableRequestError untouched if raised here.
        wait_seconds = limiter.allow_wait(key, cost)

        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                raise RateLimitTimeoutError(
                    f"timed out after {timeout}s waiting for key {key!r} "
                    f"(cost={cost})"
                )
            wait_seconds = min(wait_seconds, remaining)

        # Always sleep, even when wait_seconds == 0 -- sleep(0) is a
        # pure yield point (thread scheduling point here; the async
        # mirror's await sleep(0) yields the event loop). Skipping the
        # sleep entirely when wait_seconds == 0 meant that, under
        # contention, a denied allow() -> allow_wait()==0 -> denied
        # allow() sequence could spin in a tight loop calling
        # allow()/allow_wait() back-to-back with no yield at all,
        # starving other threads waiting on the same lock instead of
        # giving them a scheduling opportunity.
        sleep(wait_seconds)

        if limiter.allow(key, cost):
            return

        if deadline is not None and clock() >= deadline:
            raise RateLimitTimeoutError(
                f"timed out after {timeout}s waiting for key {key!r} "
                f"(cost={cost})"
            )
        # Otherwise: allow() was denied again after sleeping (another
        # caller took the capacity being waited on). Loop back and ask
        # allow_wait() again rather than trusting the previous wait.


class wait:
    """Context manager: blocks until `limiter.allow(key, cost)` succeeds,
    then runs the wrapped block.

    Usage:
        with wait(limiter, "user:123"):
            do_the_rate_limited_thing()

    With a timeout:
        with wait(limiter, "user:123", timeout=5.0):
            do_the_rate_limited_thing()  # raises RateLimitTimeoutError
                                          # if not admitted within 5s

    __exit__ is a no-op. Unlike a lock, there is nothing to release when
    the block finishes -- allow() already committed the cost the moment
    __enter__ returned successfully, and the limiter has no "give back
    the capacity" operation.
    """

    def __init__(
        self,
        limiter: RateLimiter,
        key: str,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limiter = limiter
        self._key = key
        self._cost = cost
        self._timeout = timeout
        self._clock = clock
        self._sleep = sleep

    def __enter__(self) -> "wait":
        block_until_allowed(
            self._limiter,
            self._key,
            cost=self._cost,
            timeout=self._timeout,
            clock=self._clock,
            sleep=self._sleep,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class KeyedLimiter:
    """Derives a rate-limit key from call arguments, enabling per-key
    (e.g. per user_id/IP) limiting instead of one global key shared by
    every call -- and, via block_until_allowed()/wait() below, is
    directly usable on its own, not just as an internal helper for the
    rate_limit() decorator.

    `key_func` receives the exact same (*args, **kwargs) a "call" is
    made with and must return a str. If `key_func` is None (the
    default), every call maps to a single fixed key -- a global limit
    shared across all callers. This is deliberately one mechanism with
    a default, not two separate code paths, matching the plan's "keyed
    wrapper for per-key ... vs global limiting" as two ends of the same
    spectrum.

    Three ways to use a KeyedLimiter, in increasing order of how much
    it does for you:

    1. key_for_call(*args, **kwargs) -> str: just derive the key,
       do whatever you want with it yourself (e.g. call the module-level
       block_until_allowed() or wait() directly, or call
       limiter.allow(key) yourself in a hand-rolled retry loop).
    2. keyed.block_until_allowed(*args, **kwargs): derive the key AND
       block until admitted, all in one call -- usable directly inside
       a method body or an ad hoc retry loop, with no decorator or
       function-wrapping required.
    3. keyed.wait(*args, **kwargs): same as #2, but as a context
       manager (`with keyed.wait(user_id): ...`).

    rate_limit() (below) is built on #2 -- it is sugar over
    KeyedLimiter.block_until_allowed(), not a second implementation of
    the key-then-block sequence.
    """

    _GLOBAL_KEY = "__global__"

    def __init__(
        self,
        limiter: RateLimiter,
        key_func: Callable[..., str] | None = None,
    ) -> None:
        self.limiter = limiter
        self._key_func = key_func

    def key_for_call(self, *args: Any, **kwargs: Any) -> str:
        if self._key_func is None:
            return self._GLOBAL_KEY
        return self._key_func(*args, **kwargs)

    def block_until_allowed(
        self,
        *args: Any,
        cost: int = 1,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> None:
        """Derive the key from (*args, **kwargs) via key_for_call(), then
        block until `self.limiter.allow(key, cost)` succeeds.

        *args/**kwargs are passed to key_func exactly as given here --
        this is NOT a decorator, so there's no wrapped function whose
        signature to match; call it directly with whatever positional/
        keyword arguments your key_func expects to receive, e.g.:

            keyed = KeyedLimiter(limiter, key_func=lambda user_id: user_id)
            keyed.block_until_allowed("alice")  # blocks on key "alice"

        Delegates entirely to the module-level block_until_allowed() --
        see that function's docstring for the full retry-loop, timeout,
        and "ADVISORY, NOT A RESERVATION" semantics, all of which apply
        unchanged here.
        """
        call_key = self.key_for_call(*args, **kwargs)
        block_until_allowed(
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
        sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> "wait":
        """Derive the key from (*args, **kwargs) via key_for_call(), then
        return a context manager that blocks until admitted:

            keyed = KeyedLimiter(limiter, key_func=lambda user_id: user_id)
            with keyed.wait("alice"):
                do_the_rate_limited_thing()

        Delegates entirely to the module-level `wait` context manager
        class -- see its docstring for the full semantics.
        """
        call_key = self.key_for_call(*args, **kwargs)
        return wait(
            self.limiter,
            call_key,
            cost=cost,
            timeout=timeout,
            clock=clock,
            sleep=sleep,
        )


def rate_limit(
    limiter: RateLimiter,
    cost: int = 1,
    key_func: Callable[..., str] | None = None,
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[F], F]:
    """Decorator: blocks each call to the wrapped function until
    `limiter.allow(key, cost)` succeeds, then calls the function.

    Global limit (one shared key across every caller):
        @rate_limit(my_limiter, cost=1)
        def do_thing():
            ...

    Per-key limit (e.g. per user_id, assuming it's the first positional
    arg of the wrapped function):
        @rate_limit(my_limiter, key_func=lambda user_id, *a, **k: user_id)
        def do_thing(user_id, ...):
            ...

    Built directly on KeyedLimiter.block_until_allowed() -- this
    function is sugar over that one method, not a separate
    implementation of the key-then-block sequence. If you want per-key
    blocking without wrapping a whole function (e.g. inside a method
    body or an ad hoc retry loop), use KeyedLimiter directly instead of
    reaching for this decorator.
    """
    keyed = KeyedLimiter(limiter, key_func)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            keyed.block_until_allowed(
                *args,
                cost=cost,
                timeout=timeout,
                clock=clock,
                sleep=sleep,
                **kwargs,
            )
            return func(*args, **kwargs)

        # cast, not `# type: ignore[return-value]`: wrapper's signature
        # is intentionally (*args: Any, **kwargs: Any) -> Any so it can
        # wrap any F, but that means mypy can't verify it matches F
        # structurally. A blanket ignore here would suppress the error
        # by making the whole return type Any, which then cascades into
        # "untyped decorator" errors at every call site using
        # @rate_limit(...). cast() instead asserts the specific,
        # correct type without leaking Any into every decorated
        # function's inferred type.
        return cast(F, wrapper)

    return decorator