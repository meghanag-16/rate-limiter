# src/limivault/exceptions.py
"""Library-level exceptions for backend failures.

BackendUnavailableError wraps storage-backend-specific failures (e.g.
redis-py's `redis.exceptions.RedisError` and its subclasses -- 
ConnectionError, TimeoutError, etc.) so the public API surface does not
leak a specific backend library's exception hierarchy. Every raise site
chains the original via `raise BackendUnavailableError(...) from exc`,
so the underlying cause is still inspectable (`err.__cause__`) rather
than being swallowed.

DESIGN DECISION (locked in for this pass, per review):
A backend failure is NEVER converted into `False` (denied) and never
silently treated the same as a normal rate-limit decision. Doing
either would make an infrastructure outage indistinguishable from
legitimate throttling, which is a correctness-hiding bug, not a
convenience. Callers that want "fail open" or "fail closed" behavior
during an outage must decide that explicitly by catching
BackendUnavailableError themselves -- the library does not make that
choice on their behalf.

BackendBusyError (0.1.2) is a subclass raised when the redis-py
connection pool -- not the Redis server -- is the problem. See its
docstring: never fail open on it.

Where this is raised: the Redis Lua/GCRA limiters
(limivault.algorithms.redis_lua_* and async_redis_lua_*), around their
Redis calls. See limivault.redis_lua_scripts's module docstring for how a
redis-py failure is classified. InMemoryStorage / AsyncInMemoryStorage
never raise this -- there's no network boundary to fail -- but a custom
StorageBackend / AsyncStorageBackend implementation may, and the
in-memory algorithms handle it the same way (see below).

Where this surfaces to metrics: every sync/async algorithm's allow()
catches this around its storage access and emits a `backend_error`
metrics event (see limivault.metrics) before re-raising it unchanged.
allow_wait() does NOT get this same metrics treatment in this way --
see each algorithm file's module docstring for that explicitly scoped
limitation (backend errors from allow_wait() still raise
BackendUnavailableError normally, they just don't additionally emit a
metrics event yet).
"""

from __future__ import annotations

POOL_EXHAUSTED_MESSAGE = "No connection available."


class BackendUnavailableError(Exception):
    """Raised when a storage backend cannot be reached or fails at the
    infrastructure level (e.g. a Redis connection is lost, refused, or
    times out).

    Distinct from a denied rate-limit decision: catching this means
    "the limiter could not determine an answer at all" (an outage),
    not "this request was rejected because it exceeded the configured
    limit." Callers that need outage-specific handling (fail-open,
    fail-closed, circuit breaking, retry/backoff) should catch this
    exception specifically rather than treating a `False` return from
    allow() as covering this case -- it does not.
    """

    ...


class BackendBusyError(BackendUnavailableError):
    """The Redis server may be perfectly healthy, but this process has no
    free connection in its redis-py connection pool right now.

    Raised instead of a plain BackendUnavailableError when the underlying
    redis-py error is a pool-exhaustion error:

    - `redis.exceptions.MaxConnectionsError` -- the default (non-blocking)
      `ConnectionPool` raises this the instant `max_connections` are in
      use. In redis-py 8 the async pool defaults to 100 connections, so a
      single burst of >100 concurrent decisions is enough.
    - `ConnectionError("No connection available.")` -- a
      `BlockingConnectionPool` raises this after its `timeout` expires.

    WHY A SEPARATE TYPE: it subclasses BackendUnavailableError, so every
    existing `except BackendUnavailableError` keeps working unchanged.
    But "Redis is down" and "my pool is too small for this burst" need
    different handling. An app that *fails open* on outages would
    otherwise also fail open on every pool-exhaustion error, admitting
    requests with no limit check at all while Redis is healthy (this was
    reproduced under load). Recommended handling:

        try:
            allowed = await limiter.allow(key)
        except BackendBusyError:
            allowed = False            # treat as throttled, never fail open
        except BackendUnavailableError:
            allowed = FAIL_OPEN        # your outage policy

    and size the pool with `redis.asyncio.BlockingConnectionPool` so
    callers wait briefly for a connection instead of erroring instantly.
    """

    ...


def is_pool_exhausted(exc: BaseException) -> bool:
    """True if `exc` is redis-py's way of saying "no free connection in the
    pool" (see BackendBusyError). Accepts any exception; returns False for
    everything else. The BlockingConnectionPool case is matched on its
    message because redis-py raises a plain ConnectionError for it
    (verified against redis-py 8.1.0)."""
    import redis.exceptions

    if isinstance(exc, redis.exceptions.MaxConnectionsError):
        return True
    return (
        isinstance(exc, redis.exceptions.ConnectionError)
        and str(exc) == POOL_EXHAUSTED_MESSAGE
    )


def backend_error(exc: BaseException, message: str) -> BackendUnavailableError:
    """Build the exception to raise (`raise backend_error(exc, msg) from exc`)
    for a failed Redis call: BackendBusyError for pool exhaustion,
    BackendUnavailableError for everything else."""
    if is_pool_exhausted(exc):
        return BackendBusyError(f"{message} (connection pool exhausted)")
    return BackendUnavailableError(message)


class MultiProcessStateWarning(RuntimeWarning):
    """Emitted (once per process) when an in-memory storage is created in
    what looks like one of several worker processes.

    In-memory limiter state lives inside ONE process. With N workers
    (uvicorn --workers N, gunicorn -w N, several containers/instances)
    every worker enforces its own separate limit, so clients get up to N
    times the configured limit. Use the Redis limiters to share one limit.

    Detection is a heuristic, so it can be wrong in both directions:
    - warns when this process was started by `multiprocessing` (uvicorn
      --workers and also uvicorn --reload, which is single-process), or
      when the WEB_CONCURRENCY environment variable is greater than 1;
    - cannot see other containers/instances, and does not notice gunicorn
      workers unless WEB_CONCURRENCY is set.
    Silence it with LIMIVAULT_SILENCE_MULTIPROCESS_WARNING=1 or a normal
    `warnings.filterwarnings("ignore", category=MultiProcessStateWarning)`.
    """
