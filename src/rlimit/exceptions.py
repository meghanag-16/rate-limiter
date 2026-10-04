# src/rlimit/exceptions.py
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

Where this is raised: the Redis Lua/GCRA limiters
(rlimit.algorithms.redis_lua_* and async_redis_lua_*), around their
Redis calls. See rlimit.redis_lua_scripts's module docstring for how a
redis-py failure is classified. InMemoryStorage / AsyncInMemoryStorage
never raise this -- there's no network boundary to fail -- but a custom
StorageBackend / AsyncStorageBackend implementation may, and the
in-memory algorithms handle it the same way (see below).

Where this surfaces to metrics: every sync/async algorithm's allow()
catches this around its storage access and emits a `backend_error`
metrics event (see rlimit.metrics) before re-raising it unchanged.
allow_wait() does NOT get this same metrics treatment in this way --
see each algorithm file's module docstring for that explicitly scoped
limitation (backend errors from allow_wait() still raise
BackendUnavailableError normally, they just don't additionally emit a
metrics event yet).
"""

from __future__ import annotations


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
