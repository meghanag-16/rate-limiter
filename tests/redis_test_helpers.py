# tests/redis_test_helpers.py
"""Shared Redis test fixtures, backed by testcontainers-python.

Kept as a plain top-level-importable module rather than living in
conftest.py, matching this project's established convention (see
mp_workers.py and async_test_helpers.py) of avoiding conftest-based
fixture sharing for cross-file test infrastructure. Import what you
need explicitly, e.g.:

    from tests.redis_test_helpers import redis_client, redis_connection_params

pytest resolves fixtures by name against a test's *importing* module's
namespace, so importing a `@pytest.fixture`-decorated function this way
and using it as a normal test parameter works without any plugin
registration.

WHY TESTCONTAINERS 
avoids depending on a real, separately-managed Redis
instance for the test suite -- a remote/shared Redis risks false test
failures from network jitter in the stress/property tests, and Redis
has no native Windows support, which matters for this project's
Windows dev environment. testcontainers-python drives a real Redis
inside a Docker container instead, so:

    - Requires Docker (Docker Desktop on Windows, with Linux
      containers) to be installed AND RUNNING wherever these tests are
      executed. There is no fallback: if Docker isn't available, every
      test that depends on these fixtures will fail at container
      startup, not skip silently. This module does not attempt to
      detect Docker's absence and skip gracefully -- flagged as a gap,
      not an oversight; add a `pytest.mark.skipif` / Docker-availability
      probe here if CI runs need to tolerate a Docker-less environment.
    - The `testcontainers.redis` module is deprecated in the installed
      testcontainers version in favor of `testcontainers.community.redis`
      -- this file imports from the latter. If your installed
      testcontainers version predates that split, this import will
      fail; check your installed version if so.

SCOPE: one Redis container per test SESSION (not per test, not per
module) -- container startup has real overhead (pulling the image the
first time, starting the process), so this is shared across the whole
test run for speed. Each test gets its own logically-clean view via
`redis_client`'s `flushdb()` before AND after (before, in case a prior
test crashed mid-test without reaching its own cleanup; after, so the
next test doesn't inherit leftover state either way).
"""

from __future__ import annotations

from typing import Iterator

import pytest
import redis as redis_sync
import redis.asyncio as redis_async
from testcontainers.community.redis import RedisContainer


@pytest.fixture(scope="session")
def redis_container() -> Iterator[RedisContainer]:
    """One real Redis container, started once for the whole test
    session and torn down at the end. `redis:7-alpine` is pinned
    explicitly (not `redis:latest`) so the test suite's behavior
    doesn't silently change out from under it when a new Redis major
    version is tagged `latest` upstream."""
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture
def redis_connection_params(
    redis_container: RedisContainer,
) -> Iterator[dict[str, object]]:
    """Host/port for connecting to the shared container -- needed
    specifically for multiprocessing tests, where a live client object
    can't be pickled across a process boundary the way it can be reused
    within one process (mirrors why mp_workers.py's workers rebuild
    their own limiter in each child process instead of receiving one
    ready-made).

    Flushes the shared database before and after each test, the same
    as `redis_client` below -- FIXED after a real collision was
    observed: two multiprocess tests using different algorithms
    (LeakyBucketMeter and LeakyBucketQueue) both used the key
    "shared-key" against this fixture with no flush of their own, and
    each silently depended on whichever *other* test happened to run
    immediately before or after it to leave the database clean via
    ITS OWN flush (e.g. a neighboring test using `redis_client`,
    which does flush). That's exactly what broke: the meter test
    wrote `{"volume": ..., "last_leak": ...}` under the same
    `limivault:data:shared-key` the queue test then read as
    `{"depth": ..., "last_drain": ...}`, raising `KeyError: 'depth'`
    inside a worker process. Flushing directly in this fixture, not
    relying on neighboring tests' side effects, is the actual fix --
    every test that requests `redis_connection_params` now gets a
    guaranteed-clean database regardless of what ran before or after
    it, the same guarantee `redis_client`-based tests already had.
    """
    host = redis_container.get_container_host_ip()
    port = int(redis_container.get_exposed_port(redis_container.port))
    cleanup_client = redis_sync.Redis(host=host, port=port)
    cleanup_client.flushdb()
    cleanup_client.close()
    try:
        yield {"host": host, "port": port}
    finally:
        cleanup_client = redis_sync.Redis(host=host, port=port)
        cleanup_client.flushdb()
        cleanup_client.close()


@pytest.fixture
def redis_client(redis_container: RedisContainer) -> Iterator[redis_sync.Redis]:
    """Sync client against the shared container, with a clean database
    before and after each test (see module docstring)."""
    client = redis_container.get_client()
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


@pytest.fixture
def async_redis_client(
    redis_connection_params: dict[str, object],
) -> Iterator[redis_async.Redis]:
    """Async client against the shared container. Built directly via
    redis.asyncio.Redis(...) rather than through RedisContainer.get_client()
    (which only ever returns a sync `redis.Redis`), using the same
    host/port as the sync client above.

    NOTE: constructing the client itself is not an async operation, so
    this fixture is a plain (not async) generator fixture -- pytest-asyncio
    doesn't need to be involved here. The flushdb()/close() cleanup calls
    below ARE async and must be awaited from within an async test via
    `await async_redis_client.flushdb()` if a test wants an explicit
    mid-test flush; the automatic pre/post-test flush here uses a
    throwaway sync client instead specifically to keep this fixture
    itself synchronous and simple.
    """
    host = redis_connection_params["host"]
    port = redis_connection_params["port"]
    assert isinstance(host, str) and isinstance(port, int)

    # Use a plain sync client just for the pre/post flushdb -- avoids
    # needing this fixture to be async itself.
    sync_cleanup_client = redis_sync.Redis(host=host, port=port)
    sync_cleanup_client.flushdb()
    sync_cleanup_client.close()

    client = redis_async.Redis(host=host, port=port)
    try:
        yield client
    finally:
        # Cleanup must be synchronous-fixture-compatible; a second
        # throwaway sync client keeps this teardown simple and avoids
        # needing an event loop to already be running at fixture
        # teardown time (fixture teardown happens outside any test's
        # own event loop).
        cleanup_client = redis_sync.Redis(host=host, port=port)
        cleanup_client.flushdb()
        cleanup_client.close()
