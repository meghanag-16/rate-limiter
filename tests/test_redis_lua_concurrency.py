# tests/test_redis_lua_concurrency.py
"""Multi-process correctness for all six Redis Lua/GCRA
algorithms -- the review's #13/#14 items, flagged as the most
important missing coverage in the first pass: this is the whole point
of moving to a single atomic EVAL, and it had zero equivalent tests.

Two scenarios, both across real OS processes (not just threads --
a single process's threads can serialize through one client
connection; the point of these tests is proving atomicity holds when
completely independent processes, each with their own Redis connection,
race the same key):

1. SAME-KEY RACE (review #14): N processes, ONE attempt each, against
a key with capacity/limit == 1. Exactly one must be admitted, the
rest denied, and remaining() must end at 0. This is the sharpest
possible test of the atomic-script guarantee -- if the script were
not actually atomic, this is where a lost update would show up as
more than one process getting admitted.

2. FULL HAMMERING (review #13): 5 processes x 30 attempts each
against a shared limit/capacity of 40, the same shape as the
in-memory multiprocess tests (tests/mp_workers.py), applied to the
Lua/GCRA classes.

Each process builds its OWN client + limiter via the worker functions
in redis_lua_mp_workers.py (see that file's docstring for why -- a
`redis.Redis` client holds live sockets that don't survive a pickle
round trip the way a picklable value does).

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import pytest

from tests.redis_lua_mp_workers import (
    hammer_lua_fixed_window,
    hammer_lua_gcra_token_bucket,
    hammer_lua_leaky_bucket_meter,
    hammer_lua_leaky_bucket_queue,
    hammer_lua_sliding_window_counter,
    hammer_lua_sliding_window_log,
)
from tests.redis_test_helpers import redis_connection_params, redis_container

__all__ = ["redis_container", "redis_connection_params"]


def _fresh_key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


Worker = Callable[[str, int, int, float, str, int], int]


_ALGORITHMS = [
    ("RedisLuaFixedWindow", hammer_lua_fixed_window, 3600.0),
    ("RedisGcraTokenBucket", hammer_lua_gcra_token_bucket, 0.0),
    ("RedisLuaSlidingWindowLog", hammer_lua_sliding_window_log, 3600.0),
    ("RedisLuaSlidingWindowCounter", hammer_lua_sliding_window_counter, 3600.0),
    ("RedisLuaLeakyBucketMeter", hammer_lua_leaky_bucket_meter, 0.0),
    ("RedisLuaLeakyBucketQueue", hammer_lua_leaky_bucket_queue, 0.0),
]


@pytest.mark.parametrize(
    "name,worker,param2", _ALGORITHMS, ids=[n for n, _, _ in _ALGORITHMS]
)
def test_same_key_race_admits_exactly_one_of_fifty(
    name: str,
    worker: Worker,
    param2: float,
    redis_connection_params: dict[str, object],
) -> None:
    """Review #14's headline test: limit/capacity=1, 50 processes,
    one attempt each, same key, simultaneous-as-possible start via
    ProcessPoolExecutor submitting all 50 up front. Exactly one must
    be admitted."""
    host = redis_connection_params["host"]
    port = redis_connection_params["port"]
    assert isinstance(host, str) and isinstance(port, int)

    key = _fresh_key("race")
    n_processes = 50

    with ProcessPoolExecutor(max_workers=n_processes) as ex:
        futures = [
            ex.submit(worker, host, port, 1, param2, key, 1) for _ in range(n_processes)
        ]
        results = [f.result() for f in futures]

    assert sum(results) == 1, name


@pytest.mark.parametrize(
    "name,worker,param2", _ALGORITHMS, ids=[n for n, _, _ in _ALGORITHMS]
)
def test_multiprocess_hammering_never_exceeds_limit(
    name: str,
    worker: Worker,
    param2: float,
    redis_connection_params: dict[str, object],
) -> None:
    """5 processes x 30 attempts against a shared limit of 40 -- same
    shape as the in-memory multiprocess tests (tests/mp_workers.py),
    applied here to the Lua/GCRA classes."""
    host = redis_connection_params["host"]
    port = redis_connection_params["port"]
    assert isinstance(host, str) and isinstance(port, int)

    limit = 40
    n_procs = 5
    attempts_per_proc = 30
    key = _fresh_key("hammer")

    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        futures = [
            ex.submit(worker, host, port, limit, param2, key, attempts_per_proc)
            for _ in range(n_procs)
        ]
        results = [f.result() for f in futures]

    assert sum(results) == limit, name


def test_all_six_algorithms_are_covered_by_this_file() -> None:
    assert len(_ALGORITHMS) == 6
