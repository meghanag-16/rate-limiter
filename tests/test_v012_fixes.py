"""Regression tests for the 0.1.2 fixes.

1. BackendBusyError: an exhausted redis-py connection pool is reported as
   its own error type (and as `kind="busy"` on BackendErrorEvent), so
   apps never mistake it for a Redis outage and fail open on it.
2. (docs only) allow()'s contract now says Redis limiters raise.
3. MultiProcessStateWarning when in-memory storage runs in a worker.
4. Optional max_keys / state_idle_seconds bounds for in-memory storage.
5. NonDecreasingClock: a clock that jumps backwards can no longer make
   an algorithm hand out a fresh quota.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest
import redis
import redis.asyncio
import redis.exceptions

import limivault.storage as storage_mod
from limivault import (
    AsyncFixedWindow,
    AsyncRedisLuaFixedWindow,
    AsyncTokenBucket,
    BackendBusyError,
    BackendUnavailableError,
    FixedWindow,
    LeakyBucketMeter,
    LeakyBucketQueue,
    MultiProcessStateWarning,
    RedisLuaFixedWindow,
    SlidingWindowCounter,
    SlidingWindowLog,
    TokenBucket,
)
from limivault._clock import NonDecreasingClock, non_decreasing
from limivault.exceptions import backend_error, is_pool_exhausted
from limivault.metrics import AllowedEvent, BackendErrorEvent, DeniedEvent
from limivault.storage import AsyncInMemoryStorage, InMemoryStorage


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class RecordingHook:
    def __init__(self) -> None:
        self.errors: list[BackendErrorEvent] = []

    def on_allowed(self, event: AllowedEvent) -> None: ...

    def on_denied(self, event: DeniedEvent) -> None: ...

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        self.errors.append(event)


# ---------------------------------------------------------------- fix 1


class TestBackendBusyError:
    def test_pool_exhaustion_errors_are_recognised(self) -> None:
        assert is_pool_exhausted(redis.exceptions.MaxConnectionsError("Too many"))
        assert is_pool_exhausted(
            redis.exceptions.ConnectionError("No connection available.")
        )
        assert not is_pool_exhausted(redis.exceptions.ConnectionError("Error 111"))
        assert not is_pool_exhausted(redis.exceptions.TimeoutError("timeout"))
        assert not is_pool_exhausted(OSError("boom"))

    def test_backend_error_picks_the_subclass(self) -> None:
        busy = backend_error(redis.exceptions.MaxConnectionsError("x"), "msg")
        down = backend_error(redis.exceptions.ConnectionError("refused"), "msg")
        assert type(busy) is BackendBusyError
        assert isinstance(busy, BackendUnavailableError)  # old handlers still work
        assert type(down) is BackendUnavailableError

    def test_sync_redis_limiter_raises_busy_and_tags_the_event(self) -> None:
        hook = RecordingHook()
        lim = RedisLuaFixedWindow(
            redis.Redis(host="127.0.0.1", port=1), 5, 10, metrics=hook
        )

        def exhausted(**_: Any) -> Any:
            raise redis.exceptions.MaxConnectionsError("Too many connections")

        lim._script = exhausted  # type: ignore[assignment]
        with pytest.raises(BackendBusyError):
            lim.allow("k")
        assert [e.kind for e in hook.errors] == ["busy"]

    def test_async_redis_limiter_distinguishes_busy_from_down(self) -> None:
        async def run() -> list[str]:
            hook = RecordingHook()
            lim = AsyncRedisLuaFixedWindow(
                redis.asyncio.Redis(host="127.0.0.1", port=1), 5, 10, metrics=hook
            )

            async def blocking_pool_timeout(**_: Any) -> Any:
                raise redis.exceptions.ConnectionError("No connection available.")

            async def refused(**_: Any) -> Any:
                raise redis.exceptions.ConnectionError("Connection refused")

            lim._script = blocking_pool_timeout  # type: ignore[assignment]
            with pytest.raises(BackendBusyError):
                await lim.allow("k")
            lim._script = refused  # type: ignore[assignment]
            with pytest.raises(BackendUnavailableError) as exc:
                await lim.allow("k")
            assert type(exc.value) is BackendUnavailableError
            return [e.kind for e in hook.errors]

        assert asyncio.run(run()) == ["busy", "unavailable"]

    def test_hand_built_events_default_to_unavailable(self) -> None:
        event = BackendErrorEvent("x", "k", 1, OSError())
        assert event.kind == "unavailable"


# ---------------------------------------------------------------- fix 3


def _make_storage_in_child() -> list[str]:
    storage_mod._multiprocess_warning_emitted = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InMemoryStorage()
    return [type(w.message).__name__ for w in caught]


class TestMultiProcessWarning:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(storage_mod, "_multiprocess_warning_emitted", False)
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.delenv("LIMIVAULT_SILENCE_MULTIPROCESS_WARNING", raising=False)

    def test_no_warning_in_a_plain_single_process(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InMemoryStorage()
            AsyncInMemoryStorage()

    def test_warns_once_when_web_concurrency_is_above_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        with pytest.warns(MultiProcessStateWarning, match="WEB_CONCURRENCY=4"):
            FixedWindow(5, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AsyncFixedWindow(5, 10)  # once per process, not per limiter

    def test_env_var_silences_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        monkeypatch.setenv("LIMIVAULT_SILENCE_MULTIPROCESS_WARNING", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InMemoryStorage()

    def test_multiprocess_safe_storage_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InMemoryStorage(multiprocess_safe=True, mp_lock_pool_size=2)

    def test_warns_inside_a_multiprocessing_worker(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            names = pool.submit(_make_storage_in_child).result(timeout=60)
        assert names == ["MultiProcessStateWarning"]


# ---------------------------------------------------------------- fix 4


class TestBoundedStorage:
    def test_max_keys_evicts_least_recently_used(self) -> None:
        s = InMemoryStorage(max_keys=3)
        for k in "abc":
            s.set(k, {"v": k})
        s.get("a")  # touch: "b" is now the oldest
        s.set("d", {"v": "d"})
        assert len(s) == 3 and s.evicted_count == 1
        assert s.get("b") is None and s.get("a") == {"v": "a"}

    def test_key_flood_stays_bounded_through_an_algorithm(self) -> None:
        storage = InMemoryStorage(max_keys=100)
        tb = TokenBucket(5, 1.0, storage=storage)
        for i in range(5000):
            tb.allow(f"ip:{i}")
        assert len(storage) == 100 and storage.evicted_count == 4900

    def test_async_key_flood_stays_bounded(self) -> None:
        async def run() -> tuple[int, int]:
            storage = AsyncInMemoryStorage(max_keys=50)
            lim = AsyncTokenBucket(5, 1.0, storage=storage)
            for i in range(1000):
                await lim.allow(f"ip:{i}")
            return len(storage), storage.evicted_count

        assert asyncio.run(run()) == (50, 950)

    def test_eviction_resets_only_the_evicted_key(self) -> None:
        clock = FakeClock()
        storage = InMemoryStorage(max_keys=2)
        fw = FixedWindow(2, 60, storage=storage, clock=clock)
        assert [fw.allow("hot"), fw.allow("hot"), fw.allow("hot")] == [
            True,
            True,
            False,
        ]
        fw.allow("x")
        fw.allow("hot")  # keeps "hot" recent
        fw.allow("y")  # evicts "x", the least recently used
        assert fw.allow("hot") is False  # hot's quota was NOT reset

    def test_state_idle_seconds_drops_idle_state_on_the_sweep(self) -> None:
        clock = FakeClock(0.0)
        s = InMemoryStorage(
            clock=clock, lock_sweep_interval=1, lock_idle_seconds=1000,
            state_idle_seconds=10,
        )
        with s.lock("old"):
            s.set("old", {"n": 1})
        clock.t = 5.0
        with s.lock("new"):
            s.set("new", {"n": 1})
        clock.t = 12.0  # "old" idle 12 s, "new" idle 7 s
        with s.lock("trigger"):
            pass
        assert s.get("old") is None and s.get("new") == {"n": 1}
        assert s.evicted_count == 1

    def test_async_state_idle_sweep(self) -> None:
        async def run() -> tuple[Any, Any]:
            clock = FakeClock(0.0)
            s = AsyncInMemoryStorage(
                clock=clock, lock_sweep_interval=1, state_idle_seconds=10
            )
            async with s.lock("old"):
                await s.set("old", {"n": 1})
            clock.t = 20.0
            async with s.lock("trigger"):
                pass
            return await s.get("old"), len(s)

        assert asyncio.run(run()) == (None, 0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_keys": 0},
            {"max_keys": -1},
            {"max_keys": 1.5},
            {"max_keys": True},
            {"state_idle_seconds": 0},
            {"state_idle_seconds": math.inf},
            {"state_idle_seconds": math.nan},
        ],
    )
    def test_invalid_bounds_are_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            InMemoryStorage(**kwargs)
        with pytest.raises(ValueError):
            AsyncInMemoryStorage(**kwargs)

    def test_bounds_not_allowed_with_multiprocess_safe(self) -> None:
        with pytest.raises(ValueError):
            InMemoryStorage(multiprocess_safe=True, max_keys=10)

    def test_default_storage_keeps_the_original_plain_dict_path(self) -> None:
        assert type(InMemoryStorage()._data) is dict
        assert type(AsyncInMemoryStorage()._data) is dict


# ---------------------------------------------------------------- fix 5


def _six(clock: FakeClock) -> dict[str, Any]:
    return {
        "fixed_window": FixedWindow(5, 10, clock=clock),
        "sliding_window_log": SlidingWindowLog(5, 10, clock=clock),
        "sliding_window_counter": SlidingWindowCounter(5, 10, clock=clock),
        "token_bucket": TokenBucket(5, 1.0, clock=clock),
        "leaky_bucket_meter": LeakyBucketMeter(5, 1.0, clock=clock),
        "leaky_bucket_queue": LeakyBucketQueue(5, 1.0, clock=clock),
    }


class TestNonDecreasingClock:
    @pytest.mark.parametrize("name", list(_six(FakeClock())))
    def test_backwards_jump_never_grants_a_fresh_quota(self, name: str) -> None:
        clock = FakeClock(100.0)
        lim = _six(clock)[name]
        first = sum(lim.allow("u") for _ in range(6))
        clock.t = 50.0  # jump back 50 s
        after_jump = sum(lim.allow("u") for _ in range(6))
        clock.t = 52.0  # still behind the last reading (100)
        still_behind = sum(lim.allow("u") for _ in range(6))
        assert (first, after_jump, still_behind) == (5, 0, 0)

    def test_time_resumes_once_the_real_clock_catches_up(self) -> None:
        clock = FakeClock(100.0)
        tb = TokenBucket(5, 1.0, clock=clock)
        assert sum(tb.allow("u") for _ in range(5)) == 5
        clock.t = 50.0
        assert tb.allow("u") is False
        clock.t = 103.0  # 3 s past the last good reading
        assert sum(tb.allow("u") for _ in range(10)) == 3

    def test_wrapper_behaviour(self) -> None:
        clock = FakeClock(10.0)
        c = non_decreasing(clock)
        assert non_decreasing(c) is c  # idempotent
        assert c() == 10.0
        clock.t = 4.0
        assert c() == 10.0 and c.backward_jumps == 1
        clock.t = math.nan
        assert c() == 10.0
        clock.t = 11.0
        assert c() == 11.0

    def test_wrapper_pickles(self) -> None:
        c = NonDecreasingClock(FakeClock(10.0))
        c()
        clone = pickle.loads(pickle.dumps(c))
        clone.wrapped.t = 1.0
        assert clone() == 10.0  # remembers the last reading
