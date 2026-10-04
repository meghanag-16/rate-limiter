# tests/test_redis_lua_simulation.py
"""Tests for rlimit.rlimit_lua_simulation — the real-time simulation
driver for Redis-native Lua limiters.

Verifies that:
  - Decisions are recorded in the LuaSimulationRecorder
  - advance_by values are respected as real sleeps
  - The CSV output matches the expected schema
  - Async driver works identically
  - The driver does NOT require a _clock attribute on the limiter

REQUIRES DOCKER -- see redis_test_helpers.py's module docstring.
"""

from __future__ import annotations

import time
import uuid

import pytest
import redis as redis_sync
import redis.asyncio as redis_async

from rlimit.algorithms.async_redis_lua_token_bucket import AsyncRedisGcraTokenBucket
from rlimit.algorithms.redis_lua_fixed_window import RedisLuaFixedWindow
from rlimit.algorithms.redis_lua_token_bucket import RedisGcraTokenBucket
from rlimit.rlimit_lua_simulation import (
    LuaSimulationRecorder,
    TrafficStep,
    async_run_lua_simulation,
    bursty_traffic,
    constant_rate_traffic,
    run_lua_simulation,
)
from tests.redis_test_helpers import (
    async_redis_client,
    redis_client,
    redis_connection_params,
    redis_container,
)

__all__ = [
    "redis_container",
    "redis_client",
    "async_redis_client",
    "redis_connection_params",
]


def _fresh_key(prefix: str = "sim") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------
# LuaSimulationRecorder
# ------------------------------------------------------------------


class TestLuaSimulationRecorder:
    def test_records_allowed_event(self, redis_client: redis_sync.Redis) -> None:
        recorder = LuaSimulationRecorder()
        limiter = RedisLuaFixedWindow(
            redis_client, limit=5, period=60.0, key_prefix="simrec:", metrics=recorder
        )
        limiter.allow(_fresh_key("sr"), cost=1)
        assert len(recorder.rows) == 1
        row = recorder.rows[0]
        assert row["decision"] == "allowed"
        assert row["algorithm"] == "RedisLuaFixedWindow"
        assert row["utilization"] == pytest.approx(0.2, abs=0.01)

    def test_records_denied_event(self, redis_client: redis_sync.Redis) -> None:
        recorder = LuaSimulationRecorder()
        limiter = RedisLuaFixedWindow(
            redis_client, limit=1, period=60.0, key_prefix="simd:", metrics=recorder
        )
        key = _fresh_key("sd")
        limiter.allow(key)
        limiter.allow(key)
        assert len(recorder.rows) == 2
        assert recorder.rows[1]["decision"] == "denied"

    def test_flush_to_csv_produces_valid_csv(
        self, redis_client: redis_sync.Redis, tmp_path: object
    ) -> None:
        import csv
        from pathlib import Path

        recorder = LuaSimulationRecorder()
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=5,
            refill_rate=0.0,
            key_prefix="simcsv:",
            metrics=recorder,
        )
        limiter.allow(_fresh_key("sc"))
        limiter.allow(_fresh_key("sc"))

        out = Path(str(tmp_path)) / "trace.csv"
        recorder.flush_to_csv(out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert {
            "timestamp",
            "algorithm",
            "key",
            "cost",
            "decision",
            "utilization",
            "raw_state",
        } == set(reader.fieldnames or [])

    def test_clear_resets_rows(self) -> None:
        recorder = LuaSimulationRecorder()
        recorder._rows.append({"dummy": True})
        recorder.clear()
        assert recorder.rows == []


# ------------------------------------------------------------------
# Traffic helpers
# ------------------------------------------------------------------


class TestTrafficHelpers:
    def test_constant_rate_traffic_length_and_spacing(self) -> None:
        steps = constant_rate_traffic("k", num_requests=5, interval=1.0)
        assert len(steps) == 5
        assert steps[0].advance_by == 0.0
        for s in steps[1:]:
            assert s.advance_by == 1.0

    def test_bursty_traffic_within_burst_no_gap(self) -> None:
        steps = bursty_traffic("k", burst_size=3, num_bursts=2, burst_interval=5.0)
        assert len(steps) == 6
        assert steps[0].advance_by == 0.0
        assert steps[1].advance_by == 0.0
        assert steps[2].advance_by == 0.0
        assert steps[3].advance_by == 5.0
        assert steps[4].advance_by == 0.0
        assert steps[5].advance_by == 0.0

    def test_traffic_step_validates_cost(self) -> None:
        with pytest.raises(ValueError, match="TrafficStep.cost must be an int"):
            TrafficStep(key="k", cost=1.5)  # type: ignore[arg-type]

    def test_traffic_step_validates_advance_by_negative(self) -> None:
        with pytest.raises(ValueError, match="advance_by must be non-negative"):
            TrafficStep(key="k", advance_by=-1.0)


# ------------------------------------------------------------------
# run_lua_simulation (sync)
# ------------------------------------------------------------------


class TestRunLuaSimulation:
    def test_records_all_decisions(
        self, redis_client: redis_sync.Redis
    ) -> None:
        recorder = LuaSimulationRecorder()
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=3,
            refill_rate=0.0,
            key_prefix="simrun:",
            metrics=recorder,
        )
        steps = constant_rate_traffic(_fresh_key("sr"), num_requests=5, interval=0.0)
        run_lua_simulation(limiter, steps, recorder)
        assert len(recorder.rows) == 5
        assert recorder.rows[0]["decision"] == "allowed"
        assert recorder.rows[2]["decision"] == "allowed"
        assert recorder.rows[3]["decision"] == "denied"
        assert recorder.rows[4]["decision"] == "denied"

    def test_advance_by_sleeps_real_time(
        self, redis_client: redis_sync.Redis
    ) -> None:
        recorder = LuaSimulationRecorder()
        limiter = RedisGcraTokenBucket(
            redis_client, capacity=5, refill_rate=0.0, key_prefix="simsleep:",
        )
        key = _fresh_key("ss")
        steps = [
            TrafficStep(key=key, cost=1, advance_by=0.0),
            TrafficStep(key=key, cost=1, advance_by=0.12),
            TrafficStep(key=key, cost=1, advance_by=0.0),
        ]
        t0 = time.monotonic()
        run_lua_simulation(limiter, steps, recorder)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.1

    def test_utilization_populated_in_rows(
        self, redis_client: redis_sync.Redis
    ) -> None:
        recorder = LuaSimulationRecorder()
        limiter = RedisGcraTokenBucket(
            redis_client,
            capacity=10,
            refill_rate=0.0,
            key_prefix="simutil:",
            metrics=recorder,
        )
        steps = constant_rate_traffic(_fresh_key("su"), num_requests=3, interval=0.0)
        run_lua_simulation(limiter, steps, recorder)
        for row in recorder.rows:
            assert isinstance(row["utilization"], float)
            assert 0.0 <= row["utilization"] <= 1.0

    def test_no_clock_attribute_required(
        self, redis_client: redis_sync.Redis
    ) -> None:
        """Verifies the driver does not touch limiter._clock."""
        limiter = RedisGcraTokenBucket(
            redis_client, capacity=5, refill_rate=0.0, key_prefix="simnoclk:"
        )
        assert not hasattr(limiter, "_clock")
        steps = constant_rate_traffic(_fresh_key("snk"), num_requests=2, interval=0.0)
        run_lua_simulation(limiter, steps)


# ------------------------------------------------------------------
# async_run_lua_simulation
# ------------------------------------------------------------------


class TestAsyncRunLuaSimulation:
    @pytest.mark.asyncio
    async def test_async_records_all_decisions(
        self, async_redis_client: redis_async.Redis
    ) -> None:
        recorder = LuaSimulationRecorder()
        limiter = AsyncRedisGcraTokenBucket(
            async_redis_client,
            capacity=3,
            refill_rate=0.0,
            key_prefix="asimrun:",
            metrics=recorder,
        )
        steps = constant_rate_traffic(_fresh_key("asr"), num_requests=5, interval=0.0)
        await async_run_lua_simulation(limiter, steps, recorder)
        assert len(recorder.rows) == 5
        decisions = [r["decision"] for r in recorder.rows]
        assert decisions.count("allowed") == 3
        assert decisions.count("denied") == 2

    @pytest.mark.asyncio
    async def test_async_advance_by_sleeps(
        self, async_redis_client: redis_async.Redis
    ) -> None:
        recorder = LuaSimulationRecorder()
        limiter = AsyncRedisGcraTokenBucket(
            async_redis_client, capacity=5, refill_rate=0.0, key_prefix="asimsleep:",
        )
        key = _fresh_key("ass")
        steps = [
            TrafficStep(key=key, cost=1, advance_by=0.0),
            TrafficStep(key=key, cost=1, advance_by=0.1),
        ]
        t0 = time.monotonic()
        await async_run_lua_simulation(limiter, steps, recorder)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.1
