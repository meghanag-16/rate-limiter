# tests/test_simulation.py
"""Tests for limivault.simulation (simulation/visualization-only)
 and the two new limivault.metrics event fields (`utilization`,
`timestamp`) that simulation mode depends on.

the file has the following components :
  - SimulationRecorder captures allowed/denied/backend_error rows with
    the documented CSV schema fields present.
  - flush_to_csv() produces a valid CSV with the exact expected header.
  - random_traffic() is deterministic given the same seed (two runs
    produce byte-identical output).
  - run_simulation() works end-to-end against a sync algorithm.
  - async_run_simulation() works end-to-end against an async algorithm.
  - `utilization` values are correct for a couple of known scenarios
    (not just "present"), matching the per-algorithm formulas
    documented in limivault.metrics's docstring section.
  - A backend_error row is recorded with no `utilization` value.

FakeClock/SimulationClock note: this file uses limivault.simulation's own
SimulationClock directly (rather than a locally-defined FakeClock, as
every other test file in this project does) since SimulationClock IS
the thing being tested here, not an unrelated test utility -- using it
end-to-end is the point.
"""

from __future__ import annotations

import csv
import json

import pytest

from limivault.algorithms.async_fixed_window import AsyncFixedWindow
from limivault.algorithms.fixed_window import FixedWindow
from limivault.algorithms.token_bucket import TokenBucket
from limivault.exceptions import BackendUnavailableError
from limivault.simulation import (
    SimulationClock,
    SimulationRecorder,
    TrafficStep,
    async_run_simulation,
    bursty_traffic,
    constant_rate_traffic,
    random_traffic,
    run_simulation,
)
from limivault.storage import StorageBackend

# ---------------------------------------------------------------------------
# SimulationClock
# ---------------------------------------------------------------------------


def test_simulation_clock_starts_at_given_value() -> None:
    clock = SimulationClock(start=5.0)
    assert clock() == 5.0


def test_simulation_clock_advances() -> None:
    clock = SimulationClock()
    clock.advance(3.0)
    clock.advance(2.0)
    assert clock() == pytest.approx(5.0)


def test_simulation_clock_rejects_negative_advance() -> None:
    clock = SimulationClock()
    with pytest.raises(ValueError):
        clock.advance(-1.0)


# ---------------------------------------------------------------------------
# Traffic pattern generators
# ---------------------------------------------------------------------------


def test_constant_rate_traffic_shape() -> None:
    steps = constant_rate_traffic("k", num_requests=3, interval=2.0)
    assert len(steps) == 3
    assert steps[0].advance_by == 0.0  # first request happens immediately
    assert steps[1].advance_by == pytest.approx(2.0)
    assert steps[2].advance_by == pytest.approx(2.0)
    assert all(s.key == "k" and s.cost == 1 for s in steps)


def test_bursty_traffic_shape() -> None:
    steps = bursty_traffic("k", burst_size=3, num_bursts=2, burst_interval=10.0)
    assert len(steps) == 6
    # First burst: no advances at all.
    assert steps[0].advance_by == 0.0
    assert steps[1].advance_by == 0.0
    assert steps[2].advance_by == 0.0
    # Second burst: only its first item advances the clock.
    assert steps[3].advance_by == pytest.approx(10.0)
    assert steps[4].advance_by == 0.0
    assert steps[5].advance_by == 0.0


def test_random_traffic_is_deterministic_given_same_seed() -> None:
    steps_a = random_traffic(
        "k", num_requests=20, min_interval=0.1, max_interval=5.0, seed=42
    )
    steps_b = random_traffic(
        "k", num_requests=20, min_interval=0.1, max_interval=5.0, seed=42
    )
    assert [s.advance_by for s in steps_a] == [s.advance_by for s in steps_b]


def test_random_traffic_different_seeds_differ() -> None:
    steps_a = random_traffic(
        "k", num_requests=20, min_interval=0.1, max_interval=5.0, seed=1
    )
    steps_b = random_traffic(
        "k", num_requests=20, min_interval=0.1, max_interval=5.0, seed=2
    )
    assert [s.advance_by for s in steps_a] != [s.advance_by for s in steps_b]


def test_random_traffic_intervals_within_bounds() -> None:
    steps = random_traffic(
        "k", num_requests=50, min_interval=1.0, max_interval=3.0, seed=7
    )
    for s in steps[1:]:  # first step's advance_by is always 0.0
        assert 1.0 <= s.advance_by <= 3.0


def test_constant_rate_traffic_rejects_negative_num_requests() -> None:
    with pytest.raises(ValueError):
        constant_rate_traffic("k", num_requests=-1, interval=1.0)


def test_random_traffic_rejects_max_below_min() -> None:
    with pytest.raises(ValueError):
        random_traffic("k", num_requests=5, min_interval=5.0, max_interval=1.0, seed=1)


# ---------------------------------------------------------------------------
# SimulationRecorder: row capture
# ---------------------------------------------------------------------------


def test_recorder_captures_allowed_row_with_expected_fields() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=recorder)

    limiter.allow("user:1", cost=2)

    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert row["algorithm"] == "FixedWindow"
    assert row["key"] == "user:1"
    assert row["cost"] == 2
    assert row["decision"] == "allowed"
    assert row["timestamp"] == 0.0
    assert row["utilization"] == pytest.approx(2 / 3)
    raw_state = json.loads(row["raw_state"])
    assert raw_state["count"] == 2
    assert raw_state["limit"] == 3


def test_recorder_captures_denied_row() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=1, period=60.0, clock=clock, metrics=recorder)

    limiter.allow("k")  # consumes the one slot, decision #1
    limiter.allow("k")  # denied, decision #2

    assert len(recorder.rows) == 2
    assert recorder.rows[0]["decision"] == "allowed"
    assert recorder.rows[1]["decision"] == "denied"
    assert recorder.rows[1]["utilization"] == pytest.approx(1.0)


def test_recorder_timestamp_reflects_clock_advances() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = TokenBucket(capacity=5, refill_rate=1.0, clock=clock, metrics=recorder)

    limiter.allow("k")
    clock.advance(3.5)
    limiter.allow("k")

    assert recorder.rows[0]["timestamp"] == pytest.approx(0.0)
    assert recorder.rows[1]["timestamp"] == pytest.approx(3.5)


def test_recorder_clear_empties_rows() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=3, period=60.0, clock=clock, metrics=recorder)

    limiter.allow("k")
    assert len(recorder.rows) == 1

    recorder.clear()
    assert recorder.rows == []


class _FailingStorage(StorageBackend):
    """Minimal StorageBackend whose get() always raises
    BackendUnavailableError, used to exercise the recorder's
    on_backend_error path without needing a real Redis fixture."""

    def get(self, key: str) -> dict[str, object] | None:
        raise BackendUnavailableError("simulated failure")

    def set(self, key: str, state: dict[str, object]) -> None:
        raise BackendUnavailableError("simulated failure")

    def lock(self, key: str):  # type: ignore[no-untyped-def]
        from contextlib import contextmanager

        @contextmanager
        def _cm():  # type: ignore[no-untyped-def]
            yield True

        return _cm()


def test_recorder_captures_backend_error_row_with_no_utilization() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(
        limit=5, period=60.0, storage=_FailingStorage(), clock=clock, metrics=recorder
    )

    with pytest.raises(BackendUnavailableError):
        limiter.allow("k")

    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert row["decision"] == "backend_error"
    assert row["utilization"] == ""
    raw_state = json.loads(row["raw_state"])
    assert "error" in raw_state


# ---------------------------------------------------------------------------
# flush_to_csv
# ---------------------------------------------------------------------------


def test_flush_to_csv_produces_expected_header_and_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=2, period=60.0, clock=clock, metrics=recorder)

    limiter.allow("k")
    limiter.allow("k")
    limiter.allow("k")  # denied

    out_path = tmp_path / "sim.csv"
    recorder.flush_to_csv(out_path)

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "timestamp",
            "algorithm",
            "key",
            "cost",
            "decision",
            "utilization",
            "raw_state",
        ]
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0]["decision"] == "allowed"
    assert rows[2]["decision"] == "denied"


def test_flush_to_csv_overwrites_existing_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock, metrics=recorder)
    out_path = tmp_path / "sim.csv"

    limiter.allow("k")
    recorder.flush_to_csv(out_path)
    with open(out_path, encoding="utf-8") as f:
        first_write = f.read()

    limiter.allow("k")
    recorder.flush_to_csv(out_path)  # now has 2 rows, must fully overwrite
    with open(out_path, encoding="utf-8") as f:
        second_write = f.read()

    assert first_write != second_write
    assert second_write.count("\n") > first_write.count("\n")


# ---------------------------------------------------------------------------
# run_simulation (sync end-to-end)
# ---------------------------------------------------------------------------


def test_run_simulation_drives_traffic_against_sync_limiter() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = TokenBucket(capacity=3, refill_rate=0.0, clock=clock, metrics=recorder)

    steps = constant_rate_traffic("user:1", num_requests=5, interval=1.0)
    run_simulation(limiter, clock, steps)

    assert len(recorder.rows) == 5
    allowed_count = sum(1 for r in recorder.rows if r["decision"] == "allowed")
    denied_count = sum(1 for r in recorder.rows if r["decision"] == "denied")
    assert allowed_count == 3  # capacity=3, refill_rate=0 -- exactly 3 admitted
    assert denied_count == 2
    assert clock() == pytest.approx(4.0)  # 4 advances of 1.0s each after step 1


def test_run_simulation_respects_explicit_traffic_step_list() -> None:
    """run_simulation also accepts a hand-built list[TrafficStep], not
    just generator output -- confirms the driver itself doesn't assume
    anything about how the steps were produced."""
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=10, period=60.0, clock=clock, metrics=recorder)

    steps = [
        TrafficStep(key="a", cost=1, advance_by=0.0),
        TrafficStep(key="b", cost=2, advance_by=5.0),
        TrafficStep(key="a", cost=3, advance_by=0.0),
    ]
    run_simulation(limiter, clock, steps)

    assert len(recorder.rows) == 3
    assert recorder.rows[0]["key"] == "a"
    assert recorder.rows[1]["key"] == "b"
    assert recorder.rows[1]["timestamp"] == pytest.approx(5.0)
    assert recorder.rows[2]["key"] == "a"


# ---------------------------------------------------------------------------
# async_run_simulation (async end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_run_simulation_drives_traffic_against_async_limiter() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = AsyncFixedWindow(limit=2, period=60.0, clock=clock, metrics=recorder)

    steps = constant_rate_traffic("user:1", num_requests=4, interval=0.5)
    await async_run_simulation(limiter, clock, steps)

    assert len(recorder.rows) == 4
    allowed_count = sum(1 for r in recorder.rows if r["decision"] == "allowed")
    assert allowed_count == 2  # limit=2 within one 60s window
    assert clock() == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_async_run_simulation_utilization_matches_fixed_window_formula() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = AsyncFixedWindow(limit=4, period=60.0, clock=clock, metrics=recorder)

    await async_run_simulation(
        limiter, clock, [TrafficStep(key="k", cost=3, advance_by=0.0)]
    )

    assert recorder.rows[0]["utilization"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# Cross-algorithm utilization formula spot checks
# ---------------------------------------------------------------------------


def test_utilization_matches_token_bucket_formula() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = TokenBucket(capacity=10, refill_rate=1.0, clock=clock, metrics=recorder)

    limiter.allow("k", cost=4)  # 6 tokens left -> utilization = (10-6)/10 = 0.4

    assert recorder.rows[0]["utilization"] == pytest.approx(0.4)


def test_utilization_is_comparable_in_shape_across_two_different_algorithms() -> None:
    """Not asserting equal values (the algorithms differ) -- just that
    both produce a plain float utilization in the same [0, 1] shape,
    which is the actual point of the field: a common axis to
    plot regardless of which algorithm produced the row."""
    clock = SimulationClock()
    recorder_a = SimulationRecorder()
    recorder_b = SimulationRecorder()

    fw = FixedWindow(limit=5, period=60.0, clock=clock, metrics=recorder_a)
    tb = TokenBucket(capacity=5, refill_rate=0.0, clock=clock, metrics=recorder_b)

    fw.allow("k", cost=2)
    tb.allow("k", cost=2)

    assert isinstance(recorder_a.rows[0]["utilization"], float)
    assert isinstance(recorder_b.rows[0]["utilization"], float)
    assert 0.0 <= recorder_a.rows[0]["utilization"] <= 1.0
    assert 0.0 <= recorder_b.rows[0]["utilization"] <= 1.0
