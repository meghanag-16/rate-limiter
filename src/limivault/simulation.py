# src/limivault/simulation.py
"""Phase 9 simulation/visualization-only mode: drive a limiter (sync or
async) through a scripted sequence of requests under a deterministic
clock, record every allow()/deny/backend-error decision, and flush the
recorded rows to CSV for external plotting (pandas, matplotlib, a
notebook, a spreadsheet -- anything the person already has).

Per the project's explicit Phase 9 scoping decision, THIS MODULE DOES
NOT PLOT ANYTHING ITSELF. It stops at CSV/log output. This mirrors the
same reasoning already applied to limivault.metrics's decision not to
import prometheus_client (see that module's docstring, point 3): a
visualization/plotting dependency is a downstream, optional concern
that does not belong baked into the library's own dependency footprint.

--------------------------------------------------------------------
DESIGN DECISIONS (confirmed for this pass)
--------------------------------------------------------------------

1. No new async hook type. `SimulationRecorder` implements the plain,
   existing, sync-only `MetricsHook` protocol from limivault.metrics and
   is used identically for both sync and async algorithms (matching
   the existing `metrics=` constructor argument every algorithm
   already accepts). Every `on_allowed`/`on_denied`/`on_backend_error`
   call does nothing but append an in-memory dict to a list -- no
   file I/O, no network I/O, no blocking work of any kind -- so it
   stays trivially fast regardless of which algorithm (sync or async)
   is calling it. This is the resolution to the Phase 8 "revisit
   trigger" for async hooks: the actual problem (a slow hook blocking
   an async algorithm's event loop) is avoided by construction, not by
   adding a second hook interface. Actual CSV writing happens once,
   explicitly, via `flush_to_csv()`, called by the person running the
   simulation after the run completes -- not from inside any hook call.

2. `SimulationClock` is a plain, manually-advanced clock (the same
   shape every test file in this project already hand-rolls locally as
   `FakeClock`: `__call__() -> float` plus `advance(seconds) -> None`).
   It is shipped here, in source, rather than left as a test-only
   pattern, specifically because simulation mode needs a clock to
   drive deterministically and reproducibly -- a simulation run against
   real wall-clock time would not be reproducible between two runs.

3. `run_simulation` / `async_run_simulation` do NOT attach the
   recorder to the limiter themselves. The caller constructs the
   limiter with `metrics=recorder` explicitly (exactly like any other
   `MetricsHook` usage elsewhere in this project), and passes the same
   limiter/clock/steps to the driver function here. This was chosen
   over having the driver reach into the limiter's private `_metrics`
   attribute (a pattern used internally in
   tests/test_metrics_all_algorithms.py for test convenience) because
   reassigning a private attribute from outside the class is a test-
   only convenience, not something a public, documented API should
   rely on.

3a. CLOCK IDENTITY IS NOW ENFORCED, NOT JUST DOCUMENTED (added after
   review). The original version of this module only documented, in
   prose, that the caller must construct the limiter with `clock=`
   pointing at the same `SimulationClock` instance passed to the
   driver. That was a real, silent footgun: if a caller forgot and
   constructed the limiter with its default `time.monotonic` (or any
   other clock), `run_simulation` would still advance the
   `SimulationClock` exactly as scripted and would still record a row
   per decision -- but the limiter's own refill/window/leak math would
   be running against a COMPLETELY DIFFERENT clock than the one the
   recorded `timestamp` column claims. The output CSV would look
   complete and plausible while being quietly wrong -- the `timestamp`
   column would show hours of simulated traffic while the limiter's
   actual admit/deny decisions reflect however many real milliseconds
   the loop took to run. This is exactly the class of "silent, no
   exception" failure this project has never tolerated elsewhere (see
   e.g. the Phase 4 NaN-limit bug, or the Phase 3 cost==0 log-growth
   bug) -- so it is not treated as acceptable here either.

   Fix: both `run_simulation` and `async_run_simulation` now assert
   `limiter._clock is clock` (identity, not equality -- the same
   pattern already used in tests/test_leaky_bucket.py's
   `test_accepts_injected_clock_and_storage`, which asserts
   `limiter._clock is fake_clock`) before running a single step, and
   raise `ValueError` immediately if they don't match. This converts a
   silent wrong-answer bug into a loud, immediate one, matching the
   tradeoff this project has made everywhere else (allow_wait() raising
   UnsatisfiableRequestError instead of returning float("inf"), cost
   type/value rejection, etc.). Reaching into `limiter._clock` (a
   private attribute) is accepted here the same way
   test_leaky_bucket.py already accepts it for an equivalent
   assertion -- there is no public accessor for an algorithm's
   injected clock, and adding one purely to support this check was
   judged out of scope for this pass.

4. Traffic is a plain `list[TrafficStep]`, not a generator or a
   stream-processing abstraction. Each step names a key, a cost, and
   how much simulated time to advance the clock by before that step
   runs. This keeps the driver itself trivial (iterate, advance,
   call allow()) and keeps all the "shape" of a traffic pattern
   (constant-rate, bursty, random) in separate, small, independently
   testable generator functions rather than baked into the driver.

4a. `TrafficStep` NOW VALIDATES ITS OWN FIELDS (added after review).
   Previously `TrafficStep` was a bare dataclass with no validation at
   all -- a hand-built step with a negative `advance_by` or a
   wrong-type `cost` wasn't rejected until it reached
   `clock.advance()` (which does reject a negative value) or
   `limiter.allow()` (which already rejects invalid cost per the
   existing cost contract). Nothing was silently corrupted either way
   -- both downstream consumers already validate -- but the failure
   surfaced far from where the bad step was actually constructed, and
   only at the point in the loop where that specific step happened to
   run, rather than immediately. `TrafficStep.__post_init__` now
   checks `cost` is a non-bool int and `advance_by` is a non-negative,
   finite float at construction time, mirroring the same
   `_is_valid_cost_type`/`_reject_non_finite` checks already duplicated
   across all 12 algorithm files, so a malformed step fails fast and
   at the right call site instead of mid-run.

5. Only `allow()` is driven, never `allow_wait()`. This was an
   explicit, confirmed scope decision -- `allow_wait()` staying outside
   metrics scope (limivault.metrics module docstring, point 9) is an open/
   deferred item for this project, not something simulation mode
   forces open. A future "queue-and-delay" overflow-behavior feature
   (skipped entirely in this project, per the confirmed Phase 9 scope)
   would have been the actual trigger for driving allow_wait() here;
   since that feature was not built, this driver has no reason to call
   allow_wait() either.

6. Randomized traffic patterns require an explicit `seed` argument
   (no default). This matches the project's established, consistent
   preference for determinism in anything time- or randomness-driven
   (every test in this suite drives time via explicit clock injection,
   never real sleeps) -- an unseeded random traffic pattern would make
   two "identical" simulation runs produce different CSVs, which
   defeats the purpose of a reproducible simulation.

7. CSV schema (confirmed): `timestamp, algorithm, key, cost, decision,
   utilization, raw_state`. `decision` is one of "allowed", "denied",
   "backend_error". `utilization` is the Phase 9 normalized field from
   limivault.metrics (empty string for backend_error rows, since there is
   no completed decision to normalize -- see that module's docstring).
   `raw_state` is the algorithm-specific state dict (or, for
   backend_error rows, just the stringified error) serialized as a
   JSON string in a single column, so nothing beyond `utilization` is
   lost, while the common columns stay directly comparable across
   algorithms for straightforward plotting (e.g. utilization vs.
   timestamp, colored/faceted by algorithm).

8. DESCOPED PHASE 9 ITEMS (recorded in Phase 11 so the plan and the
   code agree). Of api-plan.txt's four Phase 9 items, only
   simulation/visualization mode (this module, plus
   redis_lua_simulation.py for the Redis/Lua family) was built. The
   other three were deliberately descoped, not forgotten and not
   deferred:
     - Configurable overflow behavior (reject / queue-and-delay /
       degrade): descoped during Phase 9 itself (see point 5 for why
       this driver only ever calls allow(), never allow_wait()).
     - Hybrid limiter that switches algorithm based on observed load
       pattern: descoped in Phase 11. Not built.
     - Adaptive token bucket with burst allowances that decay over
       time: descoped in Phase 11. Not built.
   None of these will be built unless the plan is explicitly reopened.
--------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from limivault.metrics import AllowedEvent, BackendErrorEvent, DeniedEvent

if TYPE_CHECKING:
    from limivault.base import AsyncRateLimiter, RateLimiter

__all__ = [
    "SimulationClock",
    "SimulationRecorder",
    "TrafficStep",
    "run_simulation",
    "async_run_simulation",
    "constant_rate_traffic",
    "bursty_traffic",
    "random_traffic",
]

_CSV_FIELDNAMES = [
    "timestamp",
    "algorithm",
    "key",
    "cost",
    "decision",
    "utilization",
    "raw_state",
]


class SimulationClock:
    """A plain, manually-advanced clock, matching the
    `Callable[[], float]` contract every algorithm's `clock=`
    constructor argument expects (see base.py's module docstring).

    Shaped identically to the `FakeClock` every test file in this
    project already hand-rolls locally (see e.g.
    tests/test_fixed_window.py) -- shipped here in source rather than
    left as a test-only pattern because simulation mode specifically
    needs a deterministic, reproducible clock to drive, not a
    production algorithm needs.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("SimulationClock cannot move backwards")
        self._now += seconds


def _is_valid_cost_type(cost: int) -> bool:
    """True if `cost` is an actual int -- not bool, not a float. Same
    check every algorithm file already applies to `allow(cost=...)`;
    duplicated here (not imported) since it's a plain, dependency-free
    predicate and importing it from a specific algorithm module would
    create an arbitrary coupling to that one file."""
    return isinstance(cost, int) and not isinstance(cost, bool)


@dataclass
class TrafficStep:
    """One scripted request in a simulation run.

    `advance_by` is how much simulated time (seconds) to advance the
    clock by BEFORE this step's `allow()` call is made -- so the first
    step in a sequence typically has `advance_by=0.0` (no time has
    passed yet), and subsequent steps encode the inter-arrival gap.

    Validates its own fields at construction time (see this module's
    docstring, point 4a) -- `cost` must be a non-bool int, `advance_by`
    must be a finite, non-negative float. This does not change what
    happens for a *valid but denied* request (that's `allow()`'s
    ordinary cost-contract behavior, untouched); it only rejects
    structurally malformed steps immediately, at the point they're
    constructed, instead of wherever in a run they happen to execute.
    """

    key: str
    cost: int = 1
    advance_by: float = 0.0

    def __post_init__(self) -> None:
        if not _is_valid_cost_type(self.cost):
            raise ValueError(
                f"TrafficStep.cost must be an int, got "
                f"{type(self.cost).__name__}: {self.cost!r}"
            )
        if isinstance(self.advance_by, float) and not math.isfinite(self.advance_by):
            raise ValueError(
                f"TrafficStep.advance_by must be finite, got {self.advance_by}"
            )
        if self.advance_by < 0:
            raise ValueError(
                f"TrafficStep.advance_by must be non-negative, got "
                f"{self.advance_by}"
            )


class SimulationRecorder:
    """A `MetricsHook` (see limivault.metrics) that buffers every
    allowed/denied/backend_error event as an in-memory row instead of
    performing any I/O. See this module's docstring, point 1, for why
    this is what lets the same recorder be used safely for both sync
    and async algorithms with no separate async hook type.

    Usage:
        recorder = SimulationRecorder()
        limiter = TokenBucket(capacity=10, refill_rate=1.0,
                               clock=clock, metrics=recorder)
        run_simulation(limiter, clock, constant_rate_traffic("user:1", 50, 0.5))
        recorder.flush_to_csv("out.csv")
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def on_allowed(self, event: AllowedEvent) -> None:
        self._rows.append(self._row_from_decision_event(event, decision="allowed"))

    def on_denied(self, event: DeniedEvent) -> None:
        self._rows.append(self._row_from_decision_event(event, decision="denied"))

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        self._rows.append(
            {
                "timestamp": event.timestamp,
                "algorithm": event.algorithm,
                "key": event.key,
                "cost": event.cost,
                "decision": "backend_error",
                "utilization": "",
                "raw_state": json.dumps({"error": str(event.error)}),
            }
        )

    def _row_from_decision_event(
        self, event: Union[AllowedEvent, DeniedEvent], decision: str
    ) -> dict[str, Any]:
        return {
            "timestamp": event.timestamp,
            "algorithm": event.algorithm,
            "key": event.key,
            "cost": event.cost,
            "decision": decision,
            "utilization": event.utilization,
            "raw_state": json.dumps(dict(event.state)),
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        """A shallow copy of every recorded row so far, in the order
        the hook received them (see limivault.metrics module docstring,
        point 10, on why that order is not guaranteed to match
        real-world decision order under concurrent access -- for
        single-threaded simulation runs, which is the only use case
        this module is built for, that caveat does not apply)."""
        return list(self._rows)

    def clear(self) -> None:
        """Discard all recorded rows, so one recorder instance can be
        reused across multiple simulation runs without carrying rows
        over between them."""
        self._rows.clear()

    def flush_to_csv(self, path: Union[str, Path]) -> None:
        """Write every recorded row to a CSV file at `path`, using the
        schema documented in this module's docstring (point 7).
        Overwrites any existing file at `path`. Does not clear the
        in-memory rows afterward -- call `clear()` explicitly if the
        recorder is going to be reused for another run."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)


def _assert_limiter_uses_this_clock(limiter: Any, clock: SimulationClock) -> None:
    """Raise ValueError immediately if `limiter` was not constructed
    with `clock=` pointing at this exact `SimulationClock` instance.

    See this module's docstring, point 3a, for the full rationale: a
    limiter built against a different clock (e.g. the default
    `time.monotonic`) would silently produce a simulation run whose
    recorded `timestamp` values (driven by `clock`) have no
    relationship to the clock the limiter's own admit/deny math is
    actually running against -- a wrong-but-plausible-looking CSV, not
    an error. Checked via identity (`is`, not `==`) on the private
    `_clock` attribute, the same pattern already used in
    tests/test_leaky_bucket.py's `test_accepts_injected_clock_and_storage`.
    """
    if limiter._clock is not clock:
        raise ValueError(
            "limiter was not constructed with clock=<this SimulationClock "
            "instance>. run_simulation()/async_run_simulation() require the "
            "limiter's own clock to be the exact same SimulationClock object "
            "passed as the `clock` argument here -- otherwise the limiter's "
            "admit/deny decisions and the recorded timestamps would be "
            "driven by two different clocks, silently producing incorrect "
            "output. Construct the limiter with clock=<that same instance> "
            "before calling this."
        )


def run_simulation(
    limiter: "RateLimiter",
    clock: SimulationClock,
    steps: list[TrafficStep],
) -> None:
    """Drive `steps` against a SYNC `limiter`, advancing `clock`
    before each step per that step's `advance_by`.

    `limiter` must already have been constructed with its `clock=`
    argument set to this same `clock` instance, and (if recording is
    wanted) `metrics=` set to a `SimulationRecorder` -- this function
    does not wire the metrics hook up itself (see this module's
    docstring, point 3, for why), but it DOES verify the clock
    argument matches before running anything (point 3a) and raises
    ValueError immediately if it does not.

    Only calls `limiter.allow(key, cost)` -- never `allow_wait()` (see
    this module's docstring, point 5).
    """
    _assert_limiter_uses_this_clock(limiter, clock)
    for step in steps:
        if step.advance_by:
            clock.advance(step.advance_by)
        limiter.allow(step.key, cost=step.cost)


async def async_run_simulation(
    limiter: "AsyncRateLimiter",
    clock: SimulationClock,
    steps: list[TrafficStep],
) -> None:
    """Async mirror of `run_simulation` -- drives `steps` against an
    ASYNC `limiter` the same way, awaiting each `allow()` call, with
    the same clock-identity check up front. See `run_simulation`'s
    docstring for the shared setup requirements and scope (only
    `allow()`, never `allow_wait()`).
    """
    _assert_limiter_uses_this_clock(limiter, clock)
    for step in steps:
        if step.advance_by:
            clock.advance(step.advance_by)
        await limiter.allow(step.key, cost=step.cost)


def constant_rate_traffic(
    key: str, num_requests: int, interval: float, cost: int = 1
) -> list[TrafficStep]:
    """`num_requests` requests for `key`, evenly spaced `interval`
    seconds apart. The very first request has `advance_by=0.0` (no
    time has passed yet at the start of a simulation run)."""
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    if interval < 0:
        raise ValueError(f"interval must be non-negative, got {interval}")
    return [
        TrafficStep(key=key, cost=cost, advance_by=0.0 if i == 0 else interval)
        for i in range(num_requests)
    ]


def bursty_traffic(
    key: str,
    burst_size: int,
    num_bursts: int,
    burst_interval: float,
    cost: int = 1,
) -> list[TrafficStep]:
    """`num_bursts` bursts of `burst_size` back-to-back requests each
    (zero advance within a burst), with `burst_interval` seconds of
    simulated time advanced before the FIRST request of every burst
    after the first."""
    if burst_size < 0:
        raise ValueError(f"burst_size must be non-negative, got {burst_size}")
    if num_bursts < 0:
        raise ValueError(f"num_bursts must be non-negative, got {num_bursts}")
    if burst_interval < 0:
        raise ValueError(f"burst_interval must be non-negative, got {burst_interval}")
    steps: list[TrafficStep] = []
    for burst_index in range(num_bursts):
        for item_index in range(burst_size):
            advance = burst_interval if (item_index == 0 and burst_index > 0) else 0.0
            steps.append(TrafficStep(key=key, cost=cost, advance_by=advance))
    return steps


def random_traffic(
    key: str,
    num_requests: int,
    min_interval: float,
    max_interval: float,
    seed: int,
    cost: int = 1,
) -> list[TrafficStep]:
    """`num_requests` requests for `key` with inter-arrival gaps drawn
    uniformly from `[min_interval, max_interval]`. `seed` is required
    (no default) -- see this module's docstring, point 6, for why an
    unseeded random pattern is not acceptable here: two "identical"
    simulation runs must produce identical output."""
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    if min_interval < 0:
        raise ValueError(f"min_interval must be non-negative, got {min_interval}")
    if max_interval < min_interval:
        raise ValueError(
            f"max_interval ({max_interval}) must be >= min_interval "
            f"({min_interval})"
        )
    rng = random.Random(seed)  # nosec B311
    steps: list[TrafficStep] = []
    for i in range(num_requests):
        advance = 0.0 if i == 0 else rng.uniform(min_interval, max_interval)
        steps.append(TrafficStep(key=key, cost=cost, advance_by=advance))
    return steps