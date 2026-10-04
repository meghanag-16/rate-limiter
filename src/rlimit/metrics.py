# src/rlimit/metrics.py
"""Metrics hook interface extended,
simulation/visualization-only with two new event fields.

- `timestamp: float` on AllowedEvent, DeniedEvent, and BackendErrorEvent
  -- the algorithm's own injected clock value read at decision time
  (already computed inside the per-key lock in every algorithm; this
  just threads it through to the event). Deliberately the algorithm's
  own `clock`, not real wall-clock time, so a simulation run driven by
  a deterministic FakeClock-style clock produces deterministic,
  reproducible timestamps in its output -- consistent with how every
  test in this project already drives time via clock injection rather
  than real sleeps.

- `utilization: float` on AllowedEvent and DeniedEvent -- a single
  normalized field (roughly 0.0-1.0 under normal operation, though not
  strictly clamped) representing "how full is this key right now" in a
  way that's comparable ACROSS algorithms, not just within one. Added
  specifically because simulation/visualization mode plots decisions
  across potentially different algorithms and needs one common axis to
  compare them on; the existing `state` mapping's fields are still
  algorithm-specific (weighted, tokens, volume, depth, count,
  entries_count) and stay exactly as they were -- `utilization` is
  purely additive, not a replacement. Computed per algorithm as:
      FixedWindow:            count / limit
      TokenBucket:             (capacity - tokens) / capacity
      LeakyBucketMeter:        volume / capacity
      LeakyBucketQueue:        depth / capacity
      SlidingWindowLog:        total_cost / limit
      SlidingWindowCounter:    weighted / limit
  using the post-decision values already computed by each algorithm's
  allow(), not a new state read.

Both new fields default to 0.0 (NOT required positional arguments) so
existing call sites that construct these events directly without the
new fields (e.g. hand-built events in test code elsewhere in the
suite) continue to work unchanged. Every production call site inside
the 12 algorithm files passes real values explicitly.

`BackendErrorEvent` gets `timestamp` but not `utilization`, since a
backend failure means no decision -- and often no state read --
actually completed; there is nothing meaningful to normalize. Its
`timestamp` is captured via a fresh `self._clock()` call in the
except-handler (not reused from a `now` computed earlier in the try
block), because a backend failure can occur before `now` is ever
assigned -- e.g. a Redis lock-acquire failure happens before the
algorithm's own `now = self._clock()` line runs at all.

Async hooks were considered but NOT added. Resolution: the
`SimulationRecorder` hook (see rlimit.simulation) buffers each event
as a plain in-memory row instead of performing any I/O inside the hook
call itself; CSV writing happens once, explicitly, via a separate
flush call after a run completes. That keeps every hook call trivially
fast (no blocking I/O), which removes the actual problem the
async-hook idea was meant to solve, without adding a second hook
interface. MetricsHook therefore remains sync-only.

--------------------------------------------------------------------
ORIGINAL DESIGN DECISIONS (unchanged, preserved verbatim)
--------------------------------------------------------------------

1. Structured Protocol, not a stringly-typed callback. A hook is
   anything implementing `MetricsHook` (three typed methods:
   on_allowed/on_denied/on_backend_error), not a single
   `callback(event_name: str, data: dict)` function. The stringly-typed
   version is more "flexible" on paper but typo-prone and gives no
   IDE/mypy support for what fields a given event actually carries --
   rejected for that reason even though it would have been less code.

2. Three event types, not two. `allowed` / `denied` alone cannot
   distinguish "this request was rejected because it exceeded the
   limit" from "the backend could not be reached and we don't actually
   know." Conflating those is a real correctness-hiding bug (see
   rlimit.exceptions.BackendUnavailableError's docstring) -- so
   `backend_error` is a first-class third event, not an afterthought.

3. No `prometheus_client` import anywhere in this module or in core.
   The library defines the protocol; users write their own hook (or an
   adapter package can exist separately later). Importing
   prometheus_client here would turn an optional integration into an
   architectural dependency of the whole library, and would force a
   decision about label cardinality (see point 5) that belongs to the
   *user's* hook, not to rlimit's core. This same reasoning is why
   rlimit.simulation writes plain CSV via the stdlib `csv` module
   rather than depending on pandas or a plotting library -- plotting
   itself stays entirely out of scope for the library.

4. One sync hook interface, used by BOTH sync and async algorithms --
   mirrors the existing `clock: Callable[[], float]` precedent from
   sync clock reused by async algorithms rather than a
   separate async-clock concept. Consequence, stated explicitly: a
   slow or blocking hook implementation WILL block the event loop when
   used with an async algorithm, the same way a slow clock function
   would. This is documented here and in every algorithm's docstring
   rather than solved by auto-threading the hook (rejected: hidden
   threading, overhead, and shutdown/error complexity for a problem
   the caller can solve trivially by keeping their hook fast). This is
   the exact reasoning that produced the buffered-recorder resolution
   for simulation mode above, rather than a new async hook type.

5. The rate-limit `key` IS included on every event. Hook
   implementations decide what to do with it. This is flagged
   explicitly: turning a raw, unbounded-cardinality key (e.g. a raw
   user ID or IP) directly into a Prometheus label is a well-known way
   to blow up a metrics backend's memory. rlimit does not do this
   automatically and does not stop a hook author from doing it
   themselves -- that decision is downstream of this library.

6. State snapshot is algorithm-specific (a plain `Mapping[str, Any]`),
   not one forced-uniform `remaining` field. Different algorithms
   genuinely have different natural state (token count vs. window
   count vs. log length vs. bucket volume) and flattening that into one
   artificial shape would either lose information or require padding
   fields that don't apply. The tradeoff (slightly weaker typing on
   `state`) is accepted deliberately. `utilization` 
   does NOT reopen this decision -- it sits alongside `state`, not in
   place of it, precisely so this tradeoff doesn't have to be revisited.

7. Hook failures are isolated. emit_allowed/emit_denied/
   emit_backend_error below catch any exception a hook implementation
   raises, log it via structlog at ERROR level (through the plain
   logger, not by trying to call back into the same hook -- that would
   risk infinite recursion if the hook itself is what's broken), and
   then continue. A broken metrics hook must never take down the rate
   limiter itself -- observability failing is a real, separate
   incident from the core function of allow()/allow_wait() breaking.

8. Metrics are emitted only for allow()'s actual rate-limiting
   decision path (the case that reads/writes storage under the
   per-key lock), not for early-return input validation (invalid cost
   type, negative cost) and not for the documented cost == 0 no-op.
   Those are client-input-validation outcomes, not decisions the
   limiter's algorithm made -- conflating them would mean "denied"
   metrics counts include malformed-call noise indistinguishable from
   real throttling. This is a deliberate scope boundary, not an
   oversight; if you need to observe invalid-cost calls specifically,
   that is a separate, not-yet-built concern.

9. Only allow() emits metrics in this, not allow_wait(). 
    This remains true in the (simulation-only) pass as well --
   simulation mode deliberately only drives allow(), not allow_wait(),
   so this scope boundary was never revisited. 

10. Event ORDERING is not guaranteed under concurrency. Metrics
   events are emitted AFTER the per-key lock is released (see point
   at the top of this file's design notes and each algorithm file's
   module docstring for why -- it keeps the lock's critical section
   short). Consequence: if two concurrent callers race the same key,
   the actual state mutations happen in one order (serialized by the
   lock), but the metrics events describing those mutations can be
   observed by a hook in a DIFFERENT order, since nothing serializes
   the post-lock emit step itself. This is fine for aggregate
   counters (a Prometheus Counter doesn't care what order increments
   arrive in) but would be a real bug if a hook implementation ever
   assumed events arrive in the same order requests were actually
   admitted. Documented here explicitly rather than left as an
   implicit consequence of point 7's lock-release-then-emit ordering.
   For simulation mode specifically: the recorded `timestamp` field
   (the algorithm's own clock reading) is the correct ordering key for
   plotting, NOT the order rows happen to appear in the recorder's
   buffer under concurrent access.
11. Event `state` is immutable. AllowedEvent/DeniedEvent wrap the
   snapshot dict built by the algorithm in `types.MappingProxyType`
   before storing it (see `_freeze` below), so a hook cannot mutate
   `event.state` and have that go unnoticed by another hook or a
   retained reference elsewhere. This does NOT protect against a hook
   mutating the *algorithm's own* internal state -- the snapshot is a
   copy taken at decision time, not a live view -- it only protects
   the event object itself from being silently altered after the fact.
--------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from rlimit.logging import get_logger

_log = get_logger(__name__)


def _freeze(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Wrap `state` in a read-only view. O(1), no copy of the
    underlying data -- MappingProxyType is a live wrapper around
    whatever mapping is passed in, not a clone. Safe to wrap without
    copying here because every algorithm builds a fresh, exclusively-
    owned `snapshot` dict right before constructing the event and
    never mutates it afterward (verified across all twelve concrete
    algorithm classes) -- there is no other live reference to alias
    against. See module docstring point 11."""
    return MappingProxyType(state)


@dataclass(frozen=True)
class AllowedEvent:
    """Fired when allow() admits a request. `state` is an
    algorithm-specific snapshot taken at the moment of the decision
    (see this module's docstring, point 6) -- e.g. FixedWindow includes
    window_start/count/limit, TokenBucket includes tokens/capacity.
    `state` is a read-only mapping (point 11) -- mutating it raises
    TypeError, the same as mutating any other MappingProxyType.

    `utilization` and `timestamp` default to 0.0
    so pre-existing code constructing this event without them keeps
    working -- see module docstring's section for the full
    rationale and per-algorithm utilization formulas.
    """

    algorithm: str
    key: str
    cost: int
    state: Mapping[str, Any]
    utilization: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _freeze(self.state))


@dataclass(frozen=True)
class DeniedEvent:
    """Fired when allow() rejects a request that reached the actual
    rate-limiting decision (i.e. NOT invalid-cost or cost==0 -- see
    this module's docstring, point 8). `state` is read-only, see
    point 11. `utilization`/`timestamp`: see AllowedEvent's docstring
    """

    algorithm: str
    key: str
    cost: int
    state: Mapping[str, Any]
    utilization: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _freeze(self.state))


@dataclass(frozen=True)
class BackendErrorEvent:
    """Fired when allow() could not reach the storage backend at all.
    `error` is the original BackendUnavailableError (already chained to
    the underlying cause via `__cause__` -- see rlimit.exceptions).

    `timestamp` defaults to 0.0 for the same backward-
    compatibility reason as AllowedEvent/DeniedEvent. No `utilization`
    field here -- see this module's docstring section for why
    a failed decision has nothing meaningful to normalize."""

    algorithm: str
    key: str
    cost: int
    error: BaseException
    timestamp: float = 0.0


@runtime_checkable
class MetricsHook(Protocol):
    """Anything implementing these three methods can be passed as the
    `metrics=` argument to any sync or async rate limiter. See this
    module's docstring for the full set of design decisions this
    interface embodies -- in particular point 4 (hooks must be fast/
    non-blocking; they are called synchronously even from async
    algorithms) and point 7 (a hook that raises will have that
    exception caught and logged by the library, not propagated).

    Remains sync-only as of now -- see this module's docstring for
    why simulation mode's recorder does not require an async variant.
    """

    def on_allowed(self, event: AllowedEvent) -> None: ...

    def on_denied(self, event: DeniedEvent) -> None: ...

    def on_backend_error(self, event: BackendErrorEvent) -> None: ...


class NoOpMetricsHook:
    """The default hook every algorithm uses when `metrics=None` (the
    default). Exists so algorithm code never has to branch on
    `self._metrics is None` at every call site -- it always has a real
    MetricsHook to call, and this one just does nothing. See this
    module's docstring, point 8's neighbor decision in storage.py's
    equivalent "optional with a real default object" pattern used for
    consistency previously."""

    def on_allowed(self, event: AllowedEvent) -> None:
        return None

    def on_denied(self, event: DeniedEvent) -> None:
        return None

    def on_backend_error(self, event: BackendErrorEvent) -> None:
        return None


_NOOP_METRICS_HOOK = NoOpMetricsHook()


def default_metrics_hook() -> MetricsHook:
    """Returns the shared no-op hook instance. Every algorithm's
    `__init__` calls this when `metrics=None` is passed (the default),
    so `self._metrics` is always a real, callable MetricsHook."""
    return _NOOP_METRICS_HOOK


def emit_allowed(hook: MetricsHook, event: AllowedEvent) -> None:
    """Call hook.on_allowed(event), isolating any exception the hook
    raises (see module docstring, point 7)."""
    try:
        hook.on_allowed(event)
    except Exception:
        _log.error(
            "metrics_hook_failed",
            callback="on_allowed",
            algorithm=event.algorithm,
            key=event.key,
            exc_info=True,
        )


def emit_denied(hook: MetricsHook, event: DeniedEvent) -> None:
    """Call hook.on_denied(event), isolating any exception the hook
    raises (see module docstring, point 7)."""
    try:
        hook.on_denied(event)
    except Exception:
        _log.error(
            "metrics_hook_failed",
            callback="on_denied",
            algorithm=event.algorithm,
            key=event.key,
            exc_info=True,
        )


def emit_backend_error(hook: MetricsHook, event: BackendErrorEvent) -> None:
    """Call hook.on_backend_error(event), isolating any exception the
    hook raises (see module docstring, point 7). Does NOT swallow or
    affect the original BackendUnavailableError -- the caller (each
    algorithm's allow()) re-raises that separately, unchanged, after
    calling this."""
    try:
        hook.on_backend_error(event)
    except Exception:
        _log.error(
            "metrics_hook_failed",
            callback="on_backend_error",
            algorithm=event.algorithm,
            key=event.key,
            exc_info=True,
        )
