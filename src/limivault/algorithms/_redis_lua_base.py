# src/limivault/algorithms/_redis_lua_base.py
"""Shared, private helpers for Redis-native Lua/GCRA
limiter classes (redis_lua_*.py / async_redis_lua_*.py in this
package). Not part of the public API.

v2 addition: `validate_ttl_seconds` -- flagged in review as a real
gap. Every other numeric constructor parameter across this project
(limit, period, capacity, refill_rate, leak_rate) is validated for
being finite and correctly signed; `ttl_seconds` (whether computed
automatically or passed explicitly by a caller) previously was not,
despite flowing straight into Redis's EXPIRE. A zero/negative/NaN/
infinite TTL there would produce confusing Redis-level behavior
(EXPIRE with a non-positive or nonsensical seconds value) instead of a
clear Python-level ValueError at construction time.

v3 addition: `emit_decision` / `emit_error` -- the metrics glue
(build an AllowedEvent/DeniedEvent/BackendErrorEvent from an
algorithm's post-decision values and call emit_allowed/emit_denied/
emit_backend_error) was previously copy-pasted verbatim into all 12
concrete algorithm files. Flagged in review as 12x drift risk: if this
pattern ever needs to change (a new event field, a different emission
rule), it would need to change in 12 places identically, with no
structural guard against one file being missed. Factored here into two
small functions instead; every algorithms/redis_lua_*.py /
async_redis_lua_*.py file's allow() now calls one of these rather than
constructing the event objects itself. Kept as plain functions (not a
class) since there's no state to hold -- the metrics hook and every
value needed are already available at each call site.

v3 addition (second pass): `log_decision` -- the same 12x-duplication
pattern existed one level down, for the DEBUG-level structlog line
each allow() emits (`"<algorithm>_decision"`, key, allowed, cost,
remaining, and either `limit=...` or `capacity=...` depending on the
algorithm). Every file had the exact same 8-line `_log.debug(...)`
call, differing only in the event-name string and whether the last
field was spelled `limit` or `capacity`. Factored into one function
that takes the event name and the algorithm-specific quota field as
`**extra`, so the call shape can never drift between files the way 12
independently-maintained copies eventually would.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from limivault.metrics import (
    AllowedEvent,
    BackendErrorEvent,
    DeniedEvent,
    MetricsHook,
    emit_allowed,
    emit_backend_error,
    emit_denied,
)


def reject_non_finite(value: float, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def is_valid_cost_type(cost: int) -> bool:
    return isinstance(cost, int) and not isinstance(cost, bool)


def validate_ttl_seconds(ttl_seconds: float) -> None:
    """Raise ValueError if `ttl_seconds` is not a positive, finite
    number. Applies whether the value was supplied explicitly by the
    caller or computed automatically from the algorithm's own
    characteristic time -- both paths funnel through this check."""
    if not math.isfinite(ttl_seconds):
        raise ValueError(f"ttl_seconds must be finite, got {ttl_seconds}")
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")


def emit_decision(
    metrics: MetricsHook,
    *,
    algorithm: str,
    key: str,
    cost: int,
    allowed: bool,
    state: Mapping[str, Any],
    utilization: float,
    timestamp: float,
) -> None:
    """Build and emit the AllowedEvent/DeniedEvent for one allow()
    decision. Every algorithms/redis_lua_*.py / async_redis_lua_*.py
    file's allow() calls this once, after its script call returns,
    instead of constructing the event object itself -- see this
    module's v3 docstring section for why this was factored out.
    """
    if allowed:
        emit_allowed(
            metrics,
            AllowedEvent(
                algorithm=algorithm,
                key=key,
                cost=cost,
                state=state,
                utilization=utilization,
                timestamp=timestamp,
            ),
        )
    else:
        emit_denied(
            metrics,
            DeniedEvent(
                algorithm=algorithm,
                key=key,
                cost=cost,
                state=state,
                utilization=utilization,
                timestamp=timestamp,
            ),
        )


def emit_error(
    metrics: MetricsHook,
    *,
    algorithm: str,
    key: str,
    cost: int,
    error: BaseException,
    timestamp: float,
) -> None:
    """Build and emit the BackendErrorEvent for one failed script
    call. See `emit_decision`'s docstring for the same rationale."""
    emit_backend_error(
        metrics,
        BackendErrorEvent(
            algorithm=algorithm,
            key=key,
            cost=cost,
            error=error,
            timestamp=timestamp,
        ),
    )


def log_decision(
    log: Any,
    event: str,
    *,
    key: str,
    allowed: bool,
    cost: int,
    remaining: int,
    **extra: Any,
) -> None:
    """Emit the shared DEBUG-level decision log line. `log` is the
    calling module's own structlog logger (`_log` in every
    algorithms/redis_lua_*.py / async_redis_lua_*.py file) -- kept as
    a parameter rather than constructed here, so each module's log
    lines still show that module's own logger name (see
    limivault.logging.get_logger's `logger=name` binding), not this
    shared helper's.

    `extra` carries the one field that differs by algorithm family --
    `limit=self._limit` for window/log/counter algorithms,
    `capacity=self._capacity` for GCRA/bucket algorithms -- passed
    through as-is rather than forced into a single fixed parameter
    name, since the two families genuinely mean different things by
    "how much quota" and flattening that into one generic kwarg name
    would lose the distinction every other DEBUG line and metrics
    snapshot in this project already preserves (see e.g.
    limivault.metrics module docstring, point 6, making the same call for
    the in-memory algorithms' `state` snapshot's algorithm-specific
    shape).
    """
    log.debug(event, key=key, allowed=allowed, cost=cost, remaining=remaining, **extra)
