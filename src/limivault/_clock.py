# src/limivault/_clock.py
"""Non-decreasing clock wrapper (0.1.2).

Every in-memory algorithm assumes `now` never goes backwards between
calls. That was a documented precondition only, and it mattered: with a
clock that jumps back (e.g. `time.time` under an NTP correction, or a
hand-written test clock), FixedWindow and SlidingWindowCounter computed
an *earlier* window, found no state for it, and handed out a whole
fresh quota -- over-admitting by up to 100% of the limit.

`NonDecreasingClock` wraps whatever clock the caller passes and never
returns a value smaller than one it already returned. A backwards jump
therefore *freezes* the limiter's time until the real clock catches up:
no refill, no new window, no over-admission. Freezing is the safe
direction for a rate limiter (it can only be stricter, never looser).

Per-limiter, not global: each algorithm instance wraps its own clock, so
two limiters sharing one clock function stay independent. Thread-safe
(one small lock), so it also holds under free-threaded Python builds.
NaN readings are ignored (the last good value is returned).
"""

from __future__ import annotations

import math
import threading
from typing import Callable

_State = tuple[Callable[[], float], float | None, int]


class NonDecreasingClock:
    __slots__ = ("_clock", "_last", "_lock", "backward_jumps")

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()
        # How many readings were clamped. Exposed for diagnostics/tests.
        self.backward_jumps = 0

    # Picklable (limiters are pickled into worker processes by the
    # multiprocess_safe tests/users): the lock is recreated, and the last
    # reading travels along so the child never starts "behind" it.
    def __getstate__(self) -> _State:
        return (self._clock, self._last, self.backward_jumps)

    def __setstate__(self, state: _State) -> None:
        self._clock, self._last, self.backward_jumps = state
        self._lock = threading.Lock()

    @property
    def wrapped(self) -> Callable[[], float]:
        return self._clock

    def __call__(self) -> float:
        t = self._clock()
        with self._lock:
            last = self._last
            if isinstance(t, float) and math.isnan(t):
                return last if last is not None else t
            if last is not None and t < last:
                self.backward_jumps += 1
                return last
            self._last = t
            return t


def non_decreasing(clock: Callable[[], float]) -> NonDecreasingClock:
    """Wrap `clock` (idempotent: an already-wrapped clock is returned as is)."""
    if isinstance(clock, NonDecreasingClock):
        return clock
    return NonDecreasingClock(clock)
