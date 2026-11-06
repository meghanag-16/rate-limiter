# src/limivault/storage.py
"""Storage backends for rate limiter algorithm state.

This module contains the sync storage backends (Phase 1, Phase 4's
multiprocess_safe addition, and Phase 8's lock cleanup), and the async
storage backends (Phase 5, plus Phase 8's mirrored lock cleanup).

InMemoryStorage (sync) has two modes, selected at construction time:

- multiprocess_safe=False (default): a plain dict plus one
  threading.Lock per key, created lazily and guarded by a small
  `_locks_guard` lock. Fast, in-process / multi-threaded only.

- multiprocess_safe=True (Phase 4): state lives in a
  multiprocessing.Manager dict, so multiple OS processes sharing the
  same InMemoryStorage instance see and mutate the same underlying
  data, via a fixed-size pool of pre-created Manager locks (see the
  original Phase 4 design notes preserved below).

--------------------------------------------------------------------
PHASE 8: LOCK CLEANUP FOR THE DEFAULT (multiprocess_safe=False) MODE
--------------------------------------------------------------------

api-plan.txt's decisions-log explicitly flagged this: "Lock cleanup
deferred to Phase 8 (observability) -- InMemoryStorage will keep
accumulating one lock per unique key with no eviction until then."
Every unique key ever passed to a limiter backed by the default
InMemoryStorage leaves one threading.Lock in `_locks` forever -- an
unbounded memory leak for any deployment with high key cardinality
(e.g. per-user or per-IP limiting with a large or churning user base).

This does NOT apply to multiprocess_safe=True: that mode's fixed-size
Manager lock pool (Phase 4) has a hard, constant memory ceiling by
construction -- no leak, no cleanup needed, unchanged in this phase.

DESIGN CHOSEN: opportunistic idle sweep (not reference counting, not a
background thread, not a hard LRU-bounded cap). Rationale: reference
counting would need precise, race-free bookkeeping of "is anyone
currently inside a critical section for this key" on top of the lock
itself: extra complexity for a memory-leak fix, not proportionate.
A background sweep thread adds a whole extra lifecycle/shutdown
concern (when does it stop? does it keep a process alive?) for a
problem that doesn't need real-time response. A hard-bounded LRU cache
gives a strict memory ceiling but risks evicting a key that's about to
be reused, trading correctness risk for a guarantee this project
doesn't need. An idle-time sweep, triggered opportunistically from
regular calls, needs no background thread, gives no strict bound
between sweeps, but keeps eventual memory use proportional to *recent*
key cardinality rather than *all-time* key cardinality -- judged the
right tradeoff for this project (see the Phase 8 planning discussion:
"eventual cleanup ... best tradeoff for current project").

DEFAULTS -- STATED EXPLICITLY AS DELIBERATE, NOT DISCOVERED VALUES:
    _DEFAULT_LOCK_SWEEP_INTERVAL = 1000   (calls between sweep attempts)
    _DEFAULT_LOCK_IDLE_SECONDS   = 300.0  (5 minutes of inactivity)

These were NOT derived from any measured production traffic pattern --
none was available. They are conservative placeholder defaults chosen
so that:
  - Sweeping "every 1000 calls" keeps per-call overhead amortized to
    near-zero (a full sweep pass is O(number of tracked keys), but it
    only runs once every 1000 calls, not on every call).
  - An idle threshold of 5 minutes is long relative to realistic
    scheduling delays (see the race-window note below) but short
    enough that a bursty, high-cardinality workload (e.g. one-shot
    keys that are each used exactly once and never again) doesn't
    accumulate unboundedly for hours.
Both are constructor arguments specifically so a deployment with a
known traffic shape can override them; there is no dynamic
auto-tuning. If your keyspace churns faster or slower than this,
override lock_sweep_interval / lock_idle_seconds explicitly rather
than relying on these defaults being right for you.

TRIGGER: every Nth call to `_get_lock()` (N = lock_sweep_interval)
triggers one sweep pass, counted via a simple counter guarded by the
same `_locks_guard` used for the lock dict itself. Sweeping on EVERY
call was rejected (adds O(tracked keys) overhead to every single
allow()/allow_wait() call, not just occasionally). Sweeping only when
manually triggered was rejected (the leak persists indefinitely unless
a caller remembers to invoke it -- defeats the purpose of a documented,
built-in fix).

ELIGIBILITY: a lock is evicted only if BOTH (a) it has been idle for at
least `lock_idle_seconds` (tracked via a `_lock_last_used[key]`
timestamp, refreshed every time `_get_lock()` is called for that key)
AND (b) it is not currently held. (b) is checked via a non-blocking
`lock.acquire(blocking=False)` -- if that succeeds, nothing else holds
it, so it is safe to discard (and the just-acquired lock is released
immediately afterward, purely as bookkeeping hygiene, since it's about
to be dropped from `_locks` anyway). If the non-blocking acquire fails,
the lock is currently in someone's critical section and is NEVER
evicted, regardless of how "idle" its last-used timestamp claims to be
(a lock can only be "in use" or its timestamp would have just been
refreshed by the caller who's using it -- see the race-window caveat
below for the one exception to that reasoning).

DOCUMENTED, ACCEPTED RACE WINDOW (read before assuming this is airtight):
`InMemoryStorage.lock(key)` (via `_get_lock`) returns a bare
`threading.Lock` object to the caller; the caller then separately does
`with storage.lock(key):`, i.e. acquisition happens OUTSIDE the
`_locks_guard`-protected section that looked up/created the lock and
stamped its last-used time. There is therefore a narrow window --
between `_get_lock()` returning a lock reference and that caller
actually calling `.acquire()` on it -- during which the lock is
technically idle-and-unlocked from the sweep's point of view, even
though a caller is about to use it. If a sweep runs during that exact
window and `lock_idle_seconds` has (implausibly) already elapsed since
the timestamp was just refreshed a moment ago, eviction could in theory
still occur, and a second caller for the same key arriving after that
point would be handed a brand-new, different Lock object -- meaning the
two callers would no longer share mutual exclusion for that key.
This is accepted as a deliberate tradeoff, not silently ignored: with
the default 300s idle threshold, this would require a sweep to land in
a window of real scheduling delay measured in microseconds, immediately
after a timestamp refresh, for the idle-check to somehow already exceed
300 seconds -- not possible under the current design as written (the
timestamp is refreshed to "now" at the same moment the lock reference
is handed out, so the elapsed time at eviction-check time can only be
suspiciously large if the *sweep itself* is running much later, at
which point the caller holding the stale reference has almost
certainly already finished or is itself stalled for an equally long
time). The theoretical race is a consequence of `lock()` and the
caller's `with` block being two separate steps rather than one atomic
operation (true since Phase 1, unrelated to this fix), and closing it
completely would require restructuring `lock()` into an async-style
context manager that holds `_locks_guard` across the full acquire --
which would serialize unrelated keys against each other and was
rejected for that reason. Flagged here explicitly rather than left
implicit, per the project's stated preference for surfacing rather
than hiding this kind of edge case.

CLOCK USED FOR SWEEP BOOKKEEPING: `InMemoryStorage` and
`AsyncInMemoryStorage` now accept their own `clock: Callable[[], float]
= time.monotonic` constructor argument, used ONLY for lock idle-time
bookkeeping. This is DELIBERATELY INDEPENDENT from whatever `clock=`
argument is passed to a `RateLimiter`/`AsyncRateLimiter` algorithm
instance that happens to use this storage -- e.g. a test using a
FakeClock for the *algorithm's* window/refill math will still have
this storage's idle-sweep bookkeeping running against real
`time.monotonic()` by default, unless a matching FakeClock is
explicitly passed to the storage constructor too. This mismatch is
intentional (storage-level housekeeping and algorithm-level rate math
are different concerns and don't have to share a clock), but is worth
knowing: tests exercising the sweep behavior deterministically must
inject a clock here explicitly, not assume the algorithm's FakeClock
also drives it. See tests/test_storage_lock_cleanup.py.

Everything else below (multiprocess_safe design notes, async storage
scope notes) is unchanged from Phase 1-5.

Design notes / tradeoffs for multiprocess_safe mode (read before
changing the locking scheme):

1. Per-key lock creation cannot be lazy across processes the way it is
   in the threading backend. A `multiprocessing.Manager()` object
   itself cannot be pickled -- Python explicitly disallows pickling
   its internal AuthenticationString "for security reasons" -- so a
   worker process that receives a copy of this InMemoryStorage never
   has a live Manager reference and cannot call `manager.Lock()` to
   mint a brand new lock for a key it sees for the first time. (Only
   the *proxy* objects a Manager has already handed out -- e.g. a
   Lock or a DictProxy -- are themselves picklable and reconnect to
   the manager's server process correctly; this was verified directly
   against a real ProcessPoolExecutor before writing this file.)

   The fix used here is a fixed-size pool of Manager locks, all
   created once in `__init__` (in the parent process, before any
   worker exists). A key is mapped to a pool slot by
   `stable_hash(key) % pool_size`, so no new lock is ever created
   after construction -- only already-created, already-picklable Lock
   proxies ever cross a process boundary. The cost: two different keys
   that hash to the same slot will contend with each other (a small
   amount of false sharing), rather than each key getting a fully
   dedicated lock as it does in the threading backend. Raising
   `mp_lock_pool_size` reduces the odds of that at the cost of one
   extra Manager Lock (a small amount of manager-process memory) per
   slot.

2. The lock-pool index MUST NOT be computed with Python's built-in
   `hash()`. `hash()` for str is salted with a per-process random seed
   (PYTHONHASHSEED) for hash-flooding-attack resistance, and while
   `fork`-based child processes happen to inherit the parent's seed,
   `spawn`-based ones do not -- each spawned interpreter picks its own
   random seed. Windows (and macOS by default) uses `spawn` as the
   multiprocessing start method. This was verified directly: the same
   string hashed differently in a spawned child than in the parent in
   this environment. Using `hash()` here would have been a real,
   silent correctness bug -- two processes could compute two different
   lock indices for the identical key, and each would merrily proceed
   unsynchronized against the same underlying data. `zlib.crc32` is
   used instead, which is deterministic across processes and platforms
   regardless of start method or hash seed.

3. `InMemoryStorage.__getstate__`/`__setstate__` drop the `_manager`
   attribute when this object is pickled to cross a process boundary
   (it isn't picklable, and isn't needed after construction -- see
   point 1). The DictProxy and the pre-built Lock proxies pickle and
   reconnect to the manager's server process normally. The Phase 8
   sweep-bookkeeping attributes (`_lock_last_used`, `_sweep_call_count`,
   `_lock_sweep_interval`, `_lock_idle_seconds`, `_clock`) are plain
   picklable values in the default `time.monotonic` case and pickle
   along with everything else in `__dict__` without special handling;
   a custom, non-picklable `clock` callable passed to a
   multiprocess_safe=True instance would fail to pickle for the same
   reason a non-picklable custom clock passed to an algorithm would --
   this is an existing, pre-existing category of caveat, not new here.

4. multiprocess_safe=True is meaningfully slower per call than the
   default threading backend, because every get()/set()/lock
   acquisition is an IPC round trip to the manager's server process
   rather than a local dict/lock operation. It exists for cross-process
   correctness (Phase 4's stress tests, or a multi-process deployment
   sharing one InMemoryStorage), not as a general-purpose replacement
   for the default backend. A real distributed deployment across
   *machines* (not just processes on one machine) needs the
   Redis-native Lua/GCRA limiters instead (see limivault.redis_lua_scripts)
   -- this only reaches across processes on a single host.

Async storage backends (Phase 5, read before touching AsyncInMemoryStorage):

- AsyncStorageBackend.lock() is a *sync* method (not `async def`) that
  returns an async context manager, mirroring the sync
  StorageBackend.lock() shape exactly so algorithm code reads the same
  way in both worlds. The lookup/creation of the underlying per-key
  asyncio.Lock (and, as of Phase 8, the idle-sweep bookkeeping) happens
  lazily inside that context manager's __aenter__, under the storage's
  own `_locks_guard` asyncio.Lock.

- There is no multiprocess_safe mode for AsyncInMemoryStorage.
  asyncio.Lock is an event-loop-local primitive and multiprocessing.
  Manager locks are OS-level; the two don't compose. See
  AsyncInMemoryStorage's own class docstring for the operational scope
  this implies.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import multiprocessing.managers
import os
import threading
import time
import warnings
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import AbstractContextManager
from typing import Any, AsyncContextManager, Callable, MutableMapping


class StorageBackend(ABC):
    """Abstract storage for per-key algorithm state (sync)."""

    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None:
        """Return stored state for `key`, or None if not present."""
        ...

    @abstractmethod
    def set(self, key: str, state: dict[str, Any]) -> None:
        """Overwrite stored state for `key`."""
        ...

    @abstractmethod
    def lock(self, key: str) -> AbstractContextManager[bool]:
        """Return a context manager providing exclusive access for `key`."""
        ...


# Default size of the cross-process lock pool used when
# multiprocess_safe=True. See module docstring point 1.
_DEFAULT_MP_LOCK_POOL_SIZE = 64

# Phase 8 lock-cleanup defaults -- see module docstring's "DEFAULTS"
# section above for the full, explicit rationale. These are
# deliberately conservative placeholders, not measured values.
_DEFAULT_LOCK_SWEEP_INTERVAL = 1000
_DEFAULT_LOCK_IDLE_SECONDS = 300.0


_SILENCE_ENV = "LIMIVAULT_SILENCE_MULTIPROCESS_WARNING"
_multiprocess_warning_emitted = False


def _multiprocess_reason() -> str | None:
    """Why this process looks like one of several workers, or None."""
    if multiprocessing.parent_process() is not None:
        return (
            "this process was started by multiprocessing (for example "
            "uvicorn --workers or --reload)"
        )
    raw = os.environ.get("WEB_CONCURRENCY", "")
    try:
        workers = int(raw) if raw else 1
    except ValueError:
        workers = 1
    if workers > 1:
        return f"WEB_CONCURRENCY={workers}"
    return None


def _warn_if_multi_process() -> None:
    """Fix for "in-memory limits multiply with workers": warn once per
    process. See limivault.exceptions.MultiProcessStateWarning."""
    global _multiprocess_warning_emitted
    if _multiprocess_warning_emitted or os.environ.get(_SILENCE_ENV) == "1":
        return
    reason = _multiprocess_reason()
    if reason is None:
        return
    _multiprocess_warning_emitted = True
    from limivault.exceptions import MultiProcessStateWarning

    warnings.warn(
        "limivault in-memory storage keeps limits inside this one process, "
        f"and {reason}. If more than one process serves traffic, each "
        "enforces its own separate limit (N workers => up to N x the limit). "
        "Use the Redis limiters to share one limit, or set "
        f"{_SILENCE_ENV}=1 if this process really is the only one.",
        MultiProcessStateWarning,
        stacklevel=4,
    )


def _validate_bounds(max_keys: int | None, state_idle_seconds: float | None) -> None:
    if max_keys is not None and (
        not isinstance(max_keys, int) or isinstance(max_keys, bool) or max_keys <= 0
    ):
        raise ValueError(f"max_keys must be a positive int, got {max_keys!r}")
    if state_idle_seconds is not None and not (
        state_idle_seconds > 0 and state_idle_seconds != float("inf")
    ):
        raise ValueError(
            "state_idle_seconds must be positive and finite, "
            f"got {state_idle_seconds!r}"
        )


def _stable_key_hash(key: str) -> int:
    """Deterministic hash of `key`, stable across processes and Python
    invocations (unlike the built-in `hash()`, which is salted with a
    random per-process seed -- see module docstring point 2)."""
    return zlib.crc32(key.encode("utf-8"))


class InMemoryStorage(StorageBackend):
    """Dict-backed storage (sync).

    See module docstring for the two modes (`multiprocess_safe=False`,
    the default, vs `multiprocess_safe=True`) and for the Phase 8 idle
    lock-sweep behavior of the default mode.

    Scope: safe to share across threads within one process. NOT safe
    to share across processes unless constructed with
    multiprocess_safe=True.

    BOUNDED STATE (0.1.2, opt-in, default mode only):
    By default one state entry is kept per key ever seen, forever --
    memory grows with the number of distinct keys (e.g. a flood of new
    IPs on a public API). Two optional limits bound it:

    - max_keys: soft cap on stored keys. Setting a NEW key beyond the
      cap evicts the least-recently-used key. Evicting a key RESETS its
      limit (that client starts fresh), so pick a cap comfortably above
      your number of concurrently active clients. "Soft": a decision
      already in flight for an evicted key re-inserts it.
    - state_idle_seconds: drop state not read or written for this long,
      on the same opportunistic sweep as the lock cleanup. Safe when set
      to at least the limiter's full-recovery time (period for window
      algorithms, capacity / rate for buckets): by then a fresh state is
      equivalent to the dropped one.

    `len(storage)` is the current key count and `storage.evicted_count`
    counts evictions by either rule. Neither option is supported with
    multiprocess_safe=True.
    """

    def __init__(
        self,
        multiprocess_safe: bool = False,
        mp_lock_pool_size: int = _DEFAULT_MP_LOCK_POOL_SIZE,
        lock_sweep_interval: int = _DEFAULT_LOCK_SWEEP_INTERVAL,
        lock_idle_seconds: float = _DEFAULT_LOCK_IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int | None = None,
        state_idle_seconds: float | None = None,
    ) -> None:
        if lock_sweep_interval <= 0:
            raise ValueError(
                f"lock_sweep_interval must be positive, got {lock_sweep_interval}"
            )
        if lock_idle_seconds <= 0:
            raise ValueError(
                f"lock_idle_seconds must be positive, got {lock_idle_seconds}"
            )
        _validate_bounds(max_keys, state_idle_seconds)
        bounded = max_keys is not None or state_idle_seconds is not None
        if multiprocess_safe and bounded:
            raise ValueError(
                "max_keys / state_idle_seconds are not supported with "
                "multiprocess_safe=True"
            )
        self._lock_sweep_interval = lock_sweep_interval
        self._lock_idle_seconds = lock_idle_seconds
        self._clock = clock
        # 0.1.2 bounded-state options (see BOUNDED STATE in the class
        # docstring). Both None = the original unbounded plain-dict path.
        self._max_keys = max_keys
        self._state_idle_seconds = state_idle_seconds
        self._bounded = bounded
        self._state_last_used: dict[str, float] = {}
        self._data_guard = threading.Lock()
        self.evicted_count = 0

        self._multiprocess_safe = multiprocess_safe
        if multiprocess_safe:
            if mp_lock_pool_size <= 0:
                raise ValueError(
                    "mp_lock_pool_size must be positive, got "
                    f"{mp_lock_pool_size}"
                )
            self._mp_lock_pool_size = mp_lock_pool_size
            self._manager: multiprocessing.managers.SyncManager | None = (
                multiprocessing.Manager()
            )
            self._data: MutableMapping[str, dict[str, Any]] = self._manager.dict()
            self._lock_pool: list[AbstractContextManager[bool]] = [
                self._manager.Lock() for _ in range(mp_lock_pool_size)
            ]
            # multiprocess_safe mode uses the fixed pool above, which
            # has no leak -- these are unused in this mode, kept as
            # None/0 so every branch of __init__ sets every attribute
            # referenced elsewhere in the class (mypy strict).
            self._locks: dict[str, threading.Lock] | None = None
            self._locks_guard: threading.Lock | None = None
            self._lock_last_used: dict[str, float] | None = None
            self._sweep_call_count = 0
        else:
            self._mp_lock_pool_size = 0
            self._manager = None
            self._data = OrderedDict() if self._bounded else {}
            self._lock_pool = []
            self._locks = {}
            self._locks_guard = threading.Lock()
            self._lock_last_used = {}
            self._sweep_call_count = 0
            _warn_if_multi_process()

    def __getstate__(self) -> dict[str, Any]:
        # See module docstring point 3: the Manager itself can't be
        # (and doesn't need to be) pickled across a process boundary.
        state = self.__dict__.copy()
        state["_manager"] = None
        state["_data_guard"] = None  # thread locks can't be pickled
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._data_guard = threading.Lock()

    def _sweep_idle_locks_locked(self, now: float) -> None:
        """Evict idle, currently-unlocked entries from `_locks`.

        PRECONDITION: caller already holds `_locks_guard`. See module
        docstring's "ELIGIBILITY" and "DOCUMENTED, ACCEPTED RACE
        WINDOW" sections for exactly what "idle and unlocked" means
        and the one race this does not fully close.
        """
        assert self._locks is not None and self._lock_last_used is not None  # nosec B101
        for key in list(self._locks.keys()):
            last_used = self._lock_last_used.get(key, now)
            if now - last_used < self._lock_idle_seconds:
                continue
            lock = self._locks[key]
            acquired = lock.acquire(blocking=False)
            if not acquired:
                # Currently in someone's critical section -- never
                # evict, regardless of how stale the timestamp looks.
                continue
            try:
                del self._locks[key]
                self._lock_last_used.pop(key, None)
            finally:
                lock.release()
        if self._state_idle_seconds is not None:
            self._sweep_idle_state(now)

    def _sweep_idle_state(self, now: float) -> None:
        """Drop stored state for keys unused for `state_idle_seconds`.
        Runs on the same opportunistic schedule as the lock sweep.
        `_data` is kept in least-recently-used order, so the scan stops
        at the first key that is still fresh. A key whose lock is held
        right now is skipped, never dropped."""
        idle = self._state_idle_seconds
        assert idle is not None and self._locks is not None  # nosec B101
        with self._data_guard:
            for key in list(self._data.keys()):
                if now - self._state_last_used.get(key, now) < idle:
                    break
                lock = self._locks.get(key)
                if lock is not None and lock.locked():
                    continue
                del self._data[key]
                self._state_last_used.pop(key, None)
                self.evicted_count += 1

    def _get_lock(self, key: str) -> AbstractContextManager[bool]:
        if self._multiprocess_safe:
            index = _stable_key_hash(key) % self._mp_lock_pool_size
            return self._lock_pool[index]
        assert self._locks_guard is not None and self._locks is not None  # nosec B101
        assert self._lock_last_used is not None  # nosec B101
        with self._locks_guard:
            now = self._clock()
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            self._lock_last_used[key] = now
            self._sweep_call_count += 1
            if self._sweep_call_count >= self._lock_sweep_interval:
                self._sweep_call_count = 0
                self._sweep_idle_locks_locked(now)
            return self._locks[key]

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._bounded:
            return self._data.get(key)
        with self._data_guard:
            state = self._data.get(key)
            if state is not None:
                self._touch_locked(key)
            return state

    def set(self, key: str, state: dict[str, Any]) -> None:
        if not self._bounded:
            self._data[key] = state
            return
        with self._data_guard:
            self._data[key] = state
            self._touch_locked(key)
            self._evict_over_cap_locked()

    def _touch_locked(self, key: str) -> None:
        assert isinstance(self._data, OrderedDict)  # nosec B101
        self._data.move_to_end(key)
        self._state_last_used[key] = self._clock()

    def _evict_over_cap_locked(self) -> None:
        if self._max_keys is None:
            return
        assert isinstance(self._data, OrderedDict)  # nosec B101
        while len(self._data) > self._max_keys:
            oldest, _ = self._data.popitem(last=False)
            self._state_last_used.pop(oldest, None)
            self.evicted_count += 1

    def __len__(self) -> int:
        """Number of keys that currently have stored state."""
        return len(self._data)

    def lock(self, key: str) -> AbstractContextManager[bool]:
        return self._get_lock(key)


class AsyncStorageBackend(ABC):
    """Abstract storage for per-key algorithm state (async).

    Mirrors StorageBackend method-for-method. get()/set() are
    coroutines so a real backend can do genuine network I/O without
    blocking the event loop. lock() stays a *sync* method returning an
    async context manager -- see module docstring for why.
    """

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Return stored state for `key`, or None if not present."""
        ...

    @abstractmethod
    async def set(self, key: str, state: dict[str, Any]) -> None:
        """Overwrite stored state for `key`."""
        ...

    @abstractmethod
    def lock(self, key: str) -> AsyncContextManager[bool]:
        """Return an async context manager providing exclusive access
        for `key`."""
        ...


class _AsyncKeyLock:
    """Async context manager for a single per-key lock acquisition.

    Returned by AsyncInMemoryStorage.lock(key). The actual per-key
    asyncio.Lock is looked up (creating it if necessary) inside
    __aenter__, under the storage's `_locks_guard` asyncio.Lock -- the
    same section that also stamps the lock's last-used time and
    opportunistically triggers a Phase 8 idle sweep (see storage.py's
    module docstring for the full rationale, shared with the sync
    version; the same documented, accepted race window applies here
    too, for the same reason: `lock()`/__aenter__ still returns before
    the caller's own `async with` body runs).
    """

    __slots__ = ("_storage", "_key", "_acquired_lock")

    def __init__(self, storage: "AsyncInMemoryStorage", key: str) -> None:
        self._storage = storage
        self._key = key
        self._acquired_lock: asyncio.Lock | None = None

    async def __aenter__(self) -> bool:
        async with self._storage._locks_guard:
            now = self._storage._clock()
            existing = self._storage._locks.get(self._key)
            if existing is None:
                existing = asyncio.Lock()
                self._storage._locks[self._key] = existing
            self._storage._lock_last_used[self._key] = now
            self._storage._sweep_call_count += 1
            if self._storage._sweep_call_count >= self._storage._lock_sweep_interval:
                self._storage._sweep_call_count = 0
                self._storage._sweep_idle_locks_locked(now)
        await existing.acquire()
        self._acquired_lock = existing
        return True

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._acquired_lock is not None  # nosec B101
        self._acquired_lock.release()
        self._acquired_lock = None


class AsyncInMemoryStorage(AsyncStorageBackend):
    """Dict-backed storage (async).

    Plain dict plus one asyncio.Lock per key, created lazily and
    guarded by an asyncio.Lock protecting the lock-creation dict
    itself (see `_AsyncKeyLock`). As of Phase 8, also does the same
    opportunistic idle-sweep lock cleanup as the sync InMemoryStorage
    -- see storage.py's module docstring for the full rationale
    (shared between both), using `asyncio.Lock.locked()` (a plain,
    non-blocking check) instead of the sync version's
    acquire(blocking=False)/release() dance, since asyncio.Lock offers
    that check directly.

    SCOPE -- READ BEFORE SHARING AN INSTANCE:
    An AsyncInMemoryStorage instance is safe to share across coroutines
    running on ONE event loop in ONE process, and nothing wider than
    that:

    - NOT safe to share across OS threads. Its locks are asyncio.Lock
      instances, which are not thread-safe primitives.
    - NOT safe to share across multiple event loops, even within a
      single process. An asyncio.Lock is bound to the loop it was
      created under.
    - NOT safe to share across processes. There is no
      multiprocess_safe mode here -- see the module docstring for why
      asyncio.Lock and multiprocessing.Manager locks don't compose.

    If you need rate limiting shared across multiple processes (with
    or without async callers), that is out of scope for this class --
    it belongs in the Redis-native Lua/GCRA limiters (see
    limivault.redis_lua_scripts).
    """

    def __init__(
        self,
        lock_sweep_interval: int = _DEFAULT_LOCK_SWEEP_INTERVAL,
        lock_idle_seconds: float = _DEFAULT_LOCK_IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int | None = None,
        state_idle_seconds: float | None = None,
    ) -> None:
        if lock_sweep_interval <= 0:
            raise ValueError(
                f"lock_sweep_interval must be positive, got {lock_sweep_interval}"
            )
        if lock_idle_seconds <= 0:
            raise ValueError(
                f"lock_idle_seconds must be positive, got {lock_idle_seconds}"
            )
        _validate_bounds(max_keys, state_idle_seconds)
        self._max_keys = max_keys
        self._state_idle_seconds = state_idle_seconds
        self._bounded = max_keys is not None or state_idle_seconds is not None
        self._data: dict[str, dict[str, Any]] = (
            OrderedDict() if self._bounded else {}
        )
        self._state_last_used: dict[str, float] = {}
        self.evicted_count = 0
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._lock_last_used: dict[str, float] = {}
        self._sweep_call_count = 0
        self._lock_sweep_interval = lock_sweep_interval
        self._lock_idle_seconds = lock_idle_seconds
        self._clock = clock
        _warn_if_multi_process()

    def _sweep_idle_locks_locked(self, now: float) -> None:
        """Evict idle, currently-unlocked entries from `_locks`.
        PRECONDITION: caller already holds `_locks_guard`. See
        storage.py's module docstring for the shared rationale with
        the sync version's equivalent method."""
        for key in list(self._locks.keys()):
            last_used = self._lock_last_used.get(key, now)
            if now - last_used < self._lock_idle_seconds:
                continue
            lock = self._locks[key]
            if lock.locked():
                continue
            del self._locks[key]
            self._lock_last_used.pop(key, None)
        if self._state_idle_seconds is not None:
            self._sweep_idle_state(now)

    def _sweep_idle_state(self, now: float) -> None:
        """Async mirror of InMemoryStorage._sweep_idle_state. No extra
        guard is needed: this runs without awaiting, on one event loop."""
        assert self._state_idle_seconds is not None  # nosec B101
        for key in list(self._data.keys()):
            if now - self._state_last_used.get(key, now) < self._state_idle_seconds:
                break
            lock = self._locks.get(key)
            if lock is not None and lock.locked():
                continue
            del self._data[key]
            self._state_last_used.pop(key, None)
            self.evicted_count += 1

    async def get(self, key: str) -> dict[str, Any] | None:
        state = self._data.get(key)
        if state is not None and self._bounded:
            self._touch(key)
        return state

    async def set(self, key: str, state: dict[str, Any]) -> None:
        self._data[key] = state
        if not self._bounded:
            return
        self._touch(key)
        if self._max_keys is not None:
            while len(self._data) > self._max_keys:
                oldest = next(iter(self._data))
                del self._data[oldest]
                self._state_last_used.pop(oldest, None)
                self.evicted_count += 1

    def _touch(self, key: str) -> None:
        assert isinstance(self._data, OrderedDict)  # nosec B101
        self._data.move_to_end(key)
        self._state_last_used[key] = self._clock()

    def __len__(self) -> int:
        """Number of keys that currently have stored state."""
        return len(self._data)

    def lock(self, key: str) -> AsyncContextManager[bool]:
        return _AsyncKeyLock(self, key)
