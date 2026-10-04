# src/rlimit/redis_lua_scripts.py
"""Lua script sources for Redis-native Lua/GCRA
limiters (see each algorithms/redis_lua_*.py / async_redis_lua_*.py
file for the concrete classes).

--------------------------------------------------------------------
CHANGELOG -- v4 (this pass): TTL MUST BE AN INTEGER BEFORE EXPIRE
--------------------------------------------------------------------
Every script parses `ttl = tonumber(ARGV[5])` and passes it straight
to `redis.call('EXPIRE', key, ttl)`. Redis's EXPIRE requires an
integer number of seconds -- a fractional ttl (routine in practice,
since ttl_seconds is normally computed as e.g. `period * 2 +
TTL_BUFFER_SECONDS` or `capacity / refill_rate + TTL_BUFFER_SECONDS`,
which very often lands on a non-whole number) made Redis reject the
call with `ResponseError: value is not an integer or out of range`.
That failure was then caught by the same broad except clause used for
genuine connection failures and re-raised as BackendUnavailableError
-- a misleading "backend is down" error for what was actually a
malformed argument. Fixed by rounding ttl up (not down, so a key is
never evicted earlier than the caller configured) the moment it's
parsed: `local ttl = math.ceil(tonumber(ARGV[5]))`, in all six
scripts. Every later use of `ttl` in that script (there may be more
than one EXPIRE call site per script, e.g. both the deny and admit
branches) is covered by this single top-of-script change.

--------------------------------------------------------------------
CHANGELOG -- v3: ONE ROUND TRIP, NOT TWO, FOR METRICS
--------------------------------------------------------------------
v2 computed `now` server-side inside the script (via `TIME`, see
below) for the algorithm's own admit/deny decision, but never handed
that value back to the caller -- so every allow() in
algorithms/redis_lua_*.py made a SECOND `TIME` round trip afterward
just to stamp the metrics event's `timestamp` field. That quietly
turned Redis Lua/GCRA's "one round trip per allow()" story into two, and
the two `now` readings weren't even guaranteed to be the same instant.

Fixed by widening every script's return array from 3 elements to 4:
`{allowed_int, cost_applied, current_or_resulting_quota_value, now}`.
The script already had `now` computed as its very first step (the
shared `_NOW_PREAMBLE` below); returning it costs nothing extra
server-side. Every algorithms/redis_lua_*.py / async_redis_lua_*.py
file now reads `result[3]` for its metrics timestamp instead of
issuing a second `client.time()` call -- see each file's `allow()`.
`remaining()` (mode="1" peek, no metrics emitted) and `allow_wait()`
(never goes through the script at all, still does its own plain
read + one `TIME` call, since it isn't inside the atomic script) are
unaffected by this change.

--------------------------------------------------------------------
CHANGELOG -- v2: ONE SHARED CLOCK, LOW-RISK FIXES
--------------------------------------------------------------------
This revision responds directly to a review of the first Redis Lua/GCRA
pass. Three changes from v1:

1. DISTRIBUTED CLOCK FIX -- "ONE CLOCK FOR BOTH [THE ATOMIC WRITE PATH
   AND THE ADVISORY READ PATH]": v1 passed the caller's own injected
   `clock()` reading into each script as `now` via ARGV, exactly like
   every in-memory algorithm in this project. That is wrong
   for a genuinely distributed deployment: `time.monotonic()` is
   process-local and not comparable across machines, so two different
   processes calling the same Redis-native limiter from two different
   hosts would each be feeding the script a different, incomparable
   notion of "now" -- silently corrupting every time-based decision
   (window boundaries, refill/leak math, TTL) the moment more than one
   machine is involved. Fixed by dropping client-side clock injection
   for this family of classes ENTIRELY and having every script compute
   `now` itself, once, via `redis.call('TIME')` -- Redis's own server
   clock, the same single source of truth for every caller regardless
   of which machine or process it's running on. The "for both" part:
   the READ-ONLY side of these classes (allow_wait()'s Python-side
   wait-time estimate, which -- see below -- is NOT computed inside a
   script) now also fetches `now` via a plain `TIME` command against
   the same Redis server, rather than a local clock, so the mutating
   path (inside the script) and the advisory estimate path (in Python)
   are never anchored to two different clocks. Previously it would
   have been possible for allow()'s decision (driven by the injected
   clock, if v1's ARGV[1] `now` were kept but only used inconsistently
   from a different code path) and allow_wait()'s estimate to disagree
   about what time it currently is; that class of bug is now
   structurally impossible.

   CALLING `redis.call('TIME')` FROM A SCRIPT -- WHY THIS IS SAFE:
   Historically, calling a nondeterministic command like TIME from a
   Lua script was flagged as dangerous because Redis used to replicate
   scripts VERBATIM to replicas/AOF, and a script re-executed on a
   replica at a different wall-clock moment could diverge from the
   master. That concern does not apply here: Redis 5.0+ replicates
   scripts by EFFECT (the actual writes the script performed), not by
   re-running the script's source on the replica -- this has been the
   only replication mode since Redis 5 and is what `redis:7-alpine`
   (the version this project's Redis test fixtures pin) uses. Calling
   TIME() inside these scripts is therefore safe and does not risk
   master/replica divergence.

   CONSEQUENCE FOR TESTING, STATED EXPLICITLY: this project's
   established testing convention -- inject a FakeClock, freeze/
   advance it deterministically, assert exact boundary behavior -- is
   NOT available for this family of classes anymore, because there is
   no client-side clock left to inject. Boundary and TTL tests for
   this family (see tests/test_redis_lua_*.py) use REAL wall-clock
   time and real `time.sleep()` calls instead. This
   is a real, deliberate tradeoff -- accepted here because a
   deterministic fake clock is fundamentally incompatible with "the
   whole point is that every distributed caller shares one real clock"
   -- not an oversight.

2. HSET replaces the deprecated HMSET. Same argument order
   (`HSET key field value field value ...`), no behavior change --
   Redis still supports HMSET, but it's been deprecated since Redis
   4.0 in favor of a single HSET that accepts multiple field/value
   pairs. Pure cleanup, flagged as low-risk in review.

3. Every class's `ttl_seconds` (whether computed automatically or
   passed explicitly) is now validated the same way every other
   numeric parameter in this project is -- positive and finite -- at
   construction time, since an unvalidated ttl_seconds flows straight
   into EXPIRE and a bad value (zero, negative, NaN, infinity) there
   would produce confusing Redis-level behavior instead of a clear
   Python-level ValueError. See each algorithms/redis_lua_*.py file's
   `_validate_ttl_seconds` call in `__init__`.

--------------------------------------------------------------------
WHY THIS EXISTS AT ALL
--------------------------------------------------------------------
Each algorithm's entire read-compute-write sequence is expressed as a
single Lua script, executed atomically server-side via EVALSHA
(register_script() handles the EVALSHA/EVAL-on-NOSCRIPT fallback
internally in redis-py) -- 1 round trip, no separate lock object, no
lock TTL, no LockNotOwnedError classification to get wrong. The
alternative this design avoids is a per-key client-side distributed
lock wrapping a plain GET -> compute in Python -> SET sequence, which
costs 4 network round trips per allow() call plus lock-contention
retry sleeps.

STANDALONE, COEXISTING: these are standalone Redis-native classes
that live ALONGSIDE the in-memory algorithms, under their own key
namespace (see DEFAULT_KEY_PREFIX below). They are not a
StorageBackend implementation: they talk to a redis.Redis /
redis.asyncio.Redis client directly. See each algorithms/redis_lua_*.py
file's own docstring for the "standalone class, not StorageBackend"
rationale.

NO CLAIM OF "FASTER" IS MADE ANYWHERE IN THIS MODULE OR ITS CALLERS.
"Fewer round trips" and "no client-side lock" are demonstrably true
from the code; whether that translates to better measured throughput/
latency is a benchmark question, not yet answered as of this pass.

--------------------------------------------------------------------
SHARED CONVENTIONS ACROSS ALL SIX SCRIPTS (v2 ARGV layout)
--------------------------------------------------------------------
- KEYS[1] is always the single Redis key holding that limiter key's
  state (a Hash for the counter-style algorithms, a plain String for
  GCRA's single TAT value, a Sorted Set for SlidingWindowLog).
- `now` is NEVER an argument anymore (see CHANGELOG item 1) -- every
  script computes it itself via `redis.call('TIME')` as its very first
  step.
- ARGV[1] is always `cost`.
- ARGV[2] is always `mode`: "0" for a real (state-mutating) allow()
  decision, "1" for a read-only "peek" used by remaining(). Peek mode
  still performs cheap, correct reads (e.g. pruning expired
  SlidingWindowLog entries is still a real mutation even in peek mode
  -- see that script's own comment -- but it never writes the
  *requested* entry/count).
- ARGV[3..] are the algorithm's own parameters (limit/period,
  capacity/refill_rate, capacity/leak_rate), in each script's own
  documented order, followed by `ttl_seconds` as the last positional
  parameter before any algorithm-specific extras (SlidingWindowLog's
  unique member suffix).
- Every script returns a 4-element array: {allowed_int, cost_applied,
  current_or_resulting_quota_value, now} -- `now` was added in v3 so
  callers get the script's own server-time reading for free instead
  of paying for a second `TIME` round trip (see CHANGELOG above). See
  each algorithms/redis_lua_*.py file and tests/test_redis_lua_script_
  contract.py for the Level 2 tests pinning this exact shape.
- Every script sets an EXPIRE on its key after any write, sized to a
  multiple of the algorithm's own characteristic time (period, or
  capacity/rate) plus a small fixed buffer -- see
  `TTL_BUFFER_SECONDS` below.

KNOWN, DOCUMENTED GAP -- GCRA WITH refill_rate == 0: unchanged from
v1 -- see redis_lua_token_bucket.py's own docstring.

KNOWN, DOCUMENTED GAP -- SlidingWindowLog remains O(n) per call:
unchanged from v1 -- see redis_lua_sliding_window_log.py's own
docstring and changes_needed.txt's original analysis.

KNOWN, DOCUMENTED GAP -- allow_wait() IS NOT ATOMIC, FOR ANY
ALGORITHM HERE: allow_wait() never goes through the Lua script at all
(it never mutates state, so it doesn't need the atomicity guarantee
allow() needs) -- it does a direct, plain read (GET/HMGET/ZRANGE) plus
a `TIME` call, then reproduces the algorithm's wait-time formula in
Python. Between that read and whatever the caller does with the
returned wait, ANY other caller (same process, another process, or a
process on another machine) can freely call allow() and change the
state the estimate was based on. This is explicitly the same
"ADVISORY, NOT A RESERVATION" contract documented in base.py for the
whole project, restated here because it is easy to assume a Redis-
backed, atomicity-flavored implementation makes allow_wait() safe to
treat as a reservation -- it does not, and SlidingWindowLog's
allow_wait() is a sharper example: it performs two separate,
non-atomic Redis operations (ZREMRANGEBYSCORE then ZRANGE) before even
starting its Python-side math, so the window a concurrent write can
land in is wider than for the single-command-read algorithms. Every
class's `allow_wait()` docstring restates this explicitly -- see each
algorithms/redis_lua_*.py file.
--------------------------------------------------------------------
"""

from __future__ import annotations

# Own key namespace for this family -- see module docstring's
# STANDALONE, COEXISTING section.
DEFAULT_KEY_PREFIX = "rlimit:lua:"

# Added on top of each algorithm's own characteristic time when
# computing a key's EXPIRE. Purely a safety margin, not derived from
# any measured production value -- same "conservative placeholder,
# explicitly overridable" spirit as storage.py's
# _DEFAULT_LOCK_IDLE_SECONDS.
TTL_BUFFER_SECONDS = 5.0

# Fallback TTL (seconds) used only when an algorithm's own
# characteristic time can't be computed (e.g. TokenBucket/LeakyBucket
# with a zero rate -- see module docstring's GCRA gap note). Also a
# placeholder, overridable by constructing any of these classes with
# an explicit `ttl_seconds=` argument.
FALLBACK_TTL_SECONDS = 3600.0


# Shared preamble, textually inlined into every script below (Lua has
# no #include -- duplicating these three lines six times is simpler
# and more transparent than a script-composition layer for a project
# this size).
_NOW_PREAMBLE = r"""
local __t = redis.call('TIME')
local now = tonumber(__t[1]) + (tonumber(__t[2]) / 1000000.0)
"""


# ---------------------------------------------------------------------------
# GCRA TokenBucket
# ---------------------------------------------------------------------------
# KEYS[1] = data key (Redis String holding the TAT, or, in the
#           refill_rate==0 special case, a plain used-cost counter --
#           see module docstring's GCRA gap note).
# ARGV[1] = cost
# ARGV[2] = mode ("0" mutate, "1" peek)
# ARGV[3] = capacity
# ARGV[4] = refill_rate (tokens/sec; may be 0)
# ARGV[5] = ttl_seconds
GCRA_TOKEN_BUCKET = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local capacity = tonumber(ARGV[3])
local refill_rate = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))

if refill_rate <= 0 then
    local used = tonumber(redis.call('GET', key) or '0')
    if used + cost > capacity then
        if mode == "0" then
            redis.call('EXPIRE', key, ttl)
        end
        return {0, 0, capacity - used, now}
    end
    if mode == "0" then
        redis.call('SET', key, used + cost)
        redis.call('EXPIRE', key, ttl)
        return {1, cost, capacity - (used + cost), now}
    else
        return {1, cost, capacity - used, now}
    end
end

local emission_interval = 1.0 / refill_rate
local burst_offset = capacity * emission_interval

local tat_raw = redis.call('GET', key)
local tat
if tat_raw then
    tat = tonumber(tat_raw)
else
    tat = now
end
if tat < now then
    tat = now
end

local increment = cost * emission_interval
local new_tat = tat + increment
local allow_at = new_tat - burst_offset

if allow_at > now then
    local tokens_now = capacity - math.max(
        0, math.ceil((tat - now) / emission_interval)
    )
    return {0, 0, tokens_now, now}
end

if mode == "0" then
    redis.call('SET', key, new_tat)
    redis.call('EXPIRE', key, ttl)
end
local tokens_after = capacity - math.max(
    0, math.ceil((new_tat - now) / emission_interval)
)
return {1, cost, tokens_after, now}
"""


# ---------------------------------------------------------------------------
# FixedWindow
# ---------------------------------------------------------------------------
# KEYS[1] = data key (Hash: window_start, count)
# ARGV[1] = cost, ARGV[2] = mode, ARGV[3] = limit, ARGV[4] = period,
# ARGV[5] = ttl_seconds
FIXED_WINDOW = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local limit = tonumber(ARGV[3])
local period = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))

local window_start = math.floor(now / period) * period

local data = redis.call('HMGET', key, 'window_start', 'count')
local stored_window = tonumber(data[1])
local count = tonumber(data[2]) or 0
if stored_window == nil or stored_window ~= window_start then
    count = 0
end

if count + cost > limit then
    if mode == "0" then
        redis.call('HSET', key, 'window_start', window_start, 'count', count)
        redis.call('EXPIRE', key, ttl)
    end
    return {0, 0, limit - count, now}
end

if mode == "0" then
    redis.call('HSET', key, 'window_start', window_start, 'count', count + cost)
    redis.call('EXPIRE', key, ttl)
    return {1, cost, limit - (count + cost), now}
else
    return {1, cost, limit - count, now}
end
"""


# ---------------------------------------------------------------------------
# SlidingWindowCounter (weighted approximation)
# ---------------------------------------------------------------------------
# KEYS[1] = data key (Hash: window_start, count, prev_count)
# ARGV[1] = cost, ARGV[2] = mode, ARGV[3] = limit, ARGV[4] = period,
# ARGV[5] = ttl_seconds
SLIDING_WINDOW_COUNTER = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local limit = tonumber(ARGV[3])
local period = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))

local window_start = math.floor(now / period) * period

local data = redis.call('HMGET', key, 'window_start', 'count', 'prev_count')
local stored_window = tonumber(data[1])
local count = tonumber(data[2]) or 0
local prev_count = tonumber(data[3]) or 0

if stored_window == nil then
    count = 0
    prev_count = 0
elseif stored_window == window_start then
    -- same window, count/prev_count already correct
elseif stored_window == window_start - period then
    prev_count = count
    count = 0
else
    count = 0
    prev_count = 0
end

local elapsed = now - window_start
local overlap = math.max(0.0, (period - elapsed) / period)
local weighted = count + prev_count * overlap

if weighted + cost > limit then
    if mode == "0" then
        redis.call(
            'HSET', key, 'window_start', window_start, 'count', count,
            'prev_count', prev_count
        )
        redis.call('EXPIRE', key, ttl)
    end
    return {0, 0, limit - weighted, now}
end

if mode == "0" then
    redis.call(
        'HSET', key, 'window_start', window_start, 'count', count + cost,
        'prev_count', prev_count
    )
    redis.call('EXPIRE', key, ttl)
    return {1, cost, limit - (weighted + cost), now}
else
    return {1, cost, limit - weighted, now}
end
"""


# ---------------------------------------------------------------------------
# LeakyBucketMeter (continuous volume)
# ---------------------------------------------------------------------------
# KEYS[1] = data key (Hash: volume, last_leak)
# ARGV[1] = cost, ARGV[2] = mode, ARGV[3] = capacity, ARGV[4] = leak_rate,
# ARGV[5] = ttl_seconds
LEAKY_BUCKET_METER = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local capacity = tonumber(ARGV[3])
local leak_rate = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))

local data = redis.call('HMGET', key, 'volume', 'last_leak')
local volume = tonumber(data[1]) or 0.0
local last_leak = tonumber(data[2])
if last_leak == nil then
    last_leak = now
    volume = 0.0
end

local elapsed = math.max(0.0, now - last_leak)
local leaked = elapsed * leak_rate
volume = math.max(0.0, volume - leaked)

if volume + cost > capacity then
    if mode == "0" then
        redis.call('HSET', key, 'volume', volume, 'last_leak', now)
        redis.call('EXPIRE', key, ttl)
    end
    return {0, 0, capacity - volume, now}
end

if mode == "0" then
    redis.call('HSET', key, 'volume', volume + cost, 'last_leak', now)
    redis.call('EXPIRE', key, ttl)
    return {1, cost, capacity - (volume + cost), now}
else
    return {1, cost, capacity - volume, now}
end
"""


# ---------------------------------------------------------------------------
# LeakyBucketQueue (discrete whole-item drains)
# ---------------------------------------------------------------------------
# KEYS[1] = data key (Hash: depth, last_drain)
# ARGV[1] = cost, ARGV[2] = mode, ARGV[3] = capacity, ARGV[4] = leak_rate,
# ARGV[5] = ttl_seconds
LEAKY_BUCKET_QUEUE = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local capacity = tonumber(ARGV[3])
local leak_rate = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))

local data = redis.call('HMGET', key, 'depth', 'last_drain')
local depth = tonumber(data[1]) or 0
local last_drain = tonumber(data[2])
if last_drain == nil then
    last_drain = now
    depth = 0
end

local elapsed = math.max(0.0, now - last_drain)
local drained = 0
if leak_rate > 0 then
    drained = math.floor(elapsed * leak_rate)
end
depth = math.max(0, depth - drained)
if drained > 0 and leak_rate > 0 then
    last_drain = last_drain + (drained / leak_rate)
end

if depth + cost > capacity then
    if mode == "0" then
        redis.call('HSET', key, 'depth', depth, 'last_drain', last_drain)
        redis.call('EXPIRE', key, ttl)
    end
    return {0, 0, capacity - depth, now}
end

if mode == "0" then
    redis.call('HSET', key, 'depth', depth + cost, 'last_drain', last_drain)
    redis.call('EXPIRE', key, ttl)
    return {1, cost, capacity - (depth + cost), now}
else
    return {1, cost, capacity - depth, now}
end
"""


# ---------------------------------------------------------------------------
# SlidingWindowLog -- Redis Sorted Set. See module docstring's
# "KNOWN, DOCUMENTED GAP" section: still O(n), now 1 round trip
# instead of 4.
# ---------------------------------------------------------------------------
# KEYS[1] = zset key
# ARGV[1] = cost, ARGV[2] = mode, ARGV[3] = limit, ARGV[4] = period,
# ARGV[5] = ttl_seconds, ARGV[6] = unique suffix for this call's member
#           (only used if admitted; pass "" in peek mode)
#
# Member encoding: "<cost>:<now>:<unique_suffix>" -- see v1 docstring
# for why cost travels as a parseable prefix rather than a second
# structure.
SLIDING_WINDOW_LOG = _NOW_PREAMBLE + r"""
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local mode = ARGV[2]
local limit = tonumber(ARGV[3])
local period = tonumber(ARGV[4])
local ttl = math.ceil(tonumber(ARGV[5]))
local suffix = ARGV[6]

local cutoff = now - period
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

local members = redis.call('ZRANGE', key, 0, -1)
local total = 0
for i = 1, #members do
    local m = members[i]
    local sep = string.find(m, ':')
    local entry_cost = tonumber(string.sub(m, 1, sep - 1))
    total = total + entry_cost
end

if total + cost > limit then
    if mode == "0" then
        redis.call('EXPIRE', key, ttl)
    end
    return {0, 0, limit - total, now}
end

if mode == "0" then
    local member = cost .. ':' .. tostring(now) .. ':' .. suffix
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, ttl)
    return {1, cost, limit - (total + cost), now}
else
    return {1, cost, limit - total, now}
end
"""
