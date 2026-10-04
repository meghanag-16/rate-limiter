# rlimit

Rate limiting for Python 3.14+, with synchronous and asynchronous APIs, in-memory and Redis Lua backends, and metrics hooks.

## Quick start

Install for development with `pip install -e ".[dev]"`.

```python
from rlimit import FixedWindow

limiter = FixedWindow(limit=100, period=60)
allowed = limiter.allow("user-123")
```

Redis example (requires a reachable Redis server):

```python
import redis
from rlimit import RedisGcraTokenBucket

client = redis.Redis.from_url("redis://localhost:6379")
limiter = RedisGcraTokenBucket(client, capacity=100, refill_rate=10)
allowed = limiter.allow("user-123")
```

## Logging

Direct library use suppresses per-decision DEBUG messages by default. To
enable them, configure logging before importing an algorithm:

```python
import logging
from rlimit.logging import configure_logging

configure_logging(level=logging.DEBUG)

from rlimit import FixedWindow
```

The command-line tools keep their existing logging setup.

## Algorithms

Each algorithm has sync/async in-memory and Redis Lua implementations.

| Algorithm | Behavior | Tradeoff |
| --- | --- | --- |
| Fixed window | Counts within period-aligned windows | Simple; boundary bursts are possible |
| Token bucket | Allows bursts and refills at a rate | Smooth refill; the Redis version stores one GCRA timestamp per key |
| Sliding window log | Counts timestamped requests in a moving window | Exact; memory and work grow with the number of requests in the window |
| Sliding window counter | Weights the previous window | Efficient approximation near boundaries |
| Leaky bucket meter | Tracks and drains continuous volume | Smooth capacity accounting |
| Leaky bucket queue | Drains whole request units | Discrete queue semantics |

Redis decisions are atomic (one Lua script call per decision) and use Redis server time. `allow_wait()` is advisory, not a reservation.

## CLI

```sh
rlimit demo --algorithm fixed_window --param1 5 --param2 60 --requests 8
rlimit lua-demo --algorithm fixed_window --param1 5 --param2 60 --redis-host localhost
rlimit benchmark -- --help
rlimit lua-benchmark -- --help
rlimit memory
```

`demo` and `lua-demo` work from any install. `benchmark`, `lua-benchmark` and `memory` need a source checkout or editable install, because `benchmarks/` is not included in the wheel. Outside a checkout they print a clear message and exit with status 1. `lua-demo` and `lua-benchmark` need a reachable Redis; start a local one with `docker compose -f docker-compose.redis.yml up -d`.

## Benchmarks

Run these from the repository root after `pip install -e ".[dev]"`.

```sh
# In-memory async limiters (no Redis needed)
rlimit benchmark -- --concurrency 1 10 100 --num-keys 1 100 --repetitions 3

# Redis Lua limiters (needs Redis on --redis-host/--redis-port, default localhost:6379)
rlimit lua-benchmark -- --concurrency 1 10 100 --repetitions 3

# Steady-state memory per key for the in-memory limiters
rlimit memory --n-keys 100 1000
```

Both benchmark scripts sweep algorithm x concurrency x number of keys, repeat each cell, and report throughput plus p50/p95/p99 latency as the median across repetitions. Pass `--output-csv PATH` (after `--`) to save results. Run `rlimit benchmark -- --help` or `rlimit lua-benchmark -- --help` for every flag.

Benchmark results are not committed because they depend heavily on the machine, Python build, workload, and Redis setup. Run the benchmarks on your own hardware; each Redis decision includes a network round trip, so compare results using the same host and workload.

By default the benchmarks use a very large quota so that almost every request is allowed, and the in-memory benchmark exposes `--limit` and `--rate` to change that. The Lua benchmark has no such flag; see `benchmarks/lua_bench_common.py` if you want it to deny.

## Testing

```sh
ruff check src tests benchmarks
mypy src tests benchmarks
bandit -c pyproject.toml -r src

# Fast lane: no Docker needed
python -m coverage run -m pytest -m "not docker and not e2e"
python -m coverage combine
python -m coverage report -m

# Redis and end-to-end lane: needs a running Docker daemon
pytest -m "docker or e2e"
```

`tests/conftest.py` marks every test that uses a Redis fixture as `docker`. Those tests start a throwaway `redis:7-alpine` container through testcontainers and fail (rather than skip) if Docker is not available. The `e2e` tests in `tests/e2e/` drive the public API, the CLI (through subprocesses), in-memory versus Redis parity, and the simulation-to-CSV flow.

GitHub Actions (`.github/workflows/ci.yml`) runs lint/type/security checks, the fast lane on Ubuntu and Windows, the Redis and end-to-end lane on Ubuntu, and a wheel-install smoke test. See CONTRIBUTING.md.

## Known limitations

These are deliberate scope decisions or documented tradeoffs, not hidden bugs.

**Backends and deployment**
- In-memory limiters keep state inside one process. Two processes each get their own independent limit. `InMemoryStorage` has an opt-in `multiprocess_safe=True` mode (backed by a `multiprocessing.Manager`) for sharing state between processes on one machine; it is slower than the default. For sharing limits across machines, use the Redis limiters.
- The Redis backend is the Lua/GCRA family only. Each limiter key maps to one Redis key (prefix `rlimit:lua:` by default), and decisions use Redis server time, so client clocks do not matter.
- It has been tested against a single Redis 7 server (`redis:7-alpine` in the test suite). Redis Cluster, Sentinel and other Redis versions have not been tested.
- When Redis is unreachable, Redis limiters raise `BackendUnavailableError`. The library does not choose between fail-open and fail-closed for you; catch the exception and decide.
- Each Redis decision costs a network round trip, so it is much slower than the in-memory path (see Benchmarks). The sliding window log does work proportional to the number of entries in its window, so its cost grows with traffic on a single key, in memory and in Redis.
- The Redis tests need Docker with Linux containers. There is no fallback for machines without Docker.

**In-memory storage**
- Per-key locks are created lazily and removed by an opportunistic idle sweep (every 1000 lock lookups, for locks idle at least 300 seconds by default; both are constructor arguments). Memory is proportional to recent key count rather than all-time key count, but there is no hard cap between sweeps.
- A narrow race between the sweep and a caller about to acquire a lock is documented and accepted in the `rlimit/storage.py` docstring.

**API behavior**
- `allow_wait()` only reports how long to wait. It does not reserve capacity, so another caller can take it first. `block_until_allowed` and `async_block_until_allowed` retry in a loop for this reason.
- `allow(key, cost=1)` takes an integer cost. A `cost` of 0 is a no-op that returns `True`. An invalid cost (non-integer, boolean, negative) or one larger than capacity makes `allow()` return `False` rather than raise. `allow_wait()` is stricter: it raises `ValueError` for an invalid cost and `UnsatisfiableRequestError` when the cost can never be satisfied.

**Metrics and observability**
- Metrics hooks receive events only from `allow()` decisions. `allow_wait()`, invalid-cost calls and zero-cost calls emit nothing.
- Events are emitted after the per-key lock is released, so a hook can observe them in a different order than requests were admitted. Counters are unaffected; do not rely on arrival order.
- Logging is unconfigured by default and noisy (see Logging above).

**Packaging**
- Requires Python 3.14 or newer.
- `benchmarks/` is not shipped in the wheel, so the benchmark and memory commands need a source checkout.

**Not built**
- A hybrid limiter that switches algorithm based on load, an adaptive token bucket with decaying burst allowance, and configurable overflow behavior (reject, queue-and-delay, degrade) were descoped. See the `simulation.py` module docstring.
