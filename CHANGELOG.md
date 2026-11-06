# Changelog

## 0.1.2 - Unreleased

### Added
- `BackendBusyError` (subclass of `BackendUnavailableError`) raised by every Redis limiter when the redis-py connection pool is exhausted (`MaxConnectionsError`, or a `BlockingConnectionPool` timeout). `BackendErrorEvent` gains `kind` (`"busy"` / `"unavailable"`). Prevents apps that fail open on outages from admitting whole bursts unchecked while Redis is healthy.
- `MultiProcessStateWarning`, emitted once per process when in-memory storage appears to run in one of several workers. Silence with `LIMIVAULT_SILENCE_MULTIPROCESS_WARNING=1`.
- `max_keys` and `state_idle_seconds` options on `InMemoryStorage` / `AsyncInMemoryStorage` to bound per-key state; `len(storage)` and `storage.evicted_count`.

### Changed
- In-memory limiters wrap their clock so it never goes backwards. A backwards jump now freezes time instead of granting a fresh quota (fixed window and sliding window counter previously over-admitted).
- The `allow()` contract and README now state that Redis limiters raise `BackendUnavailableError` / `BackendBusyError`; "never raises" applies to bad input only. The README Redis example uses a `BlockingConnectionPool` and shows the error handling.
## 0.1.1 - 2026-10-08

- Fixed README badges and links so they display correctly on PyPI.

## 0.1.0 - 2026-10-07

### Added
- End-to-end checks for the public API, CLI, backend parity, and simulation-to-CSV flows.
- Multi-job GitHub Actions for Linux, Windows, Redis containers, and wheel installs.
- Package classifiers and a documented known-limitations section in the README.

### Changed
- Root imports lazily expose algorithms, exceptions, metrics, ergonomics, and simulation APIs.
- Redis-backed rate limiting uses direct Lua algorithms alongside in-memory implementations.

### Fixed
- Normalised mixed CRLF/LF line endings and ignored parallel coverage data files.

### Removed
- The lock-backed Redis storage implementation and its dedicated tests.
