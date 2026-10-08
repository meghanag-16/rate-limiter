# Changelog
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
