"""Structured logging setup for rlimit.

Configures structlog to emit JSON log lines. Every event includes:
- timestamp (ISO 8601, UTC)
- level (log level name)
- logger (name of the logger that emitted the event)
- event (the log message / event name)
- key (the rate-limiter key being acted on, when applicable)
- count (current request count for that key, when applicable)
- any additional context passed via kwargs

Usage:
    from rlimit.logging import get_logger

    log = get_logger(__name__)
    log.info("request_allowed", key="user:123", count=5, limit=10)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog for JSON output.

    Call this once, early in the application/library lifecycle
    (e.g. at process startup). Safe to call multiple times; the
    last call wins.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        # Keep logger proxies live so a later configure_logging() call (or
        # structlog.testing.capture_logs()) can change the active processors
        # and level even when algorithm modules imported their loggers first.
        cache_logger_on_first_use=False,
    )


def get_logger(name: str, **initial_context: Any) -> structlog.BoundLogger:
    """Return a structlog logger bound with a `logger` name field.

    Additional keyword arguments are bound as permanent context
    (e.g. get_logger(__name__, key="user:123")).
    """
    return cast(structlog.BoundLogger, _LiveLogger(name, initial_context))


class _LiveLogger:
    """Resolve the structlog logger when an event is emitted.

    Algorithm modules keep these objects at import time. Resolving the
    underlying bound logger per method access lets later configuration and
    temporary processors such as ``capture_logs()`` take effect.
    """

    __slots__ = ("_name", "_initial_context")

    def __init__(self, name: str, initial_context: dict[str, Any]) -> None:
        self._name = name
        self._initial_context = dict(initial_context)

    def __getattr__(self, method_name: str) -> Any:
        values = {"logger": self._name, **self._initial_context}
        bound_logger = structlog.get_logger().bind(**values)
        return getattr(bound_logger, method_name)


# Structlog's unconfigured default does not filter DEBUG events. Limiters emit
# per-decision details at DEBUG, so install a quiet default only when the
# application has not configured structlog itself. Callers can enable these
# events with configure_logging(logging.DEBUG).
if not structlog.is_configured():
    configure_logging(level=logging.WARNING)
