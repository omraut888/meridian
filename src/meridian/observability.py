"""Structured logging setup.

Every module logs through ``structlog.get_logger(__name__)``. Context such as a
request ID is bound with :func:`structlog.contextvars.bind_contextvars` and is
attached to every log line emitted while handling that request, including lines
from third-party libraries routed through the stdlib ``logging`` module.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """Configure structlog and route stdlib logging through the same pipeline.

    Args:
        level: Minimum log level name, e.g. ``"INFO"``.
        json: Emit JSON lines when true; human-readable console output otherwise.
    """
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # Chatty third-party loggers stay at WARNING unless we are debugging.
    for noisy in ("httpx", "httpcore", "httpx2", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if level == "DEBUG" else logging.WARNING)
