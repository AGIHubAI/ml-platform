"""Structured logging (structlog) with correlation ids.

Usage::

    from ml_platform.logging import get_logger, bind_ids
    log = get_logger(run_id="abc", round_id=3)
    log.info("training.epoch_end", epoch=1, val_rmse=0.4)

    with bound_ids(round_id=3, dataset_id="..."):   # contextvars; visible to all loggers
        ...

Renderer: ``ML_LOG_FORMAT=json`` gives one JSON object per line (for
aggregation), anything else (default ``console``) pretty console output.
``ML_LOG_LEVEL`` sets the level (default INFO).
"""

from __future__ import annotations

import logging as _stdlib_logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

CORRELATION_IDS = ("run_id", "round_id", "dataset_id", "model_version", "ray_job_id")

_configured = False


class _StderrProxy:
    """Resolve ``sys.stderr`` at write time (robust to stream swapping, e.g. pytest capture)."""

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()


def configure_logging(
    fmt: str | None = None, level: str | None = None, force: bool = False
) -> None:
    global _configured
    if _configured and not force:
        return
    fmt = (fmt or os.environ.get("ML_LOG_FORMAT", "console")).lower()
    level_name = (level or os.environ.get("ML_LOG_LEVEL", "INFO")).upper()
    level_no = getattr(_stdlib_logging, level_name, _stdlib_logging.INFO)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer(sort_keys=True, default=str)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=_StderrProxy()),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str | None = None, **bound_ids: Any) -> Any:
    """Return a structlog logger with the given ids bound (only non-None values)."""
    configure_logging()
    log = structlog.get_logger(name) if name else structlog.get_logger()
    ids = {k: v for k, v in bound_ids.items() if v is not None}
    if name:
        ids.setdefault("logger", name)
    return log.bind(**ids) if ids else log


def bind_ids(**ids: Any) -> None:
    """Bind correlation ids into the current context (all loggers see them)."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in ids.items() if v is not None})


def clear_ids(*names: str) -> None:
    if names:
        structlog.contextvars.unbind_contextvars(*names)
    else:
        structlog.contextvars.clear_contextvars()


@contextmanager
def bound_ids(**ids: Any) -> Iterator[None]:
    """Context manager that binds ids for the duration of a block."""
    tokens = structlog.contextvars.bind_contextvars(
        **{k: v for k, v in ids.items() if v is not None}
    )
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
