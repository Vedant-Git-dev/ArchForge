"""Centralized logging for ArchForge.

A single named logger tree rooted at ``archforge``. By default the root
package logger carries a ``NullHandler`` only — this is the standard
library pattern: importing the package produces no log output unless the
application (here, the CLI entry point in ``archforge.main``) explicitly
calls :func:`configure_logging`. That keeps unit tests, which import
execution modules freely, quiet and deterministic.

Levels are picked from, in order of precedence:
    1. The ``level`` argument to :func:`configure_logging` (CLI ``--verbose``).
    2. The ``ARCHFORGE_LOG_LEVEL`` environment variable.
    3. ``INFO`` (the CLI default).

Use :func:`get_logger` from any module instead of importing ``logging``
directly, so every record lives under the ``archforge.`` namespace and
inherits the configured handler + level.
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "archforge"
LEVEL_ENV = "ARCHFORGE_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"

#: module-level guard so a second configure call doesn't stack handlers.
_configured = False


def _attach_null_handler() -> None:
    """Install a NullHandler on the root package logger so the library is
    silent when no application handler is configured."""
    root = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())
    # Stop records reaching Python's root logger — ArchForge owns its own
    # handler tree. Child loggers (archforge.engine, …) still propagate
    # *up* to here; we just don't let them spill past this node.
    root.propagate = False


def _resolve_level(level: str | int | None) -> int:
    """Coerce a level name/number (or env var) to a numeric logging level."""
    if level is None:
        level = os.environ.get(LEVEL_ENV, DEFAULT_LEVEL)
    if isinstance(level, int):
        return level
    candidate = str(level).strip().upper()
    numeric = logging.getLevelName(candidate)
    if isinstance(numeric, int):
        return numeric
    # Unknown name — fall back rather than crash the run.
    return logging.INFO


def configure_logging(level: str | int | None = None) -> int:
    """Configure the ``archforge`` logger to emit to stderr.

    Idempotent: safe to call multiple times (the CLI, a notebook, tests).
    Returns the resolved numeric level for the caller's convenience.
    """
    global _configured
    root = logging.getLogger(LOGGER_NAME)
    numeric_level = _resolve_level(level)
    root.setLevel(numeric_level)

    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        _configured = True
    else:
        # Already configured — keep the existing StreamHandler but honour a
        # newly requested level (e.g. a follow-up --verbose bump).
        for h in root.handlers:
            if not isinstance(h, logging.NullHandler):
                h.setLevel(numeric_level)

    return numeric_level


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``archforge.`` namespace."""
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


# Ensure the NullHandler is present the moment this module is imported,
# so any ``get_logger(...).debug(...)`` call before configure_logging is a
# silent no-op rather than a "No handlers could be found" warning.
_attach_null_handler()


__all__ = ["configure_logging", "get_logger", "LOGGER_NAME", "LEVEL_ENV"]
