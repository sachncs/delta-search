"""Logging and seeding helpers for delta-search.

Implements the env-var contract documented in the README:

- ``DELTA_SEARCH_LOG_LEVEL`` -- log level (``DEBUG``, ``INFO``, ``WARNING``,
  ``ERROR``, or a numeric level).  Default ``WARNING``.
- ``DELTA_SEARCH_SEED`` -- default RNG seed for stochastic solvers when
  none is supplied to the constructor.

Library code never calls ``logging.basicConfig`` (which would mutate the
root logger).  Instead it installs a :class:`logging.NullHandler` on the
``delta_search`` package logger per the standard Python library
recommendation, and lets the embedding application configure its own
root logger if it wants output.
"""

from __future__ import annotations

import logging
import os
from typing import Final

__all__ = [
    "configure_logging",
    "get_logger",
    "default_seed",
]

_PACKAGE_LOGGER_NAME: Final = "delta_search"
_LOG_LEVEL_ENV: Final = "DELTA_SEARCH_LOG_LEVEL"
_SEED_ENV: Final = "DELTA_SEARCH_SEED"

_LEVEL_NAMES: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_log_level() -> int:
    """Resolve the configured log level from the env var.

    Returns:
        The numeric logging level to use.
    """
    raw = os.environ.get(_LOG_LEVEL_ENV)
    if raw is None or raw == "":
        return logging.WARNING
    raw = raw.strip().upper()
    if raw in _LEVEL_NAMES:
        return _LEVEL_NAMES[raw]
    try:
        return int(raw)
    except ValueError:
        return logging.WARNING


def _package_logger() -> logging.Logger:
    """Return the ``delta_search`` package logger, with NullHandler installed.

    Returns:
        The package logger, configured to respect the standard library
        NullHandler pattern so we don't pollute the root logger.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(level: int | str | None = None) -> None:
    """Configure the ``delta_search`` package logger level.

    Args:
        level: Optional explicit level.  When ``None`` (default) the
            level is read from ``DELTA_SEARCH_LOG_LEVEL`` *if set*; if
            the env var is unset the package logger is left at NOTSET
            so it inherits from the embedding application's root
            configuration.
    """
    if level is None:
        if not os.environ.get(_LOG_LEVEL_ENV):
            return
        resolved = _resolve_log_level()
    elif isinstance(level, str):
        resolved = _LEVEL_NAMES.get(level.upper(), logging.WARNING)
    else:
        resolved = level
    _package_logger().setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``delta_search`` namespace.

    Args:
        name: Sub-name, typically ``__name__``.

    Returns:
        A logger whose parent is the ``delta_search`` package logger,
        so its effective level respects ``DELTA_SEARCH_LOG_LEVEL``.
    """
    return logging.getLogger(name)


def default_seed() -> int | None:
    """Return the configured default seed, if any.

    Returns:
        The integer seed from ``DELTA_SEARCH_SEED``, or ``None`` when
        the user did not set one.
    """
    raw = os.environ.get(_SEED_ENV)
    if raw is None or raw == "":
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


_package_logger()
