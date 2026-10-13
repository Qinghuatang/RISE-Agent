"""Minimal logging setup shared by future runners and CLI commands."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single simple stderr handler for command-line use."""

    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(handler, "_research_agent_handler", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler._research_agent_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)

