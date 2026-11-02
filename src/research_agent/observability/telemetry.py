"""Failure-isolated sink for structured operational measurements."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .logging import emit_event


class Telemetry:
    """Write operational observations without affecting the measured work."""

    def __init__(
        self,
        emitter: Callable[..., None] | None = None,
    ) -> None:
        self._emitter = emitter or self._emit_log

    @staticmethod
    def _emit_log(event: str, **fields: Any) -> None:
        emit_event(logging.getLogger("research_agent.metrics"), logging.INFO, event, **fields)

    def record(self, event: str, **fields: Any) -> bool:
        """Return false if telemetry fails; never raise into RISE execution."""

        try:
            self._emitter(event, **fields)
        except Exception:
            return False
        return True


DEFAULT_TELEMETRY = Telemetry()


__all__ = ["DEFAULT_TELEMETRY", "Telemetry"]
