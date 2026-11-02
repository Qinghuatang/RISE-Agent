"""Lightweight, privacy-first execution telemetry for RISE services."""

from .instrumentation import (
    ObservedLLMClient,
    ObservedRetrievalProvider,
    instrument_trajectory_modules,
)
from .logging import (
    bind_log_context,
    configure_logging,
    current_log_context,
    emit_event,
)
from .telemetry import DEFAULT_TELEMETRY, Telemetry

__all__ = [
    "DEFAULT_TELEMETRY",
    "ObservedLLMClient",
    "ObservedRetrievalProvider",
    "Telemetry",
    "bind_log_context",
    "configure_logging",
    "current_log_context",
    "emit_event",
    "instrument_trajectory_modules",
]
