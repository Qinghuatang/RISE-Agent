"""Structured logging helpers with an allowlist of safe fields."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import sys
import threading
from typing import Any, Iterator, Mapping
from uuid import uuid4


_CONTEXT_FIELDS = frozenset(
    {
        "service",
        "request_id",
        "task_id",
        "thread_id",
        "worker_id",
        "attempt_count",
        "recovery_count",
        "graph_node",
        "action",
    }
)
_STRING_FIELDS = frozenset(
    {
        "service",
        "request_id",
        "task_id",
        "thread_id",
        "worker_id",
        "graph_node",
        "action",
        "route",
        "method",
        "status",
        "task_status",
        "rise_outcome",
        "failure_kind",
        "error_category",
        "error_type",
        "provider",
        "model",
        "operation",
        "metric",
        "node_status",
    }
)
_INTEGER_FIELDS = frozenset(
    {
        "attempt_count",
        "recovery_count",
        "status_code",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "provider_retry_count",
        "format_retry_count",
        "request_count",
        "event_count",
        "llm_client_count",
        "retrieval_provider_count",
        "http_request_count",
        "action_steps_used",
        "action_steps_remaining",
        "searches_used",
        "searches_remaining",
        "trajectory_event_count",
        "replay_record_count",
    }
)
_BOOLEAN_FIELDS = frozenset({"success", "token_usage_known", "cache_hit", "recovered"})
_FLOAT_FIELDS = frozenset({"duration_ms"})
_SAFE_CATEGORIES = frozenset(
    {"input", "infrastructure", "llm", "retrieval", "scientific", "ownership", "cancelled", "unknown"}
)
_SAFE_TASK_STATUSES = frozenset(
    {"queued", "running", "cancel_requested", "completed", "failed", "cancelled"}
)
_SAFE_RISE_OUTCOMES = frozenset(
    {"initialized", "running", "committed", "stopped", "failed", "finalization_failed"}
)
_SAFE_FAILURE_KINDS = frozenset({"input", "infrastructure"})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:/{}-]{1,160}$")
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")

_log_context: ContextVar[dict[str, Any]] = ContextVar("rise_log_context", default={})
_configuration_lock = threading.RLock()
_HANDLER_MARKER = "_rise_structured_logging_handler"


def _safe_field(name: str, value: Any) -> Any:
    if name not in _STRING_FIELDS | _INTEGER_FIELDS | _BOOLEAN_FIELDS | _FLOAT_FIELDS:
        return None
    if value is None:
        return None
    if name in _STRING_FIELDS:
        if not isinstance(value, str):
            return None
        limit = 96 if name == "request_id" else 160
        value = value.strip()[:limit]
        if name == "error_type":
            return value if _ERROR_TYPE_RE.fullmatch(value) else None
        if name == "error_category":
            return value if value in _SAFE_CATEGORIES else "unknown"
        if name == "task_status":
            return value if value in _SAFE_TASK_STATUSES else None
        if name == "rise_outcome":
            return value if value in _SAFE_RISE_OUTCOMES else None
        if name == "failure_kind":
            return value if value in _SAFE_FAILURE_KINDS else None
        if name == "request_id":
            return value if _REQUEST_ID_RE.fullmatch(value) else None
        return value if _IDENTIFIER_RE.fullmatch(value) else None
    if name in _INTEGER_FIELDS:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    if name in _BOOLEAN_FIELDS:
        return value if isinstance(value, bool) else None
    if name in _FLOAT_FIELDS:
        return round(value, 3) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None
    return None


def _safe_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        safe_value = _safe_field(name, value)
        if safe_value is not None:
            result[name] = safe_value
        elif name in _STRING_FIELDS | _INTEGER_FIELDS | _BOOLEAN_FIELDS | _FLOAT_FIELDS and value is None:
            result[name] = None
    return result


@contextmanager
def bind_log_context(**values: Any) -> Iterator[None]:
    """Set allowlisted context fields and restore the previous context."""

    updated = dict(_log_context.get())
    for name, value in values.items():
        if name not in _CONTEXT_FIELDS:
            continue
        safe_value = _safe_field(name, value)
        if safe_value is None:
            updated.pop(name, None)
        else:
            updated[name] = safe_value
    token = _log_context.set(updated)
    try:
        yield
    finally:
        _log_context.reset(token)


def current_log_context() -> dict[str, Any]:
    """Return a safe copy of the current context for explicit thread handoff."""

    return dict(_log_context.get())


def emit_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured event; arbitrary message text is intentionally omitted."""

    if not _IDENTIFIER_RE.fullmatch(event):
        event = "invalid_event"
    payload = _safe_fields(_log_context.get())
    payload.update(_safe_fields(fields))
    logger.log(level, event, extra={"rise_event": event, "rise_fields": payload})


class JSONFormatter(logging.Formatter):
    """Serialize only event names and explicitly allowlisted scalar fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "rise_event", None)
        if not isinstance(event, str) or not _IDENTIFIER_RE.fullmatch(event):
            event = "unstructured_log"
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name[:160],
            "event": event,
        }
        event_id = getattr(record, "rise_event_id", None)
        if not isinstance(event_id, str):
            event_id = uuid4().hex
            record.rise_event_id = event_id
        payload["event_id"] = event_id
        fields = getattr(record, "rise_fields", {})
        if isinstance(fields, Mapping):
            payload.update(_safe_fields(fields))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class PlainEventFormatter(logging.Formatter):
    """Readable non-JSON format that still excludes free-form log messages."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "rise_event", None)
        if not isinstance(event, str) or not _IDENTIFIER_RE.fullmatch(event):
            event = "unstructured_log"
        fields = getattr(record, "rise_fields", {})
        safe = _safe_fields(fields) if isinstance(fields, Mapping) else {}
        details = " ".join(f"{key}={value}" for key, value in safe.items())
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds")
        return f"{timestamp} {record.levelname} {record.name} {event}" + (f" {details}" if details else "")


def configure_logging(
    *,
    level: str | int | None = None,
    json_output: bool | None = None,
) -> logging.Handler:
    """Configure one root console handler without registering duplicates."""

    resolved_level = level if level is not None else os.environ.get("RISE_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError("RISE_LOG_LEVEL must be a valid logging level")
    if json_output is None:
        raw_json = os.environ.get("RISE_LOG_JSON", "true").strip().lower()
        if raw_json not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError("RISE_LOG_JSON must be a boolean value")
        json_output = raw_json in {"true", "1", "yes", "on"}

    root = logging.getLogger()
    with _configuration_lock:
        root.setLevel(resolved_level)
        handler = next(
            (item for item in root.handlers if getattr(item, _HANDLER_MARKER, False)),
            None,
        )
        if handler is None:
            handler = next(
                (item for item in root.handlers if not isinstance(item, logging.NullHandler)),
                None,
            )
        if handler is None:
            handler = logging.StreamHandler(sys.stderr)
            root.addHandler(handler)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(JSONFormatter() if json_output else PlainEventFormatter())
    return handler


__all__ = [
    "JSONFormatter",
    "PlainEventFormatter",
    "bind_log_context",
    "configure_logging",
    "current_log_context",
    "emit_event",
]
