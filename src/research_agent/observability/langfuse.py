"""Opt-in Langfuse tracing with explicit metadata and failure isolation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import math
import os
import re
import threading
import time
from typing import Any, Iterator, Mapping
from uuid import UUID

from .telemetry import DEFAULT_TELEMETRY, Telemetry


_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/{}-]{1,160}$")
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,159}$")
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})
_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_ACTIVE_TRACE_ID: ContextVar[str | None] = ContextVar(
    "rise_langfuse_trace_id", default=None
)

_METADATA_KEYS = {
    "task_id": "taskId",
    "thread_id": "threadId",
    "request_id": "requestId",
    "worker_id": "workerId",
    "attempt_count": "attemptCount",
    "recovery_count": "recoveryCount",
    "graph_node": "graphNode",
    "node": "node",
    "action": "action",
    "operation": "operation",
    "status": "status",
    "task_status": "taskStatus",
    "rise_outcome": "riseOutcome",
    "provider": "provider",
    "error_category": "errorCategory",
    "error_type": "errorType",
    "duration_ms": "durationMs",
    "prompt_tokens": "promptTokens",
    "completion_tokens": "completionTokens",
    "total_tokens": "totalTokens",
    "provider_retry_count": "providerRetryCount",
    "format_retry_count": "formatRetryCount",
    "http_request_count": "httpRequestCount",
    "cache_hit": "cacheHit",
    "idempotent_replay": "idempotentReplay",
}
_INTEGER_FIELDS = frozenset(
    {
        "attempt_count",
        "recovery_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "provider_retry_count",
        "format_retry_count",
        "http_request_count",
    }
)
_BOOLEAN_FIELDS = frozenset({"cache_hit", "idempotent_replay"})
_FLOAT_FIELDS = frozenset({"duration_ms"})
_SAFE_CATEGORIES = frozenset(
    {"input", "infrastructure", "llm", "retrieval", "scientific", "ownership", "cancelled", "unknown"}
)


def task_trace_id(task_id: str) -> str:
    """Return a stable W3C-compatible trace ID for a task identifier."""

    try:
        return UUID(task_id).hex
    except (ValueError, AttributeError, TypeError):
        return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]


def _safe_trace_id(value: str | None) -> str | None:
    if value is None:
        return None
    if _TRACE_ID_RE.fullmatch(value):
        return value
    return task_trace_id(value)


def _safe_metadata(values: Mapping[str, Any] | None) -> dict[str, str]:
    if not values:
        return {}
    safe: dict[str, str] = {}
    for field, value in values.items():
        key = _METADATA_KEYS.get(field)
        if key is None or value is None:
            continue
        if field == "error_category":
            if value not in _SAFE_CATEGORIES:
                continue
            safe[key] = value
        elif field == "error_type":
            if isinstance(value, str) and _ERROR_TYPE_RE.fullmatch(value):
                safe[key] = value
        elif field in _INTEGER_FIELDS:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                safe[key] = str(value)
        elif field in _BOOLEAN_FIELDS:
            if isinstance(value, bool):
                safe[key] = str(value).lower()
        elif field in _FLOAT_FIELDS:
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                safe[key] = f"{value:.3f}"
        elif isinstance(value, str) and _SAFE_VALUE_RE.fullmatch(value):
            safe[key] = value
    return safe


class _NoopObservation:
    def update(self, **_fields: Any) -> None:
        return None

    def mark_error(self, **_fields: Any) -> None:
        return None


class _SafeObservation:
    def __init__(self, observation: Any, tracer: "LangfuseTracer") -> None:
        self._observation = observation
        self._tracer = tracer
        self._has_error = False

    def update(self, **fields: Any) -> None:
        self._update(fields)

    def mark_error(self, *, error_type: str | None = None, error_category: str | None = None) -> None:
        self._has_error = True
        self._update(
            {
                "status": "error",
                "error_type": error_type,
                "error_category": error_category,
            },
            level="ERROR",
            status_message="operation_failed",
        )

    def _update(
        self,
        fields: Mapping[str, Any],
        *,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        metadata = _safe_metadata(fields)
        if metadata:
            kwargs["metadata"] = metadata

        prompt_tokens = fields.get("prompt_tokens")
        completion_tokens = fields.get("completion_tokens")
        usage: dict[str, int] = {}
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0:
            usage["input"] = prompt_tokens
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens >= 0:
            usage["output"] = completion_tokens
        if usage:
            kwargs["usage_details"] = usage

        model = fields.get("model")
        if isinstance(model, str) and _SAFE_VALUE_RE.fullmatch(model):
            kwargs["model"] = model
        if level in {"DEBUG", "DEFAULT", "WARNING", "ERROR"}:
            kwargs["level"] = level
        if status_message in {"operation_failed"}:
            kwargs["status_message"] = status_message
        if not kwargs:
            return
        try:
            self._observation.update(**kwargs)
        except Exception as exc:
            self._tracer._report_failure("langfuse.observation_update_failed", exc)


class LangfuseTracer:
    """Small optional adapter around Langfuse's manual observation API.

    The SDK is imported only when `RISE_LANGFUSE_ENABLED` is explicitly true.
    Manual observations never receive input/output arguments; only allowlisted
    metadata and numeric generation usage are passed to the SDK.
    """

    def __init__(self, client: Any | None = None, telemetry: Telemetry = DEFAULT_TELEMETRY) -> None:
        self._client = client
        self._telemetry = telemetry
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._closed

    @classmethod
    def from_env(cls, telemetry: Telemetry = DEFAULT_TELEMETRY) -> "LangfuseTracer":
        setting = os.environ.get("RISE_LANGFUSE_ENABLED", "false").strip().lower()
        if setting in _FALSE_VALUES:
            return cls(None, telemetry)
        if setting not in _TRUE_VALUES:
            telemetry.record(
                "langfuse.initialization_failed",
                operation="configuration",
                status="disabled",
                error_category="unknown",
                error_type="InvalidEnableSetting",
            )
            return cls(None, telemetry)

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            telemetry.record(
                "langfuse.initialization_failed",
                operation="configuration",
                status="disabled",
                error_category="infrastructure",
                error_type="LangfuseCredentialsMissing",
            )
            return cls(None, telemetry)

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
                tracing_enabled=True,
                timeout=5,
            )
        except Exception as exc:
            telemetry.record(
                "langfuse.initialization_failed",
                operation="client_initialization",
                status="disabled",
                error_category="infrastructure",
                error_type=type(exc).__name__,
            )
            return cls(None, telemetry)
        return cls(client, telemetry)

    @staticmethod
    def current_trace_id() -> str | None:
        return _ACTIVE_TRACE_ID.get()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        observation_type: str = "span",
        trace_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[_SafeObservation | _NoopObservation]:
        """Create a safe observation and keep the original work authoritative."""

        if not self.enabled:
            yield _NoopObservation()
            return

        active_trace_id = _ACTIVE_TRACE_ID.get()
        selected_trace_id = _safe_trace_id(trace_id) or active_trace_id
        if not _SAFE_VALUE_RE.fullmatch(name):
            name = "rise.operation"
        if observation_type not in {"span", "agent", "tool", "chain", "retriever", "generation"}:
            observation_type = "span"
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": observation_type,
        }
        safe_metadata = _safe_metadata(metadata)
        if safe_metadata:
            kwargs["metadata"] = safe_metadata
        if isinstance(model, str) and _SAFE_VALUE_RE.fullmatch(model):
            kwargs["model"] = model
        # Avoid asking the SDK for an ambient trace when none is active: the
        # v4 API logs a warning for that normal root-span case. Our own context
        # variable is authoritative for spans created by this adapter. An
        # explicitly supplied ID starts (or re-roots) the same task trace when
        # crossing a thread/process boundary.
        if selected_trace_id and active_trace_id != selected_trace_id:
            kwargs["trace_context"] = {"trace_id": selected_trace_id}

        try:
            observation_context = self._client.start_as_current_observation(**kwargs)
            observation = observation_context.__enter__()
        except Exception as exc:
            self._report_failure("langfuse.span_start_failed", exc)
            yield _NoopObservation()
            return

        actual_trace_id = selected_trace_id
        if actual_trace_id is None:
            try:
                actual_trace_id = self._client.get_current_trace_id()
            except Exception:
                actual_trace_id = None
        context_token = _ACTIVE_TRACE_ID.set(actual_trace_id)
        handle = _SafeObservation(observation, self)
        started = time.monotonic_ns()
        operation_error: BaseException | None = None
        try:
            yield handle
        except BaseException as exc:
            operation_error = exc
            if not handle._has_error:
                handle.mark_error(error_type=type(exc).__name__, error_category="unknown")
            raise
        finally:
            elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
            handle.update(
                status="error" if operation_error is not None else "completed",
                duration_ms=elapsed_ms,
            )
            try:
                observation_context.__exit__(None, None, None)
            except Exception as exc:
                self._report_failure("langfuse.span_end_failed", exc)
            finally:
                _ACTIVE_TRACE_ID.reset(context_token)

    def flush(self) -> bool:
        if not self.enabled:
            return True
        try:
            self._client.flush()
            return True
        except Exception as exc:
            self._report_failure("langfuse.flush_failed", exc)
            return False

    def shutdown(self) -> bool:
        with self._close_lock:
            if self._closed:
                return True
            self._closed = True
        if self._client is None:
            return True
        try:
            shutdown = getattr(self._client, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                self._client.flush()
            return True
        except Exception as exc:
            self._report_failure("langfuse.shutdown_failed", exc)
            return False

    def _report_failure(self, event: str, error: Exception) -> None:
        self._telemetry.record(
            event,
            operation="export",
            status="error",
            error_category="infrastructure",
            error_type=type(error).__name__,
        )


__all__ = ["LangfuseTracer", "task_trace_id"]
