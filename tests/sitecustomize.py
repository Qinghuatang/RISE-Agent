"""Optional file-backed Langfuse test exporter for independent P3-E processes.

Python imports this module automatically when ``tests`` is on ``PYTHONPATH``.
It has no effect unless a P3-E child explicitly enables trace capture.
"""

from __future__ import annotations

if __import__("os").environ.get("RISE_E2E_TRACE_CAPTURE") == "1":
    import contextvars
    import fcntl
    import json
    import os
    from pathlib import Path
    import threading
    from uuid import uuid4

    _active: contextvars.ContextVar[tuple[dict, ...]] = contextvars.ContextVar(
        "rise_e2e_langfuse_stack", default=()
    )
    _write_lock = threading.Lock()

    def _append(row: dict) -> None:
        destination = Path(os.environ["RISE_E2E_TRACE_FILE"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n"
        with _write_lock, destination.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    class _Observation:
        def __init__(self, span: dict) -> None:
            self.span = span

        def update(self, **kwargs) -> None:
            row = {
                "record": "span_update",
                "process_id": os.getpid(),
                "trace_id": self.span["trace_id"],
                "span_id": self.span["span_id"],
            }
            for key in ("metadata", "model", "usage_details", "level", "status_message"):
                if kwargs.get(key) is not None:
                    row[key] = kwargs[key]
            _append(row)

    class _ObservationContext:
        def __init__(self, kwargs: dict) -> None:
            self.kwargs = kwargs
            self.token = None
            self.span = None

        def __enter__(self):
            failures = set(os.environ.get("RISE_E2E_TRACE_FAILURES", "").split(","))
            if "start" in failures:
                raise RuntimeError("test exporter start failure")
            stack = _active.get()
            parent = stack[-1] if stack else None
            trace_context = self.kwargs.get("trace_context") or {}
            self.span = {
                "name": self.kwargs["name"],
                "as_type": self.kwargs["as_type"],
                "trace_id": trace_context.get("trace_id")
                or (parent["trace_id"] if parent else uuid4().hex),
                "span_id": uuid4().hex[:16],
                "parent_span_id": parent["span_id"] if parent else None,
                "metadata": self.kwargs.get("metadata") or {},
                "process_id": os.getpid(),
            }
            _append({"record": "span_start", **self.span})
            self.token = _active.set((*stack, self.span))
            return _Observation(self.span)

        def __exit__(self, exc_type, _exc, _tb):
            try:
                if self.span is not None:
                    _append(
                        {
                            "record": "span_end",
                            "process_id": os.getpid(),
                            "trace_id": self.span["trace_id"],
                            "span_id": self.span["span_id"],
                            "status": "error" if exc_type else "completed",
                        }
                    )
            finally:
                if self.token is not None:
                    _active.reset(self.token)
            if "end" in set(os.environ.get("RISE_E2E_TRACE_FAILURES", "").split(",")):
                raise RuntimeError("test exporter end failure")
            return False

    class _ProcessExporter:
        def start_as_current_observation(self, **kwargs):
            return _ObservationContext(kwargs)

        def get_current_trace_id(self):
            stack = _active.get()
            return stack[-1]["trace_id"] if stack else None

        def flush(self) -> None:
            if "flush" in set(os.environ.get("RISE_E2E_TRACE_FAILURES", "").split(",")):
                raise RuntimeError("test exporter flush failure")

        def shutdown(self) -> None:
            if "shutdown" in set(os.environ.get("RISE_E2E_TRACE_FAILURES", "").split(",")):
                raise RuntimeError("test exporter shutdown failure")

    from research_agent.observability.langfuse import LangfuseTracer
    from research_agent.observability.telemetry import DEFAULT_TELEMETRY

    def _from_env(cls, telemetry=None):
        return cls(_ProcessExporter(), telemetry or DEFAULT_TELEMETRY)

    LangfuseTracer.from_env = classmethod(_from_env)
