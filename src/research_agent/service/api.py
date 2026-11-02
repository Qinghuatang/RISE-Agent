"""FastAPI application for submitting and inspecting durable RISE tasks.

This application only writes task metadata and reads persisted results/events.
Scientific execution remains exclusively in the independent Worker process.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from ..runtime.checkpointer import open_postgres_checkpointer
from ..runtime.graph_state import RISEGraphState
from ..runtime.paper_store import PaperInputIntegrityError, PaperInputStore
from ..observability.langfuse import LangfuseTracer, task_trace_id
from ..observability.logging import bind_log_context
from ..observability.telemetry import DEFAULT_TELEMETRY, Telemetry
from .checkpoint_results import CheckpointResultError, read_checkpoint_state
from .contracts import CreateArtifact, TaskEventRecord, TaskPageCursor, TaskRecord
from .db import (
    create_business_engine,
    create_session_factory,
    database_url_from_env,
    upgrade_business_schema,
)
from .errors import IdempotencyConflict, InvalidTaskTransition, TaskNotFound
from .repository import TaskRepository
from .api_schemas import TaskAccepted, TaskList, TaskProgress, TaskResult, TaskSummary

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_RISE_TERMINAL_OUTCOMES = frozenset(
    {"committed", "stopped", "failed", "finalization_failed"}
)
_ACCEPTED_MARKDOWN_TYPES = frozenset(
    {"", "text/markdown", "text/x-markdown", "text/plain", "application/markdown", "application/octet-stream"}
)
_KEY_RE = re.compile(r"^[^\r\n]{1,255}$")
_SAFE_EVENT_PAYLOAD_KEYS = frozenset(
    {
        "task_status",
        "attempt",
        "trajectory_index",
        "trajectory_event_type",
        "state_fields",
        "memory_writes",
        "budget",
        "replay_sequence",
        "step",
        "kind",
        "rise_outcome",
        "failure_kind",
    }
)
_SAFE_EVENT_COUNTERS = frozenset(
    {"max_steps", "max_searches", "steps_used", "searches_used"}
)


@dataclass(frozen=True)
class APISettings:
    database_url: str
    paper_store_root: Path
    max_markdown_bytes: int = 10 * 1024 * 1024
    sse_poll_interval: float = 1.0
    sse_heartbeat_interval: float = 15.0
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.max_markdown_bytes < 1:
            raise ValueError("max_markdown_bytes must be positive")
        if self.sse_poll_interval <= 0 or self.sse_heartbeat_interval <= 0:
            raise ValueError("SSE intervals must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            database_url=database_url_from_env(),
            paper_store_root=Path(os.environ.get("RISE_PAPER_STORE_ROOT", "./var/papers")),
            max_markdown_bytes=int(os.environ.get("RISE_MAX_MARKDOWN_BYTES", 10 * 1024 * 1024)),
            sse_poll_interval=float(os.environ.get("RISE_SSE_POLL_INTERVAL", "1.0")),
            sse_heartbeat_interval=float(os.environ.get("RISE_SSE_HEARTBEAT_INTERVAL", "15.0")),
            host=os.environ.get("RISE_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("RISE_API_PORT", "8000")),
        )


@dataclass
class APIResources:
    settings: APISettings
    engine: Any
    repository: TaskRepository
    paper_store: PaperInputStore
    checkpointer: Any
    checkpointer_lock: threading.Lock
    telemetry: Telemetry
    tracer: LangfuseTracer


class _Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    markdown: str = Field(min_length=1)
    execution_config: dict[str, Any] = Field(default_factory=dict)


def create_app(
    settings: APISettings | None = None,
    *,
    telemetry: Telemetry | None = None,
    tracer: LangfuseTracer | None = None,
) -> FastAPI:
    """Create an API app whose lifespan owns database resources."""

    resolved_settings = settings or APISettings.from_env()
    resolved_telemetry = telemetry or DEFAULT_TELEMETRY
    injected_tracer = tracer

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_business_engine(resolved_settings.database_url)
        app_tracer = (
            injected_tracer
            if injected_tracer is not None
            else LangfuseTracer.from_env(resolved_telemetry)
        )
        saver_context = None
        saver_entered = False
        try:
            await run_in_threadpool(upgrade_business_schema, engine)
            sessions = create_session_factory(engine)
            saver_context = open_postgres_checkpointer(resolved_settings.database_url)
            checkpointer = await run_in_threadpool(saver_context.__enter__)
            saver_entered = True
            app.state.resources = APIResources(
                settings=resolved_settings,
                engine=engine,
                repository=TaskRepository(sessions),
                paper_store=PaperInputStore(resolved_settings.paper_store_root),
                checkpointer=checkpointer,
                checkpointer_lock=threading.Lock(),
                telemetry=resolved_telemetry,
                tracer=app_tracer,
            )
            yield
        finally:
            try:
                if saver_context is not None and saver_entered:
                    await run_in_threadpool(saver_context.__exit__, None, None, None)
            finally:
                app_tracer.shutdown()
                engine.dispose()

    app = FastAPI(
        title="RISE Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", request_id):
            request_id = uuid4().hex
        request.state.request_id = request_id
        task_match = re.match(r"^/api/tasks/([0-9a-fA-F-]{36})(?:/|$)", request.url.path)
        if task_match:
            try:
                request.state.task_id = str(UUID(task_match.group(1)))
                request.state.thread_id = request.state.task_id
            except ValueError:
                pass

        started = time.monotonic_ns()
        response = None
        status_code = 500
        with bind_log_context(
            service="api",
            request_id=request_id,
            task_id=getattr(request.state, "task_id", None),
            thread_id=getattr(request.state, "thread_id", None),
        ):
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                route = getattr(request.scope.get("route"), "path", "unknown")
                with bind_log_context(
                    task_id=getattr(request.state, "task_id", None),
                    thread_id=getattr(request.state, "thread_id", None),
                ):
                    resolved_telemetry.record(
                        "api.request.finished",
                        operation=(
                            "http_response_started"
                            if route == "/api/tasks/{task_id}/events"
                            else "http_request"
                        ),
                        method=request.method,
                        route=route,
                        status="completed" if status_code < 500 else "error",
                        status_code=status_code,
                        duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                        error_category="infrastructure" if status_code >= 500 else None,
                    )
                if response is not None:
                    response.headers["X-Request-ID"] = request_id

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "request validation failed"}},
        )

    @app.exception_handler(TaskNotFound)
    async def not_found_handler(_request: Request, _exc: TaskNotFound) -> JSONResponse:
        return _error(404, "task_not_found", "task not found")

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_handler(_request: Request, _exc: IdempotencyConflict) -> JSONResponse:
        return _error(409, "idempotency_conflict", "Idempotency-Key was used for different content")

    @app.exception_handler(InvalidTaskTransition)
    async def transition_handler(_request: Request, _exc: InvalidTaskTransition) -> JSONResponse:
        return _error(409, "invalid_task_transition", "task state does not allow this operation")

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(_request: Request, _exc: SQLAlchemyError) -> JSONResponse:
        resolved_telemetry.record(
            "api.database_operation_failed",
            status="error",
            error_category="infrastructure",
            error_type=type(_exc).__name__,
        )
        return _error(503, "database_unavailable", "database operation is temporarily unavailable")

    @app.exception_handler(Exception)
    async def internal_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        resolved_telemetry.record(
            "api.request_failed",
            status="error",
            error_category="infrastructure",
            error_type=type(exc).__name__,
        )
        return _error(500, "internal_error", "request could not be completed")

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        resources = _resources(request)

        def probe() -> None:
            with resources.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            with resources.checkpointer_lock:
                resources.checkpointer.get_tuple(
                    {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000000"}}
                )

        try:
            await run_in_threadpool(probe)
        except Exception as exc:
            resolved_telemetry.record(
                "api.readiness_probe_failed",
                status="error",
                error_category="infrastructure",
                error_type=type(exc).__name__,
            )
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "database_unavailable"},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.post(
        "/api/tasks",
        response_model=TaskAccepted,
        status_code=202,
        description="Submit JSON Markdown text or a multipart .md upload. The request is queued; execution is handled by the Worker.",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["markdown"],
                            "properties": {
                                "markdown": {"type": "string"},
                                "execution_config": {"type": "object"},
                            },
                        }
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "markdown": {"type": "string"},
                                "file": {"type": "string", "format": "binary"},
                                "execution_config": {"type": "string"},
                            },
                        }
                    },
                },
            }
        },
    )
    async def create_task(
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> TaskAccepted:
        resources = _resources(request)
        started = time.monotonic_ns()

        def accepted(task: TaskRecord, *, replay: bool) -> TaskAccepted:
            request.state.task_id = str(task.id)
            request.state.thread_id = str(task.thread_id)
            with bind_log_context(task_id=str(task.id), thread_id=str(task.thread_id)):
                duration_ms = (time.monotonic_ns() - started) / 1_000_000
                event = "api.task.submission_replayed" if replay else "api.task.submitted"
                with resources.tracer.span(
                    "rise.api.task_submission",
                    trace_id=task_trace_id(str(task.id)),
                    metadata={
                        "task_id": str(task.id),
                        "thread_id": str(task.thread_id),
                        "operation": "task_submission",
                        "task_status": task.task_status,
                        "idempotent_replay": replay,
                        "duration_ms": duration_ms,
                    },
                ):
                    resources.telemetry.record(
                        event,
                        operation="task_submission",
                        status=task.task_status,
                        task_status=task.task_status,
                        duration_ms=duration_ms,
                    )
            return _accepted(task, replay=replay)

        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not _KEY_RE.fullmatch(idempotency_key):
                raise HTTPException(422, detail="Idempotency-Key must contain 1 to 255 characters")

        raw_markdown, execution_config = await _read_submission(request, resources.settings)
        if len(raw_markdown) > resources.settings.max_markdown_bytes:
            raise HTTPException(413, detail="Markdown exceeds the configured size limit")
        try:
            markdown = raw_markdown.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(422, detail="Markdown must be valid UTF-8") from exc
        if not markdown.strip():
            raise HTTPException(422, detail="Markdown must not be empty")

        input_sha256 = hashlib.sha256(raw_markdown).hexdigest()
        request_sha256 = _request_digest(input_sha256, execution_config)
        repo = resources.repository
        existing = None
        if idempotency_key is not None:
            existing = await run_in_threadpool(repo.get_by_idempotency_key, idempotency_key)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise IdempotencyConflict
                return accepted(existing, replay=True)

        try:
            paper_ref = await run_in_threadpool(resources.paper_store.persist, raw_markdown)
        except PaperInputIntegrityError as exc:
            raise HTTPException(422, detail="Markdown input could not be safely stored") from exc
        artifact = CreateArtifact(
            artifact_type="input_markdown",
            locator=paper_ref.locator,
            byte_size=len(raw_markdown),
            sha256=paper_ref.sha256,
            media_type="text/markdown",
        )
        try:
            task = await run_in_threadpool(
                partial(
                    repo.create_task,
                    input_locator=paper_ref.locator,
                    input_sha256=paper_ref.sha256,
                    request_sha256=request_sha256,
                    idempotency_key=idempotency_key,
                    execution_config=execution_config,
                    input_artifact=artifact,
                )
            )
        except ValueError as exc:
            raise HTTPException(422, detail="execution_config is invalid") from exc
        return accepted(task, replay=existing is not None)

    @app.get("/api/tasks", response_model=TaskList)
    async def list_tasks(
        request: Request,
        limit: int = 50,
        cursor: str | None = None,
    ) -> TaskList:
        if not 1 <= limit <= 100:
            raise HTTPException(422, detail="limit must be between 1 and 100")
        page_cursor = _decode_cursor(cursor) if cursor else None
        records = await run_in_threadpool(
            partial(_resources(request).repository.list_tasks, limit=limit, cursor=page_cursor)
        )
        has_more = len(records) == limit
        next_cursor = _encode_cursor(records[-1]) if has_more and records else None
        return TaskList(items=[await _task_summary(_resources(request), row) for row in records], next_cursor=next_cursor)

    @app.get("/api/tasks/{task_id}", response_model=TaskSummary)
    async def get_task(task_id: UUID, request: Request) -> TaskSummary:
        resources = _resources(request)
        record = await run_in_threadpool(resources.repository.get_task, task_id)
        return await _task_summary(resources, record)

    @app.get("/api/tasks/{task_id}/result", response_model=TaskResult)
    async def get_task_result(task_id: UUID, request: Request) -> TaskResult:
        resources = _resources(request)
        task = await run_in_threadpool(resources.repository.get_task, task_id)
        if task.task_status not in _TERMINAL_STATUSES:
            raise HTTPException(
                409,
                detail={"code": "result_not_ready", "message": "task has not reached a terminal state"},
            )
        state: RISEGraphState | None = None
        try:
            def read_state() -> RISEGraphState | None:
                with resources.checkpointer_lock:
                    return read_checkpoint_state(resources.checkpointer, task.thread_id)

            state = await run_in_threadpool(read_state)
        except CheckpointResultError as exc:
            resources.telemetry.record(
                "api.result_checkpoint_invalid",
                status="error",
                error_category="infrastructure",
                error_type=type(exc).__name__,
            )
            raise HTTPException(503, detail="persisted result is temporarily unavailable") from exc
        if task.task_status == "completed" and state is None:
            raise HTTPException(503, detail="persisted result is temporarily unavailable")
        if state is not None:
            if task.rise_outcome is not None and task.rise_outcome != state.rise_status:
                raise HTTPException(503, detail="task result is inconsistent with its persisted state")
            if task.rise_outcome is None and state.rise_status in _RISE_TERMINAL_OUTCOMES:
                raise HTTPException(503, detail="task result is inconsistent with its persisted state")
        return TaskResult(
            task_id=task.id,
            task_status=task.task_status,
            rise_outcome=task.rise_outcome,
            final_proposal=(state.final_proposal.model_dump(mode="json") if state and state.final_proposal else None),
            commit_snapshot=(state.commit_snapshot.model_dump(mode="json") if state and state.commit_snapshot else None),
            unresolved_issues=list(state.unresolved_issues) if state else [],
            budget=asdict(state.budget) if state else None,
            trajectory_event_count=len(state.trajectory_events) if state else 0,
            replay_record_count=len(state.replay_records) if state else 0,
            error_type=_public_error_type(state.error_type if state else task.error_type),
            failure_kind=task.failure_kind,
            error_summary=_public_error_summary(task),
        )

    @app.post("/api/tasks/{task_id}/cancel", response_model=TaskSummary)
    async def cancel_task(task_id: UUID, request: Request) -> TaskSummary:
        resources = _resources(request)
        task = await run_in_threadpool(resources.repository.request_cancel, task_id)
        request.state.task_id = str(task.id)
        request.state.thread_id = str(task.thread_id)
        resources.telemetry.record(
            "api.task.cancel_requested",
            operation="cancel_request",
            status=task.task_status,
            task_status=task.task_status,
            error_category="cancelled" if task.task_status == "cancel_requested" else None,
        )
        return await _task_summary(resources, task)

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(
        task_id: UUID,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        resources = _resources(request)
        await run_in_threadpool(resources.repository.get_task, task_id)
        after_seq = _parse_last_event_id(last_event_id)
        return StreamingResponse(
            _event_stream(request, resources, task_id, after_seq),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


async def _read_submission(request: Request, settings: APISettings) -> tuple[bytes, dict[str, Any]]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        try:
            parsed = _Submission.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(422, detail="JSON body must contain non-empty Markdown text") from exc
        raw = parsed.markdown.encode("utf-8")
        return raw, parsed.execution_config
    if content_type != "multipart/form-data":
        raise HTTPException(415, detail="use application/json or multipart/form-data")
    try:
        form = await request.form(max_files=1, max_fields=3, max_part_size=settings.max_markdown_bytes + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, detail="invalid multipart request") from exc
    try:
        if set(form.keys()) - {"markdown", "file", "execution_config"}:
            raise HTTPException(422, detail="multipart request contains unsupported fields")
        supplied = [name for name in ("markdown", "file") if name in form]
        if len(supplied) != 1:
            raise HTTPException(422, detail="provide exactly one Markdown text field or .md file")
        execution_config: dict[str, Any] = {}
        config_value = form.get("execution_config")
        if config_value is not None:
            try:
                execution_config = json.loads(str(config_value))
                if not isinstance(execution_config, dict):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, detail="execution_config must be a JSON object") from exc
        value = form[supplied[0]]
        if supplied[0] == "markdown":
            if not isinstance(value, str):
                raise HTTPException(422, detail="markdown must be text")
            return value.encode("utf-8"), execution_config
        if not hasattr(value, "filename") or not value.filename:
            raise HTTPException(422, detail="uploaded Markdown file is required")
        if Path(value.filename).suffix.casefold() != ".md":
            raise HTTPException(422, detail="uploaded file must use the .md extension")
        media_type = (value.content_type or "").split(";", 1)[0].strip().lower()
        if media_type not in _ACCEPTED_MARKDOWN_TYPES:
            raise HTTPException(415, detail="uploaded file must have a Markdown or plain-text media type")
        raw = await value.read(settings.max_markdown_bytes + 1)
        if len(raw) > settings.max_markdown_bytes:
            raise HTTPException(413, detail="Markdown exceeds the configured size limit")
        return raw, execution_config
    finally:
        await form.close()


def _request_digest(input_sha256: str, execution_config: dict[str, Any]) -> str:
    try:
        payload = json.dumps(
            {"input_sha256": input_sha256, "execution_config": execution_config},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, detail="execution_config must contain finite JSON values") from exc
    return hashlib.sha256(payload).hexdigest()


async def _task_summary(resources: APIResources, task: TaskRecord) -> TaskSummary:
    latest: TaskEventRecord | None = None
    if task.last_event_seq:
        events = await run_in_threadpool(
            partial(resources.repository.list_events, task.id, after_seq=task.last_event_seq - 1, limit=1)
        )
        latest = events[0] if events else None
    return TaskSummary(
        task_id=task.id,
        task_status=task.task_status,
        rise_outcome=task.rise_outcome,
        failure_kind=task.failure_kind,
        error_type=_public_error_type(task.error_type),
        error_summary=_public_error_summary(task),
        attempt_count=task.attempt_count,
        recovery_count=task.recovery_count,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        progress=TaskProgress(
            event_seq=task.last_event_seq,
            latest_event_type=latest.event_type if latest else None,
            latest_action=_safe_event_name(latest.action) if latest and latest.action else None,
            updated_at=task.updated_at,
        ),
    )


async def _event_stream(
    request: Request,
    resources: APIResources,
    task_id: UUID,
    after_seq: int,
) -> AsyncIterator[str]:
    cursor = after_seq
    last_heartbeat = time.monotonic()
    started = time.monotonic_ns()
    emitted_events = 0
    stream_status = "disconnected"
    with bind_log_context(
        service="api",
        request_id=getattr(request.state, "request_id", None),
        task_id=str(task_id),
        thread_id=str(task_id),
    ):
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    batch = await run_in_threadpool(
                        partial(resources.repository.list_events, task_id, after_seq=cursor, limit=500)
                    )
                    for event in batch:
                        cursor = event.seq
                        emitted_events += 1
                        yield _format_sse(event)
                    if len(batch) == 500:
                        continue
                    task = await run_in_threadpool(resources.repository.get_task, task_id)
                    if task.task_status in _TERMINAL_STATUSES:
                        # Re-read after observing the terminal commit. This closes the
                        # race where the first read started just before its final event
                        # and the terminal status became visible immediately after it.
                        final_events = await run_in_threadpool(
                            partial(resources.repository.list_events, task_id, after_seq=cursor, limit=500)
                        )
                        if final_events:
                            for event in final_events:
                                cursor = event.seq
                                emitted_events += 1
                                yield _format_sse(event)
                            if len(final_events) == 500:
                                continue
                        stream_status = "completed"
                        return
                except TaskNotFound:
                    stream_status = "error"
                    yield _format_sse_error("task_not_found")
                    return
                except Exception as exc:
                    stream_status = "error"
                    resources.telemetry.record(
                        "api.sse_poll_failed",
                        operation="sse_poll",
                        status="error",
                        error_category="infrastructure",
                        error_type=type(exc).__name__,
                        task_id=str(task_id),
                        thread_id=str(task_id),
                    )
                    yield _format_sse_error("event_store_unavailable")
                    return

                now = time.monotonic()
                if now - last_heartbeat >= resources.settings.sse_heartbeat_interval:
                    yield ": keep-alive\n\n"
                    last_heartbeat = now
                await asyncio.sleep(resources.settings.sse_poll_interval)
        finally:
            resources.telemetry.record(
                "api.sse.stream_finished",
                operation="sse_stream",
                status=stream_status,
                event_count=emitted_events,
                duration_ms=(time.monotonic_ns() - started) / 1_000_000,
            )


def _format_sse(event: TaskEventRecord) -> str:
    data = {
        "seq": event.seq,
        "event_id": str(event.event_id),
        "task_id": str(event.task_id),
        "event_type": _safe_event_name(event.event_type),
        "action": _safe_event_name(event.action) if event.action else None,
        "payload": _safe_event_payload(event.payload),
        "created_at": event.created_at.isoformat(),
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.seq}\nevent: {data['event_type']}\ndata: {encoded}\n\n"


def _format_sse_error(code: str) -> str:
    return f"event: stream_error\ndata: {json.dumps({'code': code}, separators=(',', ':'))}\n\n"


def _accepted(task: TaskRecord, *, replay: bool) -> TaskAccepted:
    return TaskAccepted(
        task_id=task.id,
        task_status=task.task_status,
        rise_outcome=task.rise_outcome,
        created_at=task.created_at,
        idempotent_replay=replay,
    )


def _resources(request: Request) -> APIResources:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(503, detail="API resources are not ready")
    return resources


def _encode_cursor(task: TaskRecord) -> str:
    raw = json.dumps(
        {"created_at": task.created_at.isoformat(), "task_id": str(task.id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> TaskPageCursor:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if set(payload) != {"created_at", "task_id"}:
            raise ValueError
        created_at = datetime.fromisoformat(payload["created_at"])
        if created_at.tzinfo is None:
            raise ValueError
        return TaskPageCursor(created_at=created_at, task_id=UUID(payload["task_id"]))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(422, detail="cursor is invalid") from exc


def _parse_last_event_id(value: str | None) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(400, detail="Last-Event-ID must be a non-negative sequence number") from exc
    if parsed < 0 or parsed > 2**63 - 1:
        raise HTTPException(400, detail="Last-Event-ID must be a non-negative sequence number")
    return parsed


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _public_error_summary(task: TaskRecord) -> str | None:
    if task.failure_kind == "input":
        return "Input validation or storage failed."
    if task.failure_kind == "infrastructure":
        return "Execution failed because of a service or infrastructure error."
    if task.failure_kind == "rise":
        return "RISE reached a scientific failure outcome."
    return None


def _public_error_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,254}", value) else "ExecutionError"


def _safe_event_name(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value) else "message"


def _safe_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in _SAFE_EVENT_PAYLOAD_KEYS:
            continue
        if key == "budget" and isinstance(value, dict):
            safe[key] = {
                counter: amount
                for counter, amount in value.items()
                if counter in _SAFE_EVENT_COUNTERS
                and isinstance(amount, int)
                and not isinstance(amount, bool)
                and amount >= 0
            }
        elif key in {"state_fields", "memory_writes"} and isinstance(value, list):
            safe[key] = [
                name
                for name in value
                if isinstance(name, str)
                and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name)
            ][:100]
        elif key in {"attempt", "trajectory_index", "replay_sequence", "step"}:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                safe[key] = value
        elif isinstance(value, str):
            safe[key] = _safe_event_name(value)
    return safe


__all__ = ["APIResources", "APISettings", "create_app"]
