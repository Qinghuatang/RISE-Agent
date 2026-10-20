"""Transactional repository for RISE task lifecycle metadata.

Each public write owns a short SQLAlchemy transaction. Graph checkpoint writes
are deliberately handled by LangGraph's separate PostgresSaver connection.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Iterable
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .contracts import (
    ACTIVE_TASK_STATUSES,
    RISE_TERMINAL_OUTCOMES,
    TERMINAL_TASK_STATUSES,
    CreateArtifact,
    ExpiredTaskClaim,
    TaskArtifactRecord,
    TaskEventRecord,
    TaskPageCursor,
    TaskRecord,
    TaskStatus,
)
from .errors import (
    IdempotencyConflict,
    InvalidTaskTransition,
    LeaseOwnershipError,
    TaskNotFound,
)
from .models import Task, TaskArtifact, TaskEvent

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_EVENT_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "prompt",
    "raw_response",
    "markdown",
    "paper_content",
    "paper_markdown",
    "full_paper",
}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|password|secret)"
    r"(\s*[:=]\s*)(?:bearer\s+)?[\"']?([^\s,;\"']+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_URL_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password|signature)=)[^&#\s]+"
)
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"glpat-[A-Za-z0-9_-]{20,})\b"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_digest(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_locator(locator: str) -> str:
    if not locator or "\\" in locator:
        raise ValueError("artifact locator must be a non-empty POSIX relative path")
    path = PurePosixPath(locator)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("artifact locator must not escape its task directory")
    return locator


def _redact_sensitive_text(value: str) -> str:
    value = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", value)
    value = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", value)
    value = _URL_SECRET_QUERY_RE.sub(r"\1[REDACTED]", value)
    return _KNOWN_TOKEN_RE.sub("[REDACTED]", value)


def _json_object(value: dict[str, Any], *, label: str, reject_sensitive: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")

    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} keys must be strings")
                normalized = key.casefold().replace("-", "_")
                if reject_sensitive and (
                    normalized in _SENSITIVE_EVENT_KEYS
                    or normalized.endswith("_api_key")
                    or normalized.endswith("_secret")
                    or normalized.endswith("_password")
                ):
                    raise ValueError(f"{label} contains a prohibited sensitive field")
                result[key] = sanitize(nested)
            return result
        if isinstance(item, list):
            return [sanitize(nested) for nested in item]
        if reject_sensitive and isinstance(item, str):
            return _redact_sensitive_text(item)
        return item

    sanitized = sanitize(value)
    try:
        return json.loads(
            json.dumps(sanitized, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _safe_summary(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(_redact_sensitive_text(value).split())[:1000]


def _task_record(row: Task) -> TaskRecord:
    return TaskRecord(
        id=row.id,
        thread_id=row.thread_id,
        task_status=row.task_status,
        rise_outcome=row.rise_outcome,
        input_locator=row.input_locator,
        input_sha256=row.input_sha256,
        request_sha256=row.request_sha256,
        idempotency_key=row.idempotency_key,
        execution_config=dict(row.execution_config),
        worker_id=row.worker_id,
        lease_token=row.lease_token,
        heartbeat_at=row.heartbeat_at,
        lease_expires_at=row.lease_expires_at,
        attempt_count=row.attempt_count,
        recovery_count=row.recovery_count,
        cancel_requested_at=row.cancel_requested_at,
        failure_kind=row.failure_kind,
        error_type=row.error_type,
        error_summary=row.error_summary,
        last_event_seq=row.last_event_seq,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _event_record(row: TaskEvent) -> TaskEventRecord:
    return TaskEventRecord(
        event_id=row.event_id,
        task_id=row.task_id,
        seq=row.seq,
        event_type=row.event_type,
        action=row.action,
        event_key=row.event_key,
        payload=dict(row.payload),
        created_at=row.created_at,
    )


def _artifact_record(row: TaskArtifact) -> TaskArtifactRecord:
    return TaskArtifactRecord(
        artifact_id=row.artifact_id,
        task_id=row.task_id,
        artifact_type=row.artifact_type,
        locator=row.locator,
        byte_size=row.byte_size,
        sha256=row.sha256,
        media_type=row.media_type,
        created_at=row.created_at,
    )


class TaskRepository:
    """Task, event and artifact persistence with explicit lifecycle checks."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create_task(
        self,
        *,
        input_locator: str,
        input_sha256: str,
        request_sha256: str,
        idempotency_key: str | None = None,
        execution_config: dict[str, Any] | None = None,
        input_artifact: CreateArtifact | None = None,
        task_id: UUID | None = None,
    ) -> TaskRecord:
        """Create a queued task and its first event/artifact atomically.

        Reusing an idempotency key with the same request digest returns the
        original row; reusing it with another digest raises a conflict.
        """

        input_locator = _validate_locator(input_locator)
        input_sha256 = _validate_digest(input_sha256, "input_sha256")
        request_sha256 = _validate_digest(request_sha256, "request_sha256")
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 255:
                raise ValueError("idempotency_key must contain 1 to 255 characters")
        safe_config = _json_object(
            execution_config or {}, label="execution_config", reject_sensitive=True
        )
        if input_artifact is not None:
            self._validate_artifact(input_artifact)
            if (
                input_artifact.locator != input_locator
                or input_artifact.sha256 != input_sha256
            ):
                raise ValueError("input artifact must match task input reference")

        task_id = task_id or uuid4()
        try:
            with self._sessions.begin() as session:
                if idempotency_key is not None:
                    existing = session.scalar(
                        select(Task).where(Task.idempotency_key == idempotency_key)
                    )
                    if existing is not None:
                        return self._existing_idempotent(existing, request_sha256)

                now = _now()
                task = Task(
                    id=task_id,
                    thread_id=task_id,
                    task_status=TaskStatus.QUEUED.value,
                    input_locator=input_locator,
                    input_sha256=input_sha256,
                    request_sha256=request_sha256,
                    idempotency_key=idempotency_key,
                    execution_config=safe_config,
                    last_event_seq=0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(task)
                session.flush()
                self._append_event_locked(
                    session,
                    task,
                    event_type="task_queued",
                    event_key=f"task:{task.id}:queued",
                    payload={"task_status": TaskStatus.QUEUED.value},
                )
                if input_artifact is not None:
                    self._insert_artifact(session, task.id, input_artifact)
                session.flush()
                return _task_record(task)
        except IntegrityError:
            # The unique key is the final arbiter when identical requests race.
            if idempotency_key is not None:
                existing = self.get_by_idempotency_key(idempotency_key)
                if existing is not None:
                    return self._existing_idempotent(existing, request_sha256)
            raise

    @staticmethod
    def _existing_idempotent(row: Task | TaskRecord, request_sha256: str) -> TaskRecord:
        if row.request_sha256 != request_sha256:
            raise IdempotencyConflict(
                "Idempotency-Key was already used for a different request"
            )
        return row if isinstance(row, TaskRecord) else _task_record(row)

    def get_by_idempotency_key(self, key: str) -> TaskRecord | None:
        with self._sessions() as session:
            row = session.scalar(select(Task).where(Task.idempotency_key == key))
            return _task_record(row) if row is not None else None

    def get_task(self, task_id: UUID) -> TaskRecord:
        with self._sessions() as session:
            row = session.get(Task, task_id)
            if row is None:
                raise TaskNotFound(f"task not found: {task_id}")
            return _task_record(row)

    def list_tasks(
        self,
        *,
        limit: int = 50,
        cursor: TaskPageCursor | None = None,
    ) -> list[TaskRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        statement = select(Task)
        if cursor is not None:
            statement = statement.where(
                or_(
                    Task.created_at < cursor.created_at,
                    and_(Task.created_at == cursor.created_at, Task.id < cursor.task_id),
                )
            )
        statement = statement.order_by(Task.created_at.desc(), Task.id.desc()).limit(limit)
        with self._sessions() as session:
            return [_task_record(row) for row in session.scalars(statement)]

    def claim_next_queued(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
    ) -> TaskRecord | None:
        """Atomically claim the oldest queued task using SKIP LOCKED."""

        if not worker_id or len(worker_id) > 128:
            raise ValueError("worker_id must contain 1 to 128 characters")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._sessions.begin() as session:
            row = session.scalar(
                select(Task)
                .where(Task.task_status == TaskStatus.QUEUED.value)
                .order_by(Task.created_at.asc(), Task.id.asc())
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if row is None:
                return None
            now = _now()
            row.task_status = TaskStatus.RUNNING.value
            row.worker_id = worker_id
            row.lease_token = uuid4()
            row.heartbeat_at = now
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.attempt_count += 1
            row.started_at = row.started_at or now
            row.updated_at = now
            self._append_event_locked(
                session,
                row,
                event_type="task_started",
                event_key=f"task:{row.id}:attempt:{row.attempt_count}:started",
                payload={"attempt": row.attempt_count, "worker_id": worker_id},
            )
            session.flush()
            return _task_record(row)

    def list_expired_tasks(self, *, limit: int = 50) -> list[TaskRecord]:
        """List expired active leases for lock-first recovery discovery.

        This method does not claim or mutate rows. A Worker must first acquire
        the thread advisory lock, then call ``reclaim_expired_task`` to fence
        out any prior executor before changing the lease.
        """

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        now = _now()
        with self._sessions() as session:
            rows = session.scalars(
                select(Task)
                .where(
                    Task.task_status.in_(ACTIVE_TASK_STATUSES),
                    Task.lease_expires_at <= now,
                )
                .order_by(Task.lease_expires_at.asc(), Task.id.asc())
                .limit(limit)
            )
            return [_task_record(row) for row in rows]

    def reclaim_expired_task(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_seconds: float,
        max_recoveries: int,
    ) -> ExpiredTaskClaim | None:
        """Replace an expired lease only after the caller holds the PG thread lock.

        The advisory lock is deliberately acquired by the Worker before this
        transaction. At the retry limit, the new lease is reconciliation-only:
        it lets the Worker inspect a terminal checkpoint, but must not replay a
        non-terminal scientific node.
        """

        if not worker_id or len(worker_id) > 128:
            raise ValueError("worker_id must contain 1 to 128 characters")
        if lease_seconds <= 0 or max_recoveries < 0:
            raise ValueError("lease_seconds must be positive and max_recoveries non-negative")
        with self._sessions.begin() as session:
            row = session.scalar(
                select(Task)
                .where(
                    Task.id == task_id,
                    Task.task_status.in_(ACTIVE_TASK_STATUSES),
                    Task.lease_expires_at <= _now(),
                )
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None

            recovery_exhausted = row.recovery_count >= max_recoveries
            now = _now()
            token = uuid4()
            row.worker_id = worker_id
            row.lease_token = token
            row.heartbeat_at = now
            row.lease_expires_at = now + timedelta(seconds=lease_seconds)
            row.updated_at = now
            if not recovery_exhausted:
                row.recovery_count += 1
                row.attempt_count += 1
                event_type = "task_recovery_started"
            else:
                event_type = "task_reconciliation_started"
            self._append_event_locked(
                session,
                row,
                event_type=event_type,
                event_key=f"task:{row.id}:lease-reclaim:{token}",
                payload={
                    "worker_id": worker_id,
                    "recovery_count": row.recovery_count,
                    "recovery_exhausted": recovery_exhausted,
                },
            )
            session.flush()
            return ExpiredTaskClaim(_task_record(row), recovery_exhausted)

    def record_infrastructure_retry_in_place(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
        lease_seconds: float,
        max_recoveries: int,
        error_type: str,
        error_summary: str,
    ) -> TaskRecord:
        """Count a bounded retry while retaining a cancel-requested lease.

        This is used when a saved CommitSnapshot requires Finalizer completion
        or checkpoint state cannot yet be inspected safely.
        """

        if lease_seconds <= 0 or max_recoveries < 0:
            raise ValueError("lease_seconds must be positive and max_recoveries non-negative")
        with self._sessions.begin() as session:
            row = self._owned_task(session, task_id, worker_id, lease_token)
            if row.recovery_count >= max_recoveries:
                self._mark_failed_locked(
                    session,
                    row,
                    failure_kind="infrastructure",
                    error_type=error_type,
                    error_summary=error_summary,
                )
            else:
                now = _now()
                row.recovery_count += 1
                row.failure_kind = "infrastructure_retry"
                row.error_type = error_type[:255]
                row.error_summary = _safe_summary(error_summary)
                row.heartbeat_at = now
                row.lease_expires_at = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                self._append_event_locked(
                    session,
                    row,
                    event_type="task_infrastructure_retry",
                    event_key=f"task:{row.id}:in_place_recovery:{row.recovery_count}",
                    payload={"recovery_count": row.recovery_count},
                )
            session.flush()
            return _task_record(row)

    def heartbeat(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
        lease_seconds: float,
    ) -> bool:
        """Extend a live lease only while its exact owner still holds it."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _now()
        with self._sessions.begin() as session:
            result = session.execute(
                update(Task)
                .where(
                    Task.id == task_id,
                    Task.worker_id == worker_id,
                    Task.lease_token == lease_token,
                    Task.task_status.in_(ACTIVE_TASK_STATUSES),
                    Task.lease_expires_at > now,
                )
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def mark_completed(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
        rise_outcome: str,
        error_type: str | None = None,
        error_summary: str | None = None,
    ) -> TaskRecord:
        """Record any confirmed RISE terminal outcome as completed work."""

        if rise_outcome not in RISE_TERMINAL_OUTCOMES:
            raise ValueError(f"unknown RISE terminal outcome: {rise_outcome!r}")
        with self._sessions.begin() as session:
            row = self._owned_task(session, task_id, worker_id, lease_token)
            if row.task_status not in ACTIVE_TASK_STATUSES:
                raise InvalidTaskTransition("only an active task can complete")
            now = _now()
            row.task_status = TaskStatus.COMPLETED.value
            row.rise_outcome = rise_outcome
            row.failure_kind = "rise" if rise_outcome in {"failed", "finalization_failed"} else None
            row.error_type = error_type[:255] if error_type else None
            row.error_summary = _safe_summary(error_summary)
            row.finished_at = now
            row.updated_at = now
            self._clear_lease(row)
            self._append_event_locked(
                session,
                row,
                event_type="task_completed",
                event_key=f"task:{row.id}:terminal",
                payload={"task_status": TaskStatus.COMPLETED.value, "rise_outcome": rise_outcome},
            )
            session.flush()
            return _task_record(row)

    def fail_before_start(
        self,
        *,
        task_id: UUID,
        failure_kind: str,
        error_type: str,
        error_summary: str,
    ) -> TaskRecord:
        """Persist preflight/input failure while the task is still queued."""

        with self._sessions.begin() as session:
            row = self._task_for_update(session, task_id)
            if row.task_status != TaskStatus.QUEUED.value:
                raise InvalidTaskTransition("pre-start failure requires a queued task")
            self._mark_failed_locked(
                session,
                row,
                failure_kind=failure_kind,
                error_type=error_type,
                error_summary=error_summary,
            )
            session.flush()
            return _task_record(row)

    def mark_failed(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
        failure_kind: str,
        error_type: str,
        error_summary: str,
    ) -> TaskRecord:
        """Record a non-RISE terminal failure owned by the active Worker."""

        with self._sessions.begin() as session:
            row = self._owned_task(session, task_id, worker_id, lease_token)
            if row.task_status not in ACTIVE_TASK_STATUSES:
                raise InvalidTaskTransition("only an active task can fail")
            self._mark_failed_locked(
                session,
                row,
                failure_kind=failure_kind,
                error_type=error_type,
                error_summary=error_summary,
            )
            session.flush()
            return _task_record(row)

    def release_for_recovery(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
        max_recoveries: int,
        error_type: str,
        error_summary: str,
    ) -> TaskRecord:
        """Atomically queue another infrastructure attempt or exhaust retries.

        This is a repository transition only; the P3-C worker decides whether
        an error is transient and when it is safe to invoke it.
        """

        if max_recoveries < 0:
            raise ValueError("max_recoveries must be non-negative")
        with self._sessions.begin() as session:
            row = self._owned_task(session, task_id, worker_id, lease_token)
            if row.task_status == TaskStatus.CANCEL_REQUESTED.value:
                self._mark_cancelled_locked(session, row)
            elif row.task_status != TaskStatus.RUNNING.value:
                raise InvalidTaskTransition("only a running task can be requeued")
            elif row.recovery_count < max_recoveries:
                now = _now()
                row.recovery_count += 1
                row.task_status = TaskStatus.QUEUED.value
                row.failure_kind = "infrastructure_retry"
                row.error_type = error_type[:255]
                row.error_summary = _safe_summary(error_summary)
                row.updated_at = now
                self._clear_lease(row)
                self._append_event_locked(
                    session,
                    row,
                    event_type="task_requeued_for_recovery",
                    event_key=f"task:{row.id}:recovery:{row.recovery_count}",
                    payload={"recovery_count": row.recovery_count},
                )
            else:
                self._mark_failed_locked(
                    session,
                    row,
                    failure_kind="infrastructure",
                    error_type=error_type,
                    error_summary=error_summary,
                )
            session.flush()
            return _task_record(row)

    def request_cancel(self, task_id: UUID) -> TaskRecord:
        """Cancel queued work immediately or flag running work for a safe edge."""

        with self._sessions.begin() as session:
            row = self._task_for_update(session, task_id)
            if row.task_status == TaskStatus.QUEUED.value:
                self._mark_cancelled_locked(session, row)
            elif row.task_status == TaskStatus.RUNNING.value:
                now = _now()
                row.task_status = TaskStatus.CANCEL_REQUESTED.value
                row.cancel_requested_at = now
                row.updated_at = now
                self._append_event_locked(
                    session,
                    row,
                    event_type="task_cancel_requested",
                    event_key=f"task:{row.id}:cancel-requested",
                    payload={"task_status": TaskStatus.CANCEL_REQUESTED.value},
                )
            elif row.task_status == TaskStatus.CANCEL_REQUESTED.value:
                pass
            elif row.task_status in TERMINAL_TASK_STATUSES:
                pass
            else:
                raise InvalidTaskTransition(f"cannot cancel task in state {row.task_status!r}")
            session.flush()
            return _task_record(row)

    def mark_cancelled(
        self,
        *,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
    ) -> TaskRecord:
        """Acknowledge a running cancellation at an execution-safe boundary."""

        with self._sessions.begin() as session:
            row = self._owned_task(session, task_id, worker_id, lease_token)
            if row.task_status != TaskStatus.CANCEL_REQUESTED.value:
                raise InvalidTaskTransition("safe cancellation requires cancel_requested")
            self._mark_cancelled_locked(session, row)
            session.flush()
            return _task_record(row)

    def append_event(
        self,
        *,
        task_id: UUID,
        event_type: str,
        event_key: str,
        payload: dict[str, Any] | None = None,
        action: str | None = None,
    ) -> TaskEventRecord:
        """Append an idempotent task event with a transactionally allocated sequence."""

        safe_payload = _json_object(
            payload or {}, label="event payload", reject_sensitive=True
        )
        self._validate_event_fields(event_type, event_key, action)
        with self._sessions.begin() as session:
            task = self._task_for_update(session, task_id)
            event = self._append_event_locked(
                session,
                task,
                event_type=event_type,
                event_key=event_key,
                payload=safe_payload,
                action=action,
            )
            session.flush()
            return event

    def list_events(
        self,
        task_id: UUID,
        *,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[TaskEventRecord]:
        if after_seq < 0 or not 1 <= limit <= 500:
            raise ValueError("after_seq must be non-negative and limit must be 1 to 500")
        with self._sessions() as session:
            if session.get(Task, task_id) is None:
                raise TaskNotFound(f"task not found: {task_id}")
            rows = session.scalars(
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id, TaskEvent.seq > after_seq)
                .order_by(TaskEvent.seq.asc())
                .limit(limit)
            )
            return [_event_record(row) for row in rows]

    def record_artifact(self, task_id: UUID, artifact: CreateArtifact) -> TaskArtifactRecord:
        """Persist immutable artifact metadata idempotently."""

        self._validate_artifact(artifact)
        try:
            with self._sessions.begin() as session:
                if session.get(Task, task_id) is None:
                    raise TaskNotFound(f"task not found: {task_id}")
                row = session.scalar(
                    select(TaskArtifact).where(
                        TaskArtifact.task_id == task_id,
                        TaskArtifact.artifact_type == artifact.artifact_type,
                        TaskArtifact.sha256 == artifact.sha256,
                    )
                )
                if row is not None:
                    self._assert_same_artifact(row, artifact)
                    return _artifact_record(row)
                row = self._insert_artifact(session, task_id, artifact)
                session.flush()
                return _artifact_record(row)
        except IntegrityError:
            with self._sessions() as session:
                row = session.scalar(
                    select(TaskArtifact).where(
                        TaskArtifact.task_id == task_id,
                        TaskArtifact.artifact_type == artifact.artifact_type,
                        TaskArtifact.sha256 == artifact.sha256,
                    )
                )
                if row is None:
                    raise
                self._assert_same_artifact(row, artifact)
                return _artifact_record(row)

    def list_artifacts(self, task_id: UUID) -> list[TaskArtifactRecord]:
        with self._sessions() as session:
            if session.get(Task, task_id) is None:
                raise TaskNotFound(f"task not found: {task_id}")
            rows = session.scalars(
                select(TaskArtifact)
                .where(TaskArtifact.task_id == task_id)
                .order_by(TaskArtifact.created_at.asc(), TaskArtifact.artifact_id.asc())
            )
            return [_artifact_record(row) for row in rows]

    def _task_for_update(self, session: Session, task_id: UUID) -> Task:
        row = session.scalar(select(Task).where(Task.id == task_id).with_for_update())
        if row is None:
            raise TaskNotFound(f"task not found: {task_id}")
        return row

    def _owned_task(
        self,
        session: Session,
        task_id: UUID,
        worker_id: str,
        lease_token: UUID,
    ) -> Task:
        row = self._task_for_update(session, task_id)
        now = _now()
        if (
            row.task_status not in ACTIVE_TASK_STATUSES
            or row.worker_id != worker_id
            or row.lease_token != lease_token
            or row.lease_expires_at is None
            or row.lease_expires_at <= now
        ):
            raise LeaseOwnershipError("worker does not hold the current live task lease")
        return row

    @staticmethod
    def _clear_lease(row: Task) -> None:
        row.worker_id = None
        row.lease_token = None
        row.heartbeat_at = None
        row.lease_expires_at = None

    def _mark_failed_locked(
        self,
        session: Session,
        row: Task,
        *,
        failure_kind: str,
        error_type: str,
        error_summary: str,
    ) -> None:
        now = _now()
        row.task_status = TaskStatus.FAILED.value
        row.rise_outcome = None
        row.failure_kind = failure_kind[:64]
        row.error_type = error_type[:255]
        row.error_summary = _safe_summary(error_summary)
        row.finished_at = now
        row.updated_at = now
        self._clear_lease(row)
        self._append_event_locked(
            session,
            row,
            event_type="task_failed",
            event_key=f"task:{row.id}:terminal",
            payload={"task_status": TaskStatus.FAILED.value, "failure_kind": row.failure_kind},
        )

    def _mark_cancelled_locked(self, session: Session, row: Task) -> None:
        now = _now()
        row.task_status = TaskStatus.CANCELLED.value
        row.rise_outcome = None
        row.finished_at = now
        row.updated_at = now
        self._clear_lease(row)
        self._append_event_locked(
            session,
            row,
            event_type="task_cancelled",
            event_key=f"task:{row.id}:terminal",
            payload={"task_status": TaskStatus.CANCELLED.value},
        )

    def _append_event_locked(
        self,
        session: Session,
        task: Task,
        *,
        event_type: str,
        event_key: str,
        payload: dict[str, Any],
        action: str | None = None,
    ) -> TaskEventRecord:
        self._validate_event_fields(event_type, event_key, action)
        safe_payload = _json_object(payload, label="event payload", reject_sensitive=True)
        existing = session.scalar(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.event_key == event_key,
            )
        )
        if existing is not None:
            if (
                existing.event_type != event_type
                or existing.action != action
                or existing.payload != safe_payload
            ):
                raise IdempotencyConflict("event_key was reused with different event data")
            return _event_record(existing)

        task.last_event_seq += 1
        task.updated_at = _now()
        event = TaskEvent(
            task_id=task.id,
            seq=task.last_event_seq,
            event_type=event_type,
            action=action,
            event_key=event_key,
            payload=safe_payload,
        )
        session.add(event)
        session.flush()
        return _event_record(event)

    @staticmethod
    def _validate_event_fields(
        event_type: str, event_key: str, action: str | None
    ) -> None:
        if not event_type or len(event_type) > 64:
            raise ValueError("event_type must contain 1 to 64 characters")
        if not event_key or len(event_key) > 256:
            raise ValueError("event_key must contain 1 to 256 characters")
        if action is not None and (not action or len(action) > 64):
            raise ValueError("action must contain 1 to 64 characters")

    @staticmethod
    def _validate_artifact(artifact: CreateArtifact) -> None:
        if not artifact.artifact_type or len(artifact.artifact_type) > 64:
            raise ValueError("artifact_type must contain 1 to 64 characters")
        _validate_locator(artifact.locator)
        _validate_digest(artifact.sha256, "artifact sha256")
        if artifact.byte_size < 0:
            raise ValueError("artifact byte_size must be non-negative")
        if artifact.media_type is not None and len(artifact.media_type) > 128:
            raise ValueError("media_type must be at most 128 characters")

    @staticmethod
    def _insert_artifact(
        session: Session, task_id: UUID, artifact: CreateArtifact
    ) -> TaskArtifact:
        row = TaskArtifact(
            task_id=task_id,
            artifact_type=artifact.artifact_type,
            locator=artifact.locator,
            byte_size=artifact.byte_size,
            sha256=artifact.sha256,
            media_type=artifact.media_type,
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def _assert_same_artifact(row: TaskArtifact, artifact: CreateArtifact) -> None:
        if (
            row.locator != artifact.locator
            or row.byte_size != artifact.byte_size
            or row.media_type != artifact.media_type
        ):
            raise IdempotencyConflict("artifact digest already has different metadata")
