"""Stable repository contracts and lifecycle values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RiseOutcome(str, Enum):
    COMMITTED = "committed"
    STOPPED = "stopped"
    FAILED = "failed"
    FINALIZATION_FAILED = "finalization_failed"


RISE_TERMINAL_OUTCOMES = frozenset(outcome.value for outcome in RiseOutcome)
ACTIVE_TASK_STATUSES = frozenset(
    {TaskStatus.RUNNING.value, TaskStatus.CANCEL_REQUESTED.value}
)
TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}
)


@dataclass(frozen=True)
class TaskRecord:
    id: UUID
    thread_id: UUID
    task_status: str
    rise_outcome: str | None
    input_locator: str
    input_sha256: str
    request_sha256: str
    idempotency_key: str | None
    execution_config: dict[str, Any]
    worker_id: str | None
    lease_token: UUID | None
    heartbeat_at: datetime | None
    lease_expires_at: datetime | None
    attempt_count: int
    recovery_count: int
    cancel_requested_at: datetime | None
    failure_kind: str | None
    error_type: str | None
    error_summary: str | None
    last_event_seq: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class ExpiredTaskClaim:
    task: TaskRecord
    recovery_exhausted: bool


@dataclass(frozen=True)
class TaskEventRecord:
    event_id: UUID
    task_id: UUID
    seq: int
    event_type: str
    action: str | None
    event_key: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class TaskArtifactRecord:
    artifact_id: UUID
    task_id: UUID
    artifact_type: str
    locator: str
    byte_size: int
    sha256: str
    media_type: str | None
    created_at: datetime


@dataclass(frozen=True)
class TaskPageCursor:
    created_at: datetime
    task_id: UUID


@dataclass(frozen=True)
class CreateArtifact:
    artifact_type: str
    locator: str
    byte_size: int
    sha256: str
    media_type: str | None = None
