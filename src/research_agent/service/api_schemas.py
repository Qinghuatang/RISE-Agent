"""Public HTTP response schemas for the RISE task API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskAccepted(APIModel):
    task_id: UUID
    task_status: str
    rise_outcome: str | None
    created_at: datetime
    idempotent_replay: bool = False


class TaskProgress(APIModel):
    event_seq: int
    latest_event_type: str | None = None
    latest_action: str | None = None
    updated_at: datetime


class TaskSummary(APIModel):
    task_id: UUID
    task_status: str
    rise_outcome: str | None
    failure_kind: str | None
    error_type: str | None
    error_summary: str | None
    attempt_count: int
    recovery_count: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: TaskProgress


class TaskList(APIModel):
    items: list[TaskSummary]
    next_cursor: str | None


class TaskResult(APIModel):
    task_id: UUID
    task_status: str
    rise_outcome: str | None
    final_proposal: dict[str, Any] | None
    commit_snapshot: dict[str, Any] | None
    unresolved_issues: list[str]
    budget: dict[str, Any] | None
    trajectory_event_count: int
    replay_record_count: int
    error_type: str | None = None
    failure_kind: str | None = None
    error_summary: str | None = None


__all__ = [
    "TaskAccepted",
    "TaskList",
    "TaskProgress",
    "TaskResult",
    "TaskSummary",
]
