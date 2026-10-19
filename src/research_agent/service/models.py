"""SQLAlchemy models for the business task database.

These tables contain task lifecycle metadata and projections only. The
LangGraph PostgreSQL checkpointer remains the sole authoritative store for
Research State, Research Memory, Trajectory, Budget, and graph execution state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for application-owned tables only."""


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "task_status IN ('queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled')",
            name="ck_tasks_status",
        ),
        CheckConstraint(
            "rise_outcome IS NULL OR rise_outcome IN "
            "('committed', 'stopped', 'failed', 'finalization_failed')",
            name="ck_tasks_rise_outcome",
        ),
        CheckConstraint(
            "(task_status = 'completed' AND rise_outcome IS NOT NULL) OR "
            "(task_status <> 'completed' AND rise_outcome IS NULL)",
            name="ck_tasks_completed_has_rise_outcome",
        ),
        CheckConstraint(
            "((task_status IN ('running', 'cancel_requested')) AND "
            "worker_id IS NOT NULL AND lease_token IS NOT NULL AND "
            "heartbeat_at IS NOT NULL AND lease_expires_at IS NOT NULL) OR "
            "((task_status NOT IN ('running', 'cancel_requested')) AND "
            "worker_id IS NULL AND lease_token IS NULL AND "
            "heartbeat_at IS NULL AND lease_expires_at IS NULL)",
            name="ck_tasks_lease_matches_status",
        ),
        CheckConstraint(
            "task_status <> 'cancel_requested' OR cancel_requested_at IS NOT NULL",
            name="ck_tasks_cancel_request_timestamp",
        ),
        UniqueConstraint("thread_id", name="uq_tasks_thread_id"),
        UniqueConstraint("idempotency_key", name="uq_tasks_idempotency_key"),
        CheckConstraint(
            "char_length(input_sha256) = 64 AND input_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_tasks_input_sha256",
        ),
        CheckConstraint(
            "char_length(request_sha256) = 64 AND request_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_tasks_request_sha256",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_tasks_attempt_count_nonnegative"),
        CheckConstraint("recovery_count >= 0", name="ck_tasks_recovery_count_nonnegative"),
        CheckConstraint("last_event_seq >= 0", name="ck_tasks_event_seq_nonnegative"),
        Index(
            "ix_tasks_queued_created",
            "created_at",
            "id",
            postgresql_where=text("task_status = 'queued'"),
        ),
        Index(
            "ix_tasks_lease_expiry",
            "lease_expires_at",
            "id",
            postgresql_where=text("task_status IN ('running', 'cancel_requested')"),
        ),
        Index("ix_tasks_created_id", "created_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    thread_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    task_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued", server_default=text("'queued'")
    )
    rise_outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)

    input_locator: Mapped[str] = mapped_column(Text, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    execution_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_token: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    recovery_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    failure_kind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    last_event_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TaskEvent(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        UniqueConstraint("task_id", "seq", name="uq_task_events_task_seq"),
        UniqueConstraint("task_id", "event_key", name="uq_task_events_task_event_key"),
        CheckConstraint("seq > 0", name="ck_task_events_seq_positive"),
        CheckConstraint("char_length(event_key) > 0", name="ck_task_events_event_key_nonempty"),
    )

    event_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_key: Mapped[str] = mapped_column(String(256), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TaskArtifact(Base):
    __tablename__ = "task_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "artifact_type", "sha256", name="uq_task_artifacts_task_type_sha256"
        ),
        CheckConstraint("char_length(artifact_type) > 0", name="ck_task_artifacts_type_nonempty"),
        CheckConstraint("byte_size >= 0", name="ck_task_artifacts_byte_size_nonnegative"),
        CheckConstraint(
            "char_length(sha256) = 64 AND sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_task_artifacts_sha256",
        ),
    )

    artifact_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    locator: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
