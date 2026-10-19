"""Create the P3 business task, event, and artifact tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261019_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_status", sa.String(length=32), server_default="queued", nullable=False),
        sa.Column("rise_outcome", sa.String(length=32), nullable=True),
        sa.Column("input_locator", sa.Text(), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column(
            "execution_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("recovery_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_kind", sa.String(length=64), nullable=True),
        sa.Column("error_type", sa.String(length=255), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("last_event_seq", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "task_status IN ('queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled')",
            name="ck_tasks_status",
        ),
        sa.CheckConstraint(
            "rise_outcome IS NULL OR rise_outcome IN "
            "('committed', 'stopped', 'failed', 'finalization_failed')",
            name="ck_tasks_rise_outcome",
        ),
        sa.CheckConstraint(
            "(task_status = 'completed' AND rise_outcome IS NOT NULL) OR "
            "(task_status <> 'completed' AND rise_outcome IS NULL)",
            name="ck_tasks_completed_has_rise_outcome",
        ),
        sa.CheckConstraint(
            "((task_status IN ('running', 'cancel_requested')) AND "
            "worker_id IS NOT NULL AND lease_token IS NOT NULL AND "
            "heartbeat_at IS NOT NULL AND lease_expires_at IS NOT NULL) OR "
            "((task_status NOT IN ('running', 'cancel_requested')) AND "
            "worker_id IS NULL AND lease_token IS NULL AND "
            "heartbeat_at IS NULL AND lease_expires_at IS NULL)",
            name="ck_tasks_lease_matches_status",
        ),
        sa.CheckConstraint(
            "task_status <> 'cancel_requested' OR cancel_requested_at IS NOT NULL",
            name="ck_tasks_cancel_request_timestamp",
        ),
        sa.CheckConstraint(
            "char_length(input_sha256) = 64 AND input_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_tasks_input_sha256",
        ),
        sa.CheckConstraint(
            "char_length(request_sha256) = 64 AND request_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_tasks_request_sha256",
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_tasks_attempt_count_nonnegative"),
        sa.CheckConstraint("recovery_count >= 0", name="ck_tasks_recovery_count_nonnegative"),
        sa.CheckConstraint("last_event_seq >= 0", name="ck_tasks_event_seq_nonnegative"),
        sa.PrimaryKeyConstraint("id", name="pk_tasks"),
        sa.UniqueConstraint("thread_id", name="uq_tasks_thread_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_tasks_idempotency_key"),
    )
    op.create_index(
        "ix_tasks_queued_created",
        "tasks",
        ["created_at", "id"],
        unique=False,
        postgresql_where=sa.text("task_status = 'queued'"),
    )
    op.create_index(
        "ix_tasks_lease_expiry",
        "tasks",
        ["lease_expires_at", "id"],
        unique=False,
        postgresql_where=sa.text("task_status IN ('running', 'cancel_requested')"),
    )
    op.create_index("ix_tasks_created_id", "tasks", ["created_at", "id"], unique=False)

    op.create_table(
        "task_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=True),
        sa.Column("event_key", sa.String(length=256), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("seq > 0", name="ck_task_events_seq_positive"),
        sa.CheckConstraint("char_length(event_key) > 0", name="ck_task_events_event_key_nonempty"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE", name="fk_task_events_task_id_tasks"),
        sa.PrimaryKeyConstraint("event_id", name="pk_task_events"),
        sa.UniqueConstraint("task_id", "seq", name="uq_task_events_task_seq"),
        sa.UniqueConstraint("task_id", "event_key", name="uq_task_events_task_event_key"),
    )

    op.create_table(
        "task_artifacts",
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("locator", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("char_length(artifact_type) > 0", name="ck_task_artifacts_type_nonempty"),
        sa.CheckConstraint("byte_size >= 0", name="ck_task_artifacts_byte_size_nonnegative"),
        sa.CheckConstraint(
            "char_length(sha256) = 64 AND sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_task_artifacts_sha256",
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE", name="fk_task_artifacts_task_id_tasks"),
        sa.PrimaryKeyConstraint("artifact_id", name="pk_task_artifacts"),
        sa.UniqueConstraint(
            "task_id", "artifact_type", "sha256", name="uq_task_artifacts_task_type_sha256"
        ),
    )


def downgrade() -> None:
    op.drop_table("task_artifacts")
    op.drop_table("task_events")
    op.drop_index("ix_tasks_created_id", table_name="tasks")
    op.drop_index("ix_tasks_lease_expiry", table_name="tasks")
    op.drop_index("ix_tasks_queued_created", table_name="tasks")
    op.drop_table("tasks")
