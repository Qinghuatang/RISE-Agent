"""Real PostgreSQL tests for P3 business persistence.

Run with ``RISE_TEST_POSTGRES_URL=postgresql://... pytest -q
tests/test_business_repository.py``. Each test uses its own schema and applies
the checked-in Alembic migration, so it never touches checkpointer tables.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from research_agent.service.contracts import CreateArtifact, RiseOutcome, TaskPageCursor, TaskStatus
from research_agent.service.db import create_business_engine, create_session_factory, upgrade_business_schema
from research_agent.service.errors import (
    IdempotencyConflict,
    LeaseOwnershipError,
    TaskNotFound,
)
from research_agent.service.repository import TaskRepository


POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set RISE_TEST_POSTGRES_URL to run real PostgreSQL business repository tests",
)


@pytest.fixture
def repository():
    assert POSTGRES_URL is not None
    admin_engine = create_business_engine(POSTGRES_URL)
    schema = f"rise_p3b_{uuid4().hex}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    scoped_engine = create_engine(
        admin_engine.url,
        connect_args={"options": f"-csearch_path={schema}"},
        pool_pre_ping=True,
    )
    try:
        upgrade_business_schema(scoped_engine)
        repo = TaskRepository(create_session_factory(scoped_engine))
        yield repo, scoped_engine, schema
    finally:
        scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin_engine.dispose()


def create_task(repo: TaskRepository, **overrides):
    values = {
        "input_locator": "inputs/paper.md",
        "input_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "execution_config": {"model": "fake", "budget": 4},
    }
    values.update(overrides)
    return repo.create_task(**values)


def claim(repo: TaskRepository, task_id, *, worker="worker-a", lease_seconds=30):
    row = repo.claim_next_queued(worker_id=worker, lease_seconds=lease_seconds)
    assert row is not None and row.id == task_id
    return row


def test_alembic_migration_creates_only_business_tables_and_matches_models(repository):
    _, engine, schema = repository
    names = set(inspect(engine).get_table_names(schema=schema))
    assert {"alembic_version", "tasks", "task_events", "task_artifacts"} <= names
    assert not any(name.startswith("checkpoint") for name in names)

    config = Config("alembic.ini")
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.check(config)


def test_create_get_list_and_transactional_initial_event_and_artifact(repository):
    repo, _, _ = repository
    input_artifact = CreateArtifact(
        artifact_type="input_markdown",
        locator="inputs/paper.md",
        byte_size=37,
        sha256="a" * 64,
        media_type="text/markdown",
    )
    task = create_task(repo, input_artifact=input_artifact)

    loaded = repo.get_task(task.id)
    assert loaded.thread_id == loaded.id
    assert loaded.task_status == TaskStatus.QUEUED.value
    assert loaded.rise_outcome is None
    assert loaded.execution_config == {"model": "fake", "budget": 4}
    assert loaded.last_event_seq == 1
    assert repo.get_by_idempotency_key("missing") is None
    events = repo.list_events(task.id)
    assert [(event.seq, event.event_type) for event in events] == [(1, "task_queued")]
    artifacts = repo.list_artifacts(task.id)
    assert [(artifact.artifact_type, artifact.byte_size, artifact.sha256) for artifact in artifacts] == [
        ("input_markdown", 37, "a" * 64)
    ]

    second = create_task(repo, input_locator="inputs/second.md", input_sha256="c" * 64)
    listed = repo.list_tasks(limit=1)
    assert len(listed) == 1
    cursor = TaskPageCursor(created_at=listed[0].created_at, task_id=listed[0].id)
    next_page = repo.list_tasks(limit=10, cursor=cursor)
    assert len(next_page) == 1
    assert listed[0].id != next_page[0].id
    assert {listed[0].id, next_page[0].id} == {task.id, second.id}


def test_idempotency_key_returns_same_task_or_conflicts_on_different_request(repository):
    repo, _, _ = repository
    first = create_task(repo, idempotency_key="submit-1")
    repeated = create_task(repo, idempotency_key="submit-1")
    assert repeated.id == first.id
    assert repo.get_by_idempotency_key("submit-1").id == first.id

    with pytest.raises(IdempotencyConflict):
        create_task(repo, idempotency_key="submit-1", request_sha256="d" * 64)


def test_concurrent_idempotency_requests_return_the_same_task(repository):
    repo, _, _ = repository
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait(timeout=5)
        return create_task(repo, idempotency_key="racing-submit")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert results[0].id == results[1].id
    assert len(repo.list_tasks(limit=10)) == 1


def test_claim_next_queued_is_safe_for_two_concurrent_claimers(repository):
    repo, _, _ = repository
    tasks = [create_task(repo, input_locator=f"inputs/{i}.md", input_sha256=chr(99 + i) * 64) for i in range(2)]
    barrier = threading.Barrier(2)

    def claim_concurrently(worker):
        barrier.wait(timeout=5)
        return repo.claim_next_queued(worker_id=worker, lease_seconds=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim_concurrently, ["worker-a", "worker-b"]))

    claimed_ids = [item.id for item in claims if item is not None]
    assert len(claimed_ids) == 2
    assert len(set(claimed_ids)) == 2
    assert set(claimed_ids) == {task.id for task in tasks}
    assert all(item.task_status == TaskStatus.RUNNING.value for item in claims)


def test_old_lease_token_cannot_heartbeat_or_complete_after_reclaim(repository):
    repo, _, _ = repository
    task = create_task(repo)
    first_owner = claim(repo, task.id)
    requeued = repo.release_for_recovery(
        task_id=task.id,
        worker_id=first_owner.worker_id,
        lease_token=first_owner.lease_token,
        max_recoveries=3,
        error_type="WorkerRestart",
        error_summary="process stopped",
    )
    assert requeued.task_status == TaskStatus.QUEUED.value
    assert requeued.recovery_count == 1

    second_owner = claim(repo, task.id, worker="worker-b")
    assert second_owner.lease_token != first_owner.lease_token
    assert not repo.heartbeat(
        task_id=task.id,
        worker_id=first_owner.worker_id,
        lease_token=first_owner.lease_token,
        lease_seconds=30,
    )
    with pytest.raises(LeaseOwnershipError):
        repo.mark_completed(
            task_id=task.id,
            worker_id=first_owner.worker_id,
            lease_token=first_owner.lease_token,
            rise_outcome=RiseOutcome.COMMITTED.value,
        )

    assert repo.get_task(task.id).lease_token == second_owner.lease_token


def test_expired_lease_is_reclaimed_only_after_expiry_and_fenced_by_new_token(repository):
    repo, _, _ = repository
    task = create_task(repo)
    owner = claim(repo, task.id, lease_seconds=0.05)
    assert repo.list_expired_tasks() == []
    time.sleep(0.08)
    expired = repo.list_expired_tasks()
    assert [row.id for row in expired] == [task.id]

    reclaimed = repo.reclaim_expired_task(
        task_id=task.id,
        worker_id="recovery-worker",
        lease_seconds=30,
        max_recoveries=3,
    )
    assert reclaimed is not None
    assert reclaimed.recovery_exhausted is False
    assert reclaimed.task.task_status == TaskStatus.RUNNING.value
    assert reclaimed.task.worker_id == "recovery-worker"
    assert reclaimed.task.lease_token != owner.lease_token
    assert reclaimed.task.recovery_count == owner.recovery_count + 1
    assert not repo.heartbeat(
        task_id=task.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        lease_seconds=30,
    )
    with pytest.raises(LeaseOwnershipError):
        repo.mark_completed(
            task_id=task.id,
            worker_id=owner.worker_id,
            lease_token=owner.lease_token,
            rise_outcome=RiseOutcome.COMMITTED.value,
        )


def test_expired_lease_at_recovery_limit_gets_reconciliation_only_claim(repository):
    repo, _, _ = repository
    task = create_task(repo)
    owner = claim(repo, task.id, lease_seconds=0.05)
    time.sleep(0.08)
    reclaimed = repo.reclaim_expired_task(
        task_id=task.id,
        worker_id="recovery-worker",
        lease_seconds=30,
        max_recoveries=0,
    )
    assert reclaimed is not None
    assert reclaimed.recovery_exhausted is True
    assert reclaimed.task.recovery_count == 0
    assert reclaimed.task.lease_token != owner.lease_token


def test_expired_lease_cannot_be_renewed_or_used_for_terminal_transition(repository):
    repo, _, _ = repository
    task = create_task(repo)
    owner = claim(repo, task.id, lease_seconds=0.05)
    time.sleep(0.08)
    assert not repo.heartbeat(
        task_id=task.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        lease_seconds=5,
    )
    with pytest.raises(LeaseOwnershipError):
        repo.mark_failed(
            task_id=task.id,
            worker_id=owner.worker_id,
            lease_token=owner.lease_token,
            failure_kind="infrastructure",
            error_type="TimeoutError",
            error_summary="lease expired",
        )


@pytest.mark.parametrize(
    "outcome",
    ["committed", "stopped", "failed", "finalization_failed"],
)
def test_all_confirmed_rise_outcomes_map_to_completed(outcome, repository):
    repo, _, _ = repository
    task = create_task(repo)
    owner = claim(repo, task.id)
    completed = repo.mark_completed(
        task_id=task.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        rise_outcome=outcome,
        error_type="FinalizerError" if outcome == "finalization_failed" else None,
        error_summary="proposal generation failed" if outcome == "finalization_failed" else None,
    )
    assert completed.task_status == TaskStatus.COMPLETED.value
    assert completed.rise_outcome == outcome
    assert completed.finished_at is not None
    assert completed.lease_token is None
    terminal = repo.list_events(task.id)[-1]
    assert terminal.payload["rise_outcome"] == outcome


def test_infrastructure_and_preflight_failures_have_no_rise_outcome(repository):
    repo, _, _ = repository
    preflight = create_task(repo)
    failed_before_start = repo.fail_before_start(
        task_id=preflight.id,
        failure_kind="input",
        error_type="PaperIntegrityError",
        error_summary="sha256 mismatch",
    )
    assert failed_before_start.task_status == TaskStatus.FAILED.value
    assert failed_before_start.rise_outcome is None
    assert failed_before_start.failure_kind == "input"

    runtime = create_task(repo, input_locator="inputs/runtime.md", input_sha256="c" * 64)
    owner = claim(repo, runtime.id)
    failed = repo.mark_failed(
        task_id=runtime.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        failure_kind="infrastructure",
        error_type="DatabaseUnavailable",
        error_summary="database disconnected",
    )
    assert failed.task_status == TaskStatus.FAILED.value
    assert failed.rise_outcome is None
    assert failed.failure_kind == "infrastructure"


def test_recovery_limit_requeues_then_fails_infrastructure_task(repository):
    repo, _, _ = repository
    task = create_task(repo)
    owner = claim(repo, task.id)
    queued = repo.release_for_recovery(
        task_id=task.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        max_recoveries=1,
        error_type="WorkerExit",
        error_summary="first failure",
    )
    assert queued.task_status == TaskStatus.QUEUED.value
    assert queued.recovery_count == 1

    owner = claim(repo, task.id, worker="worker-b")
    failed = repo.release_for_recovery(
        task_id=task.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        max_recoveries=1,
        error_type="WorkerExit",
        error_summary="recovery limit reached",
    )
    assert failed.task_status == TaskStatus.FAILED.value
    assert failed.rise_outcome is None
    assert failed.recovery_count == 1


def test_cancel_queued_and_running_tasks_and_preserve_confirmed_rise_terminal(repository):
    repo, _, _ = repository
    queued = create_task(repo)
    cancelled_queued = repo.request_cancel(queued.id)
    assert cancelled_queued.task_status == TaskStatus.CANCELLED.value
    assert cancelled_queued.rise_outcome is None

    running = create_task(repo, input_locator="inputs/running.md", input_sha256="c" * 64)
    owner = claim(repo, running.id)
    requested = repo.request_cancel(running.id)
    assert requested.task_status == TaskStatus.CANCEL_REQUESTED.value
    assert requested.lease_token == owner.lease_token
    cancelled = repo.mark_cancelled(
        task_id=running.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
    )
    assert cancelled.task_status == TaskStatus.CANCELLED.value
    assert cancelled.lease_token is None

    cancellation_race = create_task(
        repo, input_locator="inputs/race.md", input_sha256="d" * 64
    )
    owner = claim(repo, cancellation_race.id, worker="worker-c")
    repo.request_cancel(cancellation_race.id)
    completed = repo.mark_completed(
        task_id=cancellation_race.id,
        worker_id=owner.worker_id,
        lease_token=owner.lease_token,
        rise_outcome=RiseOutcome.COMMITTED.value,
    )
    assert completed.task_status == TaskStatus.COMPLETED.value
    assert completed.rise_outcome == RiseOutcome.COMMITTED.value


def test_event_sequence_is_transactional_idempotent_and_rejects_sensitive_payload(repository):
    repo, _, _ = repository
    task = create_task(repo)
    first = repo.append_event(
        task_id=task.id,
        event_type="progress",
        event_key="progress:1",
        action="gap_discovery",
        payload={"step": 1, "summary": "Found a gap"},
    )
    repeated = repo.append_event(
        task_id=task.id,
        event_type="progress",
        event_key="progress:1",
        action="gap_discovery",
        payload={"step": 1, "summary": "Found a gap"},
    )
    assert repeated.event_id == first.event_id
    assert repeated.seq == first.seq == 2
    assert repo.get_task(task.id).last_event_seq == 2

    with pytest.raises(IdempotencyConflict):
        repo.append_event(
            task_id=task.id,
            event_type="progress",
            event_key="progress:1",
            payload={"step": 2},
        )
    with pytest.raises(ValueError, match="sensitive"):
        repo.append_event(
            task_id=task.id,
            event_type="debug",
            event_key="debug:secret",
            payload={"nested": {"api_key": "redacted"}},
        )

    second = repo.append_event(
        task_id=task.id,
        event_type="progress",
        event_key="progress:2",
        payload={
            "step": 2,
            "summary": "request Authorization: Bearer test_token_value_1234567890",
            "url": "https://example.test/?access_token=another_secret_value",
        },
    )
    assert second.seq == 3
    assert "test_token_value_1234567890" not in str(second.payload)
    assert "another_secret_value" not in str(second.payload)
    assert "[REDACTED]" in str(second.payload)
    assert [event.seq for event in repo.list_events(task.id, after_seq=1)] == [2, 3]


def test_concurrent_events_receive_unique_increasing_sequences(repository):
    repo, _, _ = repository
    task = create_task(repo)
    barrier = threading.Barrier(6)

    def append(index):
        barrier.wait(timeout=5)
        return repo.append_event(
            task_id=task.id,
            event_type="progress",
            event_key=f"concurrent:{index}",
            payload={"index": index},
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        events = list(pool.map(append, range(6)))
    assert sorted(event.seq for event in events) == list(range(2, 8))
    assert repo.get_task(task.id).last_event_seq == 7


def test_artifact_metadata_is_idempotent_and_validated(repository):
    repo, _, _ = repository
    task = create_task(repo)
    artifact = CreateArtifact(
        artifact_type="proposal",
        locator="artifacts/proposal.md",
        byte_size=120,
        sha256="c" * 64,
        media_type="text/markdown",
    )
    first = repo.record_artifact(task.id, artifact)
    second = repo.record_artifact(task.id, artifact)
    assert first.artifact_id == second.artifact_id
    assert repo.list_artifacts(task.id) == [first]

    with pytest.raises(IdempotencyConflict, match="different metadata"):
        repo.record_artifact(
            task.id,
            CreateArtifact(
                artifact_type="proposal",
                locator="artifacts/other.md",
                byte_size=120,
                sha256="c" * 64,
            ),
        )
    with pytest.raises(ValueError, match="escape"):
        repo.record_artifact(
            task.id,
            CreateArtifact("proposal", "../proposal.md", 1, "d" * 64),
        )
    with pytest.raises(TaskNotFound):
        repo.record_artifact(uuid4(), artifact)


def test_database_constraints_reject_invalid_status_outcome_and_artifact_size(repository):
    repo, engine, schema = repository
    task = create_task(repo)
    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                text("UPDATE tasks SET task_status='completed' WHERE id=:task_id"),
                {"task_id": task.id},
            )
    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                text("UPDATE tasks SET task_status='not-a-status' WHERE id=:task_id"),
                {"task_id": task.id},
            )
    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO task_artifacts "
                    "(artifact_id, task_id, artifact_type, locator, byte_size, sha256) "
                    "VALUES (gen_random_uuid(), :task_id, 'proposal', 'x.md', -1, :digest)"
                ),
                {"task_id": task.id, "digest": "e" * 64},
            )
