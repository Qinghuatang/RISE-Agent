"""P3-C Worker tests backed by real PostgreSQL business/checkpoint stores."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from research_agent.controller.budget import BudgetTracker
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.paper_store import PaperInputStore
from research_agent.service.advisory_lock import ThreadAdvisoryLock
from research_agent.service.contracts import TaskStatus
from research_agent.service.db import create_business_engine, create_session_factory, upgrade_business_schema
from research_agent.service.repository import TaskRepository
from research_agent.service.worker import TaskWorker, WorkerRuntime

from test_m11 import FakeLLMClient, modules, reading, seed, solution_verification

POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set RISE_TEST_POSTGRES_URL to run real PostgreSQL Worker tests",
)
PAPER_MARKDOWN = "\n".join(reading().parsed.lines) + "\n"
PAPER_BYTES = PAPER_MARKDOWN.encode("utf-8")
PAPER_SHA256 = hashlib.sha256(PAPER_BYTES).hexdigest()
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker_database():
    assert POSTGRES_URL is not None
    admin = create_business_engine(POSTGRES_URL)
    schema = f"rise_p3c_{uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped_url = make_url(POSTGRES_URL).update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)
    engine = create_business_engine(scoped_url)
    try:
        upgrade_business_schema(engine)
        yield TaskRepository(create_session_factory(engine)), scoped_url, schema
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def create_task(repository: TaskRepository, name: str = "paper"):
    request_digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return repository.create_task(
        input_locator=f"inputs/{name}.md",
        input_sha256=PAPER_SHA256,
        request_sha256=request_digest,
        execution_config={"max_steps": 5, "max_searches": 1},
    )


class RuntimeFactory:
    def __init__(
        self,
        saver,
        paper_root: Path,
        action_counter: Path | None = None,
        *,
        resumed_after_gap_verification: bool = False,
        checkpoint_recovery_only: bool = False,
    ):
        self.saver = saver
        self.paper_root = paper_root
        self.action_counter = action_counter
        self.resumed_after_gap_verification = resumed_after_gap_verification
        self.checkpoint_recovery_only = checkpoint_recovery_only
        self.runners: dict[str, LangGraphTrajectoryRunner] = {}
        self.bundles = {}

    def __call__(self, task):
        bundle = modules()
        if self.resumed_after_gap_verification:
            bundle.related_work_verification.llm_client = FakeLLMClient(
                [solution_verification()]
            )
        if self.action_counter is not None:
            original = bundle.related_work_verification.verify_gap

            def count_gap_verification(*args, **kwargs):
                with self.action_counter.open("a", encoding="utf-8") as stream:
                    stream.write("verify_gap\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                return original(*args, **kwargs)

            bundle.related_work_verification.verify_gap = count_gap_verification
        runner = LangGraphTrajectoryRunner(
            bundle,
            budget=BudgetTracker(max_steps=5, max_searches=1),
            checkpointer=self.saver,
            paper_store=PaperInputStore(self.paper_root),
        )
        self.runners[str(task.id)] = runner
        self.bundles[str(task.id)] = bundle
        if self.checkpoint_recovery_only:
            return WorkerRuntime(
                runner,
                lambda: pytest.fail("recovery must not re-run Seed Context Extraction"),
                lambda: pytest.fail("recovery must use the persisted paper reference"),
            )
        return WorkerRuntime(runner, seed(), PAPER_BYTES)


def make_worker(repository, database_url, factory, **kwargs):
    return TaskWorker(
        repository,
        factory,
        database_url=database_url,
        worker_id=kwargs.pop("worker_id", f"test-worker-{uuid4().hex[:8]}"),
        lease_seconds=kwargs.pop("lease_seconds", 3.0),
        heartbeat_interval=kwargs.pop("heartbeat_interval", 0.1),
        poll_interval=0.02,
        **kwargs,
    )


def test_worker_executes_queued_tasks_serially_and_projects_confirmed_actions(worker_database, tmp_path):
    repository, url, _ = worker_database
    tasks = [create_task(repository, f"paper-{index}") for index in range(2)]

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(saver, tmp_path / "papers")
        worker = make_worker(repository, url, factory)
        assert worker.run_once() is True
        statuses_after_first = [repository.get_task(task.id).task_status for task in tasks]
        assert statuses_after_first.count(TaskStatus.COMPLETED.value) == 1
        assert statuses_after_first.count(TaskStatus.QUEUED.value) == 1

        assert worker.run_once() is True
        assert all(repository.get_task(task.id).task_status == TaskStatus.COMPLETED.value for task in tasks)
        for task in tasks:
            events = repository.list_events(task.id)
            assert sum(event.event_type == "rise_action_confirmed" for event in events) == 5
            assert sum(event.event_type == "rise_bootstrap_confirmed" for event in events) == 1
            assert repository.get_task(task.id).attempt_count == 1


def test_advisory_lock_blocks_cross_process_execution_and_releases_cleanly(worker_database):
    _, url, _ = worker_database
    thread_id = str(uuid4())
    sqlalchemy_url = make_url(url).set(drivername="postgresql+psycopg").render_as_string(
        hide_password=False
    )
    lock = ThreadAdvisoryLock(sqlalchemy_url, thread_id)
    assert lock.acquire(timeout=0) is True

    child_code = """
import os
from research_agent.service.advisory_lock import ThreadAdvisoryLock
lock = ThreadAdvisoryLock(os.environ['RISE_TEST_POSTGRES_URL'], os.environ['RISE_THREAD_ID'])
print(lock.acquire(timeout=0))
lock.close()
"""
    environment = {
        **os.environ,
        "RISE_TEST_POSTGRES_URL": url,
        "RISE_THREAD_ID": thread_id,
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
    }
    blocked = subprocess.run(
        [sys.executable, "-c", child_code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert blocked.stdout.strip() == "False"
    lock.close()

    acquired = subprocess.run(
        [sys.executable, "-c", child_code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert acquired.stdout.strip() == "True"


def test_worker_process_fails_stop_when_lease_ownership_is_lost():
    child_code = """
import time
from types import SimpleNamespace
from uuid import uuid4
from research_agent.service.worker import _LeaseHeartbeat
class Repository:
    def heartbeat(self, **kwargs):
        return False
heartbeat = _LeaseHeartbeat(
    Repository(),
    SimpleNamespace(id=uuid4(), worker_id="lost-owner", lease_token=uuid4()),
    lease_seconds=2,
    interval=0.01,
    fail_stop_on_ownership_loss=True,
)
heartbeat.start()
time.sleep(2)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 75


def test_expired_lease_waits_for_advisory_lock_before_reclaim(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository)
    owner = repository.claim_next_queued(worker_id="old-worker", lease_seconds=0.08)
    assert owner is not None
    lock = ThreadAdvisoryLock(url, str(task.thread_id))
    assert lock.acquire(timeout=0) is True
    time.sleep(0.11)

    with open_postgres_checkpointer(url) as saver:
        worker = make_worker(repository, url, RuntimeFactory(saver, tmp_path / "papers"))
        assert worker.run_once() is False
        still_owned = repository.get_task(task.id)
        assert still_owned.lease_token == owner.lease_token
        assert still_owned.recovery_count == 0

        lock.close()
        assert worker.run_once() is True
        completed = repository.get_task(task.id)
        assert completed.task_status == TaskStatus.COMPLETED.value
        assert completed.recovery_count == 1


def test_action_boundary_cancellation_stops_before_the_next_scientific_action(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository)

    class CancelAfterFirstAction(TaskWorker):
        cancelled = False

        def _after_graph_checkpoint(self, saved_task, state):
            if state.budget.steps_used == 1 and not self.cancelled:
                self.cancelled = True
                repository.request_cancel(saved_task.id)

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(saver, tmp_path / "papers")
        worker = CancelAfterFirstAction(
            repository,
            factory,
            database_url=url,
            worker_id="cancel-worker",
            lease_seconds=3,
            heartbeat_interval=0.1,
            poll_interval=0.02,
        )
        assert worker.run_once() is True
        cancelled = repository.get_task(task.id)
        assert cancelled.task_status == TaskStatus.CANCELLED.value
        assert cancelled.rise_outcome is None
        runner = factory.runners[str(task.id)]
        state = runner.load_persistent_state(thread_id=str(task.thread_id))
        assert state is not None
        assert state.budget.steps_used == 1
        assert len([record for record in state.replay_records if record.kind == "action"]) == 1
        assert sum(event.event_type == "rise_action_confirmed" for event in repository.list_events(task.id)) == 1


def test_commit_snapshot_wins_cancellation_race_and_finalizer_completes(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository)

    class CancelAfterCommit(TaskWorker):
        cancelled = False

        def _after_graph_checkpoint(self, saved_task, state):
            if state.commit_snapshot is not None and state.rise_status == "running" and not self.cancelled:
                self.cancelled = True
                repository.request_cancel(saved_task.id)

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(saver, tmp_path / "papers")
        worker = CancelAfterCommit(
            repository,
            factory,
            database_url=url,
            worker_id="commit-cancel-worker",
            lease_seconds=3,
            heartbeat_interval=0.1,
            poll_interval=0.02,
        )
        assert worker.run_once() is True
        finished = repository.get_task(task.id)
        assert finished.task_status == TaskStatus.COMPLETED.value
        assert finished.rise_outcome == "committed"
        state = factory.runners[str(task.id)].load_persistent_state(thread_id=str(task.thread_id))
        assert state is not None and state.final_proposal is not None
        assert factory.bundles[str(task.id)].finalizer.call_count == 1


def test_commit_snapshot_survives_projection_failure_during_cancellation(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository, "commit-projection-race")

    class FailProjectionAfterCommit(TaskWorker):
        cancel_sent = False
        projection_failed = False

        def _after_graph_checkpoint(self, saved_task, state):
            if state.commit_snapshot is not None and state.rise_status == "running" and not self.cancel_sent:
                self.cancel_sent = True
                repository.request_cancel(saved_task.id)

        def _project_state(self, saved_task, state):
            if state.commit_snapshot is not None and state.rise_status == "running" and not self.projection_failed:
                self.projection_failed = True
                raise ConnectionError("simulated task-event database interruption")
            return super()._project_state(saved_task, state)

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(saver, tmp_path / "papers")
        worker = FailProjectionAfterCommit(
            repository,
            factory,
            database_url=url,
            worker_id="commit-projection-worker",
            lease_seconds=3,
            heartbeat_interval=0.1,
            poll_interval=0.02,
        )
        assert worker.run_once() is True
        finished = repository.get_task(task.id)
        assert finished.task_status == TaskStatus.COMPLETED.value
        assert finished.rise_outcome == "committed"
        assert finished.recovery_count == 1
        state = factory.runners[str(task.id)].load_persistent_state(thread_id=str(task.thread_id))
        assert state is not None and state.final_proposal is not None
        assert sum(event.event_type == "rise_action_confirmed" for event in repository.list_events(task.id)) == 5


def test_heartbeat_runs_while_fake_retrieval_action_is_blocked(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(saver, tmp_path / "papers")
        runtime = factory(task)
        original = runtime.runner.modules.related_work_verification.verify_gap

        def block_action(*args, **kwargs):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test did not release blocked action")
            return original(*args, **kwargs)

        runtime.runner.modules.related_work_verification.verify_gap = block_action
        worker = make_worker(
            repository,
            url,
            lambda _: runtime,
            lease_seconds=0.6,
            heartbeat_interval=0.08,
            worker_id="heartbeat-worker",
        )

        def run_worker():
            try:
                worker.run_once()
            except BaseException as exc:  # surfaced in the main test thread
                errors.append(exc)

        thread = threading.Thread(target=run_worker)
        thread.start()
        assert entered.wait(5)
        first = repository.get_task(task.id)
        time.sleep(0.25)
        refreshed = repository.get_task(task.id)
        assert refreshed.heartbeat_at > first.heartbeat_at
        assert refreshed.lease_expires_at > refreshed.heartbeat_at
        release.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
        assert repository.get_task(task.id).task_status == TaskStatus.COMPLETED.value


def test_worker_infrastructure_retry_is_distinct_from_scientific_failure(worker_database, tmp_path):
    repository, url, _ = worker_database
    infra_task = create_task(repository, "infra")
    scientific_task = create_task(repository, "science")
    calls = {str(infra_task.id): 0}

    with open_postgres_checkpointer(url) as saver:
        working_factory = RuntimeFactory(saver, tmp_path / "papers")

        def factory(task):
            if task.id == infra_task.id:
                calls[str(task.id)] += 1
                if calls[str(task.id)] == 1:
                    raise ConnectionError("temporary runtime configuration outage")
            bundle = modules()
            if task.id == scientific_task.id:
                def fail_action(*args, **kwargs):
                    raise RuntimeError("deterministic scientific Action failure")
                bundle.related_work_verification.verify_gap = fail_action
            runner = LangGraphTrajectoryRunner(
                bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=saver,
                paper_store=PaperInputStore(tmp_path / f"papers-{task.id}"),
            )
            working_factory.runners[str(task.id)] = runner
            working_factory.bundles[str(task.id)] = bundle
            return WorkerRuntime(runner, seed(), PAPER_BYTES)

        worker = make_worker(repository, url, factory, worker_id="retry-worker")
        assert worker.run_once() is True
        assert repository.get_task(infra_task.id).task_status == TaskStatus.QUEUED.value
        assert repository.get_task(infra_task.id).recovery_count == 1
        # FIFO order may choose either task after the first; drain both.
        while worker.run_once():
            if all(repository.get_task(item.id).task_status in {"completed", "failed"} for item in (infra_task, scientific_task)):
                break

        infra_result = repository.get_task(infra_task.id)
        science_result = repository.get_task(scientific_task.id)
        assert infra_result.task_status == TaskStatus.COMPLETED.value
        assert infra_result.rise_outcome == "committed"
        assert infra_result.recovery_count == 1, [
            (event.event_type, event.payload)
            for event in repository.list_events(infra_task.id)
        ]
        assert calls[str(infra_task.id)] == 2
        assert science_result.task_status == TaskStatus.COMPLETED.value
        assert science_result.rise_outcome == "failed"
        assert science_result.recovery_count == 0
        assert "Action failure" in (science_result.error_summary or "")


def test_worker_marks_infrastructure_failed_after_recovery_limit(worker_database):
    repository, url, _ = worker_database
    task = create_task(repository, "recovery-limit")

    def unavailable_runtime(_task):
        raise ConnectionError("deterministic infrastructure outage")

    worker = make_worker(
        repository,
        url,
        unavailable_runtime,
        max_recoveries=2,
        worker_id="recovery-limit-worker",
    )
    assert worker.run_once() is True
    assert repository.get_task(task.id).task_status == TaskStatus.QUEUED.value
    assert worker.run_once() is True
    assert repository.get_task(task.id).task_status == TaskStatus.QUEUED.value
    assert worker.run_once() is True
    failed = repository.get_task(task.id)
    assert failed.task_status == TaskStatus.FAILED.value
    assert failed.rise_outcome is None
    assert failed.failure_kind == "infrastructure"
    assert failed.recovery_count == 2


def test_worker_stops_after_advisory_session_loss_and_recovers_saved_action(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository, "lock-loss")
    entered = threading.Event()
    release = threading.Event()
    call_counter = tmp_path / "verify-gap-calls.txt"

    class CaptureLockWorker(TaskWorker):
        execution_lock = None

        def _execute_claimed(self, claimed, heartbeat, lock, recovery_exhausted):
            self.execution_lock = lock
            return super()._execute_claimed(claimed, heartbeat, lock, recovery_exhausted)

    with open_postgres_checkpointer(url) as saver:
        initial_factory = RuntimeFactory(saver, tmp_path / "papers", call_counter)
        runtime = initial_factory(task)
        verify_gap = runtime.runner.modules.related_work_verification.verify_gap

        def block_verified_action(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the scientific Action")
            return verify_gap(*args, **kwargs)

        runtime.runner.modules.related_work_verification.verify_gap = block_verified_action
        first_worker = make_worker(
            repository,
            url,
            lambda _task: runtime,
            lease_seconds=1.2,
            heartbeat_interval=0.08,
            worker_id="lock-loss-worker",
            fail_stop_on_ownership_loss=False,
        )
        captured = CaptureLockWorker(
            repository,
            lambda _task: runtime,
            database_url=url,
            worker_id=first_worker.worker_id,
            lease_seconds=1.2,
            heartbeat_interval=0.08,
            poll_interval=0.02,
            fail_stop_on_ownership_loss=False,
        )
        errors: list[BaseException] = []

        def run_first_attempt():
            try:
                captured.run_once()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run_first_attempt)
        thread.start()
        assert entered.wait(5)
        assert captured.execution_lock is not None
        captured.execution_lock.close()
        time.sleep(0.2)
        release.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
        suspended = repository.get_task(task.id)
        assert suspended.task_status == TaskStatus.RUNNING.value
        assert suspended.lease_expires_at is not None
        saved = runtime.runner.load_persistent_state(thread_id=str(task.thread_id))
        assert saved is not None and saved.budget.steps_used == 1

        deadline = time.monotonic() + 5
        while not repository.list_expired_tasks() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert [item.id for item in repository.list_expired_tasks()] == [task.id]

        resumed_factory = RuntimeFactory(
            saver,
            tmp_path / "papers",
            call_counter,
            resumed_after_gap_verification=True,
            checkpoint_recovery_only=True,
        )
        resumed_worker = make_worker(
            repository,
            url,
            resumed_factory,
            lease_seconds=1.2,
            heartbeat_interval=0.08,
            worker_id="lock-loss-recovery-worker",
        )
        assert resumed_worker.run_once() is True
        completed = repository.get_task(task.id)
        assert completed.task_status == TaskStatus.COMPLETED.value
        assert completed.rise_outcome == "committed"
        assert completed.recovery_count == 1
        assert call_counter.read_text(encoding="utf-8").splitlines() == ["verify_gap"]


def test_worker_process_crash_recovers_after_action_checkpoint_and_backfills_events(worker_database, tmp_path):
    repository, url, _ = worker_database
    task = create_task(repository, "process-crash")
    counter_path = tmp_path / "verify-gap-calls.txt"
    paper_root = tmp_path / "papers"
    child_code = r'''
import os
from pathlib import Path
from research_agent.controller.budget import BudgetTracker
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.paper_store import PaperInputStore
from research_agent.service.db import create_business_engine, create_session_factory
from research_agent.service.repository import TaskRepository
from research_agent.service.worker import TaskWorker, WorkerRuntime
from test_m11 import FakeLLMClient, modules, reading, seed, solution_verification

paper_markdown = "\n".join(reading().parsed.lines) + "\n"
class Factory:
    def __call__(self, task):
        bundle = modules()
        original = bundle.related_work_verification.verify_gap
        def count(*args, **kwargs):
            with open(os.environ["RISE_ACTION_COUNTER"], "a", encoding="utf-8") as stream:
                stream.write("verify_gap\n")
                stream.flush()
                os.fsync(stream.fileno())
            return original(*args, **kwargs)
        bundle.related_work_verification.verify_gap = count
        runner = LangGraphTrajectoryRunner(
            bundle,
            budget=BudgetTracker(max_steps=5, max_searches=1),
            checkpointer=saver,
            paper_store=PaperInputStore(os.environ["RISE_PAPER_ROOT"]),
        )
        return WorkerRuntime(runner, seed(), paper_markdown)
class CrashWorker(TaskWorker):
    def _after_graph_checkpoint(self, task, state):
        if state.budget.steps_used >= 1:
            os._exit(73)
with open_postgres_checkpointer(os.environ["RISE_TEST_POSTGRES_URL"]) as saver:
    engine = create_business_engine(os.environ["RISE_TEST_POSTGRES_URL"])
    repository = TaskRepository(create_session_factory(engine))
    worker = CrashWorker(repository, Factory(), database_url=os.environ["RISE_TEST_POSTGRES_URL"], worker_id="killed-worker", lease_seconds=3.0, heartbeat_interval=0.1, poll_interval=0.02)
    worker.run_once()
os._exit(0)
'''
    environment = {
        **os.environ,
        "RISE_TEST_POSTGRES_URL": url,
        "RISE_ACTION_COUNTER": str(counter_path),
        "RISE_PAPER_ROOT": str(paper_root),
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
    }
    crashed = subprocess.run(
        [sys.executable, "-c", child_code],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert crashed.returncode == 73, crashed.stderr

    deadline = time.monotonic() + 5
    while not repository.list_expired_tasks() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [row.id for row in repository.list_expired_tasks()] == [task.id]

    with open_postgres_checkpointer(url) as saver:
        factory = RuntimeFactory(
            saver,
            paper_root,
            counter_path,
            resumed_after_gap_verification=True,
            checkpoint_recovery_only=True,
        )
        worker = make_worker(
            repository,
            url,
            factory,
            worker_id="restart-worker",
            lease_seconds=3.0,
            heartbeat_interval=0.1,
        )
        assert worker.run_once() is True
        finished = repository.get_task(task.id)
        assert finished.task_status == TaskStatus.COMPLETED.value
        assert finished.rise_outcome == "committed"
        assert finished.recovery_count == 1
        state = factory.runners[str(task.id)].load_persistent_state(thread_id=str(task.thread_id))
        assert state is not None
        assert state.budget.steps_used == 5
        assert len([record for record in state.replay_records if record.kind == "action"]) == 5
        assert counter_path.read_text(encoding="utf-8").splitlines() == ["verify_gap"]
        events = repository.list_events(task.id)
        assert sum(event.event_type == "rise_action_confirmed" for event in events) == 5
        first_action_key = f"rise:{task.id}:replay:2"
        assert sum(event.event_key == first_action_key for event in events) == 1
