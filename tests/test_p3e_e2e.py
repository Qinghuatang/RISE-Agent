"""P3-E tests across independent API/Worker processes and real PostgreSQL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import UUID, uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from research_agent.controller.budget import BudgetTracker
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.paper_store import PaperInputStore
from research_agent.service.db import create_business_engine, create_session_factory
from research_agent.service.repository import TaskRepository
from test_m11 import modules, seed


POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set RISE_TEST_POSTGRES_URL to run real PostgreSQL P3-E process tests",
)
ROOT = Path(__file__).resolve().parents[1]
PAPER_MARKDOWN = """# Generic supplied seed

## ABSTRACT
A supplied abstract.

## 1 METHOD
A method with interacting units. Unit A and Unit B.

## 2 FINDINGS
A supplied finding and a supplied limitation.

## REFERENCES
Reference-only content.
"""
EXPECTED_ACTIONS = [
    "VERIFY_RELATED_WORK",
    "SOLUTION_IDEATION",
    "VERIFY_RELATED_WORK",
    "DESIGN_EXPERIMENT",
    "COMMIT",
]


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class ProcessSystem:
    def __init__(self, root: Path, database_url: str, schema: str):
        self.root = root
        self.paper_root = root / "papers"
        self.action_log = root / "actions.jsonl"
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.processes: list[tuple[str, subprocess.Popen, Path]] = []
        self.hook_releases: list[Path] = []
        self.schema_url = make_url(database_url).update_query_dict(
            {"options": f"-csearch_path={schema}"}
        ).render_as_string(hide_password=False)
        self.env = {
            **os.environ,
            "RISE_DATABASE_URL": self.schema_url,
            "RISE_PAPER_STORE_ROOT": str(self.paper_root),
            "RISE_API_HOST": "127.0.0.1",
            "RISE_API_PORT": str(self.port),
            "RISE_SSE_POLL_INTERVAL": "0.02",
            "RISE_SSE_HEARTBEAT_INTERVAL": "0.1",
            "RISE_WORKER_FACTORY": "p3e_worker_factory:build_worker",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]
            ),
        }

    def start_api(self) -> None:
        self._start("api", [sys.executable, "-m", "research_agent.service.api_cli"])
        deadline = time.monotonic() + 20
        last_error = "API did not become ready"
        while time.monotonic() < deadline:
            process = self.processes[-1][1]
            if process.poll() is not None:
                raise AssertionError(self.logs("api"))
            try:
                status, payload = self.request("GET", "/health/ready")
                if status == 200 and payload.get("status") == "ready":
                    return
                last_error = f"readiness returned {status}: {payload}"
            except (OSError, URLError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(0.05)
        raise AssertionError(f"API failed to become ready: {last_error}\n{self.logs('api')}")

    def start_worker(self, *, mode: str = "normal", **environment: str) -> subprocess.Popen:
        worker_env = {
            "RISE_E2E_WORKER_MODE": mode,
            "RISE_E2E_ACTION_LOG": str(self.action_log),
            "RISE_E2E_LEASE_SECONDS": "5",
            "RISE_E2E_HEARTBEAT_INTERVAL": "0.1",
            "RISE_E2E_POLL_INTERVAL": "0.02",
            **environment,
        }
        release = worker_env.get("RISE_E2E_HOOK_RELEASE")
        if release:
            self.hook_releases.append(Path(release))
        return self._start(
            f"worker-{mode}",
            [sys.executable, "-m", "research_agent.service.worker_cli"],
            worker_env,
        )

    def _start(
        self,
        name: str,
        command: list[str],
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        log_path = self.root / f"{name}-{len(self.processes)}.log"
        stream = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**self.env, **(extra_environment or {})},
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.close()
        self.processes.append((name, process, log_path))
        return process

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 10,
    ) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = dict(headers or {})
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {"raw": ""}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return exc.code, payload

    def submit(self, markdown: str = PAPER_MARKDOWN) -> str:
        status, payload = self.request(
            "POST", "/api/tasks", {"markdown": markdown, "execution_config": {}}
        )
        assert status == 202, payload
        return str(UUID(payload["task_id"]))

    def wait_marker(self, marker: Path, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if marker.exists():
                return
            for name, process, _ in self.processes:
                if name.startswith("worker") and process.poll() is not None:
                    raise AssertionError(f"{name} exited before marker:\n{self.logs(name)}")
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {marker}\n{self.all_logs()}")

    def wait_task(self, task_id: str, expected: str = "completed", timeout: float = 20) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            status, last = self.request("GET", f"/api/tasks/{task_id}")
            assert status == 200, last
            if last["task_status"] in {"completed", "failed", "cancelled"}:
                assert last["task_status"] == expected, last
                return last
            time.sleep(0.04)
        raise AssertionError(f"task {task_id} did not reach {expected}: {last}\n{self.all_logs()}")

    def action_rows(self) -> list[dict]:
        if not self.action_log.exists():
            return []
        return [json.loads(line) for line in self.action_log.read_text().splitlines()]

    def event_ids(self, task_id: str, *, last_event_id: int | None = None) -> tuple[list[int], str]:
        headers = {} if last_event_id is None else {"Last-Event-ID": str(last_event_id)}
        request = Request(self.base_url + f"/api/tasks/{task_id}/events", headers=headers)
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
        ids = [int(line.removeprefix("id: ")) for line in body.splitlines() if line.startswith("id: ")]
        return ids, body

    def logs(self, name: str) -> str:
        matches = [path for current_name, _, path in self.processes if current_name == name]
        if not matches:
            matches = [path for current_name, _, path in self.processes if current_name.startswith(name)]
        return "\n".join(path.read_text(errors="replace") for path in matches)

    def all_logs(self) -> str:
        return "\n".join(f"[{name}]\n{path.read_text(errors='replace')}" for name, _, path in self.processes)

    def stop_processes(self) -> None:
        for release in self.hook_releases:
            release.parent.mkdir(parents=True, exist_ok=True)
            release.touch(exist_ok=True)
        for _, process, _ in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.fixture
def process_system(tmp_path: Path):
    assert POSTGRES_URL is not None
    admin_engine = create_business_engine(POSTGRES_URL)
    schema = f"rise_p3e_{uuid4().hex}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    system = ProcessSystem(tmp_path, POSTGRES_URL, schema)
    try:
        yield system, admin_engine
    finally:
        system.stop_processes()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin_engine.dispose()


def _task_actions(rows: list[dict], task_id: str, kind: str = "action_returned") -> list[str]:
    return [row["action"] for row in rows if row["task_id"] == task_id and row["kind"] == kind]


def _assert_checkpoint_matches_deterministic_baseline(
    system: ProcessSystem,
    task_id: str,
    markdown: str,
) -> None:
    paper_store = PaperInputStore(system.paper_root)
    reference = paper_store.persist(markdown.encode("utf-8"))
    reading_context = paper_store.build_reading_context(
        reference,
        paper_id=seed().paper_id,
    )
    baseline = LangGraphTrajectoryRunner(
        modules(), budget=BudgetTracker(max_steps=5, max_searches=1), run_id="baseline"
    ).run(seed(), full_paper_reading_context=reading_context)
    with open_postgres_checkpointer(system.schema_url) as saver:
        runner = LangGraphTrajectoryRunner(
            modules(),
            budget=BudgetTracker(max_steps=5, max_searches=1),
            checkpointer=saver,
            paper_store=paper_store,
        )
        durable = runner.load_persistent_state(thread_id=task_id)

    assert durable is not None
    assert durable.rise_status == baseline.rise_status
    assert durable.research_state == baseline.research_state
    assert durable.research_memory == baseline.research_memory
    assert durable.budget == baseline.budget
    assert durable.replay_records == baseline.replay_records
    assert durable.commit_snapshot == baseline.commit_snapshot
    assert durable.final_proposal == baseline.final_proposal
    assert durable.unresolved_issues == baseline.unresolved_issues
    assert [
        {key: value for key, value in event.model_dump(mode="json").items() if key != "timestamp"}
        for event in durable.trajectory_events
    ] == [
        {key: value for key, value in event.model_dump(mode="json").items() if key != "timestamp"}
        for event in baseline.trajectory_events
    ]


def test_api_worker_graph_checkpoint_result_sse_and_serial_queue(process_system):
    system, _ = process_system
    system.start_api()
    marker = system.root / "first-action-checkpoint"
    release = system.root / "release-first-action"
    worker = system.start_worker(
        mode="hold_after_action",
        RISE_E2E_HOOK_MARKER=str(marker),
        RISE_E2E_HOOK_RELEASE=str(release),
        RISE_E2E_ACTION_DELAY="0.02",
    )

    first_id = system.submit()
    system.wait_marker(marker)
    second_id = system.submit(PAPER_MARKDOWN.replace("Generic supplied seed", "Second supplied seed"))
    first_status = system.request("GET", f"/api/tasks/{first_id}")[1]["task_status"]
    second_status = system.request("GET", f"/api/tasks/{second_id}")[1]["task_status"]
    assert first_status == "running"
    assert second_status == "queued"
    assert _task_actions(system.action_rows(), first_id) == ["VERIFY_RELATED_WORK"]
    release.touch()

    task_markdowns = {
        first_id: PAPER_MARKDOWN,
        second_id: PAPER_MARKDOWN.replace("Generic supplied seed", "Second supplied seed"),
    }
    for task_id in (first_id, second_id):
        summary = system.wait_task(task_id)
        assert summary["rise_outcome"] == "committed"
        status, result = system.request("GET", f"/api/tasks/{task_id}/result")
        assert status == 200, result
        assert result["task_status"] == "completed"
        assert result["rise_outcome"] == "committed"
        assert result["final_proposal"] is not None
        assert result["commit_snapshot"] is not None
        assert result["budget"]["steps_used"] == 5
        assert result["budget"]["searches_used"] == 1
        assert result["replay_record_count"] == 6
        ids, stream = system.event_ids(task_id)
        assert ids == list(range(1, len(ids) + 1))
        assert "rise_action_confirmed" in stream
        resumed_ids, _ = system.event_ids(task_id, last_event_id=ids[len(ids) // 2])
        assert resumed_ids == ids[len(ids) // 2 + 1 :]
        assert "Generic supplied seed" not in stream
        _assert_checkpoint_matches_deterministic_baseline(
            system,
            task_id,
            task_markdowns[task_id],
        )

    rows = system.action_rows()
    assert _task_actions(rows, first_id) == EXPECTED_ACTIONS
    assert _task_actions(rows, second_id) == EXPECTED_ACTIONS
    returned_order = [
        row["task_id"]
        for row in rows
        if row["kind"] == "action_returned"
    ]
    assert returned_order == [first_id] * 5 + [second_id] * 5
    assert worker.poll() is None


def test_worker_process_crash_recovers_checkpoint_and_api_backfills_sse(process_system):
    system, _ = process_system
    system.start_api()
    marker = system.root / "crashed-after-action"
    crashed = system.start_worker(
        mode="crash_after_action",
        RISE_E2E_HOOK_MARKER=str(marker),
        RISE_E2E_LEASE_SECONDS="1.2",
        RISE_E2E_HEARTBEAT_INTERVAL="0.1",
    )
    task_id = system.submit()
    system.wait_marker(marker)
    assert crashed.wait(timeout=10) == 86
    assert _task_actions(system.action_rows(), task_id) == ["VERIFY_RELATED_WORK"]

    engine = create_business_engine(system.schema_url)
    repository = TaskRepository(create_session_factory(engine))
    try:
        deadline = time.monotonic() + 8
        while not repository.list_expired_tasks() and time.monotonic() < deadline:
            time.sleep(0.04)
        assert [row.id for row in repository.list_expired_tasks()] == [UUID(task_id)]
    finally:
        engine.dispose()

    restarted = system.start_worker(
        mode="normal",
        RISE_E2E_RECOVERY_ONLY="1",
        RISE_E2E_LEASE_SECONDS="4",
        RISE_E2E_HEARTBEAT_INTERVAL="0.1",
    )
    summary = system.wait_task(task_id)
    assert summary["task_status"] == "completed"
    assert summary["rise_outcome"] == "committed"
    assert summary["recovery_count"] == 1
    assert _task_actions(system.action_rows(), task_id) == EXPECTED_ACTIONS
    status, result = system.request("GET", f"/api/tasks/{task_id}/result")
    assert status == 200, result
    assert result["final_proposal"] is not None
    ids, stream = system.event_ids(task_id)
    assert ids == list(range(1, len(ids) + 1))
    assert stream.count("event: rise_action_confirmed") == 5
    _assert_checkpoint_matches_deterministic_baseline(system, task_id, PAPER_MARKDOWN)
    assert restarted.poll() is None


def test_expired_lease_cannot_bypass_live_workers_thread_advisory_lock(process_system):
    system, _ = process_system
    system.start_api()
    marker = system.root / "old-worker-holds-thread"
    release = system.root / "release-old-worker"
    old_worker = system.start_worker(
        mode="hold_after_action",
        RISE_E2E_HOOK_MARKER=str(marker),
        RISE_E2E_HOOK_RELEASE=str(release),
        RISE_E2E_LEASE_SECONDS="60",
        RISE_E2E_HEARTBEAT_INTERVAL="30",
    )
    task_id = system.submit()
    system.wait_marker(marker)

    engine = create_business_engine(system.schema_url)
    repository = TaskRepository(create_session_factory(engine))
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE tasks SET lease_expires_at = now() - interval '1 minute' "
                    "WHERE id = :task_id"
                ),
                {"task_id": UUID(task_id)},
            )
        contender = system.start_worker(
            mode="normal",
            RISE_E2E_WORKER_ID=f"contender-{uuid4().hex[:10]}",
            RISE_E2E_RECOVERY_ONLY="1",
            RISE_E2E_LEASE_SECONDS="5",
            RISE_E2E_HEARTBEAT_INTERVAL="0.1",
        )
        time.sleep(0.5)
        row = repository.get_task(UUID(task_id))
        assert row.task_status == "running"
        assert row.recovery_count == 0
        assert _task_actions(system.action_rows(), task_id) == ["VERIFY_RELATED_WORK"]
        assert contender.poll() is None
        # Kill the stale process while it still owns the session-level lock.
        # The contender may only continue after PostgreSQL releases that lock.
        old_worker.kill()
        assert old_worker.wait(timeout=5) < 0
        summary = system.wait_task(task_id)
        assert summary["rise_outcome"] == "committed"
        assert summary["recovery_count"] == 1
        assert _task_actions(system.action_rows(), task_id) == EXPECTED_ACTIONS
        assert contender.poll() is None
    finally:
        engine.dispose()


def test_api_cancellation_stops_at_action_boundary_and_preserves_commit_finalizer_rule(process_system):
    system, _ = process_system
    system.start_api()

    marker = system.root / "action-boundary"
    release = system.root / "release-action-boundary"
    worker = system.start_worker(
        mode="hold_after_action",
        RISE_E2E_HOOK_MARKER=str(marker),
        RISE_E2E_HOOK_RELEASE=str(release),
    )
    task_id = system.submit()
    system.wait_marker(marker)
    status, cancellation = system.request("POST", f"/api/tasks/{task_id}/cancel")
    assert status == 200 and cancellation["task_status"] == "cancel_requested"
    release.touch()
    summary = system.wait_task(task_id, expected="cancelled")
    assert summary["rise_outcome"] is None
    assert _task_actions(system.action_rows(), task_id) == ["VERIFY_RELATED_WORK"]
    assert worker.poll() is None

    system.stop_processes()
    # Launch a fresh API/Worker pair in the same isolated schema. The committed
    # checkpoint is held after Commit and before Finalizer to exercise the race.
    system.processes.clear()
    system.start_api()
    marker = system.root / "commit-before-finalizer"
    release = system.root / "release-finalizer"
    worker = system.start_worker(
        mode="hold_after_commit",
        RISE_E2E_HOOK_MARKER=str(marker),
        RISE_E2E_HOOK_RELEASE=str(release),
    )
    committed_task_id = system.submit(PAPER_MARKDOWN.replace("Generic supplied seed", "Commit race seed"))
    system.wait_marker(marker)
    status, cancellation = system.request(
        "POST", f"/api/tasks/{committed_task_id}/cancel"
    )
    assert status == 200 and cancellation["task_status"] == "cancel_requested"
    release.touch()
    summary = system.wait_task(committed_task_id, expected="completed")
    assert summary["rise_outcome"] == "committed"
    status, result = system.request("GET", f"/api/tasks/{committed_task_id}/result")
    assert status == 200 and result["final_proposal"] is not None
    assert _task_actions(system.action_rows(), committed_task_id) == EXPECTED_ACTIONS
    assert worker.poll() is None


def test_queued_cancel_is_not_claimed_by_independent_worker(process_system):
    system, _ = process_system
    system.start_api()
    task_id = system.submit()
    status, cancellation = system.request("POST", f"/api/tasks/{task_id}/cancel")
    assert status == 200 and cancellation["task_status"] == "cancelled"
    worker = system.start_worker(mode="normal")
    time.sleep(0.3)
    assert worker.poll() is None
    assert system.request("GET", f"/api/tasks/{task_id}")[1]["task_status"] == "cancelled"
    assert system.action_rows() == []
