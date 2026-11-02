"""Deterministic Worker factory used by independent-process P3-E tests.

This is test-only wiring: it exercises the production Worker CLI contract with
the repository's fixed FakeLLM/FakeRetrieval modules.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time

from research_agent.controller.budget import BudgetTracker
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.graph_state import PaperInputReference
from research_agent.runtime.paper_store import PaperInputStore
from research_agent.service.db import create_business_engine, create_session_factory
from research_agent.service.repository import TaskRepository
from research_agent.service.worker import TaskWorker, WorkerRuntime

from test_m11 import FakeLLMClient, modules, reading, seed, solution_verification


def _journal(kind: str, task_id: str, action: str | None = None) -> None:
    path_text = os.environ.get("RISE_E2E_ACTION_LOG")
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = json.dumps(
        {"kind": kind, "task_id": task_id, "action": action},
        separators=(",", ":"),
    ) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(row)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_worker() -> TaskWorker:
    database_url = os.environ["RISE_DATABASE_URL"]
    engine = create_business_engine(database_url)
    repository = TaskRepository(create_session_factory(engine))
    saver_context = open_postgres_checkpointer(database_url)
    saver = saver_context.__enter__()
    paper_store = PaperInputStore(os.environ["RISE_PAPER_STORE_ROOT"])
    paper_markdown = "\n".join(reading().parsed.lines) + "\n"
    recovery_only = os.environ.get("RISE_E2E_RECOVERY_ONLY") == "1"

    class RuntimeFactory:
        def __call__(self, task):
            bundle = modules()
            if recovery_only:
                # The first gap-verification response was consumed before the
                # injected crash. Resume with the fixed solution-verification
                # fixture used by the legacy P3-C recovery contract.
                bundle.related_work_verification.llm_client = FakeLLMClient(
                    [solution_verification()]
                )
            runner = LangGraphTrajectoryRunner(
                bundle,
                budget=BudgetTracker(
                    max_steps=int(task.execution_config.get("max_steps", 5)),
                    max_searches=int(task.execution_config.get("max_searches", 1)),
                ),
                checkpointer=saver,
                paper_store=paper_store,
            )
            original = runner.adapter.execute_selected_action

            def tracked_action(state):
                action = state.controller_routing.decision.action
                action_name = str(getattr(action, "value", action))
                task_id = str(task.id)
                _journal("action_started", task_id, action_name)
                delay = float(os.environ.get("RISE_E2E_ACTION_DELAY", "0"))
                if delay:
                    time.sleep(delay)
                result = original(state)
                _journal("action_returned", task_id, action_name)
                return result

            runner.adapter.execute_selected_action = tracked_action

            def missing_seed():
                raise AssertionError("checkpoint recovery must not re-create Seed Context")

            def load_markdown():
                if recovery_only:
                    raise AssertionError("checkpoint recovery must not reload task Markdown")
                reference = PaperInputReference(
                    locator=task.input_locator,
                    sha256=task.input_sha256,
                )
                return paper_store.read(reference)

            return WorkerRuntime(
                runner,
                missing_seed if recovery_only else seed(),
                load_markdown,
            )

    class ProcessWorker(TaskWorker):
        def _after_graph_checkpoint(self, task, state) -> None:
            mode = os.environ.get("RISE_E2E_WORKER_MODE", "normal")
            marker_text = os.environ.get("RISE_E2E_HOOK_MARKER")
            marker = Path(marker_text) if marker_text else None
            reached_first_action = state.budget.steps_used >= 1
            reached_commit_before_finalizer = (
                state.commit_snapshot is not None
                and state.final_proposal is None
                and state.rise_status == "running"
            )
            should_hook = (
                mode in {"crash_after_action", "hold_after_action"}
                and reached_first_action
            ) or (mode in {"hold_after_commit"} and reached_commit_before_finalizer)
            if not should_hook or marker is None or marker.exists():
                return

            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(task.id), encoding="utf-8")
            if mode == "crash_after_action":
                os._exit(86)
            if mode in {"hold_after_action", "hold_after_commit"}:
                release_text = os.environ.get("RISE_E2E_HOOK_RELEASE")
                if not release_text:
                    raise RuntimeError("hold mode requires RISE_E2E_HOOK_RELEASE")
                release = Path(release_text)
                deadline = time.monotonic() + 45
                while not release.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not release.exists():
                    raise TimeoutError("test did not release the Worker hook")

        def run_forever(self) -> None:
            try:
                super().run_forever()
            finally:
                saver_context.__exit__(None, None, None)
                engine.dispose()

    return ProcessWorker(
        repository,
        RuntimeFactory(),
        database_url=database_url,
        worker_id=os.environ.get("RISE_E2E_WORKER_ID"),
        lease_seconds=float(os.environ.get("RISE_E2E_LEASE_SECONDS", "5")),
        heartbeat_interval=float(os.environ.get("RISE_E2E_HEARTBEAT_INTERVAL", "0.1")),
        poll_interval=float(os.environ.get("RISE_E2E_POLL_INTERVAL", "0.03")),
        max_recoveries=int(os.environ.get("RISE_E2E_MAX_RECOVERIES", "3")),
    )
