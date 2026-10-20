"""Single-process Worker for durable LangGraph-backed RISE tasks."""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from ..models.seed import SeedPaperContext
from ..runtime.graph_runner import ACTION_NODE_NAMES, LangGraphTrajectoryRunner
from ..runtime.graph_state import RISEGraphState
from ..runtime.paper_store import PaperInputIntegrityError
from .advisory_lock import ThreadAdvisoryLock
from .contracts import ExpiredTaskClaim, TaskRecord, TaskStatus
from .repository import TaskRepository

logger = logging.getLogger(__name__)

WORKER_BOUNDARY_NODES = (
    "bootstrap",
    "controller",
    *ACTION_NODE_NAMES.values(),
    "finalizer",
)
RISE_TERMINAL_OUTCOMES = frozenset(
    {"committed", "stopped", "failed", "finalization_failed"}
)


PaperMarkdown = str | bytes | Path
SeedContextSource = SeedPaperContext | Callable[[], SeedPaperContext] | None
MarkdownSource = PaperMarkdown | Callable[[], PaperMarkdown] | None


@dataclass(frozen=True)
class WorkerRuntime:
    """Per-task RISE dependencies built by the host process configuration.

    Seed extraction and Markdown loading may be supplied lazily. The Worker
    resolves them only for a Thread with no checkpoint, so recovery never
    repeats Seed Context Extraction or re-reads submitted input unnecessarily.
    Runtime factories must not initialize or persist a second Research State.
    """

    runner: LangGraphTrajectoryRunner
    seed_context: SeedContextSource
    markdown: MarkdownSource


class WorkerRuntimeFactory(Protocol):
    def __call__(self, task: TaskRecord) -> WorkerRuntime: ...


class _LeaseHeartbeat:
    def __init__(
        self,
        repository: TaskRepository,
        task: TaskRecord,
        *,
        lease_seconds: float,
        interval: float,
        fail_stop_on_ownership_loss: bool,
    ) -> None:
        self.repository = repository
        self.task = task
        self.lease_seconds = lease_seconds
        self.interval = interval
        self.fail_stop_on_ownership_loss = fail_stop_on_ownership_loss
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._execution_lock: ThreadAdvisoryLock | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"rise-heartbeat-{task.id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def monitor_execution_lock(self, lock: ThreadAdvisoryLock) -> None:
        self._execution_lock = lock

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            # Wait for an in-flight heartbeat before the Worker clears its
            # lease; otherwise that update could look like a stolen lease.
            self._thread.join()

    def _lose_ownership(self, reason: str) -> None:
        self.lost.set()
        if self.fail_stop_on_ownership_loss:
            logger.critical(
                "Worker is terminating after losing task execution ownership: %s",
                reason,
            )
            # The Worker is a dedicated process. If a node is blocked in an
            # external call, exiting the process prevents it from racing a new
            # owner after PostgreSQL has released the session advisory lock.
            os._exit(75)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                alive = self.repository.heartbeat(
                    task_id=self.task.id,
                    worker_id=self.task.worker_id or "",
                    lease_token=self.task.lease_token,
                    lease_seconds=self.lease_seconds,
                )
            except Exception:
                logger.exception("Worker heartbeat failed for task %s", self.task.id)
                self._lose_ownership("database heartbeat failure")
                return
            if not alive:
                logger.warning("Worker lease was lost for task %s", self.task.id)
                self._lose_ownership("lease token is no longer current")
                return
            lock = self._execution_lock
            if lock is not None and not lock.check_connection():
                logger.warning("Worker advisory-lock session was lost for task %s", self.task.id)
                self._lose_ownership("PostgreSQL advisory-lock session was lost")
                return


class TaskWorker:
    """Claim and execute one RISE task at a time in an independent process."""

    def __init__(
        self,
        repository: TaskRepository,
        runtime_factory: WorkerRuntimeFactory,
        *,
        database_url: str,
        worker_id: str | None = None,
        lease_seconds: float = 30.0,
        heartbeat_interval: float = 5.0,
        poll_interval: float = 1.0,
        max_recoveries: int = 3,
        fail_stop_on_ownership_loss: bool = True,
        recovery_scan_limit: int = 20,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if heartbeat_interval <= 0 or heartbeat_interval >= lease_seconds:
            raise ValueError("heartbeat_interval must be positive and shorter than the lease")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if max_recoveries < 0:
            raise ValueError("max_recoveries must be non-negative")
        if not 1 <= recovery_scan_limit <= 100:
            raise ValueError("recovery_scan_limit must be between 1 and 100")

        self.repository = repository
        self.runtime_factory = runtime_factory
        self.database_url = database_url
        self.worker_id = worker_id or self._default_worker_id()
        if len(self.worker_id) > 128:
            raise ValueError("worker_id must not exceed 128 characters")
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.poll_interval = poll_interval
        self.max_recoveries = max_recoveries
        self.fail_stop_on_ownership_loss = fail_stop_on_ownership_loss
        self.recovery_scan_limit = recovery_scan_limit
        self.stop_event = threading.Event()
        self._run_guard = threading.Lock()

    @staticmethod
    def _default_worker_id() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:12]}"

    def request_shutdown(self) -> None:
        """Stop claiming tasks; an in-flight Graph call ends at its next node edge."""

        self.stop_event.set()

    def run_forever(self) -> None:
        logger.info("RISE Worker %s started", self.worker_id)
        while not self.stop_event.is_set():
            try:
                worked = self.run_once()
            except Exception:
                logger.exception("RISE Worker loop failed")
                worked = False
            if not worked:
                self.stop_event.wait(self.poll_interval)
        logger.info("RISE Worker %s stopped", self.worker_id)

    def run_once(self) -> bool:
        """Process one queued or expired task, returning whether one was claimed."""

        if self.stop_event.is_set() or not self._run_guard.acquire(blocking=False):
            return False
        try:
            for expired in self.repository.list_expired_tasks(
                limit=self.recovery_scan_limit
            ):
                lock = ThreadAdvisoryLock(self.database_url, str(expired.thread_id))
                if not lock.acquire(timeout=0):
                    continue
                try:
                    claim = self.repository.reclaim_expired_task(
                        task_id=expired.id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                        max_recoveries=self.max_recoveries,
                    )
                except Exception:
                    lock.close()
                    raise
                if claim is None:
                    lock.close()
                    continue
                self._process_claim(
                    claim.task,
                    recovery_exhausted=claim.recovery_exhausted,
                    preacquired_lock=lock,
                )
                return True

            task = self.repository.claim_next_queued(
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if task is not None:
                self._process_claim(task, recovery_exhausted=False)
                return True
            return False
        finally:
            self._run_guard.release()

    def _process_claim(
        self,
        task: TaskRecord,
        *,
        recovery_exhausted: bool,
        preacquired_lock: ThreadAdvisoryLock | None = None,
    ) -> None:
        heartbeat = _LeaseHeartbeat(
            self.repository,
            task,
            lease_seconds=self.lease_seconds,
            interval=self.heartbeat_interval,
            fail_stop_on_ownership_loss=self.fail_stop_on_ownership_loss,
        )
        heartbeat.start()
        lock = preacquired_lock or ThreadAdvisoryLock(
            self.database_url, str(task.thread_id)
        )
        try:
            if preacquired_lock is None and not lock.acquire(
                timeout=None,
                poll_interval=min(0.2, self.heartbeat_interval),
                should_abort=lambda: (
                    self.stop_event.is_set() or heartbeat.lost.is_set()
                ),
            ):
                return
            if heartbeat.lost.is_set():
                return
            heartbeat.monitor_execution_lock(lock)
            self._execute_claimed(task, heartbeat, lock, recovery_exhausted)
        except PaperInputIntegrityError as exc:
            self._fail_input(task, heartbeat, exc)
        except Exception as exc:
            self._handle_runtime_creation_error(task, heartbeat, exc)
        finally:
            heartbeat.stop()
            lock.close()

    def _execute_claimed(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        lock: ThreadAdvisoryLock,
        recovery_exhausted: bool,
    ) -> None:
        runtime = self.runtime_factory(task)
        runner = runtime.runner
        state = runner.load_persistent_state(thread_id=str(task.thread_id))

        current = self.repository.get_task(task.id)
        if state is not None and state.rise_status in RISE_TERMINAL_OUTCOMES:
            self._project_state(task, state)
            self._complete(task, heartbeat, state)
            return
        if current.task_status == TaskStatus.CANCEL_REQUESTED.value and (
            state is None or state.commit_snapshot is None
        ):
            self._cancel(task, heartbeat)
            return
        if recovery_exhausted and (state is None or state.commit_snapshot is None):
            self._mark_infrastructure_failed(
                task,
                heartbeat,
                RuntimeError("infrastructure recovery limit exhausted before a terminal checkpoint"),
            )
            return

        while True:
            if heartbeat.lost.is_set():
                return
            if self.stop_event.is_set():
                return
            try:
                if state is None:
                    if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
                        self._cancel(task, heartbeat)
                        return
                    seed_context = self._resolve_seed_context(runtime.seed_context)
                    markdown = self._resolve_markdown(runtime.markdown)
                    if seed_context is None or markdown is None:
                        raise PaperInputIntegrityError(
                            "new task is missing its prepared SeedPaperContext or Markdown"
                        )
                    markdown_bytes = self._markdown_bytes(markdown)
                    if hashlib.sha256(markdown_bytes).hexdigest() != task.input_sha256:
                        raise PaperInputIntegrityError(
                            "task Markdown SHA-256 does not match the submitted input"
                        )
                    state = runner.start_persistent(
                        seed_context,
                        thread_id=str(task.thread_id),
                        markdown=markdown_bytes,
                        interrupt_before_nodes=WORKER_BOUNDARY_NODES,
                        boundary_check=lambda node, saved: self._boundary_reason(
                            task, heartbeat, node, saved
                        ),
                    )
                else:
                    state = runner.resume_persistent(
                        thread_id=str(task.thread_id),
                        interrupt_before_nodes=WORKER_BOUNDARY_NODES,
                        boundary_check=lambda node, saved: self._boundary_reason(
                            task, heartbeat, node, saved
                        ),
                    )
            except PaperInputIntegrityError as exc:
                self._fail_input(task, heartbeat, exc)
                return
            except Exception as exc:
                if heartbeat.lost.is_set():
                    return
                logger.exception("Graph execution failed for task %s", task.id)
                # The only in-place retry is retained under the same lease when
                # cancellation and a saved CommitSnapshot require Finalizer work.
                try:
                    state = runner.load_persistent_state(thread_id=str(task.thread_id))
                    current = self.repository.get_task(task.id)
                except Exception:
                    if heartbeat.lost.is_set():
                        return
                    current = self.repository.get_task(task.id)
                    if current.task_status != TaskStatus.CANCEL_REQUESTED.value:
                        return
                    updated = self._record_in_place_retry(task, heartbeat, exc)
                    if updated.task_status in {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
                        return
                    recovery_exhausted = updated.recovery_count >= self.max_recoveries
                    continue

                if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
                    if state is None or state.commit_snapshot is None:
                        self._cancel(task, heartbeat)
                        return
                    updated = self._record_in_place_retry(task, heartbeat, exc)
                    if updated.task_status in {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
                        return
                    recovery_exhausted = updated.recovery_count >= self.max_recoveries
                    continue
                if state is not None and state.rise_status in RISE_TERMINAL_OUTCOMES:
                    self._project_state(task, state)
                    self._complete(task, heartbeat, state)
                    return
                if recovery_exhausted:
                    self._mark_infrastructure_failed(task, heartbeat, exc)
                    return
                try:
                    heartbeat.stop()
                    if heartbeat.lost.is_set():
                        return
                    self.repository.release_for_recovery(
                        task_id=task.id,
                        worker_id=self.worker_id,
                        lease_token=task.lease_token,
                        max_recoveries=self.max_recoveries,
                        error_type=type(exc).__name__,
                        error_summary=str(exc),
                    )
                except Exception:
                    logger.exception("Could not release task %s after runtime error", task.id)
                return

            self._after_graph_checkpoint(task, state)
            if heartbeat.lost.is_set():
                return
            try:
                self._project_state(task, state)
                current = self.repository.get_task(task.id)
            except Exception as exc:
                logger.exception("Could not project saved Graph State for task %s", task.id)
                if heartbeat.lost.is_set():
                    return
                try:
                    current = self.repository.get_task(task.id)
                    if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
                        if state.commit_snapshot is None:
                            self._cancel(task, heartbeat)
                            return
                        # A saved CommitSnapshot wins over cancellation. Keep
                        # the live lease and continue toward Finalizer; events
                        # can be idempotently backfilled from the final state.
                        updated = self._record_in_place_retry(task, heartbeat, exc)
                        if updated.task_status in {
                            TaskStatus.FAILED.value,
                            TaskStatus.CANCELLED.value,
                        }:
                            return
                        recovery_exhausted = updated.recovery_count >= self.max_recoveries
                        continue

                    # A saved graph state can be projected again from the checkpoint.
                    heartbeat.stop()
                    if heartbeat.lost.is_set():
                        return
                    self.repository.release_for_recovery(
                        task_id=task.id,
                        worker_id=self.worker_id,
                        lease_token=task.lease_token,
                        max_recoveries=self.max_recoveries,
                        error_type=type(exc).__name__,
                        error_summary=str(exc),
                    )
                except Exception:
                    logger.exception("Could not schedule projection retry for task %s", task.id)
                return

            if state.rise_status in RISE_TERMINAL_OUTCOMES:
                self._complete(task, heartbeat, state)
                return
            if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
                if state.commit_snapshot is None:
                    self._cancel(task, heartbeat)
                    return
            if recovery_exhausted and state.commit_snapshot is None:
                self._mark_infrastructure_failed(
                    task,
                    heartbeat,
                    RuntimeError("infrastructure recovery limit exhausted before a terminal checkpoint"),
                )
                return

    def _boundary_reason(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        node_name: str,
        state: RISEGraphState,
    ) -> str | None:
        if heartbeat.lost.is_set() or self.stop_event.is_set():
            return "execution_ownership_lost" if heartbeat.lost.is_set() else "worker_shutdown"
        try:
            current = self.repository.get_task(task.id)
        except Exception:
            heartbeat.lost.set()
            return "business_database_unavailable"
        if (
            current.task_status == TaskStatus.CANCEL_REQUESTED.value
            and state.commit_snapshot is None
        ):
            return "user_cancellation"
        return None

    def _after_graph_checkpoint(self, task: TaskRecord, state: RISEGraphState) -> None:
        """Hook called only after LangGraph returns at a durable node boundary."""

    def _project_state(self, task: TaskRecord, state: RISEGraphState) -> None:
        for index, event in enumerate(state.trajectory_events, start=1):
            payload = {
                "trajectory_index": index,
                "step": event.step,
                "trajectory_event_type": event.event_type,
                "result_ref": event.result_ref,
                "state_fields": sorted(event.state_diff),
                "memory_writes": list(event.memory_writes),
                "budget": event.budget.model_dump(mode="json") if event.budget else None,
                "error": event.error,
            }
            self.repository.append_event(
                task_id=task.id,
                event_type="rise_trajectory_event",
                event_key=f"rise:{task.id}:trajectory:{index}",
                action=event.action,
                payload=payload,
            )

        for record in state.replay_records:
            event_type = {
                "bootstrap": "rise_bootstrap_confirmed",
                "action": "rise_action_confirmed",
                "internal": "rise_internal_step_confirmed",
            }[record.kind]
            self.repository.append_event(
                task_id=task.id,
                event_type=event_type,
                event_key=f"rise:{task.id}:replay:{record.sequence}",
                action=record.action,
                payload={
                    "replay_sequence": record.sequence,
                    "step": record.step,
                    "kind": record.kind,
                    "budget": asdict(record.budget_after),
                },
            )

    def _complete(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        state: RISEGraphState,
    ) -> None:
        if heartbeat.lost.is_set():
            return
        heartbeat.stop()
        if heartbeat.lost.is_set():
            return
        self.repository.mark_completed(
            task_id=task.id,
            worker_id=self.worker_id,
            lease_token=task.lease_token,
            rise_outcome=state.rise_status,
            error_type=state.error_type,
            error_summary=state.error_message,
        )

    def _cancel(self, task: TaskRecord, heartbeat: _LeaseHeartbeat) -> None:
        if heartbeat.lost.is_set():
            return
        heartbeat.stop()
        if heartbeat.lost.is_set():
            return
        self.repository.mark_cancelled(
            task_id=task.id,
            worker_id=self.worker_id,
            lease_token=task.lease_token,
        )

    def _fail_input(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        error: Exception,
    ) -> None:
        if heartbeat.lost.is_set():
            return
        current = self.repository.get_task(task.id)
        if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
            self._cancel(task, heartbeat)
            return
        heartbeat.stop()
        if heartbeat.lost.is_set():
            return
        self.repository.mark_failed(
            task_id=task.id,
            worker_id=self.worker_id,
            lease_token=task.lease_token,
            failure_kind="input",
            error_type=type(error).__name__,
            error_summary=str(error),
        )

    def _mark_infrastructure_failed(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        error: Exception,
    ) -> None:
        if heartbeat.lost.is_set():
            return
        heartbeat.stop()
        if heartbeat.lost.is_set():
            return
        self.repository.mark_failed(
            task_id=task.id,
            worker_id=self.worker_id,
            lease_token=task.lease_token,
            failure_kind="infrastructure",
            error_type=type(error).__name__,
            error_summary=str(error),
        )

    def _record_in_place_retry(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        error: Exception,
    ) -> TaskRecord:
        current = self.repository.get_task(task.id)
        if current.recovery_count >= self.max_recoveries:
            heartbeat.stop()
            if heartbeat.lost.is_set():
                return current
        return self.repository.record_infrastructure_retry_in_place(
            task_id=task.id,
            worker_id=self.worker_id,
            lease_token=task.lease_token,
            lease_seconds=self.lease_seconds,
            max_recoveries=self.max_recoveries,
            error_type=type(error).__name__,
            error_summary=str(error),
        )

    def _handle_runtime_creation_error(
        self,
        task: TaskRecord,
        heartbeat: _LeaseHeartbeat,
        error: Exception,
    ) -> None:
        if heartbeat.lost.is_set():
            return
        try:
            current = self.repository.get_task(task.id)
            if current.task_status == TaskStatus.CANCEL_REQUESTED.value:
                updated = self._record_in_place_retry(task, heartbeat, error)
                logger.error(
                    "Could not build runtime for cancel-requested task %s; status=%s",
                    task.id,
                    updated.task_status,
                )
                return
            heartbeat.stop()
            if heartbeat.lost.is_set():
                return
            self.repository.release_for_recovery(
                task_id=task.id,
                worker_id=self.worker_id,
                lease_token=task.lease_token,
                max_recoveries=self.max_recoveries,
                error_type=type(error).__name__,
                error_summary=str(error),
            )
        except Exception:
            logger.exception("Could not record runtime setup failure for task %s", task.id)

    @staticmethod
    def _resolve_seed_context(
        source: SeedContextSource,
    ) -> SeedPaperContext | None:
        return source() if callable(source) else source

    @staticmethod
    def _resolve_markdown(source: MarkdownSource) -> PaperMarkdown | None:
        return source() if callable(source) else source

    @staticmethod
    def _markdown_bytes(markdown: PaperMarkdown) -> bytes:
        if isinstance(markdown, Path):
            return markdown.read_bytes()
        if isinstance(markdown, str):
            return markdown.encode("utf-8")
        if isinstance(markdown, bytes):
            return markdown
        raise PaperInputIntegrityError("worker runtime returned an unsupported Markdown value")


__all__ = ["TaskWorker", "WorkerRuntime", "WorkerRuntimeFactory", "WORKER_BOUNDARY_NODES"]
