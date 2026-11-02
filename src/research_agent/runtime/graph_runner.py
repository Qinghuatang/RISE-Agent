"""LangGraph orchestration for one RISE scientific trajectory.

The graph owns execution routing only. Scientific decisions and action
semantics remain in ``ScientificController``, ``TrajectoryRunner`` helpers,
and ``RISEActionAdapter``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from threading import Lock, RLock
import time
from typing import Any, Callable, Sequence
from uuid import UUID, uuid4

from ..controller.budget import BudgetTracker, ControllerBudget
from ..ingestion.markdown import FullPaperReadingContext
from ..models.base import ScientificAction
from ..models.seed import SeedPaperContext
from ..runner.trajectory_runner import TrajectoryModules
from ..observability.logging import bind_log_context
from ..observability.langfuse import LangfuseTracer
from ..observability.telemetry import DEFAULT_TELEMETRY, Telemetry
from .action_adapter import RISEActionAdapter
from .graph_state import (
    GraphStateValidationError,
    PaperInputReference,
    RISEGraphState,
)
from .paper_store import PaperInputIntegrityError, PaperInputStore


ACTION_NODE_NAMES: dict[str, str] = {
    "SOLUTION_IDEATION": "solution_ideation",
    "SEEK_INSPIRATION": "seek_inspiration",
    "VERIFY_RELATED_WORK": "verify_related_work",
    "TARGETED_REVISION": "targeted_revision",
    "DESIGN_EXPERIMENT": "design_experiment",
    "SWITCH_GAP": "switch_gap",
    "COMMIT": "commit",
}

_TERMINAL_STATUSES = {"committed", "stopped", "failed", "finalization_failed"}
_THREAD_LOCKS: dict[str, RLock] = {}
_THREAD_LOCKS_GUARD = Lock()


class GraphRoutingInvariantError(RuntimeError):
    """Raised when a validated graph snapshot cannot select a legal next node."""


class PersistentExecutionError(RuntimeError):
    """Raised when a checkpoint cannot safely be used to start or resume a run."""


BoundaryCheck = Callable[[str, RISEGraphState], str | None]


class ExistingThreadError(PersistentExecutionError):
    """Raised when new work is submitted with a thread that already has state."""


class MissingCheckpointError(PersistentExecutionError):
    """Raised when resume is requested for a thread without a checkpoint."""


@contextmanager
def _lock_thread(thread_id: str):
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(thread_id, RLock())
    with lock:
        yield


def _canonical_thread_id(thread_id: str) -> str:
    try:
        return str(UUID(thread_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("thread_id must be a valid UUID") from exc


def _recursion_limit_for_remaining_steps(remaining_steps: int) -> int:
    return max(3, 2 * remaining_steps + 3)


def _as_graph_state(value: Any) -> RISEGraphState:
    if isinstance(value, RISEGraphState):
        return value
    return RISEGraphState.from_payload(value)


class LangGraphTrajectoryRunner:
    """Execute RISE with LangGraph while preserving the legacy Runner contract.

    ``BudgetTracker`` is read once to create the initial immutable graph
    snapshot. Every later budget value comes from ``RISEGraphState`` and is
    reconstructed by the adapter at node boundaries.
    """

    def __init__(
        self,
        modules: TrajectoryModules,
        *,
        budget: BudgetTracker | ControllerBudget,
        run_id: str | None = None,
        paper_input: PaperInputReference | None = None,
        checkpointer: Any | None = None,
        paper_store: PaperInputStore | None = None,
        telemetry: Telemetry | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.modules = modules
        self.adapter = RISEActionAdapter(modules)
        self.initial_budget = budget.snapshot()
        self.run_id = run_id or uuid4().hex
        self.paper_input = paper_input
        self.checkpointer = checkpointer
        self.paper_store = paper_store
        self.telemetry = telemetry or DEFAULT_TELEMETRY
        self.tracer = tracer

    @property
    def recursion_limit(self) -> int:
        """Bound graph supersteps from remaining RISE Action Budget.

        A trajectory needs at most one Bootstrap, one Controller and one
        Action node per remaining action, one final Controller stop check, or
        a Finalizer after Commit. The small fixed margin covers either ending
        path without imposing LangGraph's unrelated default step ceiling.
        """

        return _recursion_limit_for_remaining_steps(self.initial_budget.remaining_steps)

    def build_graph(
        self,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
        checkpointer: Any | None = None,
        interrupt_before_nodes: Sequence[str] | None = None,
        boundary_check: BoundaryCheck | None = None,
    ) -> Any:
        """Compile the graph without adding Action retries."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "LangGraph is optional; install the project with `.[agent]`."
            ) from exc

        graph = StateGraph(RISEGraphState)

        def observed_node(
            node_name: str,
            function: Callable[[Any], dict[str, Any]],
            *,
            action: str | None = None,
        ) -> Callable[[Any], dict[str, Any]]:
            def execute(raw_state: Any) -> dict[str, Any]:
                state = _as_graph_state(raw_state)
                started = time.monotonic_ns()
                trace_scope = (
                    self.tracer.span(
                        f"rise.graph.{node_name}",
                        observation_type="agent" if node_name in {"controller", "bootstrap"} else "span",
                        trace_id=state.run_id,
                        metadata={
                            "task_id": state.run_id,
                            "thread_id": state.run_id,
                            "graph_node": node_name,
                            "action": action,
                            "operation": "graph_node",
                        },
                    )
                    if self.tracer is not None
                    else nullcontext(None)
                )
                with trace_scope as trace_span, bind_log_context(
                    thread_id=state.run_id, graph_node=node_name, action=action
                ):
                    self.telemetry.record(
                        "graph.node.started",
                        operation=node_name,
                        status="started",
                    )
                    try:
                        result = function(raw_state)
                    except Exception as exc:
                        self.telemetry.record(
                            "graph.node.finished",
                            operation=node_name,
                            node_status="error",
                            status="error",
                            duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                            error_category="infrastructure",
                            error_type=type(exc).__name__,
                        )
                        raise

                    updated = _as_graph_state(result)
                    budget = updated.budget
                    selected_action = action
                    if (
                        node_name == "controller"
                        and updated.controller_routing is not None
                        and updated.controller_routing.decision is not None
                    ):
                        decision_action = updated.controller_routing.decision.action
                        selected_action = str(getattr(decision_action, "value", decision_action))
                    with bind_log_context(action=selected_action):
                        if trace_span is not None:
                            trace_span.update(
                                action=selected_action,
                                rise_outcome=updated.rise_status,
                                status="completed",
                            )
                        self.telemetry.record(
                            "graph.node.finished",
                            operation=node_name,
                            node_status="returned",
                            status="completed",
                            duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                            rise_outcome=updated.rise_status,
                            action_steps_used=budget.steps_used,
                            action_steps_remaining=budget.remaining_steps,
                            searches_used=budget.searches_used,
                            searches_remaining=budget.remaining_searches,
                            trajectory_event_count=len(updated.trajectory_events),
                            replay_record_count=len(updated.replay_records),
                        )
                    return result

            return execute

        def check_boundary(node_name: str, raw_state: Any) -> None:
            if boundary_check is None:
                return
            state = _as_graph_state(raw_state)
            reason = boundary_check(node_name, state)
            if reason is not None:
                from langgraph.types import interrupt

                interrupt({"reason": reason, "node": node_name})

        def bootstrap_node(raw_state: Any) -> dict[str, Any]:
            check_boundary("bootstrap", raw_state)
            state = _as_graph_state(raw_state)
            updated = self.adapter.bootstrap(
                state,
                full_paper_reading_context=full_paper_reading_context,
            )
            return updated.to_payload()

        def controller_node(raw_state: Any) -> dict[str, Any]:
            check_boundary("controller", raw_state)
            return self.adapter.decide(_as_graph_state(raw_state)).to_payload()

        graph.add_node("bootstrap", observed_node("bootstrap", bootstrap_node))
        graph.add_node("controller", observed_node("controller", controller_node))

        for action, node_name in ACTION_NODE_NAMES.items():
            graph.add_node(
                node_name,
                observed_node(
                    node_name,
                    self._action_node(action, check_boundary),
                    action=str(getattr(action, "value", action)),
                ),
            )

        def finalizer_node(raw_state: Any) -> dict[str, Any]:
            check_boundary("finalizer", raw_state)
            return self.adapter.finalize(_as_graph_state(raw_state)).to_payload()

        graph.add_node("finalizer", observed_node("finalizer", finalizer_node))
        graph.add_edge(START, "bootstrap")
        graph.add_conditional_edges(
            "bootstrap",
            self._route_after_bootstrap,
            {"controller": "controller", "end": END},
        )

        controller_destinations = {
            "end": END,
            **{
                action: node_name
                for action, node_name in ACTION_NODE_NAMES.items()
            },
        }
        graph.add_conditional_edges(
            "controller",
            self._route_controller_decision,
            controller_destinations,
        )

        for action, node_name in ACTION_NODE_NAMES.items():
            if action == "COMMIT":
                graph.add_conditional_edges(
                    node_name,
                    self._route_after_commit,
                    {"finalizer": "finalizer", "end": END},
                )
            else:
                graph.add_conditional_edges(
                    node_name,
                    self._route_after_action,
                    {"controller": "controller", "end": END},
                )
        graph.add_edge("finalizer", END)
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=(
                list(interrupt_before_nodes) if interrupt_before_nodes is not None else None
            ),
        )

    def _action_node(
        self,
        expected_action: ScientificAction,
        check_boundary: Callable[[str, Any], None] | None = None,
    ) -> Callable[[Any], dict[str, Any]]:
        def execute(raw_state: Any) -> dict[str, Any]:
            if check_boundary is not None:
                check_boundary(ACTION_NODE_NAMES[expected_action], raw_state)
            state = _as_graph_state(raw_state)
            routing = state.controller_routing
            if (
                not state.decision_pending
                or routing is None
                or routing.decision is None
                or routing.decision.action != expected_action
            ):
                raise GraphRoutingInvariantError(
                    f"{expected_action} node received a different saved decision"
                )
            return self.adapter.execute_selected_action(state).to_payload()

        return execute

    @staticmethod
    def _route_after_bootstrap(raw_state: Any) -> str:
        state = _as_graph_state(raw_state)
        if state.rise_status in _TERMINAL_STATUSES:
            return "end"
        if state.rise_status != "running" or not state.bootstrap_complete:
            raise GraphRoutingInvariantError(
                "Bootstrap did not complete or produce a terminal result"
            )
        return "controller"

    @staticmethod
    def _route_controller_decision(raw_state: Any) -> str:
        state = _as_graph_state(raw_state)
        if state.rise_status in _TERMINAL_STATUSES:
            return "end"
        if state.rise_status != "running" or not state.decision_pending:
            raise GraphRoutingInvariantError(
                "Controller did not produce a pending decision or terminal result"
            )
        routing = state.controller_routing
        if routing is None or routing.decision is None:
            raise GraphRoutingInvariantError(
                "pending Controller decision is missing from Graph State"
            )
        action = routing.decision.action
        if action not in ACTION_NODE_NAMES:
            raise GraphRoutingInvariantError(
                f"saved Controller decision has no graph node: {action!r}"
            )
        return action

    @staticmethod
    def _route_after_action(raw_state: Any) -> str:
        state = _as_graph_state(raw_state)
        if state.rise_status in _TERMINAL_STATUSES:
            return "end"
        if state.rise_status != "running":
            raise GraphRoutingInvariantError(
                f"unexpected RISE status after Action: {state.rise_status!r}"
            )
        return "controller"

    @staticmethod
    def _route_after_commit(raw_state: Any) -> str:
        state = _as_graph_state(raw_state)
        if state.rise_status in _TERMINAL_STATUSES:
            return "end"
        if state.rise_status != "running" or state.commit_snapshot is None:
            raise GraphRoutingInvariantError(
                "Commit completed without a running state and CommitSnapshot"
            )
        return "finalizer"

    @staticmethod
    def _has_durable_completed_node_result(snapshot: Any, state: RISEGraphState) -> bool:
        """Accept a completed task result saved before the next root checkpoint.

        With synchronous durability, LangGraph may have committed a node's
        ``put_writes`` record while a subsequent checkpoint ``put`` fails.
        In that case ``get_state`` exposes the completed node result and its
        updated channels, but ``snapshot.next`` is empty. LangGraph can resume
        from that durable task result without invoking the node again.
        """

        tasks = getattr(snapshot, "tasks", ())
        if len(tasks) != 1:
            return False
        task = tasks[0]
        if task.result is None or task.error is not None or task.interrupts:
            return False
        try:
            result_state = _as_graph_state(task.result)
        except GraphStateValidationError:
            return False
        if result_state != state:
            return False

        if state.decision_pending:
            expected_completed_node = "controller"
        elif state.commit_snapshot is not None:
            expected_completed_node = "commit"
        elif state.rise_status == "running" and state.bootstrap_complete:
            routing = state.controller_routing
            if routing is None:
                expected_completed_node = "bootstrap"
            elif routing.decision is not None:
                expected_completed_node = ACTION_NODE_NAMES.get(
                    routing.decision.action
                )
            else:
                return False
        else:
            return False
        return task.name == expected_completed_node

    @staticmethod
    def _expected_resume_node(state: RISEGraphState) -> str:
        """Derive the only legal continuation from the saved RISE snapshot."""

        if state.rise_status == "initialized":
            if state.bootstrap_complete or state.decision_pending:
                raise PersistentExecutionError(
                    "initialized checkpoint has inconsistent Bootstrap state"
                )
            return "bootstrap"

        if state.rise_status != "running" or not state.bootstrap_complete:
            raise PersistentExecutionError(
                "non-terminal checkpoint has inconsistent RISE execution state"
            )

        if state.decision_pending:
            routing = state.controller_routing
            if routing is None or routing.decision is None:
                raise PersistentExecutionError(
                    "pending Action checkpoint has no saved Controller decision"
                )
            node = ACTION_NODE_NAMES.get(routing.decision.action)
            if node is None:
                raise PersistentExecutionError(
                    "pending Action checkpoint selects an unknown Scientific Action"
                )
            if state.commit_snapshot is not None:
                raise PersistentExecutionError(
                    "pending Action checkpoint cannot already contain a CommitSnapshot"
                )
            return node

        if state.commit_snapshot is not None:
            routing = state.controller_routing
            if (
                routing is None
                or routing.decision is None
                or routing.decision.action != "COMMIT"
            ):
                raise PersistentExecutionError(
                    "CommitSnapshot checkpoint has no matching saved Commit decision"
                )
            return "finalizer"

        return "controller"

    def has_persistent_checkpoint(self, *, thread_id: str) -> bool:
        """Return whether the configured saver has a checkpoint for this thread."""

        if self.checkpointer is None:
            raise PersistentExecutionError("persistent execution requires a checkpointer")
        canonical_id = _canonical_thread_id(thread_id)
        with _lock_thread(canonical_id):
            return self.checkpointer.get_tuple(self._thread_config(canonical_id)) is not None

    def load_persistent_state(self, *, thread_id: str) -> RISEGraphState | None:
        """Read and validate a saved state without advancing the graph."""

        if self.checkpointer is None:
            raise PersistentExecutionError("persistent execution requires a checkpointer")
        canonical_id = _canonical_thread_id(thread_id)
        config = self._thread_config(canonical_id)
        with _lock_thread(canonical_id):
            if self.checkpointer.get_tuple(config) is None:
                return None
            graph = self.build_graph(checkpointer=self.checkpointer)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise MissingCheckpointError(
                    f"checkpoint contains no Graph State for thread: {canonical_id}"
                )
            schema_version = snapshot.values.get("schema_version")
            if schema_version != 1:
                raise PersistentExecutionError(
                    f"unsupported checkpoint Graph State schema_version: {schema_version!r}"
                )
            try:
                state = _as_graph_state(snapshot.values)
            except GraphStateValidationError as exc:
                raise PersistentExecutionError(
                    f"checkpoint Graph State is invalid for thread {canonical_id}"
                ) from exc
            if state.run_id != canonical_id:
                raise PersistentExecutionError(
                    "checkpoint run_id does not match its stable thread_id"
                )
            return state

    def run(
        self,
        seed_context: SeedPaperContext,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> RISEGraphState:
        """Run one trajectory without persistence, preserving the P1 entry point.

        Checkpoint-backed callers should use ``start_persistent`` or
        ``resume_persistent`` instead.
        """

        initial_state = RISEGraphState.initialize(
            seed_context,
            self.initial_budget,
            run_id=self.run_id,
            paper_input=self.paper_input,
        )
        graph = self.build_graph(
            full_paper_reading_context=full_paper_reading_context
        )
        result = self._invoke_observed(
            graph,
            initial_state.to_payload(),
            config={"recursion_limit": self.recursion_limit},
            thread_id=initial_state.run_id,
        )
        return RISEGraphState.from_payload(result)

    def start_persistent(
        self,
        seed_context: SeedPaperContext,
        *,
        thread_id: str,
        markdown: str | bytes | Path,
        interrupt_before_nodes: Sequence[str] | None = None,
        boundary_check: BoundaryCheck | None = None,
    ) -> RISEGraphState:
        """Create and run a new PostgreSQL-checkpointed trajectory.

        ``markdown`` accepts UTF-8 source text/bytes or a ``pathlib.Path``.
        The task UUID is also stored as ``run_id`` so restored snapshots keep
        the same identity. Existing threads are never overwritten.
        """

        checkpointer, paper_store = self._require_persistence()
        canonical_id = _canonical_thread_id(thread_id)
        config = self._thread_config(canonical_id)
        with _lock_thread(canonical_id):
            if checkpointer.get_tuple(config) is not None:
                raise ExistingThreadError(
                    f"thread already has a checkpoint: {canonical_id}"
                )

            paper_input = paper_store.persist(markdown)
            reading_context = paper_store.build_reading_context(
                paper_input,
                paper_id=seed_context.paper_id,
            )
            initial_state = RISEGraphState.initialize(
                seed_context,
                self.initial_budget,
                run_id=canonical_id,
                paper_input=paper_input,
            )
            graph = self.build_graph(
                full_paper_reading_context=reading_context,
                checkpointer=checkpointer,
                interrupt_before_nodes=interrupt_before_nodes,
                boundary_check=boundary_check,
            )
            result = self._invoke_observed(
                graph,
                initial_state.to_payload(),
                config={**config, "recursion_limit": self.recursion_limit},
                durability="sync",
                thread_id=canonical_id,
            )
            return RISEGraphState.from_payload(result)

    def resume_persistent(
        self,
        *,
        thread_id: str,
        interrupt_before_nodes: Sequence[str] | None = None,
        boundary_check: BoundaryCheck | None = None,
    ) -> RISEGraphState:
        """Resume the latest reliable checkpoint without reinitializing RISE."""

        checkpointer, paper_store = self._require_persistence()
        canonical_id = _canonical_thread_id(thread_id)
        config = self._thread_config(canonical_id)
        with _lock_thread(canonical_id):
            if checkpointer.get_tuple(config) is None:
                raise MissingCheckpointError(
                    f"no checkpoint exists for thread: {canonical_id}"
                )

            # Read the saved snapshot before rebuilding the runtime closure.
            # This graph is used only for state inspection; it executes no node.
            inspection_graph = self.build_graph(checkpointer=checkpointer)
            snapshot = inspection_graph.get_state(config)
            if not snapshot.values:
                raise MissingCheckpointError(
                    f"checkpoint contains no Graph State for thread: {canonical_id}"
                )
            schema_version = snapshot.values.get("schema_version")
            if schema_version != 1:
                raise PersistentExecutionError(
                    f"unsupported checkpoint Graph State schema_version: {schema_version!r}"
                )
            try:
                state = _as_graph_state(snapshot.values)
            except GraphStateValidationError as exc:
                raise PersistentExecutionError(
                    f"checkpoint Graph State is invalid for thread {canonical_id}"
                ) from exc

            if state.run_id != canonical_id:
                raise PersistentExecutionError(
                    "checkpoint run_id does not match its stable thread_id"
                )
            if state.rise_status in _TERMINAL_STATUSES:
                return state
            if not snapshot.next and not self._has_durable_completed_node_result(
                snapshot, state
            ):
                raise PersistentExecutionError(
                    "non-terminal checkpoint has no pending LangGraph node or "
                    "durable completed task result"
                )
            expected_node = self._expected_resume_node(state)
            if tuple(snapshot.next) != (expected_node,):
                if snapshot.next or not self._has_durable_completed_node_result(
                    snapshot, state
                ):
                    raise PersistentExecutionError(
                        "checkpoint pending node does not match saved RISE state: "
                        f"expected {(expected_node,)!r}, got {tuple(snapshot.next)!r}"
                    )
            if state.paper_input is None:
                raise PaperInputIntegrityError(
                    "non-terminal checkpoint has no persisted paper input reference"
                )

            reading_context = paper_store.build_reading_context(
                state.paper_input,
                paper_id=state.seed_context.paper_id,
            )
            graph = self.build_graph(
                full_paper_reading_context=reading_context,
                checkpointer=checkpointer,
                interrupt_before_nodes=interrupt_before_nodes,
                boundary_check=boundary_check,
            )
            result = self._invoke_observed(
                graph,
                None,
                config={
                    **config,
                    "recursion_limit": _recursion_limit_for_remaining_steps(
                        state.budget.remaining_steps
                    ),
                },
                durability="sync",
                thread_id=canonical_id,
            )
            try:
                return RISEGraphState.from_payload(result)
            except GraphStateValidationError as exc:
                raise PersistentExecutionError(
                    f"resumed Graph State is invalid for thread {canonical_id}"
                ) from exc

    def _require_persistence(self) -> tuple[Any, PaperInputStore]:
        if self.checkpointer is None:
            raise PersistentExecutionError(
                "persistent execution requires a checkpointer"
            )
        if self.paper_store is None:
            raise PersistentExecutionError(
                "persistent execution requires a stable PaperInputStore"
            )
        return self.checkpointer, self.paper_store

    def _invoke_observed(
        self,
        graph: Any,
        input_value: Any,
        *,
        config: dict[str, Any],
        thread_id: str,
        durability: str | None = None,
    ) -> Any:
        started = time.monotonic_ns()
        with bind_log_context(thread_id=thread_id):
            try:
                if durability is None:
                    result = graph.invoke(input_value, config=config)
                else:
                    result = graph.invoke(
                        input_value,
                        config=config,
                        durability=durability,
                    )
            except Exception as exc:
                self.telemetry.record(
                    "graph.execution.finished",
                    operation="invoke",
                    status="error",
                    duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    error_category="infrastructure",
                    error_type=type(exc).__name__,
                )
                raise
            try:
                state = _as_graph_state(result)
            except Exception:
                state = None
            fields: dict[str, Any] = {
                "operation": "invoke",
                "status": "completed",
                "duration_ms": (time.monotonic_ns() - started) / 1_000_000,
            }
            if state is not None:
                fields.update(
                    rise_outcome=state.rise_status,
                    action_steps_used=state.budget.steps_used,
                    action_steps_remaining=state.budget.remaining_steps,
                    searches_used=state.budget.searches_used,
                    searches_remaining=state.budget.remaining_searches,
                    trajectory_event_count=len(state.trajectory_events),
                    replay_record_count=len(state.replay_records),
                )
            self.telemetry.record("graph.execution.finished", **fields)
            return result

    @staticmethod
    def _thread_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}


__all__ = [
    "ACTION_NODE_NAMES",
    "ExistingThreadError",
    "GraphRoutingInvariantError",
    "LangGraphTrajectoryRunner",
    "MissingCheckpointError",
    "PersistentExecutionError",
]
