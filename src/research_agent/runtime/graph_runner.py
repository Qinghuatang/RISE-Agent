"""LangGraph orchestration for one RISE scientific trajectory.

The graph owns execution routing only. Scientific decisions and action
semantics remain in ``ScientificController``, ``TrajectoryRunner`` helpers,
and ``RISEActionAdapter``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable
from uuid import UUID, uuid4

from ..controller.budget import BudgetTracker, ControllerBudget
from ..ingestion.markdown import FullPaperReadingContext
from ..models.base import ScientificAction
from ..models.seed import SeedPaperContext
from ..runner.trajectory_runner import TrajectoryModules
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
    ) -> None:
        self.modules = modules
        self.adapter = RISEActionAdapter(modules)
        self.initial_budget = budget.snapshot()
        self.run_id = run_id or uuid4().hex
        self.paper_input = paper_input
        self.checkpointer = checkpointer
        self.paper_store = paper_store

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
    ) -> Any:
        """Compile the graph without adding Action retries."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "LangGraph is optional; install the project with `.[agent]`."
            ) from exc

        graph = StateGraph(RISEGraphState)

        def bootstrap_node(raw_state: Any) -> dict[str, Any]:
            state = _as_graph_state(raw_state)
            updated = self.adapter.bootstrap(
                state,
                full_paper_reading_context=full_paper_reading_context,
            )
            return updated.to_payload()

        def controller_node(raw_state: Any) -> dict[str, Any]:
            return self.adapter.decide(_as_graph_state(raw_state)).to_payload()

        graph.add_node("bootstrap", bootstrap_node)
        graph.add_node("controller", controller_node)

        for action, node_name in ACTION_NODE_NAMES.items():
            graph.add_node(node_name, self._action_node(action))

        def finalizer_node(raw_state: Any) -> dict[str, Any]:
            return self.adapter.finalize(_as_graph_state(raw_state)).to_payload()

        graph.add_node("finalizer", finalizer_node)
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
        return graph.compile(checkpointer=checkpointer)

    def _action_node(self, expected_action: ScientificAction) -> Callable[[Any], dict[str, Any]]:
        def execute(raw_state: Any) -> dict[str, Any]:
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
        result = graph.invoke(
            initial_state.to_payload(),
            config={"recursion_limit": self.recursion_limit},
        )
        return RISEGraphState.from_payload(result)

    def start_persistent(
        self,
        seed_context: SeedPaperContext,
        *,
        thread_id: str,
        markdown: str | bytes | Path,
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
            )
            result = graph.invoke(
                initial_state.to_payload(),
                config={**config, "recursion_limit": self.recursion_limit},
                durability="sync",
            )
            return RISEGraphState.from_payload(result)

    def resume_persistent(self, *, thread_id: str) -> RISEGraphState:
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
            if not snapshot.next:
                raise PersistentExecutionError(
                    "non-terminal checkpoint has no pending LangGraph node"
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
            )
            result = graph.invoke(
                None,
                config={
                    **config,
                    "recursion_limit": _recursion_limit_for_remaining_steps(
                        state.budget.remaining_steps
                    ),
                },
                durability="sync",
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
