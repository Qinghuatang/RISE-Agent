"""LangGraph orchestration for one RISE scientific trajectory.

The graph owns execution routing only. Scientific decisions and action
semantics remain in ``ScientificController``, ``TrajectoryRunner`` helpers,
and ``RISEActionAdapter``.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from ..controller.budget import BudgetTracker, ControllerBudget
from ..ingestion.markdown import FullPaperReadingContext
from ..models.base import ScientificAction
from ..models.seed import SeedPaperContext
from ..runner.trajectory_runner import TrajectoryModules
from .action_adapter import RISEActionAdapter
from .graph_state import PaperInputReference, RISEGraphState


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


class GraphRoutingInvariantError(RuntimeError):
    """Raised when a validated graph snapshot cannot select a legal next node."""


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
    ) -> None:
        self.modules = modules
        self.adapter = RISEActionAdapter(modules)
        self.initial_budget = budget.snapshot()
        self.run_id = run_id or uuid4().hex
        self.paper_input = paper_input

    @property
    def recursion_limit(self) -> int:
        """Bound graph supersteps from remaining RISE Action Budget.

        A trajectory needs at most one Bootstrap, one Controller and one
        Action node per remaining action, one final Controller stop check, or
        a Finalizer after Commit. The small fixed margin covers either ending
        path without imposing LangGraph's unrelated default step ceiling.
        """

        remaining_steps = self.initial_budget.remaining_steps
        return max(3, 2 * remaining_steps + 3)

    def build_graph(
        self,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> Any:
        """Compile the P1-C graph without a checkpointer or Action retries."""

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
        graph.add_edge("bootstrap", "controller")

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
        return graph.compile()

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
        """Initialize and execute one full LangGraph-managed trajectory."""

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


__all__ = [
    "ACTION_NODE_NAMES",
    "GraphRoutingInvariantError",
    "LangGraphTrajectoryRunner",
]
