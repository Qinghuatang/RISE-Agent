"""Adapter from validated Graph State snapshots to legacy RISE execution."""

from __future__ import annotations

from typing import Any

from ..controller.budget import BudgetTracker, ControllerBudget
from ..errors import RunnerInvariantError
from ..ingestion.markdown import FullPaperReadingContext
from ..models.commit import CommitSnapshot
from ..models.controller import ControllerRoutingResult
from ..models.runner import TrajectoryReplayRecord
from ..models.seed import SeedPaperContext
from ..models.trajectory import BudgetMetadata
from ..modules.commit import check_commit_gate
from ..runner.trajectory_runner import TrajectoryModules, TrajectoryRunner
from ..trajectory.context import TrajectoryContext
from .graph_state import PaperInputReference, RISEGraphState


def _event_step(context: TrajectoryContext) -> int:
    events = context.trajectory.get_events()
    return events[-1].step + 1 if events else 0


def _budget_metadata(budget: ControllerBudget) -> BudgetMetadata:
    return BudgetMetadata(
        step_budget_remaining=budget.remaining_steps,
        search_budget_remaining=budget.remaining_searches,
    )


class RISEActionAdapter:
    """Reuse TrajectoryRunner's existing execution boundaries for Graph nodes.

    This class delegates Action dispatch, special verification and inspiration
    handoffs, budget updates, trajectory events, and replay creation to the
    legacy runner's existing helpers. It adds state conversion and terminal
    result mapping only.
    """

    def __init__(self, modules: TrajectoryModules) -> None:
        self.modules = modules

    @staticmethod
    def initialize(
        seed_context: SeedPaperContext,
        *,
        max_steps: int,
        max_searches: int,
        run_id: str,
        paper_input: PaperInputReference | None = None,
    ) -> RISEGraphState:
        return RISEGraphState.initialize(
            seed_context,
            ControllerBudget(max_steps=max_steps, max_searches=max_searches),
            run_id=run_id,
            paper_input=paper_input,
        )

    def _runner(self, state: RISEGraphState) -> tuple[TrajectoryRunner, BudgetTracker]:
        budget = BudgetTracker(
            state.budget.max_steps,
            state.budget.max_searches,
            steps_used=state.budget.steps_used,
            searches_used=state.budget.searches_used,
        )
        runner = TrajectoryRunner(self.modules, budget=budget)
        # The legacy runner appends these records by sequence. Seed it from the
        # snapshot so every adapter operation replaces the complete journal.
        runner._replay_records = list(state.replay_records)
        return runner, budget

    @staticmethod
    def _capture(
        base: RISEGraphState,
        context: TrajectoryContext,
        budget: BudgetTracker,
        replay_records: list[TrajectoryReplayRecord],
        **updates: Any,
    ) -> RISEGraphState:
        payload = base.model_dump(mode="python")
        payload.update(
            research_state=context.get_state(),
            research_memory=context.get_memory(),
            trajectory_events=context.trajectory.get_events(),
            budget=budget.snapshot(),
            replay_records=list(replay_records),
        )
        payload.update(updates)
        return RISEGraphState.model_validate(payload)

    @staticmethod
    def _failed(
        state: RISEGraphState,
        context: TrajectoryContext,
        budget: BudgetTracker,
        replay_records: list[TrajectoryReplayRecord],
        error: Exception,
        **updates: Any,
    ) -> RISEGraphState:
        message = str(error).strip().replace("\n", " ")[:500]
        if not message:
            message = error.__class__.__name__
        failed_updates = dict(updates)
        failed_updates.update(
            rise_status="failed",
            decision_pending=False,
            unresolved_issues=[message],
            error_type=error.__class__.__name__,
            error_message=message,
        )
        return RISEActionAdapter._capture(
            state,
            context,
            budget,
            replay_records,
            **failed_updates,
        )

    def bootstrap(
        self,
        state: RISEGraphState,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> RISEGraphState:
        """Run the existing, non-budgeted Gap Discovery bootstrap exactly once."""

        if state.rise_status != "initialized" or state.bootstrap_complete:
            raise ValueError("Bootstrap may only run for an initialized Graph State")
        context = state.to_trajectory_context()
        runner, budget = self._runner(state)
        before_memory = context.get_memory()
        step = _event_step(context)
        try:
            self.modules.gap_discovery.discover(
                state.seed_context,
                full_paper_reading_context=full_paper_reading_context,
                context=context,
                step=step,
            )
            runner._append_replay(
                context,
                seed_paper_id=state.seed_context.paper_id,
                kind="bootstrap",
                action=None,
                step=step,
                before_memory=before_memory,
                budget=budget.snapshot(),
            )
        except Exception as exc:
            return self._failed(
                state,
                context,
                budget,
                runner._replay_records,
                exc,
            )
        return self._capture(
            state,
            context,
            budget,
            runner._replay_records,
            bootstrap_complete=True,
            rise_status="running",
        )

    def decide(self, state: RISEGraphState) -> RISEGraphState:
        """Call the unchanged Scientific Controller and persist its exact route."""

        if state.rise_status != "running" or not state.bootstrap_complete:
            raise ValueError("Controller requires a bootstrapped running trajectory")
        if state.decision_pending:
            raise ValueError("the saved Controller decision must be dispatched first")

        context = state.to_trajectory_context()
        runner, budget = self._runner(state)
        try:
            gate = check_commit_gate(context.get_state(), context.memory_store)
            routing: ControllerRoutingResult = self.modules.controller.decide(
                context,
                budget=budget,
                commit_allowed=gate.commit_allowed,
                unmet_commit_conditions=gate.unmet_conditions,
            )
        except Exception as exc:
            return self._failed(state, context, budget, runner._replay_records, exc)

        if routing.status == "stopped":
            context.trajectory.append_event(
                _event_step(context),
                "controller_stopped",
                action=None,
                result_payload={"unresolved_issues": routing.unresolved_issues},
                budget=_budget_metadata(budget.snapshot()),
            )
            return self._capture(
                state,
                context,
                budget,
                runner._replay_records,
                controller_routing=routing,
                commit_check=gate,
                decision_pending=False,
                rise_status="stopped",
                unresolved_issues=list(routing.unresolved_issues),
                error_type=None,
                error_message=None,
            )

        if routing.decision is None:
            error = RunnerInvariantError(
                "Controller returned decision status without a decision"
            )
            return self._failed(
                state,
                context,
                budget,
                runner._replay_records,
                error,
                controller_routing=routing,
                commit_check=gate,
            )

        return self._capture(
            state,
            context,
            budget,
            runner._replay_records,
            controller_routing=routing,
            commit_check=gate,
            decision_pending=True,
            unresolved_issues=[],
            error_type=None,
            error_message=None,
        )

    def execute_selected_action(self, state: RISEGraphState) -> RISEGraphState:
        """Execute the already-saved Controller decision without re-deciding."""

        routing = state.controller_routing
        if (
            state.rise_status != "running"
            or not state.decision_pending
            or routing is None
            or routing.status != "decision"
            or routing.decision is None
        ):
            raise ValueError("Graph State has no pending Controller Action")

        context = state.to_trajectory_context()
        runner, budget = self._runner(state)
        outcome, error, _attempt_started, _step, _search_delta = runner._run_action(
            context,
            routing.decision,
            allowed_actions=list(routing.allowed_actions),
            used_llm=routing.used_llm,
        )
        if error is not None:
            return self._failed(
                state,
                context,
                budget,
                runner._replay_records,
                error,
                decision_pending=False,
            )

        assert outcome is not None
        updates: dict[str, Any] = {"decision_pending": False}
        if routing.decision.action == "COMMIT":
            if not isinstance(outcome.result, CommitSnapshot):
                error = RunnerInvariantError("Commit action returned an invalid snapshot")
                return self._failed(
                    state,
                    context,
                    budget,
                    runner._replay_records,
                    error,
                    decision_pending=False,
                )
            updates["commit_snapshot"] = outcome.result

        return self._capture(
            state,
            context,
            budget,
            runner._replay_records,
            **updates,
        )

    def finalize(self, state: RISEGraphState) -> RISEGraphState:
        """Run the read-only Finalizer after a successful Commit action."""

        routing = state.controller_routing
        if (
            state.rise_status != "running"
            or state.decision_pending
            or state.commit_snapshot is None
            or routing is None
            or routing.decision is None
            or routing.decision.action != "COMMIT"
        ):
            raise ValueError("Finalizer requires a completed Commit action")

        context = state.to_trajectory_context()
        runner, budget = self._runner(state)
        try:
            proposal = self.modules.finalizer.finalize(
                state.commit_snapshot,
                state.seed_context,
            )
        except Exception as exc:
            message = str(exc).strip().replace("\n", " ")[:500]
            if not message:
                message = exc.__class__.__name__
            return self._capture(
                state,
                context,
                budget,
                runner._replay_records,
                rise_status="finalization_failed",
                unresolved_issues=[message],
                error_type=exc.__class__.__name__,
                error_message=message,
            )
        return self._capture(
            state,
            context,
            budget,
            runner._replay_records,
            rise_status="committed",
            final_proposal=proposal,
            unresolved_issues=[],
            error_type=None,
            error_message=None,
        )


__all__ = ["RISEActionAdapter"]
