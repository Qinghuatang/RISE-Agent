"""Offline end-to-end orchestration of the existing scientific modules.

The runner is deliberately an execution boundary.  It does not create
scientific judgments, choose a gap, synthesize a solution, or add actions to
the controller's fixed action space.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..controller.budget import BudgetTracker, ControllerBudget
from ..controller.scientific_controller import (
    record_decision_result,
)
from ..errors import RunnerInvariantError
from ..ingestion.markdown import FullPaperReadingContext
from ..memory.store import ResearchMemoryStore
from ..models.commit import CommitSnapshot, FinalResearchProposal
from ..models.controller import ControllerDecision, ControllerRoutingResult
from ..models.memory import ResearchMemory
from ..models.runner import (
    RunBudget,
    RunManifest,
    TrajectoryReplayRecord,
    TrajectoryRunResult,
)
from ..models.seed import SeedPaperContext
from ..models.state import ResearchState
from ..models.trajectory import BudgetMetadata
from ..models.verification import GapVerificationResult, VerificationFocus
from ..modules.commit import CommitModule, check_commit_gate
from ..modules.experiment_design import ExperimentDesignModule
from ..modules.finalizer import FinalizerModule, save_final_research_proposal
from ..modules.gap_discovery import GapDiscoveryModule
from ..modules.inspiration_retrieval import InspirationRetrievalModule
from ..modules.related_work_verification import RelatedWorkVerificationModule
from ..modules.solution_ideation import SolutionIdeationModule
from ..modules.targeted_revision import TargetedRevisionModule
from ..serialization import write_json
from ..serialization.trajectory import save_replay_journal
from ..trajectory.context import TrajectoryContext


@dataclass(frozen=True)
class TrajectoryModules:
    """Explicit dependency bundle; all dependencies are injected by callers."""

    gap_discovery: GapDiscoveryModule
    solution_ideation: SolutionIdeationModule
    related_work_verification: RelatedWorkVerificationModule
    inspiration_retrieval: InspirationRetrievalModule
    targeted_revision: TargetedRevisionModule
    experiment_design: ExperimentDesignModule
    controller: Any
    commit: CommitModule
    finalizer: FinalizerModule


@dataclass(frozen=True)
class _DispatchOutcome:
    result: Any
    boundary_step: int
    verify_kind: str | None = None


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _budget_metadata(budget: ControllerBudget) -> BudgetMetadata:
    return BudgetMetadata(
        step_budget_remaining=budget.remaining_steps,
        search_budget_remaining=budget.remaining_searches,
    )


def _event_step(context: TrajectoryContext) -> int:
    events = context.trajectory.get_events()
    return (events[-1].step + 1) if events else 0


def _search_count(modules: TrajectoryModules) -> int:
    return sum(
        max(0, int(getattr(module, "scientific_search_count", 0)))
        for module in (modules.related_work_verification, modules.inspiration_retrieval)
    )


def _memory_patch(before: ResearchMemory, after: ResearchMemory) -> dict[str, Any]:
    before_payload = before.model_dump(mode="json")
    after_payload = after.model_dump(mode="json")
    return {
        category: deepcopy(after_payload[category])
        for category in after_payload
        if before_payload.get(category) != after_payload.get(category)
    }


def _error_text(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    return message[:500] if message else exc.__class__.__name__


def _structured_error_details(exc: Exception) -> dict[str, Any] | None:
    """Return safe structured-output diagnostics for trajectory auditing."""

    schema_name = getattr(exc, "schema_name", None)
    validation_errors = getattr(exc, "validation_errors", None)
    parsed_payload = getattr(exc, "parsed_payload", None)
    raw_response_excerpt = getattr(exc, "raw_response_excerpt", None)
    if (
        schema_name is None
        and not validation_errors
        and parsed_payload is None
        and raw_response_excerpt is None
    ):
        return None
    details: dict[str, Any] = {
        "schema_name": schema_name,
        "validation_errors": list(validation_errors or []),
    }
    if isinstance(parsed_payload, dict):
        details["parsed_payload"] = deepcopy(parsed_payload)
    if isinstance(raw_response_excerpt, str) and raw_response_excerpt:
        details["raw_response_excerpt"] = raw_response_excerpt[:4_000]
    return details


class TrajectoryRunner:
    """Run one prepared seed context through the existing M0–M10 contracts."""

    def __init__(
        self,
        modules: TrajectoryModules,
        *,
        budget: BudgetTracker,
        output_dir: str | Path | None = None,
    ) -> None:
        self.modules = modules
        self.budget = budget
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._replay_records: list[TrajectoryReplayRecord] = []

    def _append_replay(
        self,
        context: TrajectoryContext,
        *,
        seed_paper_id: str,
        kind: str,
        action: str | None,
        step: int,
        before_memory: ResearchMemory,
        budget: ControllerBudget,
    ) -> None:
        record = TrajectoryReplayRecord(
            seed_paper_id=seed_paper_id,
            sequence=len(self._replay_records) + 1,
            step=step,
            kind=kind,
            action=action,
            state_after=context.get_state(),
            memory_patch=_memory_patch(before_memory, context.get_memory()),
            budget_after=budget,
        )
        self._replay_records.append(record)

    def _resolve_verification_target(
        self,
        context: TrajectoryContext,
    ) -> tuple[str, VerificationFocus | None]:
        state = context.get_state()
        memory = context.get_memory()
        gap = state.active_gap
        if gap is None:
            raise RunnerInvariantError("VERIFY_RELATED_WORK requires an active gap")
        latest_gap = context.memory_store.get_latest_gap_verification(gap.id)
        if latest_gap is None:
            return "gap", None
        refined_after = any(
            event.event_type == "gap_refined"
            and event.result_ref == gap.id
            and event.step > latest_gap.created_at_step
            for event in context.trajectory.get_events()
        )
        if refined_after:
            return "gap", None
        if not isinstance(latest_gap.result, GapVerificationResult):
            raise RunnerInvariantError("latest Gap Verification has an invalid result")

        solution = state.current_solution
        if solution is None:
            raise RunnerInvariantError(
                "VERIFY_RELATED_WORK has no unique Gap or Solution target"
            )
        latest_solution = context.memory_store.get_latest_verification(
            solution.id, task="VERIFY_SOLUTION"
        )
        if solution.novelty_status == "unchecked" or latest_solution is None:
            return "solution", self._latest_revision_focus(context, solution.id)
        raise RunnerInvariantError(
            "VERIFY_RELATED_WORK has no unique unresolved verification target"
        )

    @staticmethod
    def _latest_revision_focus(
        context: TrajectoryContext,
        solution_id: str,
    ) -> VerificationFocus:
        for event in reversed(context.trajectory.get_events()):
            if event.event_type != "solution_revised" or event.result_ref != solution_id:
                continue
            payload = event.result_payload or {}
            focus_payload = payload.get("verification_focus")
            if isinstance(focus_payload, dict):
                try:
                    return VerificationFocus.model_validate(focus_payload)
                except ValidationError as exc:
                    raise RunnerInvariantError(
                        "solution_revised contains an invalid verification_focus"
                    ) from exc
        return VerificationFocus(mode="full_solution")

    @staticmethod
    def _latest_inspiration_trigger(
        context: TrajectoryContext,
    ) -> str | None:
        # Inspiration is a direct handoff from the immediately preceding
        # Scientific Action only.  Historical ``inspiration_retrieved``
        # events must never be reused after another action has run.
        latest_action = next(
            (event for event in reversed(context.trajectory.get_events())
             if event.action is not None),
            None,
        )
        if latest_action is None or latest_action.action != "SEEK_INSPIRATION":
            return None
        if latest_action.event_type != "inspiration_retrieved":
            return None
        record_id = latest_action.result_ref
        if not record_id:
            return None
        records = context.memory_store.get_relevant_inspirations(record_id=record_id)
        if len(records) != 1 or not records[0].inspirations:
            return None
        result = (latest_action.result_payload or {}).get("result")
        if isinstance(result, dict) and result.get("status") == "found":
            return record_id
        return None

    @staticmethod
    def _latest_inspiration_exploration_rationale(
        context: TrajectoryContext,
    ) -> str | None:
        """Recover the preceding Controller reason for an Inspiration handoff.

        The reason is an execution-only value.  It is already present in the
        controller_decided trajectory event, so it does not need a new State
        or Research Memory field.
        """

        for event in reversed(context.trajectory.get_events()):
            if event.event_type != "controller_decided":
                continue
            payload = event.result_payload or {}
            decision = payload.get("decision")
            if not isinstance(decision, dict):
                continue
            if decision.get("action") != "SEEK_INSPIRATION":
                # _run_action records the current TARGETED_REVISION decision
                # before dispatch.  Continue to the preceding controller
                # decision to recover the Inspiration rationale.
                continue
            reason = decision.get("reason")
            return reason if isinstance(reason, str) and reason.strip() else None
        return None

    @staticmethod
    def _switch_gap_terminal_status(context: TrajectoryContext) -> str:
        """Derive the old Gap status from verification, never from prose.

        A resolved verification permits ``resolved``.  Every other open-gap
        abandonment path is a rejection, including valid and partially
        resolved gaps whose solution space no longer merits investment.
        """

        gap = context.get_state().active_gap
        if gap is None:
            raise RunnerInvariantError("SWITCH_GAP requires an active gap")
        verification = context.memory_store.get_latest_gap_verification(gap.id)
        if verification is None or not isinstance(verification.result, GapVerificationResult):
            raise RunnerInvariantError(
                "SWITCH_GAP requires a valid latest Gap Verification"
            )
        return "resolved" if verification.result.status == "resolved" else "rejected"

    def _dispatch(
        self,
        decision: ControllerDecision,
        context: TrajectoryContext,
        step: int,
    ) -> _DispatchOutcome:
        action = decision.action
        if action == "SOLUTION_IDEATION":
            result = self.modules.solution_ideation.ideate(context, step=step)
            return _DispatchOutcome(result, step)
        if action == "SEEK_INSPIRATION":
            result = self.modules.inspiration_retrieval.retrieve(
                context=context,
                controller_exploration_rationale=decision.reason,
                step=step,
            )
            return _DispatchOutcome(result, step)
        if action == "VERIFY_RELATED_WORK":
            target, focus = self._resolve_verification_target(context)
            gap = context.get_state().active_gap
            assert gap is not None
            if target == "gap":
                result = self.modules.related_work_verification.verify_gap(
                    gap, context=context, step=step
                )
            else:
                result = self.modules.related_work_verification.verify_solution(
                    gap,
                    context=context,
                    verification_focus=focus,
                    step=step,
                )
            return _DispatchOutcome(result, step, verify_kind=target)
        if action == "TARGETED_REVISION":
            result = self.modules.targeted_revision.revise(
                context=context,
                inspiration_record_id=self._latest_inspiration_trigger(context),
                controller_exploration_rationale=(
                    self._latest_inspiration_exploration_rationale(context)
                    if self._latest_inspiration_trigger(context) is not None
                    else None
                ),
                step=step,
            )
            return _DispatchOutcome(result, step)
        if action == "DESIGN_EXPERIMENT":
            result = self.modules.experiment_design.design(context=context, step=step)
            return _DispatchOutcome(result, step)
        if action == "SWITCH_GAP":
            result = self.modules.gap_discovery.switch_gap(
                context=context,
                terminal_status=self._switch_gap_terminal_status(context),
                step=step,
            )
            return _DispatchOutcome(result, step)
        if action == "COMMIT":
            result = self.modules.commit.commit(context=context, step=step)
            return _DispatchOutcome(result, step)
        raise RunnerInvariantError(f"unsupported Scientific Action: {action!r}")

    def _post_action_refine(
        self,
        outcome: _DispatchOutcome,
        context: TrajectoryContext,
    ) -> _DispatchOutcome:
        if outcome.verify_kind != "gap":
            return outcome
        result = outcome.result
        if not isinstance(result, GapVerificationResult):
            return outcome
        if result.status != "partially_resolved":
            return outcome
        gap = context.get_state().active_gap
        if gap is None:
            raise RunnerInvariantError("partial Gap Verification left no active gap")
        # RefineGap is bounded to one call.  A later re-verification may still
        # be partially resolved, but it must not invoke RefineGap a second
        # time; the Controller will deterministically reject/switch the gap
        # on its next turn.
        if context.has_refined_gap(gap.id):
            return outcome
        refine_step = max(_event_step(context), outcome.boundary_step + 1)
        refined = self.modules.gap_discovery.refine_gap(
            gap, result, context=context, step=refine_step
        )
        return _DispatchOutcome(result, refine_step, verify_kind="gap")

    @staticmethod
    def _result_summary(action: str, outcome: _DispatchOutcome) -> str:
        result = outcome.result
        if action == "VERIFY_RELATED_WORK":
            if outcome.verify_kind == "gap":
                return f"gap verification: {getattr(result, 'status', 'unknown')}"
            return f"solution verification: {getattr(result, 'novelty_status', 'unknown')}"
        if action == "SOLUTION_IDEATION":
            return f"developed solution {getattr(result, 'selected_solution_id', 'unknown')}"
        if action == "SEEK_INSPIRATION":
            return f"inspiration: {getattr(result, 'status', 'unknown')}"
        if action == "TARGETED_REVISION":
            revision = getattr(result, "revision_result", result)
            changed = getattr(revision, "changed_units", [])
            suffix = "; changed " + ",".join(changed) if changed else ""
            return f"revision: {getattr(revision, 'status', 'unknown')}{suffix}"
        if action == "DESIGN_EXPERIMENT":
            if getattr(result, "status", None) == "unable_to_design":
                return (
                    "experiment design unresolved: "
                    + str(getattr(result, "failure_reason", "unresolved identifiability"))
                )
            experiments = getattr(result, "experiments", [])
            return "experiments ready: " + ",".join(experiment.id for experiment in experiments)
        if action == "SWITCH_GAP":
            active = getattr(result, "active_gap", None)
            return f"active gap switched to {active.id if active else 'none'}"
        if action == "COMMIT":
            return "research committed"
        raise RunnerInvariantError(f"cannot summarize unsupported action: {action!r}")

    def _record_budget_update(
        self,
        context: TrajectoryContext,
        action: str,
        scientific_searches: int,
    ) -> None:
        budget = self.budget.snapshot()
        context.trajectory.append_event(
            _event_step(context) - 1,
            "budget_updated",
            action=None,
            result_payload={
                "scientific_action": action,
                "scientific_searches_this_action": scientific_searches,
                "steps_used": budget.steps_used,
                "searches_used": budget.searches_used,
            },
            budget=_budget_metadata(budget),
        )

    def _run_action(
        self,
        context: TrajectoryContext,
        decision: ControllerDecision,
        *,
        allowed_actions: list[str],
        used_llm: bool,
    ) -> tuple[_DispatchOutcome | None, Exception | None, bool, int, int]:
        action = decision.action
        step = _event_step(context)
        context.trajectory.append_event(
            step,
            "controller_decided",
            action=None,
            result_payload={
                "decision": decision.model_dump(mode="json"),
                "allowed_actions": allowed_actions,
                "used_llm": used_llm,
            },
            budget=_budget_metadata(self.budget.snapshot()),
        )
        before_state = context.get_state()
        before_memory = context.get_memory()
        try:
            if decision.action not in {
                "SOLUTION_IDEATION",
                "SEEK_INSPIRATION",
                "VERIFY_RELATED_WORK",
                "TARGETED_REVISION",
                "DESIGN_EXPERIMENT",
                "SWITCH_GAP",
                "COMMIT",
            }:
                raise RunnerInvariantError(
                    f"unsupported Scientific Action: {decision.action!r}"
                )
            # Target resolution is a runner invariant.  Validate it before
            # entering module execution so an internal routing error is not
            # counted as a scientific Action attempt.
            if decision.action == "VERIFY_RELATED_WORK":
                self._resolve_verification_target(context)
        except Exception as exc:
            context.trajectory.append_event(
                _event_step(context),
                "runner_invariant_failed",
                action=None,
                result_payload={
                    "action": decision.action,
                    "error_type": exc.__class__.__name__,
                    "error": _error_text(exc),
                },
                budget=_budget_metadata(self.budget.snapshot()),
                error=_error_text(exc),
            )
            return None, exc, False, step, 0
        before_searches = _search_count(self.modules)
        error: Exception | None = None
        outcome: _DispatchOutcome | None = None
        started = False
        try:
            started = True
            outcome = self._dispatch(decision, context, step)
            outcome = self._post_action_refine(outcome, context)
        except Exception as exc:  # record an attempted action and stop safely
            error = exc
        after_searches = _search_count(self.modules)
        search_delta = after_searches - before_searches
        if search_delta < 0:
            error = error or RunnerInvariantError("module scientific search counter moved backwards")
            search_delta = 0
        try:
            self.budget.record_execution(scientific_searches=search_delta)
        except Exception as exc:
            error = error or exc
        budget = self.budget.snapshot()
        if error is not None:
            after_state = context.get_state()
            result_payload = {
                "action": action,
                "error_type": error.__class__.__name__,
                "error": _error_text(error),
                "attempt_started": started,
            }
            structured_details = _structured_error_details(error)
            if structured_details is not None:
                result_payload["structured_output"] = structured_details
            context.record_transition(
                _event_step(context),
                "runner_action_failed",
                after_state,
                after_state,
                action=action,
                result_payload=result_payload,
                budget=_budget_metadata(budget),
                error=_error_text(error),
            )
            try:
                record_decision_result(
                    step,
                    decision,
                    f"failed: {error.__class__.__name__}: {_error_text(error)}",
                    context.memory_store,
                )
            finally:
                self._record_budget_update(context, action, search_delta)
                self._append_replay(
                    context,
                    seed_paper_id=context.paper_id,
                    kind="action",
                    action=action,
                    step=_event_step(context) - 1,
                    before_memory=before_memory,
                    budget=budget,
                )
            return None, error, started, step, search_delta

        assert outcome is not None
        summary = self._result_summary(action, outcome)
        record_decision_result(step, decision, summary, context.memory_store)
        self._record_budget_update(context, action, search_delta)
        self._append_replay(
            context,
            seed_paper_id=context.paper_id,
            kind="action",
            action=action,
            step=outcome.boundary_step,
            before_memory=before_memory,
            budget=budget,
        )
        return outcome, None, True, step, search_delta

    def _write_artifacts(
        self,
        context: TrajectoryContext,
        result: TrajectoryRunResult,
        *,
        commit_gate_passed: bool,
        commit_snapshot: CommitSnapshot | None,
        final_proposal: FinalResearchProposal | None,
    ) -> None:
        if self.output_dir is None:
            return
        root = self.output_dir
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "seed_context.json", "seed_context", context.seed_context)
        context.persist(root)
        save_replay_journal(self._replay_records, root / "trajectory_replay.jsonl")
        if commit_snapshot is not None:
            from ..modules.commit import save_commit_snapshot

            save_commit_snapshot(commit_snapshot, root / "commit_snapshot.json")
        if final_proposal is not None:
            save_final_research_proposal(final_proposal, root / "final_research_proposal.json")
        if result.status != "committed":
            write_json(root / "unresolved_result.json", "trajectory_run_result", result)
        budget = result.budget
        manifest = RunManifest(
            paper_id=context.paper_id,
            status=result.status,
            budget=RunBudget(
                max_steps=budget.max_steps,
                max_searches=budget.max_searches,
                steps_used=budget.steps_used,
                searches_used=budget.searches_used,
                remaining_steps=budget.remaining_steps,
                remaining_searches=budget.remaining_searches,
            ),
            trajectory_event_count=len(context.trajectory.get_events()),
            replay_record_count=len(self._replay_records),
            commit_gate_passed=commit_gate_passed,
            commit_snapshot_written=commit_snapshot is not None,
            final_proposal_written=final_proposal is not None,
            unresolved_issues=list(result.unresolved_issues),
            error_type=result.error_type,
            error_message=result.error_message,
        )
        write_json(root / "run_manifest.json", "run_manifest", manifest)

    def _finish(
        self,
        context: TrajectoryContext,
        *,
        status: str,
        unresolved_issues: list[str] | None = None,
        commit_snapshot: CommitSnapshot | None = None,
        final_proposal: FinalResearchProposal | None = None,
        error: Exception | None = None,
        commit_gate_passed: bool = False,
    ) -> TrajectoryRunResult:
        result = TrajectoryRunResult(
            status=status,
            unresolved_issues=list(unresolved_issues or []),
            final_state=context.get_state(),
            budget=self.budget.snapshot(),
            commit_snapshot=commit_snapshot,
            final_proposal=final_proposal,
            error_type=error.__class__.__name__ if error else None,
            error_message=_error_text(error) if error else None,
            output_dir=str(self.output_dir) if self.output_dir is not None else None,
        )
        self._write_artifacts(
            context,
            result,
            commit_gate_passed=commit_gate_passed,
            commit_snapshot=commit_snapshot,
            final_proposal=final_proposal,
        )
        return result

    def run(
        self,
        seed_context: SeedPaperContext,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> TrajectoryRunResult:
        """Execute one offline trajectory; all module dependencies are injected."""

        seed = SeedPaperContext.model_validate(seed_context.model_dump(mode="python"))
        self._replay_records = []
        context = TrajectoryContext.initialize(seed)
        try:
            before_memory = context.get_memory()
            bootstrap_step = _event_step(context)
            self.modules.gap_discovery.discover(
                seed,
                full_paper_reading_context=full_paper_reading_context,
                context=context,
                step=bootstrap_step,
            )
            self._append_replay(
                context,
                seed_paper_id=seed.paper_id,
                kind="bootstrap",
                action=None,
                step=bootstrap_step,
                before_memory=before_memory,
                budget=self.budget.snapshot(),
            )
        except Exception as exc:
            return self._finish(context, status="failed", unresolved_issues=[_error_text(exc)], error=exc)

        while True:
            try:
                gate = check_commit_gate(context.get_state(), context.memory_store)
                routing: ControllerRoutingResult = self.modules.controller.decide(
                    context,
                    budget=self.budget,
                    commit_allowed=gate.commit_allowed,
                    unmet_commit_conditions=gate.unmet_conditions,
                )
            except Exception as exc:
                return self._finish(
                    context,
                    status="failed",
                    unresolved_issues=[_error_text(exc)],
                    error=exc,
                )
            if routing.status == "stopped":
                context.trajectory.append_event(
                    _event_step(context),
                    "controller_stopped",
                    action=None,
                    result_payload={"unresolved_issues": routing.unresolved_issues},
                    budget=_budget_metadata(self.budget.snapshot()),
                )
                return self._finish(
                    context,
                    status="stopped",
                    unresolved_issues=routing.unresolved_issues,
                    commit_gate_passed=gate.commit_allowed,
                )
            if routing.decision is None:
                error = RunnerInvariantError("Controller returned decision status without a decision")
                return self._finish(context, status="failed", unresolved_issues=[_error_text(error)], error=error)
            outcome, error, _, _, _ = self._run_action(
                context,
                routing.decision,
                allowed_actions=list(routing.allowed_actions),
                used_llm=routing.used_llm,
            )
            if error is not None:
                return self._finish(
                    context,
                    status="failed",
                    unresolved_issues=[_error_text(error)],
                    error=error,
                    commit_gate_passed=gate.commit_allowed,
                )
            assert outcome is not None
            if routing.decision.action != "COMMIT":
                continue
            snapshot = outcome.result
            if not isinstance(snapshot, CommitSnapshot):
                error = RunnerInvariantError("Commit action returned an invalid snapshot")
                return self._finish(
                    context,
                    status="failed",
                    unresolved_issues=[_error_text(error)],
                    error=error,
                    commit_gate_passed=True,
                )
            try:
                proposal = self.modules.finalizer.finalize(snapshot, seed)
            except Exception as exc:
                return self._finish(
                    context,
                    status="finalization_failed",
                    unresolved_issues=[_error_text(exc)],
                    commit_snapshot=snapshot,
                    error=exc,
                    commit_gate_passed=True,
                )
            return self._finish(
                context,
                status="committed",
                commit_snapshot=snapshot,
                final_proposal=proposal,
                commit_gate_passed=True,
            )


__all__ = ["TrajectoryModules", "TrajectoryRunner"]
