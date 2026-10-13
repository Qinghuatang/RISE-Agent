"""Deterministic-first Scientific Controller.

The controller only routes to an existing Scientific Action.  It never calls
or imports a scientific module, writes Research State, or records a pending
decision as if an action had already executed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from ..errors import ControllerRoutingError, ModuleContractError, PreconditionError
from ..llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
)
from ..memory.store import ResearchMemoryStore
from ..models.controller import (
    ControllerContext,
    ControllerDecision,
    ControllerRoutingResult,
)
from ..models.memory import ResearchMemory, VerificationRecord
from ..models.state import ResearchState
from ..models.trajectory import TrajectoryEvent
from ..models.verification import GapVerificationResult, SolutionVerificationResult
from ..trajectory.recorder import TrajectoryRecorder
from .budget import BudgetTracker, ControllerBudget


ACTION_SPACE: tuple[str, ...] = (
    "SOLUTION_IDEATION",
    "SEEK_INSPIRATION",
    "VERIFY_RELATED_WORK",
    "TARGETED_REVISION",
    "DESIGN_EXPERIMENT",
    "SWITCH_GAP",
    "COMMIT",
)

_RESERVED_FOLLOW_UP = {
    "SOLUTION_IDEATION",
    "TARGETED_REVISION",
    "SWITCH_GAP",
}
_RELEVANT_FAILURE_TYPES = {"rejected_solution", "rejected_mechanism", "failed_revision"}

@dataclass(frozen=True)
class ScientificControllerConfig:
    max_tokens: int = 1_000
    recent_failure_count: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (
            isinstance(self.recent_failure_count, bool)
            or not isinstance(self.recent_failure_count, int)
            or self.recent_failure_count < 0
        ):
            raise ValueError("recent_failure_count must be non-negative")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ScientificControllerConfig":
        values = os.environ if environ is None else environ
        raw_tokens = values.get("CONTROLLER_MAX_TOKENS")
        raw_failures = values.get("CONTROLLER_RECENT_FAILURES")
        try:
            return cls(
                max_tokens=int(raw_tokens) if raw_tokens and raw_tokens.strip() else 1_000,
                recent_failure_count=(
                    int(raw_failures)
                    if raw_failures and raw_failures.strip()
                    else 5
                ),
            )
        except ValueError as exc:
            raise ValueError(
                "CONTROLLER_MAX_TOKENS and CONTROLLER_RECENT_FAILURES must be integers"
            ) from exc


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "controller.md"


def _controller_prompt() -> str:
    template = _prompt_path().read_text(encoding="utf-8")
    schema = json.dumps(
        ControllerDecision.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.replace("{{CONTROLLER_DECISION_JSON_SCHEMA}}", schema)


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _copy_record(record: VerificationRecord) -> VerificationRecord:
    return VerificationRecord.model_validate(record.model_dump(mode="python"))


def _latest_verification(
    memory: ResearchMemory,
    target_ref: str,
    task: str,
) -> VerificationRecord | None:
    matches = [
        (index, record)
        for index, record in enumerate(memory.verification_memory)
        if record.target_ref == target_ref and record.task == task
    ]
    if not matches:
        return None
    return _copy_record(max(matches, key=lambda item: (item[1].created_at_step, item[0]))[1])


def _latest_action_event(
    trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
) -> TrajectoryEvent | None:
    if trajectory is None:
        return None
    events = trajectory.get_events() if hasattr(trajectory, "get_events") else list(trajectory)
    for event in reversed(events):
        if event.action is not None:
            return event
    return None


def _result_status(payload: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, None
    # Action semantics live in the action result.  In particular, Inspiration
    # events also carry a provider-level ``grounding_status`` (for example
    # ``arxiv``), which is not a Scientific Action status and must never mask
    # ``result.status == "found"``.
    candidates: list[Mapping[str, Any]] = []
    for key in ("result", "revision_result", "inspiration_result"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    candidates.append(payload)
    status: str | None = None
    summary: str | None = None
    for candidate in candidates:
        if status is None:
            for key in ("status", "novelty_status"):
                if isinstance(candidate.get(key), str):
                    status = candidate[key]
                    break
        if summary is None:
            for key in ("summary", "revision_summary", "selection_reason"):
                if isinstance(candidate.get(key), str):
                    summary = candidate[key]
                    break
    return status, summary


def _compact_event(event: TrajectoryEvent | None) -> tuple[str | None, dict[str, Any] | None]:
    if event is None or event.action is None:
        return None, None
    status, summary = _result_status(event.result_payload)
    result: dict[str, Any] = {
        "event_type": event.event_type,
        "result_ref": event.result_ref,
    }
    if status is not None:
        result["status"] = status
    if summary is not None:
        result["summary"] = summary
    return event.action, result


def _store_memory(memory_or_store: ResearchMemory | ResearchMemoryStore) -> ResearchMemory:
    if isinstance(memory_or_store, ResearchMemoryStore):
        return memory_or_store.get_memory()
    if isinstance(memory_or_store, ResearchMemory):
        return _copy_memory(memory_or_store)
    raise TypeError("memory must be ResearchMemory or ResearchMemoryStore")


def _pool_exists(memory: ResearchMemory) -> bool:
    return memory.paper_memory.related_paper_pool is not None


def _gap_refined_after(
    trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
    gap_id: str,
    verification_step: int,
) -> bool:
    if trajectory is None:
        return False
    events = trajectory.get_events() if hasattr(trajectory, "get_events") else list(trajectory)
    for event in events:
        if event.event_type != "gap_refined" or event.step <= verification_step:
            continue
        payload = event.result_payload
        if isinstance(payload, Mapping) and payload.get("gap_id") == gap_id:
            return True
    return False


def _gap_has_refinement(
    trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
    gap_id: str,
) -> bool:
    if trajectory is None:
        return False
    events = trajectory.get_events() if hasattr(trajectory, "get_events") else list(trajectory)
    for event in events:
        if event.event_type != "gap_refined":
            continue
        payload = event.result_payload
        if isinstance(payload, Mapping) and payload.get("gap_id") == gap_id:
            return True
    return False


def _candidate_gap_exists(
    trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
) -> bool:
    if trajectory is None:
        return False
    if hasattr(trajectory, "get_latest_gap_agenda"):
        agenda = trajectory.get_latest_gap_agenda()
        return bool(agenda and any(gap.status == "candidate" for gap in agenda.gaps))
    events = trajectory.get_events() if hasattr(trajectory, "get_events") else list(trajectory)
    for event in reversed(events):
        if event.event_type != "gap_agenda_updated":
            continue
        try:
            from ..models.gap import GapAgenda

            agenda = GapAgenda.model_validate(event.result_payload or {})
        except Exception:
            return False
        return any(gap.status == "candidate" for gap in agenda.gaps)
    return False


def _untried_solution_exists(state: ResearchState) -> bool:
    return any(candidate.status == "candidate" for candidate in state.candidate_solutions)


def _solution_space_exhausted(
    state: ResearchState,
    memory: ResearchMemory,
) -> bool:
    """Conservative deterministic signal for gap reconsideration.

    A gap is not switched merely because the first solution is weak. The
    signal requires a non-empty candidate pool with no untried candidates and
    at least two failure records tied to that pool/current solution. A single
    failed revision is not enough to justify abandoning the active gap.
    """

    if not state.candidate_solutions or _untried_solution_exists(state):
        return False
    ids = {candidate.id for candidate in state.candidate_solutions}
    failures = [
        record
        for record in memory.failure_memory
        if record.object_ref in ids
        and record.type in {"rejected_solution", "failed_revision", "rejected_mechanism"}
    ]
    return len(failures) >= 2


def _partial_overlap_revision_stagnated(
    trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
    solution_id: str,
) -> bool:
    """Detect repeated effective revisions without novelty improvement.

    This is deliberately an execution-only routing signal.  It does not add
    State or Memory fields: successful ``solution_revised`` events and their
    subsequent ``solution_verified`` events are the existing authoritative
    records.  Only a trailing run of at least two revision/partial-overlap
    pairs is considered stagnation, so one partial overlap still permits a
    normal targeted revision.
    """

    if trajectory is None:
        return False
    events = trajectory.get_events() if hasattr(trajectory, "get_events") else list(trajectory)
    revision_outcomes: list[bool] = []
    for index, event in enumerate(events):
        if event.event_type != "solution_revised" or event.result_ref != solution_id:
            continue
        next_verification = next(
            (
                candidate
                for candidate in events[index + 1 :]
                if candidate.event_type == "solution_verified"
                and candidate.result_ref == solution_id
            ),
            None,
        )
        if next_verification is None:
            continue
        status, _ = _result_status(next_verification.result_payload)
        revision_outcomes.append(status == "partial_overlap")

    trailing_partial = 0
    for remained_partial in reversed(revision_outcomes):
        if not remained_partial:
            break
        trailing_partial += 1
    return trailing_partial >= 2


def _action_order(actions: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for action in actions:
        if action not in ACTION_SPACE:
            raise ModuleContractError(f"unknown Scientific Action: {action!r}")
        if action not in ordered:
            ordered.append(action)
    return ordered


def record_decision_result(
    step: int,
    decision: ControllerDecision,
    result_summary: str,
    memory_store: ResearchMemoryStore,
):
    """Write one authoritative DecisionRecord after execution completed."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("decision step must be a non-negative integer")
    if not isinstance(decision, ControllerDecision):
        decision = ControllerDecision.model_validate(decision)
    if not isinstance(result_summary, str) or not result_summary.strip():
        raise ValueError("result_summary must be non-empty")
    if not isinstance(memory_store, ResearchMemoryStore):
        raise TypeError("memory_store must be ResearchMemoryStore")
    from ..models.memory import DecisionRecord

    record = DecisionRecord(
        step=step,
        action=decision.action,
        reason=decision.reason,
        result_summary=result_summary,
    )
    memory_store.save_decision(record)
    return record


class ScientificController:
    """Choose the next action without executing or mutating any module/state."""

    action_space = ACTION_SPACE

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        budget_tracker: BudgetTracker | None = None,
        budget: BudgetTracker | None = None,
        config: ScientificControllerConfig | None = None,
    ) -> None:
        if budget_tracker is not None and budget is not None:
            raise ValueError("provide budget_tracker or budget, not both")
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker or budget
        self.config = config or ScientificControllerConfig.from_env()
        self.prompt = _controller_prompt()
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_context: ControllerContext | None = None

    @staticmethod
    def _resolve_inputs(
        runtime_context: Any,
        *,
        state: ResearchState | None,
        memory: ResearchMemory | None,
        memory_store: ResearchMemoryStore | None,
        trajectory: Any,
    ) -> tuple[
        ResearchState,
        ResearchMemory | ResearchMemoryStore,
        TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
    ]:
        if runtime_context is not None:
            if state is not None or memory is not None or memory_store is not None:
                raise TypeError("runtime context cannot be combined with state/memory inputs")
            if not hasattr(runtime_context, "get_state") or not hasattr(runtime_context, "get_memory"):
                raise TypeError("runtime context must expose get_state and get_memory")
            state = runtime_context.get_state()
            memory_store = getattr(runtime_context, "memory_store", None)
            memory = None if memory_store is not None else runtime_context.get_memory()
            trajectory = getattr(runtime_context, "trajectory", trajectory)
        if state is None:
            raise TypeError("ScientificController requires ResearchState or runtime context")
        if memory is None and memory_store is None:
            raise TypeError("ScientificController requires ResearchMemory or memory_store")
        if memory is not None and memory_store is not None:
            raise TypeError("provide memory or memory_store, not both")
        return (
            _copy_state(state),
            memory_store if memory_store is not None else memory,  # type: ignore[return-value]
            trajectory,
        )

    def _budget(self, budget: ControllerBudget | BudgetTracker | None) -> ControllerBudget:
        selected = budget or self.budget_tracker
        if selected is None:
            # The controller remains usable as a pure router in small offline
            # fixtures, while production callers should provide explicit limits.
            return ControllerBudget(max_steps=100, max_searches=100)
        if isinstance(selected, BudgetTracker):
            return selected.snapshot()
        if isinstance(selected, ControllerBudget):
            return selected
        raise TypeError("budget must be ControllerBudget or BudgetTracker")

    @staticmethod
    def build_context(
        state: ResearchState,
        memory: ResearchMemory,
        trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
        budget: ControllerBudget,
        *,
        allowed_actions: Sequence[str] = (),
        commit_allowed: bool = False,
        unmet_commit_conditions: Sequence[str] = (),
        recent_failure_count: int = 5,
        solution_verification: VerificationRecord | None = None,
    ) -> ControllerContext:
        active_gap = state.active_gap
        solution = state.current_solution
        event = _latest_action_event(trajectory)
        last_action, last_result = _compact_event(event)
        relevant_failures = [
            record.model_dump(mode="json")
            for record in memory.failure_memory
            if record.type in _RELEVANT_FAILURE_TYPES
        ]
        failures = (
            relevant_failures[-recent_failure_count:]
            if recent_failure_count
            else []
        )
        current_solution_summary = (
            {
                "id": solution.id,
                "status": solution.status,
                "core_idea": solution.core_idea,
                "rationale": solution.rationale,
                "components": [
                    {
                        "id": component.id,
                        "description": component.description,
                        "status": component.status,
                    }
                    for component in solution.components
                ],
                "novelty_status": solution.novelty_status,
            }
            if solution is not None
            else None
        )
        candidate_solutions = [
            {
                "id": candidate.id,
                "core_idea": candidate.core_idea,
                "rationale": candidate.rationale,
                "status": candidate.status,
            }
            for candidate in state.candidate_solutions
        ]
        if current_solution_summary is not None and solution_verification is not None:
            if isinstance(solution_verification.result, SolutionVerificationResult):
                current_solution_summary["verification_summary"] = (
                    solution_verification.result.summary
                )
                current_solution_summary["verification_key_differences"] = list(
                    solution_verification.result.key_differences
                )
                current_solution_summary["verification_distinctive_elements"] = list(
                    solution_verification.result.distinctive_elements
                )
        return ControllerContext(
            active_gap_summary=(
                {
                    "id": active_gap.id,
                    "status": active_gap.status,
                    "confidence": active_gap.confidence,
                    "priority": active_gap.priority,
                    "priority_reason": active_gap.priority_reason,
                    "statement": active_gap.statement,
                    "evidence_refs": list(active_gap.evidence_refs),
                }
                if active_gap is not None
                else None
            ),
            candidate_solutions=candidate_solutions,
            current_solution_summary=current_solution_summary,
            component_status=(
                {component.id: component.status for component in solution.components}
                if solution is not None
                else {}
            ),
            novelty_status=solution.novelty_status if solution is not None else None,
            experiment_status={experiment.id: experiment.status for experiment in state.experiments},
            last_action=last_action,
            last_result=last_result,
            relevant_failure_memory=failures,
            commit_allowed=commit_allowed,
            unmet_commit_conditions=list(unmet_commit_conditions),
            budget={
                "max_steps": budget.max_steps,
                "max_searches": budget.max_searches,
                "steps_used": budget.steps_used,
                "searches_used": budget.searches_used,
                "remaining_steps": budget.remaining_steps,
                "remaining_searches": budget.remaining_searches,
            },
            allowed_actions=list(_action_order(allowed_actions)),
        )

    @staticmethod
    def _validate_consistency(
        state: ResearchState,
        memory: ResearchMemory,
    ) -> None:
        gap = state.active_gap
        if gap is not None and gap.status != "active":
            raise ControllerRoutingError("Research State active_gap must have status=active")
        solution = state.current_solution
        if solution is None:
            return
        if solution.novelty_status == "core_overlap":
            raise ControllerRoutingError(
                "core_overlap cannot remain as an active Current Solution"
            )
        if solution.novelty_status in {"distinct", "partial_overlap"} and any(
            component.status != "ready" for component in solution.components
        ):
            raise ControllerRoutingError(
                f"{solution.novelty_status} Current Solution cannot contain modified or stale Components"
            )
        verification = _latest_verification(memory, solution.id, "VERIFY_SOLUTION")
        if (
            solution.novelty_status != "unchecked"
            and verification is not None
            and isinstance(verification.result, SolutionVerificationResult)
            and verification.result.novelty_status != solution.novelty_status
        ):
            raise ControllerRoutingError(
                "Current Solution novelty_status conflicts with latest verification"
            )

    @staticmethod
    def _can_action(
        action: str,
        budget: ControllerBudget,
        memory: ResearchMemory,
        *,
        reserve_follow_up: bool = False,
    ) -> bool:
        if action == "SEEK_INSPIRATION":
            # Forward reserve for the worst productive path:
            # inspiration found -> targeted revision -> verification.
            required_steps = 3
        else:
            required_steps = 1 + (1 if reserve_follow_up else 0)
        if budget.remaining_steps < required_steps:
            return False
        if action == "SEEK_INSPIRATION" and budget.remaining_searches < 1:
            return False
        if action == "VERIFY_RELATED_WORK" and not _pool_exists(memory):
            return budget.remaining_searches >= 1
        return True

    def _stopped(
        self,
        *,
        budget: ControllerBudget,
        state: ResearchState,
        memory: ResearchMemory,
        trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
        issues: Sequence[str],
        commit_allowed: bool,
        unmet_commit_conditions: Sequence[str],
        solution_verification: VerificationRecord | None = None,
    ) -> ControllerRoutingResult:
        context = self.build_context(
            state,
            memory,
            trajectory,
            budget,
            allowed_actions=(),
            commit_allowed=commit_allowed,
            unmet_commit_conditions=unmet_commit_conditions,
            recent_failure_count=self.config.recent_failure_count,
            solution_verification=solution_verification,
        )
        self.last_context = context
        return ControllerRoutingResult(
            status="stopped",
            decision=None,
            allowed_actions=[],
            unresolved_issues=list(dict.fromkeys(issues)),
            used_llm=False,
        )

    def _decision(
        self,
        *,
        allowed_actions: Sequence[str],
        reason: str,
        state: ResearchState,
        memory: ResearchMemory,
        trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None,
        budget: ControllerBudget,
        commit_allowed: bool,
        unmet_commit_conditions: Sequence[str],
        solution_verification: VerificationRecord | None = None,
    ) -> ControllerRoutingResult:
        allowed = _action_order(allowed_actions)
        context = self.build_context(
            state,
            memory,
            trajectory,
            budget,
            allowed_actions=allowed,
            commit_allowed=commit_allowed,
            unmet_commit_conditions=unmet_commit_conditions,
            recent_failure_count=self.config.recent_failure_count,
            solution_verification=solution_verification,
        )
        self.last_context = context
        if not allowed:
            return self._stopped(
                budget=budget,
                state=state,
                memory=memory,
                trajectory=trajectory,
                issues=unmet_commit_conditions,
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if len(allowed) == 1:
            return ControllerRoutingResult(
                status="decision",
                decision=ControllerDecision(action=allowed[0], reason=reason),
                allowed_actions=allowed,
                unresolved_issues=[],
                used_llm=False,
            )

        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=self.prompt),
                LLMMessage(
                    role="user",
                    content=json.dumps(
                        {"controller_context": context.model_dump(mode="json")},
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "scientific_controller"},
        )
        response = self.llm_client.complete_structured(request, ControllerDecision)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        decision = response.output
        if not isinstance(decision, ControllerDecision):
            raise ModuleContractError("Controller response has the wrong schema")
        if decision.action not in allowed:
            raise ModuleContractError(
                f"Controller selected illegal action {decision.action!r}; allowed={allowed}"
            )
        return ControllerRoutingResult(
            status="decision",
            decision=decision,
            allowed_actions=allowed,
            unresolved_issues=[],
            used_llm=True,
        )

    def decide(
        self,
        runtime_context: Any = None,
        *,
        state: ResearchState | None = None,
        memory: ResearchMemory | None = None,
        memory_store: ResearchMemoryStore | None = None,
        trajectory: TrajectoryRecorder | Sequence[TrajectoryEvent] | None = None,
        budget: ControllerBudget | BudgetTracker | None = None,
        commit_allowed: bool = False,
        unmet_commit_conditions: Sequence[str] = (),
    ) -> ControllerRoutingResult:
        """Route one turn; this method has no State/Memory/trajectory writes."""

        state, memory_source, trajectory = self._resolve_inputs(
            runtime_context,
            state=state,
            memory=memory,
            memory_store=memory_store,
            trajectory=trajectory,
        )
        memory_value = _store_memory(memory_source)
        budget_value = self._budget(budget)
        self._validate_consistency(state, memory_value)

        if budget_value.remaining_steps == 0:
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["step budget exhausted"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if state.active_gap is None:
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["no active gap"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )

        gap = state.active_gap
        gap_verification = _latest_verification(memory_value, gap.id, "VERIFY_GAP")
        if gap_verification is None:
            if self._can_action("VERIFY_RELATED_WORK", budget_value, memory_value):
                return self._decision(
                    allowed_actions=["VERIFY_RELATED_WORK"],
                    reason="The active gap has no matching latest Gap Verification.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            issue = "gap verification missing"
            if not _pool_exists(memory_value) and budget_value.remaining_searches == 0:
                issue = "gap verification missing and search budget exhausted"
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=[issue],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )

        if not isinstance(gap_verification.result, GapVerificationResult):
            raise ControllerRoutingError("latest Gap Verification has an invalid result")
        gap_status = gap_verification.result.status
        if gap_status == "partially_resolved":
            refined_after = _gap_refined_after(
                trajectory, gap.id, gap_verification.created_at_step
            )
            if _gap_has_refinement(trajectory, gap.id) and not refined_after:
                # RefineGap is intentionally bounded to one call.  A second
                # partial verification therefore rejects/deactivates this
                # gap through the existing SWITCH_GAP action; GapDiscovery's
                # deterministic switch leaves active_gap=None when there is
                # no remaining candidate, and the next turn stops cleanly.
                if self._can_action("SWITCH_GAP", budget_value, memory_value):
                    reason = (
                        "The refined gap remains partially resolved after the one allowed "
                        "refinement; reject it and switch to the next candidate gap."
                    )
                    if not _candidate_gap_exists(trajectory):
                        reason = (
                            "The refined gap remains partially resolved after the one allowed "
                            "refinement; reject it and terminate because no candidate gap remains."
                        )
                    return self._decision(
                        allowed_actions=["SWITCH_GAP"],
                        reason=reason,
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                    )
                return self._stopped(
                    budget=budget_value,
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    issues=[
                        "refined gap remained partially resolved and cannot fit the "
                        "deterministic rejection/switch action"
                    ],
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            if not refined_after:
                raise ControllerRoutingError(
                    "Gap Module internal refinement must consume partially_resolved "
                    "before the next Controller turn"
                )
            if self._can_action("VERIFY_RELATED_WORK", budget_value, memory_value):
                return self._decision(
                    allowed_actions=["VERIFY_RELATED_WORK"],
                    reason="The gap was refined after partial resolution and requires verification.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["refined gap verification cannot fit the remaining budget"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if gap_status == "resolved":
            if self._can_action(
                "SWITCH_GAP", budget_value, memory_value, reserve_follow_up=True
            ):
                return self._decision(
                    allowed_actions=["SWITCH_GAP"],
                    reason="The latest Gap Verification resolved the active gap.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["resolved gap switch requires a verification reserve"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if gap_status != "valid":
            raise ControllerRoutingError(f"unsupported Gap Verification status: {gap_status}")

        solution = state.current_solution
        if solution is None:
            if _solution_space_exhausted(state, memory_value):
                reconsideration_actions: list[str] = []
                if self._can_action(
                    "SOLUTION_IDEATION", budget_value, memory_value, reserve_follow_up=True
                ):
                    reconsideration_actions.append("SOLUTION_IDEATION")
                if _candidate_gap_exists(trajectory) and self._can_action(
                    "SWITCH_GAP", budget_value, memory_value, reserve_follow_up=True
                ):
                    reconsideration_actions.append("SWITCH_GAP")
                if reconsideration_actions:
                    return self._decision(
                        allowed_actions=reconsideration_actions,
                        reason=(
                            "The active gap's initial solution pool has been explored after "
                            "repeated failures. Reconsider its value using the gap priority, "
                            "failure evidence, remaining budget, and the next candidate gap; "
                            "continue solution ideation only if further headroom remains."
                        ),
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                    )
            if self._can_action(
                "SOLUTION_IDEATION", budget_value, memory_value, reserve_follow_up=True
            ):
                return self._decision(
                    allowed_actions=["SOLUTION_IDEATION"],
                    reason="The active gap is valid and has no Current Solution.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["current solution missing and Solution Ideation needs a verification reserve"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )

        solution_verification = _latest_verification(
            memory_value, solution.id, "VERIFY_SOLUTION"
        )
        if solution.novelty_status == "unchecked":
            if self._can_action("VERIFY_RELATED_WORK", budget_value, memory_value):
                return self._decision(
                    allowed_actions=["VERIFY_RELATED_WORK"],
                    reason="The Current Solution novelty is unchecked.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["solution verification pending"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if solution_verification is None:
            if self._can_action("VERIFY_RELATED_WORK", budget_value, memory_value):
                return self._decision(
                    allowed_actions=["VERIFY_RELATED_WORK"],
                    reason="The Current Solution has no matching latest Solution Verification.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["solution verification pending"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )
        if not isinstance(solution_verification.result, SolutionVerificationResult):
            raise ControllerRoutingError("latest Solution Verification has an invalid result")

        last_event = _latest_action_event(trajectory)
        last_status, _ = _result_status(last_event.result_payload if last_event else None)
        if (
            last_event is not None
            and last_event.action == "SEEK_INSPIRATION"
            and last_status == "found"
            and solution.novelty_status != "core_overlap"
        ):
            if self._can_action(
                "TARGETED_REVISION", budget_value, memory_value, reserve_follow_up=True
            ):
                return self._decision(
                    allowed_actions=["TARGETED_REVISION"],
                    reason="The latest Inspiration search found a candidate for targeted revision.",
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    budget=budget_value,
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._stopped(
                budget=budget_value,
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                issues=["targeted revision requires a verification reserve"],
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )

        if solution.novelty_status == "partial_overlap":
            if _partial_overlap_revision_stagnated(trajectory, solution.id):
                if _untried_solution_exists(state) and self._can_action(
                    "SOLUTION_IDEATION", budget_value, memory_value, reserve_follow_up=True
                ):
                    return self._decision(
                        allowed_actions=["SOLUTION_IDEATION"],
                        reason=(
                            "The current solution has undergone at least two effective "
                            "revisions whose subsequent verifications remain partial_overlap. "
                            "Treat this revision path as stagnant and develop an untried "
                            "candidate instead of applying another default patch."
                        ),
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                        solution_verification=solution_verification,
                    )
                stagnation_actions: list[str] = []
                if self._can_action("SEEK_INSPIRATION", budget_value, memory_value):
                    stagnation_actions.append("SEEK_INSPIRATION")
                if (
                    not stagnation_actions
                    and _solution_space_exhausted(state, memory_value)
                    and _candidate_gap_exists(trajectory)
                    and self._can_action(
                        "SWITCH_GAP", budget_value, memory_value, reserve_follow_up=True
                    )
                ):
                    stagnation_actions.append("SWITCH_GAP")
                if stagnation_actions:
                    return self._decision(
                        allowed_actions=stagnation_actions,
                        reason=(
                            "The current solution has undergone at least two effective "
                            "revisions whose subsequent verifications remain partial_overlap. "
                            "Treat this revision path as stagnant and seek a different "
                            "transferable mechanism instead of applying another default patch."
                        ),
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                        solution_verification=solution_verification,
                    )
            revision_failed = (
                last_event is not None
                and last_event.action == "TARGETED_REVISION"
                and last_status == "unable_to_revise"
            )
            if revision_failed:
                # Do not immediately repeat a revision that just failed.  If
                # another candidate remains, reuse the existing ideation
                # action. Once the initial pool is exhausted, expose all
                # existing scientifically meaningful routes to the Controller
                # instead of making a failure count synonymous with switching.
                if _solution_space_exhausted(state, memory_value):
                    reconsideration_actions: list[str] = []
                    if self._can_action(
                        "SOLUTION_IDEATION", budget_value, memory_value, reserve_follow_up=True
                    ):
                        reconsideration_actions.append("SOLUTION_IDEATION")
                    if self._can_action("SEEK_INSPIRATION", budget_value, memory_value):
                        reconsideration_actions.append("SEEK_INSPIRATION")
                    if _candidate_gap_exists(trajectory) and self._can_action(
                        "SWITCH_GAP", budget_value, memory_value, reserve_follow_up=True
                    ):
                        reconsideration_actions.append("SWITCH_GAP")
                    if reconsideration_actions:
                        return self._decision(
                            allowed_actions=reconsideration_actions,
                            reason=(
                                "The active gap's initial solution space has been explored after "
                                "repeated failures. Compare further solution ideation, external "
                                "mechanism search, and switching using gap value, failure evidence, "
                                "and remaining budget."
                            ),
                            state=state,
                            memory=memory_value,
                            trajectory=trajectory,
                            budget=budget_value,
                            commit_allowed=commit_allowed,
                            unmet_commit_conditions=unmet_commit_conditions,
                        )
                if _untried_solution_exists(state) and self._can_action(
                    "SOLUTION_IDEATION", budget_value, memory_value, reserve_follow_up=True
                ):
                    return self._decision(
                        allowed_actions=["SOLUTION_IDEATION"],
                        reason=(
                            "The latest targeted revision was unable to produce a defensible "
                            "repair; try an untested candidate solution."
                        ),
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                    )
                if self._can_action("SEEK_INSPIRATION", budget_value, memory_value):
                    return self._decision(
                        allowed_actions=["SEEK_INSPIRATION"],
                        reason=(
                            "The latest targeted revision was unable to produce a defensible "
                            "repair; seek a different transferable mechanism before switching gaps."
                        ),
                        state=state,
                        memory=memory_value,
                        trajectory=trajectory,
                        budget=budget_value,
                        commit_allowed=commit_allowed,
                        unmet_commit_conditions=unmet_commit_conditions,
                    )
                return self._stopped(
                    budget=budget_value,
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    issues=["targeted revision failed and no follow-up action fits the budget"],
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            actions: list[str] = []
            if self._can_action(
                "TARGETED_REVISION", budget_value, memory_value, reserve_follow_up=True
            ):
                actions.append("TARGETED_REVISION")
            if self._can_action("SEEK_INSPIRATION", budget_value, memory_value):
                actions.append("SEEK_INSPIRATION")
            if not actions:
                return self._stopped(
                    budget=budget_value,
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    issues=["partial-overlap revision/search actions are budget-exhausted"],
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._decision(
                allowed_actions=actions,
                reason="Partial novelty overlap permits targeted revision or inspiration.",
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                budget=budget_value,
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
            )

        if solution.novelty_status != "distinct":
            raise ControllerRoutingError(
                f"unsupported Current Solution novelty status: {solution.novelty_status}"
            )
        stale = any(experiment.status == "stale" for experiment in state.experiments)
        experiments_ready = (
            2 <= len(state.experiments) <= 4
            and bool(state.experiments)
            and all(experiment.status == "ready" for experiment in state.experiments)
        )
        if not experiments_ready or stale:
            actions = []
            if self._can_action("DESIGN_EXPERIMENT", budget_value, memory_value):
                actions.append("DESIGN_EXPERIMENT")
            if self._can_action(
                "SEEK_INSPIRATION", budget_value, memory_value
            ):
                actions.append("SEEK_INSPIRATION")
            if not actions:
                return self._stopped(
                    budget=budget_value,
                    state=state,
                    memory=memory_value,
                    trajectory=trajectory,
                    issues=["experiments missing or stale and no action fits budget"],
                    commit_allowed=commit_allowed,
                    unmet_commit_conditions=unmet_commit_conditions,
                )
            return self._decision(
                allowed_actions=actions,
                reason="Distinct solution still needs experiments or may seek inspiration.",
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                budget=budget_value,
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
                solution_verification=solution_verification,
            )

        if commit_allowed:
            # A mature commit-ready state is an exploitation checkpoint.  It
            # should not reopen ordinary proactive exploration without new
            # scientific evidence; the existing Commit Gate is sufficient.
            actions = ["COMMIT"]
            return self._decision(
                allowed_actions=actions,
                reason="The mature state is commit-eligible; further exploration is optional.",
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                budget=budget_value,
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
                solution_verification=solution_verification,
            )
        if self._can_action(
            "SEEK_INSPIRATION", budget_value, memory_value
        ):
            return self._decision(
                allowed_actions=["SEEK_INSPIRATION"],
                reason=(
                    "Experiments are ready, but the external Commit Gate is not satisfied "
                    "and further mechanism exploration is worthwhile."
                ),
                state=state,
                memory=memory_value,
                trajectory=trajectory,
                budget=budget_value,
                commit_allowed=commit_allowed,
                unmet_commit_conditions=unmet_commit_conditions,
                solution_verification=solution_verification,
            )
        return self._stopped(
            budget=budget_value,
            state=state,
            memory=memory_value,
            trajectory=trajectory,
            issues=list(unmet_commit_conditions) or ["commit gate unmet and search budget exhausted"],
            commit_allowed=commit_allowed,
            unmet_commit_conditions=unmet_commit_conditions,
            solution_verification=solution_verification,
        )


__all__ = [
    "ACTION_SPACE",
    "ScientificController",
    "ScientificControllerConfig",
    "record_decision_result",
]
