"""Deterministic Commit Gate, isolated CommitSnapshot, and COMMIT action."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from ..errors import PreconditionError, SerializationError
from ..memory.store import ResearchMemoryStore
from ..models.commit import CommitCheck, CommitSnapshot
from ..models.memory import ResearchMemory, VerificationRecord
from ..models.state import ResearchState
from ..models.verification import GapVerificationResult, SolutionVerificationResult
from ..runtime.state_store import ResearchStateStore
from ..serialization import read_json, write_json


COMMIT_SNAPSHOT_SCHEMA = "commit_snapshot"


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _memory_value(memory: ResearchMemory | ResearchMemoryStore) -> ResearchMemory:
    if isinstance(memory, ResearchMemoryStore):
        return memory.get_memory()
    if isinstance(memory, ResearchMemory):
        return _copy_memory(memory)
    raise TypeError("memory must be ResearchMemory or ResearchMemoryStore")


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
    record = max(matches, key=lambda item: (item[1].created_at_step, item[0]))[1]
    return VerificationRecord.model_validate(record.model_dump(mode="python"))


def _latest_gap_verification(
    memory: ResearchMemory | ResearchMemoryStore,
    gap_id: str,
) -> VerificationRecord | None:
    if isinstance(memory, ResearchMemoryStore):
        return memory.get_latest_gap_verification(gap_id)
    return _latest_verification(_memory_value(memory), gap_id, "VERIFY_GAP")


def _latest_solution_verification(
    memory: ResearchMemory | ResearchMemoryStore,
    solution_id: str,
) -> VerificationRecord | None:
    if isinstance(memory, ResearchMemoryStore):
        return memory.get_latest_verification(solution_id, task="VERIFY_SOLUTION")
    return _latest_verification(_memory_value(memory), solution_id, "VERIFY_SOLUTION")


def _append_unique(values: list[str], candidates: Sequence[str]) -> None:
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        if value not in values:
            values.append(value)


def _check_consistency(
    state: ResearchState,
    solution_verification: VerificationRecord | None,
) -> None:
    solution = state.current_solution
    if solution is None or solution_verification is None:
        return
    result = solution_verification.result
    if not isinstance(result, SolutionVerificationResult):
        raise PreconditionError("latest Solution Verification has an invalid result")
    if solution.novelty_status != "unchecked" and result.novelty_status != solution.novelty_status:
        raise PreconditionError(
            "Research State novelty_status conflicts with the latest Solution Verification"
        )


def check_commit_gate(
    state: ResearchState,
    memory: ResearchMemory | ResearchMemoryStore,
) -> CommitCheck:
    """Apply the five deterministic Commit Gate conditions without side effects."""

    state_copy = _copy_state(state)
    memory_copy = _memory_value(memory)
    unmet: list[str] = []

    gap = state_copy.active_gap
    if gap is None or gap.status != "active":
        unmet.append("Active Gap must exist and have status=active.")

    gap_verification: VerificationRecord | None = None
    if gap is not None:
        gap_verification = _latest_gap_verification(memory, gap.id)
    if gap_verification is None:
        unmet.append("A matching latest Gap Verification is required.")
    elif not isinstance(gap_verification.result, GapVerificationResult):
        raise PreconditionError("latest Gap Verification has an invalid result")
    elif gap_verification.result.status != "valid":
        unmet.append("The latest Gap Verification must have status=valid.")

    solution = state_copy.current_solution
    if solution is None:
        unmet.append("Current Solution is required.")
        solution_verification = None
    else:
        if (
            solution.status != "developed"
            or not solution.core_idea.strip()
            or not solution.rationale.strip()
            or not 2 <= len(solution.components) <= 5
        ):
            unmet.append("Current Solution must be a complete developed solution.")
        solution_verification = _latest_solution_verification(memory, solution.id)
        _check_consistency(state_copy, solution_verification)

        if solution.novelty_status != "distinct":
            unmet.append("Current Solution novelty_status must be distinct.")
        if solution_verification is None:
            unmet.append("A matching latest Solution Verification is required.")
        elif not isinstance(solution_verification.result, SolutionVerificationResult):
            raise PreconditionError("latest Solution Verification has an invalid result")
        elif solution_verification.result.novelty_status != "distinct":
            unmet.append("The latest Solution Verification must report distinct.")

        pending_components = [
            component.id
            for component in solution.components
            if component.status != "ready"
        ]
        if pending_components:
            unmet.append(
                "All Current Solution Components must be ready: "
                + ", ".join(pending_components)
                + "."
            )

    experiments = state_copy.experiments
    if not 2 <= len(experiments) <= 4:
        unmet.append("Research State must contain 2-4 Experiments.")
    elif any(experiment.status != "ready" for experiment in experiments):
        unmet.append("All Experiments must have status=ready.")

    return CommitCheck(
        commit_allowed=not unmet,
        unmet_conditions=unmet,
    )


def _snapshot_evidence_refs(
    state: ResearchState,
    gap_verification: VerificationRecord,
    solution_verification: VerificationRecord,
) -> list[str]:
    refs: list[str] = []
    assert state.active_gap is not None
    assert state.current_solution is not None
    _append_unique(refs, state.active_gap.evidence_refs)
    _append_unique(refs, state.current_solution.evidence_refs)
    for component in state.current_solution.components:
        _append_unique(refs, component.evidence_refs)
    gap_result = gap_verification.result
    solution_result = solution_verification.result
    if isinstance(gap_result, GapVerificationResult):
        _append_unique(refs, gap_result.related_papers)
    if isinstance(solution_result, SolutionVerificationResult):
        _append_unique(refs, solution_result.related_papers)
        for overlap in solution_result.novelty_relevant_overlaps:
            _append_unique(refs, [overlap.matched_paper_ref])
    return refs


def create_commit_snapshot(
    state: ResearchState,
    memory: ResearchMemory | ResearchMemoryStore,
) -> CommitSnapshot:
    """Re-run the Gate and create an independent snapshot only when allowed."""

    state_copy = _copy_state(state)
    check = check_commit_gate(state_copy, memory)
    if not check.commit_allowed:
        raise PreconditionError(
            "Commit Gate is not satisfied: " + "; ".join(check.unmet_conditions)
        )
    assert state_copy.active_gap is not None
    assert state_copy.current_solution is not None
    gap_record = _latest_gap_verification(memory, state_copy.active_gap.id)
    solution_record = _latest_solution_verification(memory, state_copy.current_solution.id)
    if gap_record is None or solution_record is None:
        # Defensive re-check in case a custom memory implementation changed
        # between the Gate call and materialization.
        raise PreconditionError("Commit Snapshot requires both latest Verification records")
    if not isinstance(gap_record.result, GapVerificationResult):
        raise PreconditionError("Commit Snapshot has an invalid Gap Verification")
    if not isinstance(solution_record.result, SolutionVerificationResult):
        raise PreconditionError("Commit Snapshot has an invalid Solution Verification")
    return CommitSnapshot(
        active_gap=state_copy.active_gap,
        current_solution=state_copy.current_solution,
        experiments=[
            experiment.model_copy(deep=True) for experiment in state_copy.experiments
        ],
        latest_gap_verification=GapVerificationResult.model_validate(
            gap_record.result.model_dump(mode="python")
        ),
        latest_solution_verification=SolutionVerificationResult.model_validate(
            solution_record.result.model_dump(mode="python")
        ),
        relevant_evidence_refs=_snapshot_evidence_refs(
            state_copy, gap_record, solution_record
        ),
    )


def save_commit_snapshot(snapshot: CommitSnapshot, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, COMMIT_SNAPSHOT_SCHEMA, snapshot)
    return target


def load_commit_snapshot(path: str | Path) -> CommitSnapshot:
    try:
        envelope = read_json(path, expected_schema=COMMIT_SNAPSHOT_SCHEMA)
        return CommitSnapshot.model_validate(envelope["payload"])
    except (SerializationError, ValidationError) as exc:
        raise SerializationError("stored CommitSnapshot is invalid") from exc


class CommitModule:
    """The COMMIT action freezes a snapshot but does not mutate Research State."""

    def check_commit_gate(
        self,
        state: ResearchState,
        memory: ResearchMemory | ResearchMemoryStore,
    ) -> CommitCheck:
        return check_commit_gate(state, memory)

    def create_commit_snapshot(
        self,
        state: ResearchState,
        memory: ResearchMemory | ResearchMemoryStore,
    ) -> CommitSnapshot:
        return create_commit_snapshot(state, memory)

    def commit(
        self,
        state: ResearchState | None = None,
        memory: ResearchMemory | ResearchMemoryStore | None = None,
        *,
        context=None,
        step: int = 1,
    ) -> CommitSnapshot:
        if context is not None:
            if state is not None or memory is not None:
                raise TypeError("context cannot be combined with state or memory")
            state = context.get_state()
            memory = context.memory_store
            context.trajectory.validate_next_step(step)
        if state is None or memory is None:
            raise TypeError("Commit requires state and memory or a TrajectoryContext")
        snapshot = create_commit_snapshot(state, memory)
        if context is not None:
            before = context.get_state()
            context.record_transition(
                step,
                "research_committed",
                before,
                before,
                action="COMMIT",
                result_ref="commit_snapshot",
                result_payload={
                    "active_gap_id": snapshot.active_gap.id,
                    "current_solution_id": snapshot.current_solution.id,
                    "experiment_ids": [experiment.id for experiment in snapshot.experiments],
                    "relevant_evidence_refs": list(snapshot.relevant_evidence_refs),
                },
                memory_writes=[],
            )
        return snapshot


__all__ = [
    "COMMIT_SNAPSHOT_SCHEMA",
    "CommitModule",
    "check_commit_gate",
    "create_commit_snapshot",
    "load_commit_snapshot",
    "save_commit_snapshot",
]
