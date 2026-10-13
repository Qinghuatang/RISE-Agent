"""Discovery and fidelity checks for historical E3 checkpoints.

The replay journal is the canonical action-boundary source.  This module only
reconstructs prefixes; it never executes a production scientific module and
never alters the source run.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from research_agent.errors import SerializationError
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models.memory import ResearchMemory, VerificationRecord
from research_agent.models.seed import SeedPaperContext
from research_agent.models.state import ResearchState
from research_agent.models.trajectory import TrajectoryEvent
from research_agent.serialization import read_json
from research_agent.serialization.trajectory import load_replay_journal
from research_agent.trajectory.recorder import TrajectoryRecorder

from .models import (
    E3BudgetSnapshot,
    E3Checkpoint,
    HistoricalLocalBranch,
    RevisionCase,
    ValidationResult,
)


def _payload(path: Path, schema: str) -> Any:
    return read_json(path, expected_schema=schema)["payload"]


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _apply_memory_patch(
    memory_store: ResearchMemoryStore,
    memory_patch: dict[str, object],
) -> None:
    """Apply the same top-level replay semantics without importing runner code."""

    allowed = {
        "paper_memory",
        "verification_memory",
        "inspiration_memory",
        "revision_memory",
        "failure_memory",
        "decision_log",
    }
    unknown = set(memory_patch) - allowed
    if unknown:
        raise SerializationError(
            "replay contains unsupported memory categories: "
            + ", ".join(sorted(unknown))
        )
    payload = memory_store.get_memory().model_dump(mode="python")
    for category, value in memory_patch.items():
        payload[category] = deepcopy(value)
    memory_store.replace_memory(ResearchMemory.model_validate(payload))


def _load_source(source_dir: Path) -> tuple[
    SeedPaperContext,
    list[TrajectoryEvent],
    list[Any],
    ResearchState,
    ResearchMemory,
]:
    seed = SeedPaperContext.model_validate(_payload(source_dir / "seed_context.json", "seed_context"))
    recorder = TrajectoryRecorder.load(source_dir / "trajectory.jsonl")
    events = recorder.get_events()
    replay = load_replay_journal(source_dir / "trajectory_replay.jsonl")
    state = ResearchState.model_validate(
        _payload(source_dir / "research_state.json", "research_state")
    )
    memory = ResearchMemory.model_validate(
        _payload(source_dir / "research_memory.json", "research_memory")
    )
    return seed, events, replay, state, memory


def _event_payload(event: TrajectoryEvent | None) -> dict[str, Any]:
    if event is None or not isinstance(event.result_payload, dict):
        return {}
    return event.result_payload


def _verification_payload(record: VerificationRecord | None) -> dict[str, Any] | None:
    return record.model_dump(mode="json") if record is not None else None


def _checkpoint_memory(
    seed_id: str,
    replay_records: Iterable[Any],
) -> ResearchMemory:
    store = ResearchMemoryStore(seed_id)
    for record in replay_records:
        _apply_memory_patch(store, record.memory_patch)
    return store.get_memory()


def reconstruct_checkpoint(
    source_dir: str | Path,
    revision_event_index: int | None = None,
) -> E3Checkpoint:
    """Reconstruct the state immediately before the first revision action."""

    source = Path(source_dir)
    seed, events, replay, _final_state, _final_memory = _load_source(source)
    revision_indices = [
        index
        for index, event in enumerate(events)
        if event.event_type == "solution_revised"
        and event.action == "TARGETED_REVISION"
    ]
    if not revision_indices:
        raise SerializationError("source trajectory contains no TARGETED_REVISION")
    event_index = revision_event_index if revision_event_index is not None else revision_indices[0]
    if event_index != revision_indices[0]:
        raise SerializationError("E3 checkpoint must use the first TARGETED_REVISION")
    revision_event = events[event_index]
    replay_index = next(
        (
            index
            for index, record in enumerate(replay)
            if record.action == "TARGETED_REVISION"
            and record.step == revision_event.step
        ),
        None,
    )
    if replay_index is None:
        raise SerializationError(
            "first TARGETED_REVISION has no matching replay action boundary"
        )
    if replay_index == 0:
        raise SerializationError("TARGETED_REVISION has no preceding replay checkpoint")
    previous = replay[replay_index - 1]
    if previous.action not in {"VERIFY_RELATED_WORK", "SEEK_INSPIRATION"}:
        raise SerializationError(
            "first revision checkpoint must follow verification or inspiration"
        )
    # The controller decision for the revision is logged immediately before
    # the revision event at the *next* step.  A replay checkpoint is an action
    # boundary, so that decision (and everything at a later step) is future
    # information and must not enter the prefix.
    prefix_events = [
        event for event in events[:event_index] if event.step <= previous.step
    ]
    if any(event.event_type == "solution_revised" for event in prefix_events):
        raise SerializationError("checkpoint prefix contains a future revision")
    if prefix_events and prefix_events[-1].step > previous.step:
        raise SerializationError("trajectory prefix extends past replay checkpoint")

    state = _copy_state(previous.state_after)
    memory = _checkpoint_memory(seed.paper_id, replay[:replay_index])
    solution = state.current_solution
    if state.active_gap is None or solution is None:
        raise SerializationError("revision checkpoint lacks active gap or current solution")

    if solution.novelty_status == "partial_overlap":
        trigger_type = "overlap_repair"
    elif (
        solution.novelty_status == "distinct"
        and previous.action == "SEEK_INSPIRATION"
    ):
        trigger_type = "inspiration_strengthening"
    else:
        raise SerializationError(
            "first revision does not match an E3 trigger classification"
        )

    return E3Checkpoint(
        paper_id=seed.paper_id,
        checkpoint_step=previous.step,
        replay_sequence=previous.sequence,
        state=state,
        memory=memory,
        budget=E3BudgetSnapshot(
            max_steps=previous.budget_after.max_steps,
            max_searches=previous.budget_after.max_searches,
            steps_used=previous.budget_after.steps_used,
            searches_used=previous.budget_after.searches_used,
            remaining_steps=previous.budget_after.remaining_steps,
            remaining_searches=previous.budget_after.remaining_searches,
        ),
        trajectory_prefix=prefix_events,
        preceding_action=previous.action,
        trigger_type=trigger_type,
    )


def _validate_checkpoint_details(source_dir: Path) -> tuple[E3Checkpoint, ValidationResult]:
    seed, events, replay, final_state, final_memory = _load_source(source_dir)
    revision_event_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.event_type == "solution_revised"
            and event.action == "TARGETED_REVISION"
        ),
        None,
    )
    if revision_event_index is None:
        raise SerializationError("source trajectory contains no first revision")
    checkpoint = reconstruct_checkpoint(source_dir, revision_event_index)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    revision_event = events[revision_event_index]
    replay_index = checkpoint.replay_sequence - 1
    preceding = replay[replay_index]
    checks["state_matches_replay_state_after"] = checkpoint.state == preceding.state_after
    checks["memory_restored_to_checkpoint"] = checkpoint.memory == _checkpoint_memory(
        seed.paper_id, replay[: replay_index + 1]
    )
    checks["budget_matches_replay_budget_after"] = (
        checkpoint.budget.max_steps == preceding.budget_after.max_steps
        and checkpoint.budget.max_searches == preceding.budget_after.max_searches
        and checkpoint.budget.steps_used == preceding.budget_after.steps_used
        and checkpoint.budget.searches_used == preceding.budget_after.searches_used
        and checkpoint.budget.remaining_steps == preceding.budget_after.remaining_steps
        and checkpoint.budget.remaining_searches == preceding.budget_after.remaining_searches
    )
    checks["prefix_stops_before_revision"] = (
        bool(not checkpoint.trajectory_prefix or checkpoint.trajectory_prefix[-1].step <= checkpoint.checkpoint_step)
        and not any(
            event.event_type == "solution_revised"
            or event.action == "TARGETED_REVISION"
            for event in checkpoint.trajectory_prefix
        )
    )
    preceding_inspiration_event = next(
        (
            event
            for event in reversed(checkpoint.trajectory_prefix)
            if event.event_type == "inspiration_retrieved"
            and event.action == "SEEK_INSPIRATION"
        ),
        None,
    )
    preceding_inspiration_ref = (
        preceding_inspiration_event.result_ref
        if preceding_inspiration_event is not None
        else None
    )
    checks["checkpoint_has_required_evidence"] = bool(
        checkpoint.state.active_gap
        and checkpoint.state.current_solution
        and (
            checkpoint.trigger_type == "overlap_repair"
            and any(
                record.task == "VERIFY_SOLUTION"
                and record.target_ref == checkpoint.state.current_solution.id
                and getattr(record.result, "novelty_status", None) == "partial_overlap"
                for record in checkpoint.memory.verification_memory
            )
            or checkpoint.trigger_type == "inspiration_strengthening"
            and checkpoint.preceding_action == "SEEK_INSPIRATION"
            and preceding_inspiration_ref is not None
            and any(
                record.id == preceding_inspiration_ref and record.inspirations
                for record in checkpoint.memory.inspiration_memory
            )
        )
    )
    replay_memory_store = ResearchMemoryStore(seed.paper_id)
    for record in replay:
        _apply_memory_patch(replay_memory_store, record.memory_patch)
    checks["replay_reaches_final_state"] = replay[-1].state_after == final_state
    checks["replay_reaches_final_memory"] = replay_memory_store.get_memory() == final_memory
    checks["replay_has_no_backward_steps"] = all(
        left.step <= right.step for left, right in zip(replay, replay[1:])
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return checkpoint, ValidationResult(
        paper_id=seed.paper_id,
        valid=not errors,
        checks=checks,
        errors=errors,
    )


def validate_checkpoint(source_dir: str | Path) -> ValidationResult:
    """Validate one source trajectory without any LLM or network activity."""

    try:
        _checkpoint, result = _validate_checkpoint_details(Path(source_dir))
        return result
    except Exception as exc:
        paper_id = Path(source_dir).name
        try:
            seed, *_ = _load_source(Path(source_dir))
            paper_id = seed.paper_id
        except Exception:
            pass
        return ValidationResult(
            paper_id=paper_id,
            valid=False,
            checks={},
            errors=[f"{type(exc).__name__}: {str(exc)[:500]}"],
        )


def extract_historical_local_branch(
    source_dir: str | Path,
    checkpoint: E3Checkpoint | None = None,
) -> HistoricalLocalBranch:
    """Extract only the first historical local revision and its next verification."""

    source = Path(source_dir)
    seed, events, _replay, _final_state, _final_memory = _load_source(source)
    revision_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.event_type == "solution_revised"
            and event.action == "TARGETED_REVISION"
        ),
        None,
    )
    if revision_index is None:
        raise SerializationError("source trajectory contains no revision event")
    cp = checkpoint or reconstruct_checkpoint(source, revision_index)
    revision_event = events[revision_index]
    revision_payload = _event_payload(revision_event)
    revision_result = revision_payload.get("revision_result")
    if not isinstance(revision_result, dict):
        revision_result = revision_payload.get("result")
    if not isinstance(revision_result, dict):
        raise SerializationError("solution_revised has no revision result object")
    selected_inspiration_ref = revision_payload.get("selected_inspiration_ref")
    if not isinstance(selected_inspiration_ref, str):
        selected_inspiration_ref = revision_result.get("selected_inspiration_ref")
    if not isinstance(selected_inspiration_ref, str):
        selected_inspiration_ref = None
    need = revision_payload.get("revision_need")
    affected = need.get("affected_units", []) if isinstance(need, dict) else []
    if not isinstance(affected, list) or not all(isinstance(item, str) for item in affected):
        raise SerializationError("revision_need.affected_units is invalid")
    changed = revision_result.get("changed_units", [])
    if not isinstance(changed, list) or not all(isinstance(item, str) for item in changed):
        raise SerializationError("revision_result.changed_units is invalid")
    memory = cp.memory
    before_verification = None
    if cp.state.current_solution is not None:
        matching = [
            record
            for record in memory.verification_memory
            if record.task == "VERIFY_SOLUTION"
            and record.target_ref == cp.state.current_solution.id
            and record.created_at_step <= cp.checkpoint_step
        ]
        if matching:
            before_verification = max(
                matching, key=lambda record: record.created_at_step
            )
    inspiration = None
    if cp.trigger_type == "inspiration_strengthening":
        inspiration_events = [
            event
            for event in cp.trajectory_prefix
            if event.event_type == "inspiration_retrieved"
            and event.action == "SEEK_INSPIRATION"
        ]
        if inspiration_events:
            ref = inspiration_events[-1].result_ref
            records = [record for record in memory.inspiration_memory if record.id == ref]
            if records:
                inspiration = records[-1]
    post_event = next(
        (
            event
            for event in events[revision_index + 1 :]
            if event.event_type == "solution_verified"
            and event.result_ref == revision_event.result_ref
        ),
        None,
    )
    focus = revision_payload.get("verification_focus")
    return HistoricalLocalBranch(
        paper_id=seed.paper_id,
        trigger_type=cp.trigger_type,
        checkpoint_step=cp.checkpoint_step,
        solution_before=cp.state.current_solution,
        verification_before=_verification_payload(before_verification),
        inspiration_before=(
            inspiration.model_dump(mode="json") if inspiration is not None else None
        ),
        selected_inspiration_ref=selected_inspiration_ref,
        affected_units=list(affected),
        historical_revision_result=revision_result,
        historical_changed_units=list(changed),
        historical_verification_focus=focus if isinstance(focus, dict) else None,
        historical_post_revision_outcome=(
            _event_payload(post_event) if post_event is not None else None
        ),
    )


def discover_revision_cases(source_root: str | Path) -> list[RevisionCase]:
    """Find all unique source papers whose historical run revised at least once."""

    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"RISE source output directory does not exist: {root}")
    cases: list[RevisionCase] = []
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            seed, events, replay, _state, _memory = _load_source(source_dir)
            revision_index = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event.event_type == "solution_revised"
                    and event.action == "TARGETED_REVISION"
                ),
                None,
            )
            if revision_index is None:
                continue
            event = events[revision_index]
            replay_record = next(
                (
                    record
                    for record in replay
                    if record.action == "TARGETED_REVISION" and record.step == event.step
                ),
                None,
            )
            if replay_record is None:
                raise SerializationError("no matching replay revision action")
            checkpoint = reconstruct_checkpoint(source_dir, revision_index)
            validation = validate_checkpoint(source_dir)
            cases.append(
                RevisionCase(
                    paper_id=seed.paper_id,
                    source_dir=str(source_dir),
                    first_revision_step=event.step,
                    first_revision_replay_sequence=replay_record.sequence,
                    trigger_type=checkpoint.trigger_type,
                    checkpoint=checkpoint if validation.valid else None,
                    valid=validation.valid,
                    validation_errors=list(validation.errors),
                )
            )
        except Exception as exc:
            try:
                seed, events, replay, *_ = _load_source(source_dir)
                revision = next(
                    event
                    for event in events
                    if event.event_type == "solution_revised"
                    and event.action == "TARGETED_REVISION"
                )
                replay_record = next(
                    record
                    for record in replay
                    if record.action == "TARGETED_REVISION" and record.step == revision.step
                )
                cases.append(
                    RevisionCase(
                        paper_id=seed.paper_id,
                        source_dir=str(source_dir),
                        first_revision_step=revision.step,
                        first_revision_replay_sequence=replay_record.sequence,
                        valid=False,
                        validation_errors=[f"{type(exc).__name__}: {str(exc)[:500]}"],
                    )
                )
            except Exception:
                # A directory without a reliable first revision is not an E3
                # case and is intentionally omitted rather than guessed.
                continue
    return cases


__all__ = [
    "discover_revision_cases",
    "extract_historical_local_branch",
    "reconstruct_checkpoint",
    "validate_checkpoint",
]
