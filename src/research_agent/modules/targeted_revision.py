"""M7 evidence-grounded Targeted Revision.

The module performs one structured LLM patch call and applies all State
changes deterministically.  It never invokes verification, retrieval,
Inspiration, experiments, or a controller.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ModuleContractError, PreconditionError
from ..llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from ..memory.store import ResearchMemoryStore
from ..models.gap import CandidateGap
from ..models.inspiration import InspirationCandidate
from ..models.memory import (
    FailureRecord,
    InspirationRecord,
    RevisionRecord,
    VerificationRecord,
)
from ..models.revision import (
    ComponentRevisionPatch,
    RevisionNeed,
    RevisionResult,
    TargetedRevisionDraft,
)
from ..models.seed import SeedPaperContext
from ..models.state import Component, CurrentSolution, Experiment, ResearchState
from ..models.verification import SolutionVerificationResult, VerificationFocus
from ..runtime.state_store import ResearchStateStore


@dataclass(frozen=True)
class TargetedRevisionConfig:
    max_tokens: int = 4_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "TargetedRevisionConfig":
        values = os.environ if environ is None else environ
        raw = values.get("TARGETED_REVISION_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError("TARGETED_REVISION_MAX_TOKENS must be an integer") from exc


@dataclass(frozen=True)
class TargetedRevisionExecutionResult:
    revision_need: RevisionNeed
    selected_inspiration_ref: str | None
    revision_result: RevisionResult
    verification_focus: VerificationFocus | None
    next_required_action: str | None


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / name


def _inject_schema(name: str, placeholder: str, schema_type: type[Any]) -> str:
    template = _prompt_path(name).read_text(encoding="utf-8")
    schema = json.dumps(
        schema_type.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return template.replace(placeholder, schema)


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_solution(solution: CurrentSolution) -> CurrentSolution:
    return CurrentSolution.model_validate(solution.model_dump(mode="python"))


def _entity_payload(candidate: InspirationCandidate, runtime_ref: str) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    payload["inspiration_ref"] = runtime_ref
    return payload


def _runtime_inspirations(
    records: list[InspirationRecord],
) -> tuple[list[tuple[str, InspirationCandidate]], list[dict[str, Any]]]:
    pairs: list[tuple[str, InspirationCandidate]] = []
    payload: list[dict[str, Any]] = []
    for record in records:
        for index, candidate in enumerate(record.inspirations, start=1):
            ref = f"{record.id}:{index}"
            pairs.append((ref, candidate))
            payload.append(_entity_payload(candidate, ref))
    return pairs, payload


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _contains_token(text: str, token: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is not None


def _verification_summary(record: VerificationRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "task": record.task,
        "target_ref": record.target_ref,
        "verification_focus": (
            record.verification_focus.model_dump(mode="json")
            if record.verification_focus is not None
            else None
        ),
        "result": record.result.model_dump(mode="json"),
        "paper_refs": record.paper_refs,
    }


def _solution_payload(solution: CurrentSolution) -> dict[str, Any]:
    return solution.model_dump(mode="json")


def _validate_revision_patch(
    current_solution: CurrentSolution,
    result: RevisionResult,
) -> tuple[bool, list[str], bool]:
    """Validate patch semantics and return deterministic change information."""

    if result.status != "revised":
        return False, [], False
    component_ids = [component.id for component in current_solution.components]
    known_components = set(component_ids)
    patches_by_id: dict[str, ComponentRevisionPatch] = {}
    structural_change = False
    for patch in result.revised_components:
        if patch.component_id in patches_by_id:
            raise ModuleContractError(
                f"revision contains duplicate patch for {patch.component_id}"
            )
        if patch.operation == "add" and patch.component_id in known_components:
            raise ModuleContractError(
                f"add patch references existing Component {patch.component_id}"
            )
        if patch.operation != "add" and patch.component_id not in known_components:
            raise ModuleContractError(
                f"revision patch references unknown Component {patch.component_id}"
            )
        if patch.operation in {"add", "remove", "replace"}:
            structural_change = True
        elif patch.description == current_solution.components[component_ids.index(patch.component_id)].description:
            raise ModuleContractError(f"Component patch for {patch.component_id} is a no-op")
        patches_by_id[patch.component_id] = patch

    core_supplied = result.revised_core_idea is not None
    rationale_supplied = result.revised_rationale is not None

    core_changed = False
    canonical_changed_units: list[str] = []
    if core_supplied:
        if not result.revised_core_idea.strip():
            raise ModuleContractError("revised_core_idea must be non-empty")
        if result.revised_core_idea == current_solution.core_idea:
            raise ModuleContractError("revised_core_idea is a no-op")
        core_changed = True
        canonical_changed_units.append("core_idea")
    if rationale_supplied:
        if not result.revised_rationale.strip():
            raise ModuleContractError("revised_rationale must be non-empty")
        if result.revised_rationale == current_solution.rationale:
            raise ModuleContractError("revised_rationale is a no-op")
        canonical_changed_units.append("rationale")

    # A central mechanism change invalidates the assumptions of the complete
    # component graph.  The existing patch contract has no dependency field,
    # so use a conservative deterministic coverage rule: every pre-existing
    # component must be explicitly reconsidered when core_idea changes.  This
    # prevents a component that still describes the replaced mechanism from
    # being silently accepted; the subsequent full-solution verification still
    # remains authoritative for scientific validity.
    if core_changed:
        missing_component_patches = [
            component_id
            for component_id in component_ids
            if component_id not in patches_by_id
        ]
        if missing_component_patches:
            raise ModuleContractError(
                "changing core_idea requires coordinated patches for all existing "
                "Components; missing: " + ", ".join(missing_component_patches)
            )
    canonical_changed_units.extend(
        component_id for component_id in component_ids if component_id in patches_by_id
    )
    canonical_changed_units.extend(
        patch.component_id
        for patch in result.revised_components
        if patch.operation == "add"
    )

    changed_component_ids = [
        unit for unit in canonical_changed_units if unit not in {"core_idea", "rationale"}
    ]
    if not core_changed and not changed_component_ids:
        raise ModuleContractError(
            "a revised Targeted Revision must change core_idea or a Component"
        )
    resulting_count = len(component_ids)
    resulting_count -= sum(
        1 for patch in result.revised_components if patch.operation == "remove"
    )
    resulting_count += sum(
        1 for patch in result.revised_components if patch.operation == "add"
    )
    if not 2 <= resulting_count <= 5:
        raise ModuleContractError("revised Current Solution must contain 2-5 Components")
    return core_changed, changed_component_ids, structural_change


def canonicalize_revision_result(
    current_solution: CurrentSolution,
    result: RevisionResult,
) -> RevisionResult:
    """Set changed_units from the actual lightweight patch, not LLM metadata."""

    core_changed, changed_component_ids, _ = _validate_revision_patch(
        current_solution, result
    )
    if result.status != "revised":
        return result.model_copy(update={"changed_units": []})
    changed: list[str] = []
    if core_changed:
        changed.append("core_idea")
    if result.revised_rationale is not None:
        changed.append("rationale")
    changed.extend(changed_component_ids)
    if not changed:
        raise ModuleContractError("a revised Targeted Revision is a no-op")
    return result.model_copy(update={"changed_units": changed})


def apply_revision(
    current_solution: CurrentSolution,
    revision_result: RevisionResult,
) -> CurrentSolution:
    """Deterministically merge a validated structural patch into a solution."""

    solution = _copy_solution(current_solution)
    canonical_result = canonicalize_revision_result(solution, revision_result)
    core_changed, changed_component_ids, _ = _validate_revision_patch(
        solution, canonical_result
    )
    if canonical_result.status != "revised":
        raise ModuleContractError("apply_revision requires a revised RevisionResult")
    if canonical_result.revised_core_idea is not None:
        solution.core_idea = canonical_result.revised_core_idea
    if canonical_result.revised_rationale is not None:
        solution.rationale = canonical_result.revised_rationale
    patch_by_id = {patch.component_id: patch for patch in canonical_result.revised_components}
    merged_components: list[Component] = []
    for component in solution.components:
        patch = patch_by_id.get(component.id)
        if patch is not None and patch.operation == "remove":
            continue
        if patch is not None and patch.operation in {"modify", "replace"}:
            assert patch.description is not None
            component.description = patch.description
            component.status = "modified"
        elif core_changed:
            component.status = "stale"
        merged_components.append(component)
    for patch in canonical_result.revised_components:
        if patch.operation == "add":
            assert patch.description is not None
            merged_components.append(
                Component(id=patch.component_id, description=patch.description, status="modified")
            )
    solution.components = merged_components
    solution.novelty_status = "unchecked"
    solution.novelty_summary = None
    # Keep the variable visible to make the merge contract explicit.
    _ = changed_component_ids
    return CurrentSolution.model_validate(solution.model_dump(mode="python"))


def _build_revision_need(
    active_gap: CandidateGap,
    solution: CurrentSolution,
    verification: VerificationRecord | None,
    inspiration_records: list[InspirationRecord],
    memory_store: ResearchMemoryStore,
    controller_exploration_rationale: str | None = None,
) -> tuple[RevisionNeed, list[tuple[str, InspirationCandidate]], list[dict[str, Any]], list[str]]:
    affected_units: list[str] = []
    problem_parts: list[str] = []
    evidence_refs: list[str] = []
    if verification is not None:
        if not isinstance(verification.result, SolutionVerificationResult):
            raise PreconditionError("latest solution verification has the wrong result type")
        result = verification.result
        problem_parts.append(result.summary)
        evidence_refs.extend(verification.paper_refs)
        for overlap in result.novelty_relevant_overlaps:
            problem_parts.append(overlap.overlap_summary)
            if overlap.matched_paper_ref:
                evidence_refs.append(overlap.matched_paper_ref)
            if overlap.component_id is not None:
                if overlap.component_id not in {c.id for c in solution.components}:
                    raise ModuleContractError(
                        f"verification overlap references unknown Component {overlap.component_id}"
                    )
                affected_units.append(overlap.component_id)
        evidence_refs.extend(result.related_papers)
    if inspiration_records and not problem_parts:
        latest = inspiration_records[-1]
        problem_parts.append(latest.inspiration_need.current_problem)
    if not problem_parts:
        raise PreconditionError("Targeted Revision has no actionable problem evidence")

    failures = memory_store.get_relevant_failures()
    constraints = [
        "Preserve the Active Gap and do not introduce a new research problem.",
        "Preserve unaffected solution units whenever possible.",
        "Make the minimum sufficient scientific method/mechanism change: include all coordinated changes needed to repair the supplied problem, while preserving unrelated valid content.",
        "Avoid the supplied rejected mechanisms and failed revision directions.",
        "Maintain internal consistency across the complete Current Solution.",
        "Do not design experiments or claim novelty without re-verification.",
    ]
    affected_units = _dedupe_preserving_order(affected_units)
    if affected_units:
        constraints.append(
            "Prioritize these affected units: " + ", ".join(affected_units) + "."
        )
    if inspiration_records:
        constraints.append(
            "Adapt only the selected transferable principle; do not copy an external method wholesale."
        )
        constraints.append(
            "The selected Inspiration must produce an identifiable mechanism-level change, not rhetorical support for a revision that would otherwise be made."
        )
        if controller_exploration_rationale:
            constraints.append(
                "Use the Controller's exploration rationale as the epistemic need being addressed: "
                + controller_exploration_rationale.strip()
            )
    rejected = [record.summary for record in failures if record.type == "rejected_mechanism"]
    if rejected:
        problem_parts.append("Rejected mechanisms to avoid: " + "; ".join(rejected))
    need = RevisionNeed(
        problem_to_fix=" ".join(part.strip() for part in problem_parts if part.strip()),
        affected_units=affected_units,
        constraints=constraints,
    )
    pairs, payload = _runtime_inspirations(inspiration_records)
    return need, pairs, payload, _dedupe_preserving_order(evidence_refs)


def _build_after_state(
    before: ResearchState,
    revision_result: RevisionResult,
) -> tuple[ResearchState, VerificationFocus]:
    if before.current_solution is None:
        raise PreconditionError("Targeted Revision requires a Current Solution")
    canonical_result = canonicalize_revision_result(before.current_solution, revision_result)
    solution = apply_revision(before.current_solution, canonical_result)
    core_changed, changed_component_ids, structural_change = _validate_revision_patch(
        before.current_solution, canonical_result
    )
    after = _copy_state(before)
    after.current_solution = solution
    broad_solution_change = structural_change or len(changed_component_ids) >= 2
    for experiment in after.experiments:
        # A structural edit changes the solution's mechanism topology.  Even
        # when the prose of core_idea is unchanged, an experiment validating
        # core_solution (or any component) may no longer identify the same
        # claim.  Revalidate the complete set for structural edits.  Ordinary
        # local text edits retain focused stale propagation below.
        if core_changed or broad_solution_change:
            experiment.status = "stale"
        elif any(
            _contains_token(experiment.what_it_validates, component_id)
            for component_id in changed_component_ids
        ):
            experiment.status = "stale"
    if core_changed or broad_solution_change:
        focus = VerificationFocus(mode="full_solution")
    else:
        focus = VerificationFocus(
            mode="changed_units", changed_units=[changed_component_ids[0]]
        )
    return ResearchState.model_validate(after.model_dump(mode="python")), focus


class TargetedRevisionModule:
    """Apply one evidence-grounded minimum-sufficient revision."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: TargetedRevisionConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or TargetedRevisionConfig.from_env()
        self._prompt = _inject_schema(
            "targeted_revision.md",
            "{{TARGETED_REVISION_DRAFT_JSON_SCHEMA}}",
            TargetedRevisionDraft,
        )
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None
        self.last_revision_need: RevisionNeed | None = None
        self.last_selected_inspiration_ref: str | None = None
        self.last_revision_record: RevisionRecord | None = None
        self.last_failure_record: FailureRecord | None = None
        self.last_verification_focus: VerificationFocus | None = None
        self.last_next_required_action: str | None = None

    def _complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        paper_id: str,
    ) -> StructuredLLMResponse[Any]:
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "targeted_revision", "paper_id": paper_id},
        )
        response = self.llm_client.complete_structured(request, TargetedRevisionDraft)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        return response

    @staticmethod
    def _resolve_inputs(
        active_gap,
        current_solution,
        seed_context: SeedPaperContext | None,
        context,
        state_store: ResearchStateStore | None,
        memory_store: ResearchMemoryStore | None,
    ) -> tuple[SeedPaperContext, ResearchMemoryStore, ResearchState | None, Any]:
        if context is not None:
            if seed_context is not None and seed_context.paper_id != context.paper_id:
                raise PreconditionError("SeedPaperContext does not match trajectory")
            seed = context.seed_context
            store = context.memory_store
            state = context.get_state()
        else:
            if seed_context is None:
                raise TypeError("Targeted Revision requires SeedPaperContext or context")
            seed = SeedPaperContext.model_validate(
                seed_context.model_dump(mode="python")
            )
            store = memory_store or ResearchMemoryStore(seed.paper_id)
            state = state_store.get_state() if state_store is not None else None
        if store.seed_paper_id != seed.paper_id:
            raise PreconditionError("Research Memory does not match Seed Paper")
        return seed, store, state, state_store

    def _preconditions(
        self,
        active_gap,
        current_solution,
        state: ResearchState | None,
    ) -> tuple[CandidateGap, CurrentSolution]:
        if state is None:
            raise PreconditionError(
                "Targeted Revision requires a TrajectoryContext or ResearchStateStore"
            )
        if state is not None:
            state_gap = state.active_gap
            state_solution = state.current_solution
            if state_gap is None or state_gap.status != "active":
                raise PreconditionError(
                    "Targeted Revision requires an active Research State gap"
                )
            if state_solution is None:
                raise PreconditionError(
                    "Targeted Revision requires a Current Solution in Research State"
                )
            if active_gap is None:
                active_gap = state_gap
            elif active_gap.id != state_gap.id:
                raise PreconditionError(
                    "Targeted Revision active_gap does not match Research State"
                )
            if current_solution is None:
                current_solution = state_solution
            elif current_solution.id != state_solution.id:
                raise PreconditionError(
                    "Targeted Revision current_solution does not match Research State"
                )
        if active_gap is None or current_solution is None:
            raise PreconditionError(
                "Targeted Revision requires active_gap and current_solution"
            )
        if active_gap.status != "active":
            raise PreconditionError("Targeted Revision requires an active gap")
        solution = CurrentSolution.model_validate(
            current_solution.model_dump(mode="python")
        )
        unfinished = [
            component.id
            for component in solution.components
            if component.status in {"modified", "stale"}
        ]
        if unfinished:
            raise PreconditionError(
                "Targeted Revision cannot run before re-verification; unfinished Components: "
                + ", ".join(unfinished)
            )
        return CandidateGap.model_validate(active_gap.model_dump(mode="python")), solution

    def revise(
        self,
        active_gap=None,
        current_solution: CurrentSolution | None = None,
        seed_context: SeedPaperContext | None = None,
        *,
        context=None,
        state_store: ResearchStateStore | None = None,
        memory_store: ResearchMemoryStore | None = None,
        inspiration_record_id: str | None = None,
        controller_exploration_rationale: str | None = None,
        step: int = 1,
    ) -> TargetedRevisionExecutionResult:
        if context is None and hasattr(active_gap, "seed_context"):
            context = active_gap
            active_gap = None
        if isinstance(active_gap, SeedPaperContext) and seed_context is None:
            seed_context = active_gap
            active_gap = None
        seed, store, state, resolved_state_store = self._resolve_inputs(
            active_gap,
            current_solution,
            seed_context,
            context,
            state_store,
            memory_store,
        )
        active_gap, solution = self._preconditions(active_gap, current_solution, state)
        verification = store.get_latest_verification(solution.id, task="VERIFY_SOLUTION")
        partial_verification = None
        if verification is not None:
            if not isinstance(verification.result, SolutionVerificationResult):
                raise PreconditionError("latest solution verification has the wrong result type")
            if verification.result.novelty_status == "partial_overlap":
                partial_verification = verification
        if inspiration_record_id is None:
            inspiration_records: list[InspirationRecord] = []
        else:
            matches = store.get_relevant_inspirations(record_id=inspiration_record_id)
            if len(matches) != 1:
                raise PreconditionError(
                    "Targeted Revision inspiration_record_id must identify exactly one InspirationRecord"
                )
            if not matches[0].inspirations:
                raise PreconditionError(
                    "Targeted Revision inspiration_record_id must identify a useful InspirationRecord"
                )
            inspiration_records = matches
        if partial_verification is None and not inspiration_records:
            raise PreconditionError(
                "Targeted Revision requires partial_overlap evidence or useful Inspiration"
            )
        need, inspiration_pairs, inspiration_payload, evidence_refs = _build_revision_need(
            active_gap,
            solution,
            partial_verification,
            inspiration_records,
            store,
            controller_exploration_rationale,
        )
        failures = [
            record.model_dump(mode="json")
            for record in store.get_relevant_failures()
            if record.type in {"rejected_mechanism", "failed_revision", "rejected_solution"}
        ]
        user_content = json.dumps(
            {
                "active_gap": active_gap.model_dump(mode="json"),
                "seed_method_constraints": list(seed.method_constraints),
                "current_solution": _solution_payload(solution),
                "revision_need": need.model_dump(mode="json"),
                "latest_solution_verification": _verification_summary(
                    verification
                ),
                "available_inspirations": inspiration_payload,
                "inspiration_need": (
                    inspiration_records[0].inspiration_need.model_dump(mode="json")
                    if inspiration_records
                    else None
                ),
                "controller_exploration_rationale": controller_exploration_rationale,
                "relevant_failure_memory": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        if context is not None:
            try:
                context.trajectory.validate_next_step(step)
            except ValueError as exc:
                raise PreconditionError(str(exc)) from exc
        before = _copy_state(state) if state is not None else None
        response = self._complete(
            system_prompt=self._prompt,
            user_content=user_content,
            paper_id=seed.paper_id,
        )
        draft = response.output
        if not isinstance(draft, TargetedRevisionDraft):
            raise ModuleContractError("Targeted Revision response has the wrong schema")
        available_refs = {ref for ref, _ in inspiration_pairs}
        selected_ref = draft.selected_inspiration_ref
        if selected_ref is not None and selected_ref not in available_refs:
            raise ModuleContractError(
                f"Targeted Revision selected unknown Inspiration ref {selected_ref}"
            )
        if draft.result.status == "revised":
            if available_refs and selected_ref is None:
                raise ModuleContractError(
                    "a revised result with available Inspirations must select exactly one"
                )
            if not available_refs and selected_ref is not None:
                raise ModuleContractError(
                    "a revised result without Inspirations must select null"
                )
        if draft.result.status == "unable_to_revise":
            _validate_unable_result(draft.result)
            canonical_result = draft.result.model_copy(update={"changed_units": []})
        else:
            canonical_result = canonicalize_revision_result(solution, draft.result)

        selected_candidate = None
        if selected_ref is not None:
            selected_candidate = next(
                candidate
                for ref, candidate in inspiration_pairs
                if ref == selected_ref
            )
            if selected_candidate.paper_ref is not None:
                evidence_refs = _dedupe_preserving_order(
                    [*evidence_refs, selected_candidate.paper_ref]
                )

        if draft.result.status == "unable_to_revise":
            record = RevisionRecord(
                id=f"R{len(store.get_memory().revision_memory) + 1}",
                revision_reason=need.problem_to_fix,
                affected_units=list(need.affected_units),
                selected_inspiration_ref=selected_ref,
                changed_units=[],
                revision_summary=canonical_result.revision_summary,
                status="unable_to_revise",
                failure_reason=canonical_result.failure_reason,
            )
            failure = FailureRecord(
                type="failed_revision",
                object_ref=solution.id,
                summary=need.problem_to_fix,
                reason=canonical_result.failure_reason or "Targeted Revision was unable to revise.",
                evidence_refs=evidence_refs,
            )
            self._write_failure(
                store,
                record,
                failure,
                context=context,
                before_state=before,
                step=step,
                need=need,
                selected_ref=selected_ref,
                revision_result=canonical_result,
            )
            self.last_revision_need = need
            self.last_selected_inspiration_ref = selected_ref
            self.last_revision_record = record
            self.last_failure_record = failure
            self.last_verification_focus = None
            self.last_next_required_action = None
            return TargetedRevisionExecutionResult(
                revision_need=need,
                selected_inspiration_ref=selected_ref,
                revision_result=canonical_result,
                verification_focus=None,
                next_required_action=None,
            )

        assert before is not None
        after, focus = _build_after_state(before, canonical_result)
        record = RevisionRecord(
            id=f"R{len(store.get_memory().revision_memory) + 1}",
            revision_reason=need.problem_to_fix,
            affected_units=list(need.affected_units),
            selected_inspiration_ref=selected_ref,
            changed_units=list(canonical_result.changed_units),
            revision_summary=canonical_result.revision_summary,
            status="revised",
            failure_reason=None,
        )
        store.save_revision(record)
        if context is not None:
            context.record_transition(
                step,
                "solution_revised",
                before,
                after,
                action="TARGETED_REVISION",
                result_ref=solution.id,
                result_payload={
                    "revision_need": need.model_dump(mode="json"),
                    "selected_inspiration_ref": selected_ref,
                    "revision_result": canonical_result.model_dump(mode="json"),
                    "llm_changed_units": list(draft.result.changed_units),
                    "verification_focus": focus.model_dump(mode="json"),
                },
                memory_writes=["revision_memory"],
            )
        elif resolved_state_store is not None:
            resolved_state_store.replace_state(after)
        self.last_revision_need = need
        self.last_selected_inspiration_ref = selected_ref
        self.last_revision_record = record
        self.last_failure_record = None
        self.last_verification_focus = focus
        self.last_next_required_action = "VERIFY_RELATED_WORK"
        return TargetedRevisionExecutionResult(
            revision_need=need,
            selected_inspiration_ref=selected_ref,
            revision_result=canonical_result,
            verification_focus=focus,
            next_required_action="VERIFY_RELATED_WORK",
        )

    def _write_failure(
        self,
        store: ResearchMemoryStore,
        record: RevisionRecord,
        failure: FailureRecord,
        *,
        context,
        before_state: ResearchState | None,
        step: int,
        need: RevisionNeed,
        selected_ref: str | None,
        revision_result: RevisionResult,
    ) -> None:
        store.save_revision(record)
        store.save_failure(failure)
        if context is not None:
            assert before_state is not None
            context.record_transition(
                step,
                "revision_failed",
                before_state,
                before_state,
                action="TARGETED_REVISION",
                result_ref=record.id,
                result_payload={
                    "revision_need": need.model_dump(mode="json"),
                    "selected_inspiration_ref": selected_ref,
                    "revision_result": revision_result.model_dump(mode="json"),
                },
                memory_writes=["revision_memory", "failure_memory"],
            )

    run = revise
    execute = revise


def _validate_unable_result(result: RevisionResult) -> None:
    if result.status != "unable_to_revise":
        raise ModuleContractError("expected unable_to_revise result")
    if result.revised_core_idea is not None or result.revised_rationale is not None:
        raise ModuleContractError("unable_to_revise cannot contain a core/rationale patch")
    if result.revised_components:
        raise ModuleContractError("unable_to_revise cannot contain a patch")
    if result.failure_reason is None or not result.failure_reason.strip():
        raise ModuleContractError("unable_to_revise requires a failure reason")


__all__ = [
    "TargetedRevisionConfig",
    "TargetedRevisionExecutionResult",
    "TargetedRevisionModule",
    "apply_revision",
    "canonicalize_revision_result",
]
