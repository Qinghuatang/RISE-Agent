"""Verified-gap Solution Ideation for M4.

The module has exactly two possible LLM operations: FormSolution and
Select+Develop.  State changes happen only after their Pydantic drafts have
validated.  Related-work verification, Inspiration, and Experiment Design are
not implemented here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import ModuleContractError, PreconditionError
from ..llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from ..memory.store import ResearchMemoryStore
from ..models.memory import FailureRecord, ResearchMemory, VerificationRecord
from ..models.seed import SeedPaperContext
from ..models.solution import (
    CandidateSolutionDraft,
    FormSolutionsDraft,
    SolutionDevelopmentDraft,
    SolutionIdeationResult,
)
from ..models.state import CandidateSolution, Component, CurrentSolution, ResearchState
from ..models.verification import GapVerificationResult
from ..runtime.state_store import ResearchStateStore


@dataclass(frozen=True)
class SolutionIdeationConfig:
    max_tokens: int = 4_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SolutionIdeationConfig":
        values = os.environ if environ is None else environ
        raw = values.get("SOLUTION_IDEATION_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError("SOLUTION_IDEATION_MAX_TOKENS must be an integer") from exc


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / name


def _prompt_with_schema(name: str, placeholder: str, schema_type: type[Any]) -> str:
    template = _prompt_path(name).read_text(encoding="utf-8")
    schema = json.dumps(
        schema_type.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return template.replace(placeholder, schema)


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_candidate(candidate: CandidateSolution) -> CandidateSolution:
    return CandidateSolution.model_validate(candidate.model_dump(mode="python"))


def _failure_records(
    failure_memory: ResearchMemory | ResearchMemoryStore | Iterable[FailureRecord] | None,
) -> list[FailureRecord]:
    if failure_memory is None:
        return []
    if isinstance(failure_memory, ResearchMemoryStore):
        records = failure_memory.get_relevant_failures(failure_type="rejected_solution")
        records.extend(
            failure_memory.get_relevant_failures(failure_type="rejected_mechanism")
        )
        return records
    if isinstance(failure_memory, ResearchMemory):
        values = failure_memory.failure_memory
    else:
        values = list(failure_memory)
    return [
        FailureRecord.model_validate(record.model_dump(mode="python"))
        for record in values
        if record.type in {"rejected_solution", "rejected_mechanism"}
    ]


def _failure_payload(
    failure_memory: ResearchMemory | ResearchMemoryStore | Iterable[FailureRecord] | None,
) -> list[dict[str, Any]]:
    return [record.model_dump(mode="json") for record in _failure_records(failure_memory)]


def _next_solution_number(candidates: Iterable[CandidateSolution]) -> int:
    numbers = []
    for candidate in candidates:
        match = re.fullmatch(r"S(\d+)", candidate.id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


class SolutionIdeationModule:
    """Form, select, and develop solutions only after a valid Gap Verification."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: SolutionIdeationConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SolutionIdeationConfig.from_env()
        self._form_prompt = _prompt_with_schema(
            "solution_form.md",
            "{{FORM_SOLUTIONS_DRAFT_JSON_SCHEMA}}",
            FormSolutionsDraft,
        )
        self._develop_prompt = _prompt_with_schema(
            "solution_select_develop.md",
            "{{SOLUTION_DEVELOPMENT_DRAFT_JSON_SCHEMA}}",
            SolutionDevelopmentDraft,
        )
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None

    def _complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        task: str,
        paper_id: str,
        output_schema: type[Any],
    ) -> StructuredLLMResponse[Any]:
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": task, "paper_id": paper_id},
        )
        response = self.llm_client.complete_structured(request, output_schema)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        return response

    @staticmethod
    def _verified_gap(
        active_gap_id: str,
        memory_store: ResearchMemoryStore,
    ) -> VerificationRecord:
        record = memory_store.get_latest_gap_verification(active_gap_id)
        if record is None:
            raise PreconditionError(
                f"Solution Ideation requires a Gap Verification for {active_gap_id}"
            )
        if not isinstance(record.result, GapVerificationResult):
            raise PreconditionError("latest gap verification has the wrong result type")
        if record.result.status != "valid":
            raise PreconditionError(
                "Solution Ideation requires the latest Gap Verification status to be valid"
            )
        return record

    def form_solutions(
        self,
        active_gap,
        seed_paper_context: SeedPaperContext,
        gap_verification: GapVerificationResult,
        failure_memory: ResearchMemory | ResearchMemoryStore | Iterable[FailureRecord] | None = None,
        *,
        starting_id: int = 1,
    ) -> list[CandidateSolution]:
        """Generate exactly three code-numbered candidate solutions."""

        if starting_id <= 0:
            raise ValueError("starting_id must be positive")
        failures = _failure_payload(failure_memory)
        user_content = (
            "ACTIVE GAP\n"
            + json.dumps(active_gap.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nSEED PAPER CONTEXT\n"
            + json.dumps(
                seed_paper_context.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nVALID GAP VERIFICATION SUMMARY\n"
            + json.dumps(gap_verification.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nRELEVANT FAILURE MEMORY (rejected_solution/rejected_mechanism only)\n"
            + json.dumps(failures, ensure_ascii=False, indent=2)
            + "\n\nReturn exactly three candidates in one JSON object."
        )
        response = self._complete(
            system_prompt=self._form_prompt,
            user_content=user_content,
            task="solution_form",
            paper_id=seed_paper_context.paper_id,
            output_schema=FormSolutionsDraft,
        )
        draft = response.output
        if not isinstance(draft, FormSolutionsDraft):
            raise ModuleContractError("FormSolution response has the wrong schema")
        return [
            CandidateSolution(
                id=f"S{starting_id + index}",
                core_idea=item.core_idea,
                rationale=item.rationale,
                status="candidate",
            )
            for index, item in enumerate(draft.candidates)
        ]

    def select_and_develop(
        self,
        active_gap,
        seed_paper_context: SeedPaperContext,
        candidate_solutions: Iterable[CandidateSolution],
        failure_memory: ResearchMemory | ResearchMemoryStore | Iterable[FailureRecord] | None = None,
    ) -> tuple[CandidateSolution, CurrentSolution, str]:
        """Select one available candidate and materialize its Current Solution."""

        candidates = [_copy_candidate(candidate) for candidate in candidate_solutions]
        available = [candidate for candidate in candidates if candidate.status == "candidate"]
        if not available:
            raise PreconditionError("Select + Develop requires at least one candidate solution")
        user_content = (
            "ACTIVE GAP\n"
            + json.dumps(active_gap.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nSEED PAPER CONTEXT\n"
            + json.dumps(
                seed_paper_context.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nAVAILABLE CANDIDATE SOLUTIONS\n"
            + json.dumps(
                [candidate.model_dump(mode="json") for candidate in available],
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nRELEVANT FAILURE MEMORY (rejected_solution/rejected_mechanism only)\n"
            + json.dumps(_failure_payload(failure_memory), ensure_ascii=False, indent=2)
            + "\n\nReturn one JSON object selecting and developing one available candidate."
        )
        response = self._complete(
            system_prompt=self._develop_prompt,
            user_content=user_content,
            task="solution_select_develop",
            paper_id=seed_paper_context.paper_id,
            output_schema=SolutionDevelopmentDraft,
        )
        draft = response.output
        if not isinstance(draft, SolutionDevelopmentDraft):
            raise ModuleContractError("Select + Develop response has the wrong schema")
        selected = next(
            (candidate for candidate in available if candidate.id == draft.selected_solution_id),
            None,
        )
        if selected is None:
            raise ModuleContractError(
                "Select + Develop selected an ID that is not an available candidate"
            )
        current_solution = CurrentSolution(
            id=selected.id,
            core_idea=draft.core_idea,
            rationale=draft.rationale,
            components=[
                Component(id=f"C{index}", description=description, status="ready")
                for index, description in enumerate(draft.components, start=1)
            ],
            status="developed",
            novelty_status="unchecked",
            novelty_summary=None,
            evidence_refs=[],
        )
        return selected, current_solution, draft.selection_reason

    def ideate(
        self,
        seed_or_context,
        state_store: ResearchStateStore | None = None,
        memory_store: ResearchMemoryStore | None = None,
        *,
        context=None,
        step: int = 1,
    ) -> SolutionIdeationResult:
        """Run Solution Ideation using a trajectory context or explicit stores.

        Passing a ``TrajectoryContext`` as the first argument is the preferred
        integration path.  Explicit ``state_store`` and ``memory_store`` are
        supported for deterministic module tests; both paths enforce the same
        latest-valid-gap-verification precondition.
        """

        if context is None and hasattr(seed_or_context, "seed_context"):
            context = seed_or_context
        if context is not None:
            seed = context.seed_context
            state_store = context.state_store
            memory_store = context.memory_store
        else:
            seed = seed_or_context
        if not isinstance(seed, SeedPaperContext):
            raise TypeError("ideate requires SeedPaperContext or TrajectoryContext")
        if state_store is None or memory_store is None:
            raise TypeError("ideate requires state_store and memory_store")

        state = _copy_state(state_store.get_state())
        active_gap = state.active_gap
        if active_gap is None or active_gap.status != "active":
            raise PreconditionError("Solution Ideation requires an active gap")
        verification_record = self._verified_gap(active_gap.id, memory_store)
        replacing_failed_revision = False
        previous_solution_id: str | None = None
        if state.current_solution is not None:
            # A failed Targeted Revision leaves the active solution available
            # for inspection, but the next untried Candidate Solution must be
            # allowed to take its place.  This is the existing
            # SOLUTION_IDEATION action, not a new replanning action.
            revisions = memory_store.get_revision_history()
            if not revisions or revisions[-1].status != "unable_to_revise":
                raise PreconditionError(
                    "Solution Ideation cannot replace an existing Current Solution"
                )
            replacing_failed_revision = True
            previous_solution_id = state.current_solution.id

        failures = _failure_records(memory_store)
        candidate_pool = [_copy_candidate(candidate) for candidate in state.candidate_solutions]
        should_form = not candidate_pool or all(
            candidate.status == "rejected" for candidate in candidate_pool
        )
        generated = False
        if should_form:
            candidates = self.form_solutions(
                active_gap,
                seed,
                verification_record.result,
                failures,
                starting_id=_next_solution_number(candidate_pool),
            )
            generated = True
            candidate_state = _copy_state(state)
            candidate_state.candidate_solutions = candidates
            if context is not None:
                context.record_transition(
                    step,
                    "solution_candidates_generated",
                    state,
                    candidate_state,
                    action="SOLUTION_IDEATION",
                    result_ref="candidate_solutions",
                    result_payload={
                        "candidate_solutions": [
                            candidate.model_dump(mode="json") for candidate in candidates
                        ]
                    },
                )
            else:
                state_store.replace_state(candidate_state)
            state = candidate_state
            candidate_pool = [_copy_candidate(candidate) for candidate in candidates]
        elif not any(candidate.status == "candidate" for candidate in candidate_pool):
            raise PreconditionError("no candidate solution is available for Select + Develop")

        selected, current_solution, selection_reason = self.select_and_develop(
            active_gap,
            seed,
            candidate_pool,
            failures,
        )
        final_state = _copy_state(state)
        final_candidates: list[CandidateSolution] = []
        for candidate in candidate_pool:
            updated = _copy_candidate(candidate)
            if replacing_failed_revision and updated.id == previous_solution_id:
                updated.status = "rejected"
            if updated.id == selected.id:
                updated.status = "selected"
            final_candidates.append(updated)
        final_state.candidate_solutions = final_candidates
        final_state.current_solution = current_solution
        final_state.experiments = []
        if context is not None:
            context.record_transition(
                step,
                "solution_developed",
                state,
                final_state,
                action="SOLUTION_IDEATION",
                result_ref=current_solution.id,
                result_payload={
                    "selected_solution_id": selected.id,
                    "selection_reason": selection_reason,
                    "current_solution": current_solution.model_dump(mode="json"),
                },
            )
        else:
            state_store.replace_state(final_state)
        return SolutionIdeationResult(
            generated_new_candidates=generated,
            selected_solution_id=selected.id,
            selection_reason=selection_reason,
            current_solution=current_solution,
            next_required_action="VERIFY_RELATED_WORK",
        )

    run = ideate
    execute = ideate


__all__ = ["SolutionIdeationConfig", "SolutionIdeationModule"]
