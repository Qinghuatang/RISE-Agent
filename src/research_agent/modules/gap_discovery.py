"""Seed-paper-grounded Gap Discovery and Gap management.

This module owns only the M4 gap operations.  The initial generation is one
structured LLM call; selection, refinement materialization, and switching are
deterministic.  Related-work verification is deliberately an input contract,
not an implementation in this module.
"""

from __future__ import annotations

import json
import os
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
from ..ingestion.markdown import (
    FullPaperReadingContext,
    build_full_paper_reading_context,
    parse_mineru_markdown,
)
from ..models.gap import (
    CandidateGap,
    GapAgenda,
    GapEvaluationDraft,
    GapGenerationDraft,
    GapDiscoveryResult,
    GapRefinementDraft,
)
from ..models.seed import SeedPaperContext
from ..models.state import ResearchState
from ..models.verification import GapVerificationResult


@dataclass(frozen=True)
class GapDiscoveryConfig:
    """Only output-budget configuration; it never limits input content."""

    max_tokens: int = 8_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GapDiscoveryConfig":
        values = os.environ if environ is None else environ
        raw = values.get("GAP_DISCOVERY_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError("GAP_DISCOVERY_MAX_TOKENS must be an integer") from exc


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / name


def _inject_schema(template_name: str, placeholder: str, schema_type: type[Any]) -> str:
    template = _prompt_path(template_name).read_text(encoding="utf-8")
    schema = json.dumps(
        schema_type.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return template.replace(placeholder, schema)


def _nonempty(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_full_paper_reading(
    seed: SeedPaperContext,
    reading: FullPaperReadingContext | str | None,
) -> FullPaperReadingContext:
    """Resolve the actual main-paper input and its legal section IDs.

    The production runner supplies the deterministic parser result.  A raw
    string is accepted as a small integration convenience, but is parsed here
    so evidence IDs still come from the supplied paper rather than from
    SeedPaperContext fields.
    """

    if isinstance(reading, FullPaperReadingContext):
        resolved = reading
    elif isinstance(reading, str) and reading.strip():
        resolved = build_full_paper_reading_context(
            parse_mineru_markdown(reading, paper_id=seed.paper_id)
        )
    else:
        raise PreconditionError(
            "Gap Discovery requires FullPaperReadingContext or full paper Markdown"
        )
    if resolved.paper_id != seed.paper_id:
        raise ModuleContractError(
            "FullPaperReadingContext does not match the SeedPaperContext"
        )
    if not resolved.body_section_ids:
        raise ModuleContractError(
            "FullPaperReadingContext contains no main-paper source sections"
        )
    return resolved


def _full_paper_content(
    seed: SeedPaperContext,
    reading: FullPaperReadingContext | str | None,
) -> str:
    """Return the complete deterministic main-paper body for Gap calls."""
    return _require_full_paper_reading(seed, reading).render()


def _copy_gap(gap: CandidateGap) -> CandidateGap:
    return CandidateGap.model_validate(gap.model_dump(mode="python"))


def _copy_agenda(agenda: GapAgenda) -> GapAgenda:
    return GapAgenda.model_validate(agenda.model_dump(mode="python"))


class GapDiscoveryModule:
    """Generate a bounded agenda and perform deterministic gap management."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: GapDiscoveryConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or GapDiscoveryConfig.from_env()
        self._discovery_prompt = _inject_schema(
            "gap_discovery.md",
            "{{GAP_GENERATION_DRAFT_JSON_SCHEMA}}",
            GapGenerationDraft,
        )
        self._evaluation_prompt = _inject_schema(
            "gap_evaluation.md",
            "{{GAP_EVALUATION_DRAFT_JSON_SCHEMA}}",
            GapEvaluationDraft,
        )
        self._refine_prompt = _inject_schema(
            "gap_refine.md",
            "{{GAP_REFINEMENT_DRAFT_JSON_SCHEMA}}",
            GapRefinementDraft,
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

    def generate_gaps(
        self,
        seed_paper_context: SeedPaperContext,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> GapAgenda:
        """Call 1: generate diverse gap statements from the full paper.

        Confidence and priority are intentionally not accepted from this call;
        they are assigned only by the independent evaluation call.
        """

        seed = SeedPaperContext.model_validate(
            seed_paper_context.model_dump(mode="python")
        )
        reading = _require_full_paper_reading(seed, full_paper_reading_context)
        user_content = (
            "FULL SEED PAPER MAIN-BODY CONTENT\n"
            "References and everything after the first exact References/Reference/"
            "Bibliography heading have already been removed deterministically.\n\n"
            "LEGAL GAP EVIDENCE SECTION IDS\n"
            + json.dumps(list(reading.body_section_ids), ensure_ascii=False)
            + "\n\n"
            + reading.render()
            + "\n\nReturn one JSON object containing zero to five diverse candidate gaps."
        )
        response = self._complete(
            system_prompt=self._discovery_prompt,
            user_content=user_content,
            task="gap_discovery",
            paper_id=seed.paper_id,
            output_schema=GapGenerationDraft,
        )
        draft = response.output
        if not isinstance(draft, GapGenerationDraft):
            raise ModuleContractError("Gap Discovery response has the wrong schema")
        allowed_refs = set(reading.body_section_ids)
        gaps: list[CandidateGap] = []
        for index, item in enumerate(draft.gaps, start=1):
            invalid = set(item.evidence_refs) - allowed_refs
            if invalid:
                raise ModuleContractError(
                    f"gap G{index} references unavailable evidence: {sorted(invalid)}"
                )
            if not item.evidence_refs:
                raise ModuleContractError(f"gap G{index} must have evidence_refs")
            gaps.append(
                CandidateGap(
                    id=f"G{index}",
                    statement=item.statement,
                    confidence="low",
                    confidence_reason="Pending independent Gap Evaluation.",
                    priority="low",
                    priority_reason="Pending independent Gap Evaluation.",
                    status="candidate",
                    evidence_refs=list(item.evidence_refs),
                )
            )
        return GapAgenda(gaps=gaps)

    def evaluate_gaps(
        self,
        candidate_gaps: GapAgenda | list[CandidateGap],
        seed_paper_context: SeedPaperContext | None = None,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
    ) -> GapAgenda:
        """Call 2: independently evaluate and rank every generated gap.

        Manually constructed agendas remain deterministically evidence-checked
        when no generation draft is pending; normal discovery always reaches
        this LLM evaluation path.
        """

        agenda = (
            candidate_gaps
            if isinstance(candidate_gaps, GapAgenda)
            else GapAgenda(gaps=list(candidate_gaps))
        )
        agenda = _copy_agenda(agenda)
        reading: FullPaperReadingContext | None = None
        if seed_paper_context is not None:
            reading = _require_full_paper_reading(seed_paper_context, full_paper_reading_context)
            allowed_refs = set(reading.body_section_ids)
            for gap in agenda.gaps:
                invalid = set(gap.evidence_refs) - allowed_refs
                if invalid:
                    raise ModuleContractError(
                        f"gap {gap.id} references unavailable evidence: {sorted(invalid)}"
                    )
                if not gap.evidence_refs:
                    raise ModuleContractError(f"gap {gap.id} must have evidence_refs")
        if seed_paper_context is None:
            return agenda
        if not agenda.gaps:
            # There is nothing to rank. Avoid an unnecessary second call while
            # preserving the valid empty GapAgenda outcome.
            return agenda

        seed = SeedPaperContext.model_validate(
            seed_paper_context.model_dump(mode="python")
        )
        assert reading is not None
        generated = [
            {
                "candidate_id": gap.id,
                "statement": gap.statement,
                "evidence_refs": list(gap.evidence_refs),
            }
            for gap in agenda.gaps
        ]
        user_content = (
            "FULL SEED PAPER MAIN-BODY CONTENT\n"
            "LEGAL GAP EVIDENCE SECTION IDS\n"
            + json.dumps(list(reading.body_section_ids), ensure_ascii=False)
            + "\n\n"
            + reading.render()
            + "\n\nGENERATED CANDIDATE GAPS\n"
            + json.dumps(generated, ensure_ascii=False, indent=2)
            + "\n\nReturn one evaluation for every candidate_id, ordered from most valuable to least valuable."
        )
        response = self._complete(
            system_prompt=self._evaluation_prompt,
            user_content=user_content,
            task="gap_evaluation",
            paper_id=seed.paper_id,
            output_schema=GapEvaluationDraft,
        )
        draft = response.output
        if not isinstance(draft, GapEvaluationDraft):
            raise ModuleContractError("Gap Evaluation response has the wrong schema")
        expected = {gap.id for gap in agenda.gaps}
        actual = [item.candidate_id for item in draft.evaluations]
        if set(actual) != expected or len(actual) != len(expected) or len(set(actual)) != len(actual):
            raise ModuleContractError(
                "Gap Evaluation must evaluate each generated candidate exactly once"
            )
        by_id = {gap.id: gap for gap in agenda.gaps}
        evaluated: list[CandidateGap] = []
        for item in draft.evaluations:
            original = by_id[item.candidate_id]
            evaluated.append(
                original.model_copy(
                    update={
                        "confidence": item.confidence,
                        "confidence_reason": item.confidence_reason,
                        "priority": item.priority,
                        "priority_reason": item.priority_reason,
                    }
                )
            )
        return GapAgenda(gaps=evaluated)

    @staticmethod
    def select_active_gap(gap_agenda: GapAgenda) -> GapDiscoveryResult:
        """Select the first candidate in model-provided worth-exploring order."""

        agenda = _copy_agenda(gap_agenda)
        selected_index: int | None = None
        for index, gap in enumerate(agenda.gaps):
            if gap.status == "candidate":
                selected_index = index
                break
        if selected_index is not None:
            agenda.gaps[selected_index].status = "active"
        active = (
            _copy_gap(agenda.gaps[selected_index])
            if selected_index is not None
            else None
        )
        return GapDiscoveryResult(
            gap_agenda=agenda,
            active_gap=active,
            next_required_action="VERIFY_RELATED_WORK" if active else None,
        )

    def discover(
        self,
        seed_paper_context: SeedPaperContext,
        *,
        full_paper_reading_context: FullPaperReadingContext | str | None = None,
        context=None,
        step: int = 1,
    ) -> GapDiscoveryResult:
        """Run initial discovery and optionally update a trajectory context."""

        agenda = self.generate_gaps(
            seed_paper_context,
            full_paper_reading_context=full_paper_reading_context,
        )
        agenda = self.evaluate_gaps(
            agenda,
            seed_paper_context,
            full_paper_reading_context=full_paper_reading_context,
        )
        result = self.select_active_gap(agenda)
        if context is None:
            return result

        before = context.get_state()
        context.record_gap_agenda(step, result.gap_agenda)
        context.trajectory.append_event(
            step,
            "gap_discovered",
            result_ref="gap_agenda",
            result_payload=result.gap_agenda.model_dump(mode="json"),
        )
        after = before.model_copy(deep=True)
        after.active_gap = (
            _copy_gap(result.active_gap) if result.active_gap is not None else None
        )
        context.record_transition(
            step,
            "active_gap_selected",
            before,
            after,
            action=None,
            result_ref=result.active_gap.id if result.active_gap else None,
            result_payload=result.model_dump(mode="json"),
        )
        return result

    def refine_gap(
        self,
        active_gap: CandidateGap,
        verification_result: GapVerificationResult,
        *,
        context=None,
        step: int = 1,
    ) -> GapDiscoveryResult:
        """Narrow one partially resolved gap at most once."""

        if verification_result.status != "partially_resolved":
            raise PreconditionError(
                "RefineGap requires a partially_resolved GapVerificationResult"
            )
        if not verification_result.remaining_gap or not verification_result.remaining_gap.strip():
            raise PreconditionError("RefineGap requires a non-empty remaining_gap")
        active = _copy_gap(active_gap)
        if active.status != "active":
            raise PreconditionError("RefineGap requires an active gap")
        if context is not None:
            if context.has_refined_gap(active.id):
                raise PreconditionError(f"gap {active.id} has already been refined")
            state_gap = context.get_state().active_gap
            if state_gap is None or state_gap.id != active.id:
                raise PreconditionError("RefineGap active gap does not match Research State")

        user_content = (
            "ACTIVE GAP\n"
            + json.dumps(active.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nGAP VERIFICATION\n"
            + json.dumps(
                verification_result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nReturn one JSON object."
        )
        if context is not None and context.seed_context.method_constraints:
            user_content += (
                "\n\nSEED METHOD CONSTRAINTS AND DESIGN RATIONALES\n"
                + json.dumps(
                    context.seed_context.method_constraints,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        response = self._complete(
            system_prompt=self._refine_prompt,
            user_content=user_content,
            task="gap_refine",
            paper_id=context.paper_id if context is not None else active.id,
            output_schema=GapRefinementDraft,
        )
        draft = response.output
        if not isinstance(draft, GapRefinementDraft):
            raise ModuleContractError("Gap Refine response has the wrong schema")
        refined = CandidateGap(
            id=active.id,
            statement=draft.refined_statement,
            confidence=active.confidence,
            confidence_reason=active.confidence_reason,
            priority=active.priority,
            priority_reason=active.priority_reason,
            status="active",
            evidence_refs=list(active.evidence_refs),
        )
        if context is None:
            agenda = GapAgenda(gaps=[refined])
            return GapDiscoveryResult(
                gap_agenda=agenda,
                active_gap=refined,
                next_required_action="VERIFY_RELATED_WORK",
            )

        current_agenda = context.get_latest_gap_agenda()
        if current_agenda is None:
            raise PreconditionError("RefineGap requires an existing GapAgenda")
        updated = _copy_agenda(current_agenda)
        matching = [index for index, gap in enumerate(updated.gaps) if gap.id == active.id]
        if not matching:
            raise PreconditionError(f"gap {active.id} is absent from the GapAgenda")
        updated.gaps[matching[0]] = refined
        before = context.get_state()
        after = before.model_copy(deep=True)
        after.active_gap = _copy_gap(refined)
        context.record_transition(
            step,
            "gap_refined",
            before,
            after,
            action=None,
            result_ref=active.id,
            result_payload={
                "gap_id": active.id,
                "refined_statement": refined.statement,
                "refinement_reason": draft.refinement_reason,
            },
        )
        context.record_gap_agenda(step, updated)
        return GapDiscoveryResult(
            gap_agenda=updated,
            active_gap=refined,
            next_required_action="VERIFY_RELATED_WORK",
        )

    def switch_gap(
        self,
        gap_agenda: GapAgenda | None = None,
        *,
        context=None,
        terminal_status: str = "resolved",
        step: int = 1,
    ) -> GapDiscoveryResult:
        """Switch deterministically to an existing candidate in agenda order."""

        if terminal_status not in {"resolved", "rejected"}:
            raise ValueError("terminal_status must be 'resolved' or 'rejected'")
        if context is not None:
            stored = context.get_latest_gap_agenda()
            if stored is None:
                raise PreconditionError("SwitchGap requires an existing GapAgenda")
            agenda = stored
            current_id = context.get_state().active_gap.id if context.get_state().active_gap else None
        elif gap_agenda is not None:
            agenda = gap_agenda
            current_id = next((gap.id for gap in agenda.gaps if gap.status == "active"), None)
        else:
            raise ValueError("gap_agenda or context is required")

        if context is not None:
            verification = context.memory_store.get_latest_gap_verification(current_id)
            if verification is None or not isinstance(
                verification.result, GapVerificationResult
            ):
                raise PreconditionError(
                    "SwitchGap requires a latest Gap Verification for the active gap"
                )
            expected_status = (
                "resolved" if verification.result.status == "resolved" else "rejected"
            )
            if terminal_status != expected_status:
                raise ModuleContractError(
                    "SwitchGap terminal status must match the latest Gap Verification: "
                    + expected_status
                )

        updated = _copy_agenda(agenda)
        if current_id is not None:
            for gap in updated.gaps:
                if gap.id == current_id and gap.status == "active":
                    gap.status = terminal_status
                    break
        selected_index: int | None = None
        for index, gap in enumerate(updated.gaps):
            if gap.status == "candidate":
                selected_index = index
                break
        active = None
        if selected_index is not None:
            updated.gaps[selected_index].status = "active"
            active = _copy_gap(updated.gaps[selected_index])
        result = GapDiscoveryResult(
            gap_agenda=updated,
            active_gap=active,
            next_required_action="VERIFY_RELATED_WORK" if active else None,
        )
        if context is None:
            return result

        before = context.get_state()
        after = before.model_copy(deep=True)
        after.active_gap = _copy_gap(active) if active is not None else None
        after.candidate_solutions = []
        after.current_solution = None
        after.experiments = []
        context.record_transition(
            step,
            "gap_switched",
            before,
            after,
            action="SWITCH_GAP",
            result_ref=active.id if active else None,
            result_payload=result.model_dump(mode="json"),
        )
        context.record_gap_agenda(step, updated)
        return result


__all__ = ["GapDiscoveryConfig", "GapDiscoveryModule"]
