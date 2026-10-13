"""M5 Related-work Verification and deterministic state application."""

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
from ..memory.store import ResearchMemoryStore
from ..models.memory import (
    FailureRecord,
    PaperRecord,
    RelatedPaperPool,
    SearchRecord,
    VerificationRecord,
)
from ..models.seed import SeedPaperContext
from ..models.state import CurrentSolution, ResearchState
from ..models.verification import (
    GapVerificationResult,
    ComparisonRecord,
    SolutionVerificationResult,
    SolutionVerificationDraft,
    VerificationFocus,
)
from ..retrieval.interfaces import RelatedPaperProvider
from ..runtime.state_store import ResearchStateStore
from .verification_refs import canonicalize_paper_refs


@dataclass(frozen=True)
class VerificationConfig:
    max_tokens: int = 4_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "VerificationConfig":
        values = os.environ if environ is None else environ
        raw = values.get("RELATED_WORK_VERIFICATION_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError(
                "RELATED_WORK_VERIFICATION_MAX_TOKENS must be an integer"
            ) from exc


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


def _copy_pool(pool: RelatedPaperPool) -> RelatedPaperPool:
    return RelatedPaperPool.model_validate(pool.model_dump(mode="python"))


def _provider_name(provider: RelatedPaperProvider) -> str:
    name = getattr(provider, "provider_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    class_name = provider.__class__.__name__.lower()
    if "semantic" in class_name:
        return "semantic_scholar"
    if "fake" in class_name:
        return "fake"
    return class_name


def _paper_payload(paper: PaperRecord) -> dict[str, Any]:
    return {
        "paper_ref": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract
        if paper.abstract is not None
        else "[ABSTRACT UNAVAILABLE — do not infer its contents]",
        "year": paper.year,
        "venue": paper.venue,
    }


def _seed_resolution_fields(provider: object) -> dict[str, Any]:
    audit = getattr(provider, "last_seed_resolution_audit", {})
    if not isinstance(audit, Mapping) or not audit:
        return {}
    return {
        "seed_resolution_candidate_count": audit.get("candidate_count"),
        "seed_resolution_selected_title": audit.get("selected_title"),
        "seed_resolution_selected_year": audit.get("selected_year"),
        "seed_resolution_selected_venue": audit.get("selected_venue"),
        "seed_resolution_reason": audit.get("selection_reason"),
    }


class RelatedWorkVerificationModule:
    """Run exactly one structured LLM call per verification task."""

    def __init__(
        self,
        llm_client: LLMClient,
        related_paper_provider: RelatedPaperProvider,
        config: VerificationConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.related_paper_provider = related_paper_provider
        self.config = config or VerificationConfig.from_env()
        self._gap_prompt = _inject_schema(
            "verify_gap.md",
            "{{GAP_VERIFICATION_RESULT_JSON_SCHEMA}}",
            GapVerificationResult,
        )
        self._solution_prompt = _inject_schema(
            "verify_solution.md",
            "{{SOLUTION_VERIFICATION_DRAFT_JSON_SCHEMA}}",
            SolutionVerificationDraft,
        )
        self.call_count = 0
        self.scientific_search_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None
        self.last_provider_request_count = 0
        self.last_provider_retry_count = 0
        self.last_provider_rate_limit_count = 0
        self.last_provider_pacing_waits: list[float] = []
        self.last_provider_retry_waits: list[float] = []
        self.last_cache_hit = False
        self.last_cache_scope = "none"
        self.last_global_cache_operations: list[str] = []
        self.last_search_record: SearchRecord | None = None
        self.last_solution_comparisons: list[ComparisonRecord] = []

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
    def _resolve_context(
        seed_context: SeedPaperContext | None,
        context,
        memory_store: ResearchMemoryStore | None,
    ) -> tuple[SeedPaperContext, ResearchMemoryStore]:
        if context is not None:
            if seed_context is not None and seed_context.paper_id != context.paper_id:
                raise PreconditionError("SeedPaperContext does not match trajectory")
            seed = context.seed_context
            store = context.memory_store
        else:
            if seed_context is None:
                raise TypeError("verification requires SeedPaperContext or context")
            seed = SeedPaperContext.model_validate(seed_context.model_dump(mode="python"))
            store = memory_store or ResearchMemoryStore(seed.paper_id)
        if store.seed_paper_id != seed.paper_id:
            raise PreconditionError("Research Memory does not match Seed Paper")
        return seed, store

    def get_related_paper_pool(
        self,
        seed_context: SeedPaperContext,
        *,
        memory_store: ResearchMemoryStore | None = None,
        context=None,
        step: int = 1,
    ) -> RelatedPaperPool:
        seed, store = self._resolve_context(seed_context, context, memory_store)
        existing = store.get_memory().paper_memory.related_paper_pool
        if existing is not None:
            if existing.seed_paper_id != seed.paper_id:
                raise ModuleContractError("cached related-paper pool has the wrong seed")
            for paper in existing.papers:
                store.save_paper(paper)
            pool = _copy_pool(existing)
            self.last_cache_hit = True
            self.last_cache_scope = "trajectory"
            self.last_global_cache_operations = []
            self.last_provider_request_count = 0
            self.last_provider_retry_count = 0
            self.last_provider_rate_limit_count = 0
            self.last_provider_pacing_waits = []
            self.last_provider_retry_waits = []
            self.last_search_record = SearchRecord(
                provider=_provider_name(self.related_paper_provider),
                seed_paper_id=seed.paper_id,
                operation="related_pool_acquisition",
                returned_paper_refs=[paper.paper_id for paper in pool.papers],
                created_at_step=step,
                cache_hit=True,
                cache_scope="trajectory",
                cached_operations=[],
                provider_request_count=0,
                resolved_provider_paper_id=getattr(
                    self.related_paper_provider, "last_resolved_provider_id", None
                ),
                **_seed_resolution_fields(self.related_paper_provider),
            )
            if context is not None:
                context.trajectory.append_event(
                    step,
                    "related_pool_cache_hit",
                    action="VERIFY_RELATED_WORK",
                    result_ref="related_paper_pool",
                    result_payload=self.last_search_record.model_dump(mode="json"),
                    memory_writes=[],
                )
            return pool

        self.last_cache_hit = False
        self.last_cache_scope = "none"
        self.last_global_cache_operations = []
        self.last_search_record = None
        self.scientific_search_count += 1
        try:
            pool = self.related_paper_provider.get_related_paper_pool(seed, limit=20)
        finally:
            self.last_provider_request_count = getattr(
                self.related_paper_provider, "last_request_count", 0
            )
            self.last_provider_retry_count = getattr(
                self.related_paper_provider, "last_retry_count", 0
            )
            self.last_provider_rate_limit_count = getattr(
                self.related_paper_provider, "last_rate_limit_count", 0
            )
            self.last_provider_pacing_waits = list(
                getattr(self.related_paper_provider, "last_pacing_waits", [])
            )
            self.last_provider_retry_waits = list(
                getattr(self.related_paper_provider, "last_retry_waits", [])
            )
            self.last_global_cache_operations = list(
                getattr(self.related_paper_provider, "last_global_cache_operations", [])
            )
            global_cache_complete = bool(
                getattr(self.related_paper_provider, "last_global_cache_hit", False)
            )
            if global_cache_complete:
                # The provider returned the complete pool from the persistent
                # cache, so this acquisition did not consume a new scientific
                # search despite entering the provider path.
                self.scientific_search_count -= 1
                self.last_cache_hit = True
                self.last_cache_scope = "global"
            elif self.last_global_cache_operations:
                self.last_cache_scope = "global_partial"
        try:
            pool = RelatedPaperPool.model_validate(pool.model_dump(mode="python"))
        except Exception as exc:
            raise ModuleContractError("RelatedPaperProvider returned an invalid pool") from exc
        if pool.seed_paper_id != seed.paper_id:
            raise ModuleContractError("RelatedPaperProvider returned the wrong seed")
        if len(pool.papers) > 20:
            raise ModuleContractError("related-paper pool cannot exceed 20 papers")
        seen: set[str] = set()
        for paper in pool.papers:
            if paper.paper_id in seen:
                raise ModuleContractError("related-paper pool contains duplicate references")
            seen.add(paper.paper_id)
            store.save_paper(paper)
        store.set_related_paper_pool(pool)
        provider_requests = getattr(self.related_paper_provider, "last_request_count", None)
        if not isinstance(provider_requests, int):
            provider_requests = None
        self.last_provider_request_count = provider_requests or 0
        self.last_search_record = SearchRecord(
            provider=_provider_name(self.related_paper_provider),
            seed_paper_id=seed.paper_id,
            operation="related_pool_acquisition",
            returned_paper_refs=[paper.paper_id for paper in pool.papers],
            created_at_step=step,
            cache_hit=self.last_cache_hit,
            cache_scope=self.last_cache_scope,
            cached_operations=self.last_global_cache_operations,
            provider_request_count=provider_requests,
            resolved_provider_paper_id=getattr(
                self.related_paper_provider, "last_resolved_provider_id", None
            ),
            **_seed_resolution_fields(self.related_paper_provider),
        )
        store.save_search_record(self.last_search_record)
        if context is not None:
            event_type = (
                "related_pool_global_cache_hit"
                if self.last_cache_scope == "global"
                else "related_pool_acquired"
            )
            context.trajectory.append_event(
                step,
                event_type,
                action="VERIFY_RELATED_WORK",
                result_ref="related_paper_pool",
                result_payload=self.last_search_record.model_dump(mode="json"),
                memory_writes=["paper_memory.related_paper_pool", "paper_memory.search_records"],
            )
        return _copy_pool(pool)

    @staticmethod
    def _known_paper_refs(store: ResearchMemoryStore) -> set[str]:
        memory = store.get_memory()
        refs = {paper.paper_id for paper in memory.paper_memory.papers}
        pool = memory.paper_memory.related_paper_pool
        if pool is not None:
            refs.update(paper.paper_id for paper in pool.papers)
        return refs

    @staticmethod
    def _validate_paper_refs(
        refs: list[str],
        store: ResearchMemoryStore,
        *,
        field_name: str = "related_papers",
    ) -> None:
        if len(refs) > 5:
            raise ModuleContractError(f"{field_name} cannot contain more than five papers")
        unknown = set(refs) - RelatedWorkVerificationModule._known_paper_refs(store)
        if unknown:
            raise ModuleContractError(
                f"{field_name} references papers absent from PaperMemory: {sorted(unknown)}"
            )

    @staticmethod
    def _validate_gap_result(
        result: GapVerificationResult,
        store: ResearchMemoryStore,
    ) -> None:
        RelatedWorkVerificationModule._validate_paper_refs(result.related_papers, store)
        if result.status == "partially_resolved" and not result.remaining_gap:
            raise ModuleContractError(
                "partially_resolved verification requires a meaningful remaining_gap"
            )
        if result.evidence_level != "abstract_only":
            raise ModuleContractError(
                "M5 related-paper verification uses abstract-level evidence only"
            )

    @staticmethod
    def _canonicalize_gap_result(
        result: GapVerificationResult,
        pool: RelatedPaperPool,
    ) -> GapVerificationResult:
        payload = result.model_dump(mode="python")
        payload["related_papers"] = canonicalize_paper_refs(
            result.related_papers, pool.papers
        )
        return GapVerificationResult.model_validate(payload)

    @staticmethod
    def _canonicalize_solution_draft(
        draft: SolutionVerificationDraft,
        pool: RelatedPaperPool,
    ) -> SolutionVerificationDraft:
        payload = draft.model_dump(mode="python")
        papers = pool.papers
        for comparison in payload["comparisons"]:
            comparison["paper_ref"] = canonicalize_paper_refs(
                [comparison["paper_ref"]], papers
            )[0]
        result = payload["result"]
        result["related_papers"] = canonicalize_paper_refs(
            result["related_papers"], papers
        )
        for overlap in result["novelty_relevant_overlaps"]:
            overlap["matched_paper_ref"] = canonicalize_paper_refs(
                [overlap["matched_paper_ref"]], papers
            )[0]
        return SolutionVerificationDraft.model_validate(payload)

    @staticmethod
    def _validate_solution_result(
        result: SolutionVerificationResult,
        comparisons: list[ComparisonRecord],
        solution: CurrentSolution,
        store: ResearchMemoryStore,
    ) -> None:
        if len(comparisons) > 5:
            raise ModuleContractError("solution verification cannot contain more than five comparisons")
        comparison_refs = [comparison.paper_ref for comparison in comparisons]
        if len(set(comparison_refs)) != len(comparison_refs):
            raise ModuleContractError("solution comparisons cannot repeat paper_ref")
        unknown_comparison_refs = set(comparison_refs) - RelatedWorkVerificationModule._known_paper_refs(store)
        if unknown_comparison_refs:
            raise ModuleContractError(
                "solution comparisons reference papers absent from PaperMemory: "
                f"{sorted(unknown_comparison_refs)}"
            )
        if result.related_papers != comparison_refs:
            raise ModuleContractError(
                "result.related_papers must equal comparison paper refs in order"
            )
        RelatedWorkVerificationModule._validate_paper_refs(result.related_papers, store)
        known_components = {component.id for component in solution.components}
        overlap_refs = []
        for overlap in result.novelty_relevant_overlaps:
            if overlap.component_id is not None and overlap.component_id not in known_components:
                raise ModuleContractError(
                    f"solution overlap references unknown component {overlap.component_id}"
                )
            overlap_refs.append(overlap.matched_paper_ref)
        RelatedWorkVerificationModule._validate_paper_refs(
            overlap_refs, store, field_name="matched_paper_ref"
        )
        if len(result.novelty_relevant_overlaps) > 5:
            raise ModuleContractError("solution verification cannot contain more than five overlaps")
        if result.novelty_status == "partial_overlap" and not result.novelty_relevant_overlaps:
            raise ModuleContractError(
                "partial_overlap requires novelty_relevant_overlaps"
            )
        if result.novelty_status == "core_overlap" and not result.novelty_relevant_overlaps:
            raise ModuleContractError(
                "core_overlap requires evidence covering the core mechanism"
            )
        comparison_novelty_refs = {
            comparison.paper_ref
            for comparison in comparisons
            if comparison.novelty_relevant_overlaps
        }
        final_overlap_refs = {
            overlap.matched_paper_ref
            for overlap in result.novelty_relevant_overlaps
        }
        if not final_overlap_refs <= set(comparison_refs):
            raise ModuleContractError(
                "final novelty overlaps must reference comparison papers"
            )
        if result.novelty_status == "distinct":
            if result.novelty_relevant_overlaps or comparison_novelty_refs:
                raise ModuleContractError(
                    "distinct result conflicts with novelty-bearing comparison evidence"
                )
        elif not final_overlap_refs <= comparison_novelty_refs:
            raise ModuleContractError(
                "final novelty overlaps lack supporting ComparisonRecord evidence"
            )
        if result.evidence_level != "abstract_only":
            raise ModuleContractError(
                "M5 related-paper verification uses abstract-level evidence only"
            )

    def _pool_for(
        self,
        seed: SeedPaperContext,
        store: ResearchMemoryStore,
        context,
        step: int,
        provided: RelatedPaperPool | None,
    ) -> RelatedPaperPool:
        if provided is not None:
            pool = _copy_pool(provided)
            if pool.seed_paper_id != seed.paper_id:
                raise ModuleContractError("provided related-paper pool has the wrong seed")
            existing = store.get_memory().paper_memory.related_paper_pool
            if existing is None:
                for paper in pool.papers:
                    store.save_paper(paper)
                store.set_related_paper_pool(pool)
            return pool
        return self.get_related_paper_pool(seed, memory_store=store, context=context, step=step)

    def verify_gap(
        self,
        active_gap,
        seed_context: SeedPaperContext | None = None,
        related_paper_pool: RelatedPaperPool | None = None,
        *,
        context=None,
        memory_store: ResearchMemoryStore | None = None,
        state_store: ResearchStateStore | None = None,
        step: int = 1,
    ) -> GapVerificationResult:
        seed, store = self._resolve_context(seed_context, context, memory_store)
        if context is not None:
            state_gap = context.get_state().active_gap
            if state_gap is None or state_gap.id != active_gap.id:
                raise PreconditionError("VERIFY_GAP target does not match active Research State gap")
        elif state_store is not None:
            state_gap = state_store.get_state().active_gap
            if state_gap is None or state_gap.id != active_gap.id:
                raise PreconditionError("VERIFY_GAP target does not match Research State")
        if active_gap.status != "active":
            raise PreconditionError("VERIFY_GAP requires an active gap")
        pool = self._pool_for(seed, store, context, step, related_paper_pool)
        user_content = (
            "ACTIVE GAP\n"
            + json.dumps(active_gap.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nSEED PAPER CONTEXT\n"
            + json.dumps(
                {
                    "paper_id": seed.paper_id,
                    "title": seed.title,
                    "abstract": seed.abstract,
                    "research_objective": seed.research_objective,
                    "core_method": seed.core_method,
                    "main_components": seed.main_components,
                    "main_findings": seed.main_findings,
                    "limitations": seed.limitations,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nRELATED PAPER POOL (title and abstract only)\n"
            + json.dumps([_paper_payload(paper) for paper in pool.papers], ensure_ascii=False, indent=2)
            + "\n\nReturn exactly one JSON object."
        )
        response = self._complete(
            system_prompt=self._gap_prompt,
            user_content=user_content,
            task="verify_gap",
            paper_id=seed.paper_id,
            output_schema=GapVerificationResult,
        )
        result = response.output
        if not isinstance(result, GapVerificationResult):
            raise ModuleContractError("VERIFY_GAP response has the wrong schema")
        result = self._canonicalize_gap_result(result, pool)
        self._validate_gap_result(result, store)
        record = VerificationRecord(
            id=f"V{len(store.get_memory().verification_memory) + 1}",
            task="VERIFY_GAP",
            target_ref=active_gap.id,
            verification_focus=None,
            result=result,
            paper_refs=list(result.related_papers),
            created_at_step=step,
        )
        store.save_verification(record)
        if context is not None:
            state = context.get_state()
            context.record_transition(
                step,
                "gap_verified",
                state,
                state,
                action="VERIFY_RELATED_WORK",
                result_ref=active_gap.id,
                result_payload=result.model_dump(mode="json"),
                memory_writes=["verification_memory"],
            )
        return result

    def verify_solution(
        self,
        active_gap,
        current_solution: CurrentSolution | None = None,
        seed_context: SeedPaperContext | None = None,
        related_paper_pool: RelatedPaperPool | None = None,
        verification_focus: VerificationFocus | None = None,
        *,
        context=None,
        memory_store: ResearchMemoryStore | None = None,
        state_store: ResearchStateStore | None = None,
        step: int = 1,
    ) -> SolutionVerificationResult:
        seed, store = self._resolve_context(seed_context, context, memory_store)
        state = None
        if context is not None:
            state = context.get_state()
            state_gap = state.active_gap
            if state_gap is None or state_gap.id != active_gap.id:
                raise PreconditionError("VERIFY_SOLUTION gap does not match active Research State gap")
        elif state_store is not None:
            state = state_store.get_state()
            state_gap = state.active_gap
            if state_gap is None or state_gap.id != active_gap.id:
                raise PreconditionError("VERIFY_SOLUTION gap does not match Research State")
        if state is not None:
            state_solution = state.current_solution
            if state_solution is None:
                raise PreconditionError("VERIFY_SOLUTION requires a Current Solution in Research State")
            if current_solution is None:
                current_solution = state_solution
            elif current_solution.id != state_solution.id:
                raise PreconditionError(
                    "VERIFY_SOLUTION current_solution does not match Research State"
                )
        if active_gap.status != "active":
            raise PreconditionError("VERIFY_SOLUTION requires an active gap")
        if current_solution is None:
            raise PreconditionError("VERIFY_SOLUTION requires a Current Solution")
        solution = CurrentSolution.model_validate(current_solution.model_dump(mode="python"))
        focus = verification_focus or VerificationFocus(mode="full_solution")
        unknown_focus = set(focus.changed_units) - {component.id for component in solution.components}
        if unknown_focus:
            raise ModuleContractError(
                f"verification focus references unknown components: {sorted(unknown_focus)}"
            )
        pool = self._pool_for(seed, store, context, step, related_paper_pool)
        user_content = (
            "ACTIVE GAP\n"
            + json.dumps(active_gap.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nSEED PAPER CONTEXT\n"
            + json.dumps(
                {
                    "paper_id": seed.paper_id,
                    "title": seed.title,
                    "abstract": seed.abstract,
                    "research_objective": seed.research_objective,
                    "core_method": seed.core_method,
                    "main_components": seed.main_components,
                    "main_findings": seed.main_findings,
                    "limitations": seed.limitations,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nCURRENT SOLUTION (always provide the complete object)\n"
            + json.dumps(solution.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nVERIFICATION FOCUS\n"
            + json.dumps(focus.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n\nRELATED PAPER POOL (title and abstract only)\n"
            + json.dumps([_paper_payload(paper) for paper in pool.papers], ensure_ascii=False, indent=2)
            + "\n\nReturn exactly one JSON object."
        )
        response = self._complete(
            system_prompt=self._solution_prompt,
            user_content=user_content,
            task="verify_solution",
            paper_id=seed.paper_id,
            output_schema=SolutionVerificationDraft,
        )
        draft = response.output
        if not isinstance(draft, SolutionVerificationDraft):
            raise ModuleContractError("VERIFY_SOLUTION response has the wrong schema")
        draft = self._canonicalize_solution_draft(draft, pool)
        comparisons = list(draft.comparisons)
        result = draft.result
        self._validate_solution_result(result, comparisons, solution, store)
        record = VerificationRecord(
            id=f"V{len(store.get_memory().verification_memory) + 1}",
            task="VERIFY_SOLUTION",
            target_ref=solution.id,
            verification_focus=focus,
            result=result,
            paper_refs=list(result.related_papers),
            created_at_step=step,
        )
        failure_record: FailureRecord | None = None
        if result.novelty_status == "core_overlap":
            failure_record = FailureRecord(
                type="rejected_solution",
                object_ref=solution.id,
                summary=f"Solution {solution.id} rejected after core novelty overlap.",
                reason=result.summary,
                evidence_refs=list(result.related_papers),
            )

        before: ResearchState | None = None
        after: ResearchState | None = None
        if context is not None or state_store is not None:
            before = context.get_state() if context is not None else state_store.get_state()
            after = _copy_state(before)
            if after.current_solution is None or after.current_solution.id != solution.id:
                raise PreconditionError("Current Solution changed during verification")
            after_solution = after.current_solution
            after_solution.novelty_status = result.novelty_status
            after_solution.novelty_summary = result.summary
            if result.novelty_status == "core_overlap":
                for candidate in after.candidate_solutions:
                    if candidate.id == solution.id:
                        candidate.status = "rejected"
                after.current_solution = None
                after.experiments = []
            elif result.novelty_status in {"distinct", "partial_overlap"}:
                if focus.mode == "changed_units":
                    if any(
                        component.status == "stale"
                        for component in after_solution.components
                    ):
                        raise ModuleContractError(
                            "changed_units verification cannot settle stale Components; "
                            "use full_solution focus"
                        )
                    pending_modified = {
                        component.id
                        for component in after_solution.components
                        if component.status == "modified"
                    }
                    focused_units = set(focus.changed_units)
                    if pending_modified != focused_units:
                        raise ModuleContractError(
                            "changed_units verification must cover all pending modified Components"
                        )
                    for component in after_solution.components:
                        if (
                            component.id in focus.changed_units
                        ):
                            component.status = "ready"
                elif focus.mode == "full_solution":
                    # A full-solution verification covers every component,
                    # including components made stale by a core-idea revision.
                    # Novelty and freshness are orthogonal: partial overlap
                    # still means the verified component versions are fresh.
                    for component in after_solution.components:
                        if component.status in {"modified", "stale"}:
                            component.status = "ready"

            # Validate the complete after-state before any memory or trajectory mutation.
            after = ResearchState.model_validate(after.model_dump(mode="python"))

        self.last_solution_comparisons = comparisons
        store.save_verification(record)
        if failure_record is not None:
            store.save_failure(failure_record)
        failure_written = failure_record is not None
        if context is not None or state_store is not None:
            assert before is not None and after is not None
            if context is not None:
                context.record_transition(
                    step,
                    "solution_verified",
                    before,
                    after,
                    action="VERIFY_RELATED_WORK",
                    result_ref=solution.id,
                    result_payload={
                        "verification_focus": focus.model_dump(mode="json"),
                        "comparisons": [
                            comparison.model_dump(mode="json")
                            for comparison in comparisons
                        ],
                        "result": result.model_dump(mode="json"),
                    },
                    memory_writes=(
                        ["verification_memory", "failure_memory"]
                        if failure_written
                        else ["verification_memory"]
                    ),
                )
            else:
                state_store.replace_state(after)
        return result


__all__ = ["RelatedWorkVerificationModule", "VerificationConfig"]
