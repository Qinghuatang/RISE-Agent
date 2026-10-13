"""M6 Dynamic Entity Bridging and Inspiration Retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ModuleContractError, PreconditionError, RetrievalError, StructuredOutputError
from ..llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from ..memory.store import ResearchMemoryStore
from ..models.inspiration import (
    BridgeEntity,
    BridgeGenerationDraft,
    InspirationCandidate,
    InspirationNeed,
    InspirationResult,
)
from ..models.memory import (
    InspirationRecord,
    InspirationSearchRecord,
    PaperRecord,
    ResearchMemory,
    VerificationRecord,
)
from ..models.seed import SeedPaperContext
from ..models.state import CurrentSolution, ResearchState
from ..retrieval.interfaces import InspirationSearchProvider
from ..runtime.state_store import ResearchStateStore


@dataclass(frozen=True)
class InspirationRetrievalConfig:
    max_tokens: int = 4_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "InspirationRetrievalConfig":
        values = os.environ if environ is None else environ
        raw = values.get("INSPIRATION_RETRIEVAL_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError(
                "INSPIRATION_RETRIEVAL_MAX_TOKENS must be an integer"
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


def _entity_key(entity: BridgeEntity | str) -> str:
    value = entity.name if isinstance(entity, BridgeEntity) else entity
    return " ".join(value.casefold().split())


def _copy_paper(paper: PaperRecord) -> PaperRecord:
    return PaperRecord.model_validate(paper.model_dump(mode="python"))


def _deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    seen: set[str] = set()
    result: list[PaperRecord] = []
    for paper in papers:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        result.append(_copy_paper(paper))
        if len(result) == 20:
            break
    return result


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


def _solution_summary(solution: CurrentSolution) -> dict[str, Any]:
    return {
        "id": solution.id,
        "core_idea": solution.core_idea,
        "rationale": solution.rationale,
        "components": [
            {"id": component.id, "description": component.description}
            for component in solution.components
        ],
        "novelty_status": solution.novelty_status,
    }


def _verification_summary(record: VerificationRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "task": record.task,
        "target_ref": record.target_ref,
        "result": record.result.model_dump(mode="json"),
        "paper_refs": record.paper_refs,
    }


class InspirationRetrievalModule:
    """Run M6 with at most two structured LLM calls and one search attempt."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_provider: InspirationSearchProvider,
        config: InspirationRetrievalConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.search_provider = search_provider
        self.config = config or InspirationRetrievalConfig.from_env()
        self._entity_prompt = _inject_schema(
            "inspiration_entities.md",
            "{{BRIDGE_GENERATION_DRAFT_JSON_SCHEMA}}",
            BridgeGenerationDraft,
        )
        self._extract_prompt = _inject_schema(
            "inspiration_extract.md",
            "{{INSPIRATION_RESULT_JSON_SCHEMA}}",
            InspirationResult,
        )
        self.call_count = 0
        self.scientific_search_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None
        self.last_provider_request_count = 0
        self.last_provider_retry_count = 0
        self.last_cache_hit = False
        self.last_cache_scope = "none"
        self.last_provider_error: str | None = None
        self.last_grounding_status = "none"
        self.last_record: InspirationRecord | None = None
        self.last_search_record: InspirationSearchRecord | None = None
        self.last_next_required_action: str | None = None

    def _complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        task: str,
        paper_id: str,
        output_schema: type[Any],
        format_retry_reason: str | None = None,
    ) -> StructuredLLMResponse[Any]:
        metadata: dict[str, Any] = {"task": task, "paper_id": paper_id}
        if format_retry_reason:
            system_prompt = (
                system_prompt
                + "\n\nFORMAT-ONLY RETRY: preserve the same scientific answer; correct only "
                "JSON structure and retrieval provenance. Do not change the search "
                "result, scientific content, or selected entities. Return one JSON object."
            )
            metadata.update(
                {
                    "format_retry_count": 1,
                    "retry_reason": format_retry_reason[:500],
                }
            )
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata=metadata,
        )
        response = self.llm_client.complete_structured(request, output_schema)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        return response

    @staticmethod
    def _resolve_seed(
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
                raise TypeError("Inspiration Retrieval requires SeedPaperContext or context")
            seed = SeedPaperContext.model_validate(
                seed_context.model_dump(mode="python")
            )
            store = memory_store or ResearchMemoryStore(seed.paper_id)
        if store.seed_paper_id != seed.paper_id:
            raise PreconditionError("Research Memory does not match Seed Paper")
        return seed, store

    @staticmethod
    def _previous_entity_names(memory: ResearchMemory) -> list[str]:
        names: dict[str, str] = {}
        for record in memory.inspiration_memory:
            entities = [*record.proposed_entities, *record.selected_entities]
            for entity in entities:
                names.setdefault(_entity_key(entity), entity.name)
        for record in memory.paper_memory.inspiration_search_records:
            for entity in record.bridge_entities:
                names.setdefault(_entity_key(entity), entity.name)
        return list(names.values())

    @staticmethod
    def _rejected_mechanisms(memory_store: ResearchMemoryStore) -> list[str]:
        records = memory_store.get_relevant_failures(
            failure_type="rejected_mechanism"
        )
        return [record.summary for record in records]

    @staticmethod
    def _compact_related_context(
        memory_store: ResearchMemoryStore,
        active_gap_id: str,
        solution_id: str,
    ) -> dict[str, Any]:
        gap_verification = memory_store.get_latest_gap_verification(active_gap_id)
        solution_verification = memory_store.get_latest_verification(
            solution_id, task="VERIFY_SOLUTION"
        )
        related_papers = memory_store.get_related_papers()[:3]
        return {
            "latest_gap_verification": _verification_summary(gap_verification),
            "latest_solution_verification": _verification_summary(solution_verification),
            "related_work_context": [
                {
                    "paper_ref": paper.paper_id,
                    "title": paper.title,
                    "abstract": (paper.abstract or "")[:500],
                }
                for paper in related_papers
            ],
        }

    @staticmethod
    def _validate_draft(
        draft: BridgeGenerationDraft,
        previous_names: list[str],
    ) -> None:
        if len(draft.proposed_entities) != 5:
            raise ModuleContractError(
                "BridgeGenerationDraft must contain exactly five proposed entities"
            )
        if len(draft.selected_entities) > 3:
            raise ModuleContractError(
                "BridgeGenerationDraft cannot select more than three entities"
            )
        proposed = {_entity_key(entity) for entity in draft.proposed_entities}
        selected = {_entity_key(entity) for entity in draft.selected_entities}
        if not selected <= proposed:
            raise ModuleContractError(
                "selected bridge entities must come from proposed entities"
            )
        # Previous entities are supplied to the prompt as an anti-repetition
        # signal.  A repeated entity can still be returned when the model has
        # a defensible reason; exact cache reuse also needs to remain possible.

    @staticmethod
    def _cache_lookup(
        memory: ResearchMemory,
        query: str,
    ) -> tuple[InspirationSearchRecord, list[PaperRecord]] | None:
        normalized_query = " ".join(query.casefold().split())
        papers_by_id = {paper.paper_id: paper for paper in memory.paper_memory.papers}
        for record in reversed(memory.paper_memory.inspiration_search_records):
            if " ".join(record.query.casefold().split()) != normalized_query:
                continue
            missing = [ref for ref in record.paper_refs if ref not in papers_by_id]
            if missing:
                raise ModuleContractError(
                    "Inspiration search cache is corrupt; missing PaperMemory refs: "
                    + ", ".join(missing)
                )
            return record, [
                _copy_paper(papers_by_id[ref]) for ref in record.paper_refs
            ]
        return None

    @staticmethod
    def _validate_result(
        result: InspirationResult,
        *,
        grounded: bool,
        papers: list[PaperRecord],
        selected_entities: list[BridgeEntity],
    ) -> InspirationResult:
        if result.status == "no_useful_inspiration":
            if result.inspirations:
                raise ModuleContractError(
                    "no_useful_inspiration cannot contain candidates"
                )
            return result
        if not 1 <= len(result.inspirations) <= 3:
            raise ModuleContractError("found InspirationResult must contain 1 to 3 candidates")
        known_papers = {paper.paper_id for paper in papers}
        selected_by_key = {
            _entity_key(entity): entity for entity in selected_entities
        }
        selected = set(selected_by_key)
        used_papers: set[str] = set()
        canonical_candidates: list[InspirationCandidate] = []
        for candidate in result.inspirations:
            candidate_key = _entity_key(candidate.bridge_entity)
            if candidate_key not in selected:
                raise ModuleContractError(
                    "inspiration candidate must use a selected bridge entity"
                )
            if grounded:
                if candidate.grounding != "arxiv" or candidate.paper_ref is None:
                    raise InspirationRetrievalModule._provenance_error(
                        result,
                        f"inspirations.{len(canonical_candidates)}.paper_ref",
                        "grounded candidates require grounding=arxiv and a paper_ref",
                    )
                if candidate.paper_ref not in known_papers:
                    raise InspirationRetrievalModule._provenance_error(
                        result,
                        f"inspirations.{len(canonical_candidates)}.paper_ref",
                        "paper_ref is not in the retrieved paper pool",
                    )
                if candidate.paper_ref in used_papers:
                    raise InspirationRetrievalModule._provenance_error(
                        result,
                        f"inspirations.{len(canonical_candidates)}.paper_ref",
                        "one retrieved paper cannot ground multiple candidates",
                    )
                used_papers.add(candidate.paper_ref)
            else:
                if candidate.grounding != "none" or candidate.paper_ref is not None:
                    raise InspirationRetrievalModule._provenance_error(
                        result,
                        f"inspirations.{len(canonical_candidates)}.grounding",
                        "fallback candidates require grounding=none and paper_ref=null",
                    )
            canonical_candidates.append(
                candidate.model_copy(
                    update={"bridge_entity": selected_by_key[candidate_key]}
                )
            )
        return result.model_copy(update={"inspirations": canonical_candidates})

    @staticmethod
    def _provenance_error(
        result: InspirationResult,
        field_path: str,
        message: str,
    ) -> StructuredOutputError:
        return StructuredOutputError(
            f"InspirationResult provenance validation failed: {message}",
            schema_name="InspirationResult",
            validation_errors=[
                {
                    "field_path": field_path,
                    "error_type": "provenance_error",
                    "message": message,
                }
            ],
            parsed_payload=result.model_dump(mode="json"),
        )

    @staticmethod
    def _discard_invalid_provenance_candidates(
        result: InspirationResult,
        *,
        grounded: bool,
        papers: list[PaperRecord],
        selected_entities: list[BridgeEntity],
    ) -> InspirationResult:
        """Keep only candidates whose provenance is mechanically defensible."""

        known_papers = {paper.paper_id for paper in papers}
        selected = {_entity_key(entity) for entity in selected_entities}
        valid: list[InspirationCandidate] = []
        seen_papers: set[str] = set()
        for candidate in result.inspirations:
            if _entity_key(candidate.bridge_entity) not in selected:
                raise ModuleContractError(
                    "inspiration candidate must use a selected bridge entity"
                )
            if grounded:
                if (
                    candidate.grounding != "arxiv"
                    or candidate.paper_ref is None
                    or candidate.paper_ref not in known_papers
                    or candidate.paper_ref in seen_papers
                ):
                    continue
                seen_papers.add(candidate.paper_ref)
            elif candidate.grounding != "none" or candidate.paper_ref is not None:
                continue
            valid.append(candidate)
        if not valid:
            return InspirationResult(status="no_useful_inspiration", inspirations=[])
        return result.model_copy(update={"inspirations": valid, "status": "found"})

    def _build_need_input(
        self,
        active_gap,
        solution: CurrentSolution,
        memory_store: ResearchMemoryStore,
        *,
        seed_method_constraints: list[str],
        controller_exploration_rationale: str | None,
    ) -> dict[str, Any]:
        memory = memory_store.get_memory()
        related_context = self._compact_related_context(
            memory_store, active_gap.id, solution.id
        )
        return {
            "active_gap": active_gap.model_dump(mode="json"),
            "current_solution": _solution_summary(solution),
            "controller_exploration_rationale": controller_exploration_rationale,
            "seed_method_constraints": list(seed_method_constraints),
            "latest_relevant_verification": related_context[
                "latest_solution_verification"
            ],
            "rejected_mechanisms": self._rejected_mechanisms(memory_store),
            "previous_bridge_entity_names": self._previous_entity_names(memory),
            "related_work_context": related_context,
        }

    def _save_record(
        self,
        *,
        seed: SeedPaperContext,
        store: ResearchMemoryStore,
        need: InspirationNeed,
        proposed: list[BridgeEntity],
        selected: list[BridgeEntity],
        query: str,
        grounding_status: str,
        result: InspirationResult,
        paper_refs: list[str],
        step: int,
        search_record: InspirationSearchRecord | None,
        context,
        before_state: ResearchState | None,
    ) -> InspirationResult:
        if grounding_status not in {"arxiv", "none"}:
            raise ModuleContractError("invalid deterministic grounding_status")
        record = InspirationRecord(
            id=f"I{len(store.get_memory().inspiration_memory) + 1}",
            inspiration_need=need,
            proposed_entities=proposed,
            selected_entities=selected,
            search_query=query,
            grounding_status=grounding_status,
            inspirations=list(result.inspirations),
            paper_refs=paper_refs,
            created_at_step=step,
        )
        if search_record is not None:
            store.save_inspiration_search(search_record)
        store.save_inspiration(record)
        self.last_record = record
        self.last_search_record = search_record
        self.last_grounding_status = grounding_status
        self.last_next_required_action = (
            "TARGETED_REVISION" if result.status == "found" else None
        )
        if context is not None:
            assert before_state is not None
            writes = ["inspiration_memory"]
            if search_record is not None:
                writes = ["paper_memory", "inspiration_search_records", *writes]
            context.record_transition(
                step,
                "inspiration_retrieved",
                before_state,
                before_state,
                action="SEEK_INSPIRATION",
                result_ref=record.id,
                result_payload={
                    "inspiration_need": need.model_dump(mode="json"),
                    "proposed_entities": [
                        entity.model_dump(mode="json") for entity in proposed
                    ],
                    "selected_entities": [
                        entity.model_dump(mode="json") for entity in selected
                    ],
                    "search_query": query,
                    "grounding_status": grounding_status,
                    "result": result.model_dump(mode="json"),
                },
                memory_writes=writes,
            )
        return result

    def retrieve(
        self,
        active_gap=None,
        current_solution: CurrentSolution | None = None,
        seed_context: SeedPaperContext | None = None,
        *,
        context=None,
        state_store: ResearchStateStore | None = None,
        memory_store: ResearchMemoryStore | None = None,
        step: int = 1,
        allow_external_search: bool = True,
        controller_exploration_rationale: str | None = None,
    ) -> InspirationResult:
        """Execute SEEK_INSPIRATION without changing Research State."""

        if context is not None:
            seed, store = self._resolve_seed(seed_context, context, memory_store)
            state = context.get_state()
        else:
            if seed_context is None:
                raise TypeError("retrieve requires SeedPaperContext or context")
            seed, store = self._resolve_seed(seed_context, None, memory_store)
            state = state_store.get_state() if state_store is not None else None

        if state is not None:
            state_gap = state.active_gap
            state_solution = state.current_solution
            if state_gap is None or state_gap.status != "active":
                raise PreconditionError(
                    "Inspiration Retrieval requires an active Research State gap"
                )
            if state_solution is None:
                raise PreconditionError(
                    "Inspiration Retrieval requires a Current Solution in Research State"
                )
            if active_gap is None:
                active_gap = state_gap
            elif active_gap.id != state_gap.id:
                raise PreconditionError(
                    "Inspiration Retrieval gap does not match Research State"
                )
            if current_solution is None:
                current_solution = state_solution
            elif current_solution.id != state_solution.id:
                raise PreconditionError(
                    "Inspiration Retrieval solution does not match Research State"
                )
        if active_gap is None or current_solution is None:
            raise PreconditionError(
                "Inspiration Retrieval requires active_gap and current_solution"
            )
        if active_gap.status != "active":
            raise PreconditionError("Inspiration Retrieval requires an active gap")
        solution = CurrentSolution.model_validate(
            current_solution.model_dump(mode="python")
        )
        before_state = _copy_state(state) if state is not None else None
        call1_input = self._build_need_input(
            active_gap,
            solution,
            store,
            seed_method_constraints=list(seed.method_constraints),
            controller_exploration_rationale=controller_exploration_rationale,
        )
        response = self._complete(
            system_prompt=self._entity_prompt,
            user_content=json.dumps(call1_input, ensure_ascii=False, indent=2),
            task="inspiration_entities",
            paper_id=seed.paper_id,
            output_schema=BridgeGenerationDraft,
        )
        draft = response.output
        if not isinstance(draft, BridgeGenerationDraft):
            raise ModuleContractError("Inspiration entity response has the wrong schema")
        previous_names = self._previous_entity_names(store.get_memory())
        self._validate_draft(draft, previous_names)
        authoritative_rejected_mechanisms = self._rejected_mechanisms(store)
        need = draft.inspiration_need.model_copy(
            update={"rejected_mechanisms": authoritative_rejected_mechanisms}
        )
        proposed = list(draft.proposed_entities)
        selected = list(draft.selected_entities)
        query = " OR ".join(entity.name for entity in selected)
        self.last_cache_hit = False
        self.last_cache_scope = "none"
        self.last_provider_error = None
        self.last_search_record = None
        self.last_provider_request_count = 0
        self.last_provider_retry_count = 0

        if not selected:
            result = InspirationResult(status="no_useful_inspiration", inspirations=[])
            return self._save_record(
                seed=seed,
                store=store,
                need=need,
                proposed=proposed,
                selected=selected,
                query=query,
                grounding_status="none",
                result=result,
                paper_refs=[],
                step=step,
                search_record=None,
                context=context,
                before_state=before_state,
            )

        memory = store.get_memory()
        cached = self._cache_lookup(memory, query)
        retrieved: list[PaperRecord] = []
        search_record: InspirationSearchRecord | None = None
        new_search_record: InspirationSearchRecord | None = None
        grounded = False
        if cached is not None:
            search_record, retrieved = cached
            self.last_cache_hit = True
            self.last_cache_scope = "trajectory"
            self.last_search_record = search_record
            grounded = bool(retrieved)
        elif allow_external_search:
            self.scientific_search_count += 1
            try:
                retrieved = _deduplicate_papers(
                    list(self.search_provider.search(query, limit=20))
                )
                self.last_provider_request_count = getattr(
                    self.search_provider, "last_request_count", 0
                )
                self.last_provider_retry_count = getattr(
                    self.search_provider, "last_retry_count", 0
                )
                grounded = True
                search_record = InspirationSearchRecord(
                    query=query,
                    bridge_entities=selected,
                    paper_refs=[paper.paper_id for paper in retrieved],
                )
                new_search_record = search_record
                grounded = bool(retrieved)
            except RetrievalError as exc:
                self.last_provider_error = str(exc)
                self.last_provider_request_count = getattr(
                    self.search_provider, "last_request_count", 0
                )
                self.last_provider_retry_count = getattr(
                    self.search_provider, "last_retry_count", 0
                )
                retrieved = []
                search_record = None
        # Cache hits and successful searches use one Call 2.  Disabled or
        # unavailable grounding also uses one Call 2 in explicit fallback mode.
        call2_input = {
            "grounding_mode": "ARXIV GROUNDED" if grounded else "LLM-ONLY FALLBACK",
            "inspiration_need": need.model_dump(mode="json"),
            "selected_bridge_entities": [
                entity.model_dump(mode="json") for entity in selected
            ],
            "current_solution": _solution_summary(solution),
            "rejected_mechanisms": self._rejected_mechanisms(store),
            "retrieved_papers": [_paper_payload(paper) for paper in retrieved],
            "grounding_unavailable_reason": self.last_provider_error,
        }
        response = self._complete(
            system_prompt=self._extract_prompt,
            user_content=json.dumps(call2_input, ensure_ascii=False, indent=2),
            task="inspiration_extract",
            paper_id=seed.paper_id,
            output_schema=InspirationResult,
        )
        result = response.output
        if not isinstance(result, InspirationResult):
            raise ModuleContractError("Inspiration extraction response has the wrong schema")
        try:
            result = self._validate_result(
                result,
                grounded=grounded,
                papers=retrieved,
                selected_entities=selected,
            )
        except StructuredOutputError as exc:
            # The pilot-wide RecordingLLMClient handles schema/JSON failures.
            # This module-only retry covers the additional provenance contract
            # that is checked after a schema-valid result is returned.  It
            # reuses the exact retrieved pool and never repeats retrieval.
            retry_enabled = bool(getattr(self.llm_client, "format_retry_enabled", False))
            retry_used = getattr(response.metadata, "format_retry_count", 0) > 0
            if not retry_enabled or retry_used:
                raise
            try:
                retry_response = self._complete(
                    system_prompt=self._extract_prompt,
                    user_content=json.dumps(call2_input, ensure_ascii=False, indent=2),
                    task="inspiration_extract",
                    paper_id=seed.paper_id,
                    output_schema=InspirationResult,
                    format_retry_reason=str(exc),
                )
            except StructuredOutputError:
                result = InspirationResult(status="no_useful_inspiration", inspirations=[])
            else:
                retry_result = retry_response.output
                if not isinstance(retry_result, InspirationResult):
                    raise ModuleContractError(
                        "Inspiration extraction retry response has the wrong schema"
                    )
                try:
                    result = self._validate_result(
                        retry_result,
                        grounded=grounded,
                        papers=retrieved,
                        selected_entities=selected,
                    )
                except StructuredOutputError:
                    result = self._discard_invalid_provenance_candidates(
                        retry_result,
                        grounded=grounded,
                        papers=retrieved,
                        selected_entities=selected,
                    )
        grounding_status = "arxiv" if grounded else "none"
        if new_search_record is not None:
            # The successful search is committed only after Call 2 passes its
            # semantic contract; failed extraction must not create a success record.
            for paper in retrieved:
                store.save_paper(paper)
        return self._save_record(
            seed=seed,
            store=store,
            need=need,
            proposed=proposed,
            selected=selected,
            query=query,
            grounding_status=grounding_status,
            result=result,
            paper_refs=[paper.paper_id for paper in retrieved],
            step=step,
            search_record=new_search_record,
            context=context,
            before_state=before_state,
        )

    run = retrieve
    execute = retrieve
    seek_inspiration = retrieve


__all__ = ["InspirationRetrievalConfig", "InspirationRetrievalModule"]
