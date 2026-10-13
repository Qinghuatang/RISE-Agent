"""Independent Flat proposal runner for the state-evolution ablation.

The Flat branch keeps only a generic proposal, evidence history, action
history, and remaining budget.  Existing RISE verification/inspiration modules
are called through short-lived adapters; their typed objects are never written
as Flat state or used for Flat routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from research_agent.errors import ModuleContractError
from research_agent.ingestion.markdown import FullPaperReadingContext
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.llm.recording import RecordingLLMClient, aggregate_token_usage
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models.gap import CandidateGap
from research_agent.models.seed import ExperimentContext, SeedPaperContext
from research_agent.models.state import Component, CurrentSolution
from research_agent.models.verification import VerificationFocus
from research_agent.modules.inspiration_retrieval import InspirationRetrievalModule
from research_agent.modules.related_work_verification import RelatedWorkVerificationModule
from research_agent.modules.seed_context import SeedContextExtractor
from research_agent.retrieval.arxiv import ArxivConfig, ArxivInspirationSearchProvider
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig, SemanticScholarProvider
from research_agent.serialization import write_json

from .flat_controller import FlatBudgetView, FlatController
from .flat_prompts import EXPERIMENT_PROMPT, PROPOSAL_PROMPT, FlatProposal, render, schema_json


@dataclass
class FlatExecutionState:
    """The complete persisted Flat state, intentionally not a RISE State."""

    current_proposal: dict[str, str] | None = None
    evidence_history: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: dict[str, int] = field(
        default_factory=lambda: {"steps_remaining": 0, "searches_remaining": 0}
    )


@dataclass(frozen=True)
class FlatRunConfig:
    markdown_path: Path
    output_dir: Path
    semantic_scholar_config: Path
    max_steps: int
    max_searches: int
    allow_real_services: bool = False
    allow_overwrite: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()[:1000] or type(exc).__name__


def _plain_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plain_jsonl_write(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _proposal_dict(proposal: FlatProposal | dict[str, str] | None) -> dict[str, str] | None:
    if proposal is None:
        return None
    if isinstance(proposal, FlatProposal):
        return proposal.model_dump(mode="json")
    return dict(proposal)


def _ephemeral_gap(proposal: FlatProposal) -> CandidateGap:
    return CandidateGap(
        id="flat-problem",
        statement=proposal.research_problem,
        confidence="medium",
        confidence_reason="Transient adapter contract for the Flat verifier.",
        priority="medium",
        priority_reason="Transient adapter contract for the Flat verifier.",
        status="active",
        evidence_refs=[],
    )


def _ephemeral_solution(proposal: FlatProposal) -> CurrentSolution:
    # These objects exist only at an adapter boundary for the existing verifier.
    # They are never serialized as Flat state or supplied to the Flat controller.
    return CurrentSolution(
        id="flat-proposal",
        core_idea=proposal.proposed_method,
        rationale=proposal.research_problem,
        components=[
            Component(id="proposal_mechanism", description=proposal.proposed_method, status="ready"),
            Component(id="proposal_evaluation", description=proposal.experimental_plan, status="ready"),
        ],
        status="developed",
        novelty_status="unchecked",
    )


class FlatRunner:
    """Run one Flat branch with the same high-level budget as Full."""

    def __init__(self, config: FlatRunConfig) -> None:
        self.config = config
        self.llm: RecordingLLMClient | None = None
        self.controller: FlatController | None = None
        self._started_at = _now()
        self._related_pool = None
        self._related_provider: SemanticScholarProvider | None = None
        self._arxiv_provider: ArxivInspirationSearchProvider | None = None
        self._flat_state: FlatExecutionState | None = None

    def _preflight(self) -> tuple[DeepSeekConfig, SemanticScholarConfig, ArxivConfig]:
        if not self.config.allow_real_services:
            raise ModuleContractError("Flat runner requires --allow-real-services")
        if not self.config.markdown_path.is_file():
            raise FileNotFoundError(f"Markdown input does not exist: {self.config.markdown_path}")
        if self.config.max_steps <= 0 or self.config.max_searches < 0:
            raise ValueError("Flat budgets must be positive/non-negative")
        if self.config.output_dir.exists() and not self.config.allow_overwrite:
            raise FileExistsError(f"Flat output already exists: {self.config.output_dir}")
        deepseek = DeepSeekConfig.from_env()
        deepseek.validate_for_request()
        semantic = SemanticScholarConfig.from_file(self.config.semantic_scholar_config)
        semantic.validate_for_request()
        arxiv = ArxivConfig.from_env()
        return deepseek, semantic, arxiv

    def _write_llm_audit(self) -> None:
        if self.llm is None:
            _plain_jsonl_write(self.config.output_dir / "llm_calls.jsonl", [])
            return
        _plain_jsonl_write(
            self.config.output_dir / "llm_calls.jsonl",
            [record.model_dump(mode="json") for record in self.llm.records],
        )

    def _write_flat_histories(self, state: FlatExecutionState) -> None:
        _plain_write(self.config.output_dir / "evidence_history.json", state.evidence_history)
        _plain_write(self.config.output_dir / "action_history.json", state.action_history)

    def _write_metadata(self, *, status: str, error: Exception | None = None) -> dict[str, Any]:
        records = self.llm.records if self.llm is not None else []
        usage = aggregate_token_usage(records)
        state = self._flat_state
        deepseek_config = getattr(getattr(self.llm, "inner", None), "config", None)
        metadata: dict[str, Any] = {
            "seed_id": self.config.markdown_path.stem,
            "system": "flat",
            "status": status,
            "source_md": str(self.config.markdown_path.resolve()),
            "max_steps": self.config.max_steps,
            "actual_steps": len(state.action_history) if state else 0,
            "max_searches": self.config.max_searches,
            "actual_searches": max(0, self.config.max_searches - (state.remaining_budget["searches_remaining"] if state else self.config.max_searches)),
            "llm_calls": len(records),
            "prompt_tokens": usage["prompt_tokens_total"],
            "completion_tokens": usage["completion_tokens_total"],
            "total_tokens": usage["total_tokens"],
            "known_prompt_tokens": usage["known_prompt_tokens"],
            "known_completion_tokens": usage["known_completion_tokens"],
            "known_total_tokens": usage["known_total_tokens"],
            "calls_with_token_usage": usage["calls_with_token_usage"],
            "calls_missing_token_usage": usage["calls_missing_token_usage"],
            "token_metadata_complete": usage["token_metadata_complete"],
            "start_time": self._started_at,
            "end_time": _now(),
            "failure": _safe_error(error) if error else None,
            "provider": "deepseek",
            "model": getattr(deepseek_config, "model", None),
            "temperature": 0.0,
            "output_limit_mode": "unbounded_like_current_full_runner",
            "flat_state_fields": [
                "current_proposal",
                "evidence_history",
                "action_history",
                "remaining_budget",
            ],
        }
        _plain_write(self.config.output_dir / "run_metadata.json", metadata)
        return metadata

    def _save_provider_audit(self) -> None:
        semantic = self._related_provider
        arxiv = self._arxiv_provider
        _plain_write(
            self.config.output_dir / "provider_audit.json",
            {
                "semantic_scholar": {
                    "http_requests": getattr(semantic, "total_request_count", 0),
                    "cache_hit": getattr(semantic, "last_global_cache_hit", False),
                },
                "arxiv": {
                    "http_requests": getattr(arxiv, "total_request_count", 0),
                    "cache_hit": getattr(arxiv, "last_cache_hit", False),
                },
            },
        )

    def _call_proposal(
        self,
        *,
        reading: FullPaperReadingContext,
        state: FlatExecutionState,
        experiment_mode: bool,
    ) -> FlatProposal:
        assert self.llm is not None
        template = EXPERIMENT_PROMPT if experiment_mode else PROPOSAL_PROMPT
        system = template.format(schema=schema_json(FlatProposal), paper=reading.render(), proposal=render(state.current_proposal), evidence=render(state.evidence_history))
        from research_agent.llm.client import LLMMessage, LLMRequest

        request = LLMRequest(
            messages=(LLMMessage(role="system", content=system),),
            temperature=0.0,
            max_tokens=4_000,
            metadata={"task": "flat_experiment_design" if experiment_mode else "flat_proposal", "paper_id": reading.paper_id},
        )
        response = self.llm.complete_structured(request, FlatProposal)
        proposal = response.output
        if not isinstance(proposal, FlatProposal):
            raise ModuleContractError("Flat proposal response has the wrong schema")
        return proposal

    def _record_action(
        self,
        *,
        state: FlatExecutionState,
        step: int,
        action: str,
        reason: str,
        before: dict[str, str] | None,
        after: dict[str, str] | None,
        tool_input: dict[str, Any] | None,
        tool_output: dict[str, Any] | None,
        evidence: dict[str, Any] | None,
        status: str,
    ) -> None:
        state.action_history.append(
            {
                "step": step,
                "action": action,
                "proposal_before": before,
                "controller_reason": reason,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "evidence": evidence,
                "proposal_after": after,
                "remaining_budget": dict(state.remaining_budget),
                "token_metadata": self._last_token_metadata(),
                "timestamp": _now(),
                "status": status,
            }
        )

    def _last_token_metadata(self) -> dict[str, Any] | None:
        if self.llm is None or not self.llm.records:
            return None
        return self.llm.records[-1].model_dump(mode="json")

    @staticmethod
    def _allowed(state: FlatExecutionState) -> list[str]:
        if state.current_proposal is None:
            return ["REFINE_PROPOSAL"]
        actions = ["VERIFY", "REFINE_PROPOSAL", "DESIGN_EXPERIMENTS", "COMMIT"]
        if state.remaining_budget["searches_remaining"] > 0:
            actions.insert(1, "SEEK_INSPIRATION")
        return actions

    def _run_tool(
        self,
        *,
        action: str,
        seed_context: SeedPaperContext,
        state: FlatExecutionState,
        step: int,
    ) -> tuple[dict[str, str] | None, dict[str, Any] | None, dict[str, Any] | None, int]:
        if action in {"REFINE_PROPOSAL", "DESIGN_EXPERIMENTS"}:
            reading = self._reading
            updated = self._call_proposal(
                reading=reading,
                state=state,
                experiment_mode=action == "DESIGN_EXPERIMENTS",
            )
            return _proposal_dict(updated), {"operation": action.lower()}, None, 0

        # The first Flat step is intentionally a REFINE_PROPOSAL action with
        # no existing proposal.  Validate only actions that consume an
        # already-materialized proposal.
        proposal = FlatProposal.model_validate(state.current_proposal)
        gap = _ephemeral_gap(proposal)
        solution = _ephemeral_solution(proposal)
        transient_memory = ResearchMemoryStore(seed_context.paper_id)
        if action == "VERIFY":
            assert self._related_provider is not None
            verifier = RelatedWorkVerificationModule(self.llm, self._related_provider)
            if self._related_pool is None:
                self._related_pool = verifier.get_related_paper_pool(
                    seed_context, memory_store=transient_memory, step=step
                )
            result = verifier.verify_solution(
                gap,
                current_solution=solution,
                seed_context=seed_context,
                related_paper_pool=self._related_pool,
                verification_focus=VerificationFocus(mode="full_solution"),
                memory_store=transient_memory,
                step=step,
            )
            evidence = {
                "kind": "related_work_verification",
                "outcome": result.novelty_status,
                "summary": result.summary,
                "related_papers": list(result.related_papers),
                "novelty_relevant_overlaps": [item.model_dump(mode="json") for item in result.novelty_relevant_overlaps],
            }
            return None, {"operation": "verify", "pool_size": len(self._related_pool.papers)}, evidence, verifier.scientific_search_count

        assert action == "SEEK_INSPIRATION"
        assert self._arxiv_provider is not None
        inspiration = InspirationRetrievalModule(self.llm, self._arxiv_provider)
        result = inspiration.retrieve(
            active_gap=gap,
            current_solution=solution,
            seed_context=seed_context,
            memory_store=transient_memory,
            step=step,
            allow_external_search=True,
        )
        evidence = {
            "kind": "inspiration",
            "status": result.status,
            "inspirations": [item.model_dump(mode="json") for item in result.inspirations],
        }
        return None, {"operation": "seek_inspiration"}, evidence, inspiration.scientific_search_count

    def run(self) -> dict[str, Any]:
        deepseek, semantic_config, arxiv_config = self._preflight()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = RecordingLLMClient(DeepSeekClient(deepseek), unbounded_output=True)
        self.controller = FlatController(self.llm)
        self._related_provider = SemanticScholarProvider(semantic_config)
        self._arxiv_provider = ArxivInspirationSearchProvider(arxiv_config)
        extractor = SeedContextExtractor(self.llm)
        extraction, self._reading = extractor.extract_file_with_reading(self.config.markdown_path)
        seed_context = extraction.context
        write_json(self.config.output_dir / "seed_context.json", "seed_paper_context", seed_context)
        write_json(self.config.output_dir / "seed_extraction_audit.json", "seed_extraction_audit", extraction.audit)
        state = FlatExecutionState(
            remaining_budget={"steps_remaining": self.config.max_steps, "searches_remaining": self.config.max_searches}
        )
        self._flat_state = state
        status = "stopped"
        try:
            while state.remaining_budget["steps_remaining"] > 0:
                step = self.config.max_steps - state.remaining_budget["steps_remaining"] + 1
                allowed = self._allowed(state)
                decision = self.controller.choose(
                    seed_paper=self._reading.render(),
                    paper_id=seed_context.paper_id,
                    current_proposal=state.current_proposal,
                    evidence_history=state.evidence_history,
                    action_history=state.action_history,
                    budget=FlatBudgetView(**state.remaining_budget),
                    allowed_actions=allowed,
                )
                action = decision.action
                before = _proposal_dict(state.current_proposal)
                after: dict[str, str] | None = before
                evidence: dict[str, Any] | None = None
                tool_input = {"proposal": before, "evidence_count": len(state.evidence_history)}
                tool_output: dict[str, Any] | None = None
                search_delta = 0
                if action == "COMMIT":
                    self._record_action(state=state, step=step, action=action, reason=decision.reason, before=before, after=before, tool_input=None, tool_output=None, evidence=None, status="committed")
                    state.remaining_budget["steps_remaining"] -= 1
                    status = "committed"
                    break
                after, tool_output, evidence, search_delta = self._run_tool(
                    action=action, seed_context=seed_context, state=state, step=step
                )
                if after is not None:
                    state.current_proposal = after
                if evidence is not None:
                    state.evidence_history.append(evidence)
                if search_delta > state.remaining_budget["searches_remaining"]:
                    raise ModuleContractError("Flat tool exceeded remaining search budget")
                state.remaining_budget["searches_remaining"] -= search_delta
                state.remaining_budget["steps_remaining"] -= 1
                self._record_action(state=state, step=step, action=action, reason=decision.reason, before=before, after=_proposal_dict(state.current_proposal), tool_input=tool_input, tool_output=tool_output, evidence=evidence, status="completed")
            else:
                status = "stopped"
            if state.current_proposal is not None:
                _plain_write(self.config.output_dir / "final_proposal.json", state.current_proposal)
                proposal = state.current_proposal
                (self.config.output_dir / "final_proposal.md").write_text(
                    "# Research Problem\n\n"
                    + proposal["research_problem"]
                    + "\n\n# Proposed Method\n\n"
                    + proposal["proposed_method"]
                    + "\n\n# Experimental Plan\n\n"
                    + proposal["experimental_plan"]
                    + "\n",
                    encoding="utf-8",
                )
            self._write_flat_histories(state)
            self._write_llm_audit()
            self._save_provider_audit()
            return self._write_metadata(status=status)
        except Exception as exc:
            self._write_flat_histories(state)
            self._write_llm_audit()
            self._save_provider_audit()
            self._write_metadata(status="failed", error=exc)
            _plain_write(self.config.output_dir / "error.json", {"type": type(exc).__name__, "message": _safe_error(exc)})
            raise


__all__ = ["FlatExecutionState", "FlatRunConfig", "FlatRunner"]
