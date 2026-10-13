"""One-call, read-only materialization of a committed research proposal."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from ..errors import ModuleContractError, SerializationError
from ..llm.client import LLMCallMetadata, LLMClient, LLMMessage, LLMRequest
from ..models.commit import CommitSnapshot, FinalResearchProposal, FinalizerDraft
from ..models.seed import SeedPaperContext
from ..serialization import read_json, write_json


FINAL_PROPOSAL_SCHEMA = "final_research_proposal"


@dataclass(frozen=True)
class FinalizerConfig:
    max_tokens: int = 2_000

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "FinalizerConfig":
        values = os.environ if environ is None else environ
        raw = values.get("FINALIZER_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError("FINALIZER_MAX_TOKENS must be an integer") from exc


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "finalizer.md"


def _inject_schema() -> str:
    template = _prompt_path().read_text(encoding="utf-8")
    schema = json.dumps(
        FinalizerDraft.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.replace("{{FINALIZER_DRAFT_JSON_SCHEMA}}", schema)


def _seed_summary(seed_context: SeedPaperContext) -> dict[str, Any]:
    return {
        "research_objective": seed_context.research_objective,
        "core_method": seed_context.core_method,
        "main_findings": list(seed_context.main_findings),
        "limitations": list(seed_context.limitations),
    }


def save_final_research_proposal(
    proposal: FinalResearchProposal,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, FINAL_PROPOSAL_SCHEMA, proposal)
    return target


def load_final_research_proposal(path: str | Path) -> FinalResearchProposal:
    try:
        envelope = read_json(path, expected_schema=FINAL_PROPOSAL_SCHEMA)
        return FinalResearchProposal.model_validate(envelope["payload"])
    except (SerializationError, ValidationError) as exc:
        raise SerializationError("stored FinalResearchProposal is invalid") from exc


class FinalizerModule:
    """Finalizer with one structured call and deterministic structural merge."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: FinalizerConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or FinalizerConfig.from_env()
        self.prompt = _inject_schema()
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None

    def finalize(
        self,
        snapshot: CommitSnapshot,
        seed_context: SeedPaperContext,
    ) -> FinalResearchProposal:
        committed = CommitSnapshot.model_validate(snapshot.model_dump(mode="python"))
        seed = SeedPaperContext.model_validate(seed_context.model_dump(mode="python"))
        user_payload = {
            "commit_snapshot": committed.model_dump(mode="json"),
            "seed_summary": _seed_summary(seed),
            "relevant_evidence_refs": list(committed.relevant_evidence_refs),
            "novelty_context": committed.latest_solution_verification.model_dump(mode="json"),
        }
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=self.prompt),
                LLMMessage(
                    role="user",
                    content=json.dumps(user_payload, ensure_ascii=False, indent=2),
                ),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "finalizer", "paper_id": seed.paper_id},
        )
        response = self.llm_client.complete_structured(request, FinalizerDraft)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        draft = response.output
        if not isinstance(draft, FinalizerDraft):
            raise ModuleContractError("Finalizer response has the wrong schema")
        return FinalResearchProposal(
            research_gap=committed.active_gap.statement,
            proposed_solution=committed.current_solution.core_idea,
            solution_rationale=committed.current_solution.rationale,
            method_components=[
                component.model_copy(deep=True)
                for component in committed.current_solution.components
            ],
            novelty_positioning=draft.novelty_positioning,
            experimental_plan=[
                experiment.model_copy(deep=True) for experiment in committed.experiments
            ],
            expected_contributions=list(draft.expected_contributions),
        )

    run = finalize
    execute = finalize


__all__ = [
    "FINAL_PROPOSAL_SCHEMA",
    "FinalizerConfig",
    "FinalizerModule",
    "load_final_research_proposal",
    "save_final_research_proposal",
]
