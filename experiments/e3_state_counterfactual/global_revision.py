"""E3-only whole-solution revision module.

This module is intentionally separate from the production TargetedRevision
module.  It performs one complete rewrite call and has no retrieval or
Controller behavior.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from research_agent.llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from research_agent.models.gap import CandidateGap
from research_agent.models.memory import FailureRecord, ResearchMemory, VerificationRecord
from research_agent.models.revision import RevisionNeed
from research_agent.models.seed import SeedPaperContext
from research_agent.models.state import CurrentSolution

from .models import GlobalRevisionDraft, materialize_global_solution


@dataclass(frozen=True)
class GlobalRevisionConfig:
    max_tokens: int = 4_000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GlobalRevisionConfig":
        values = os.environ if environ is None else environ
        raw = values.get("TARGETED_REVISION_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        value = int(raw)
        if value <= 0:
            raise ValueError("TARGETED_REVISION_MAX_TOKENS must be positive")
        return cls(max_tokens=value)


@dataclass(frozen=True)
class GlobalRevisionExecutionResult:
    draft: GlobalRevisionDraft
    revised_solution: CurrentSolution | None
    metadata: LLMCallMetadata
    input_chars: int
    call_count: int


def _prompt_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "global_revision.md"


def _prompt() -> str:
    template = _prompt_path().read_text(encoding="utf-8")
    schema = json.dumps(
        GlobalRevisionDraft.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.replace("{{E3_GLOBAL_REVISION_DRAFT_JSON_SCHEMA}}", schema)


def _model_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _verification_payload(value: Any) -> Any:
    if isinstance(value, VerificationRecord):
        return value.model_dump(mode="json")
    return _model_payload(value)


def _failure_payload(value: Any) -> Any:
    if isinstance(value, FailureRecord):
        return value.model_dump(mode="json")
    return _model_payload(value)


class GlobalRevisionModule:
    """One-call E3 complete-solution rewrite."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: GlobalRevisionConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or GlobalRevisionConfig.from_env()
        self.system_prompt = _prompt()
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None

    def revise(
        self,
        *,
        seed_context: SeedPaperContext,
        active_gap: CandidateGap,
        current_solution: CurrentSolution,
        revision_need: RevisionNeed | Mapping[str, Any],
        verification_evidence: VerificationRecord | Mapping[str, Any] | None,
        selected_inspiration: Mapping[str, Any] | None,
        relevant_failure_memory: list[FailureRecord | Mapping[str, Any]],
    ) -> GlobalRevisionExecutionResult:
        user_payload = {
            "active_gap": _model_payload(active_gap),
            "current_solution": _model_payload(current_solution),
            "revision_need": _model_payload(revision_need),
            "verification_evidence": _verification_payload(verification_evidence),
            "selected_inspiration": selected_inspiration,
            "seed_method_constraints": list(seed_context.method_constraints),
            "relevant_failure_memory": [
                _failure_payload(record) for record in relevant_failure_memory
            ],
        }
        user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "e3_global_revision", "paper_id": seed_context.paper_id},
        )
        response: StructuredLLMResponse[Any] = self.llm_client.complete_structured(
            request, GlobalRevisionDraft
        )
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        draft = response.output
        if not isinstance(draft, GlobalRevisionDraft):
            raise TypeError("Global Revision response has the wrong schema")
        revised = (
            materialize_global_solution(current_solution, draft)
            if draft.status == "revised"
            else None
        )
        return GlobalRevisionExecutionResult(
            draft=draft,
            revised_solution=revised,
            metadata=response.metadata,
            input_chars=self.last_input_chars,
            call_count=self.call_count,
        )


__all__ = [
    "GlobalRevisionConfig",
    "GlobalRevisionExecutionResult",
    "GlobalRevisionModule",
]
