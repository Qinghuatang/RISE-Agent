"""Flat controller used only by the state-evolution ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from typing import Any, Sequence

from research_agent.llm.client import LLMClient, LLMMessage, LLMRequest
from research_agent.errors import ModuleContractError

from .flat_prompts import CONTROLLER_PROMPT, FLAT_ACTIONS, FlatControllerDecision, schema_json


@dataclass(frozen=True)
class FlatBudgetView:
    steps_remaining: int
    searches_remaining: int


@dataclass(frozen=True)
class FlatControllerConfig:
    max_tokens: int = 1_000

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("Flat controller max_tokens must be a positive integer")

    @classmethod
    def from_env(cls) -> "FlatControllerConfig":
        raw = os.environ.get("CONTROLLER_MAX_TOKENS", "").strip()
        return cls(max_tokens=int(raw)) if raw else cls()


class FlatController:
    """Select a flat action without constructing a RISE ResearchState.

    The only deterministic routing rule is the ordinary one-action case.  If
    multiple actions are legal, the controller LLM chooses among that mask.
    """

    def __init__(self, llm_client: LLMClient, config: FlatControllerConfig | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or FlatControllerConfig.from_env()
        self.prompt = CONTROLLER_PROMPT.format(schema=schema_json(FlatControllerDecision))
        self.call_count = 0
        self.last_call_metadata = None

    def choose(
        self,
        *,
        seed_paper: str,
        paper_id: str | None = None,
        current_proposal: dict[str, str] | None,
        evidence_history: Sequence[dict[str, Any]],
        action_history: Sequence[dict[str, Any]],
        budget: FlatBudgetView,
        allowed_actions: Sequence[str],
    ) -> FlatControllerDecision:
        allowed = tuple(dict.fromkeys(allowed_actions))
        if not allowed:
            raise ModuleContractError("Flat Controller received an empty action mask")
        invalid = set(allowed) - set(FLAT_ACTIONS)
        if invalid:
            raise ModuleContractError(f"Flat Controller received invalid actions: {sorted(invalid)}")
        if len(allowed) == 1:
            return FlatControllerDecision(action=allowed[0], reason="Only one legal flat action remains.")

        user_content = json.dumps(
            {
                "allowed_actions": list(allowed),
                "seed_paper": seed_paper,
                "current_proposal": current_proposal,
                "evidence_history": list(evidence_history),
                "action_history": list(action_history),
                "remaining_budget": {
                    "steps_remaining": budget.steps_remaining,
                    "searches_remaining": budget.searches_remaining,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=self.prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "flat_controller", "paper_id": paper_id},
        )
        response = self.llm_client.complete_structured(request, FlatControllerDecision)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        decision = response.output
        if not isinstance(decision, FlatControllerDecision):
            raise ModuleContractError("Flat Controller returned the wrong schema")
        if decision.action not in allowed:
            raise ModuleContractError(
                f"Flat Controller selected {decision.action!r}, outside allowed actions {list(allowed)!r}"
            )
        return decision


__all__ = ["FlatBudgetView", "FlatController", "FlatControllerConfig"]
