"""Closed contracts used only by the E3 experiment harness.

These contracts deliberately do not extend the production Research State or
Research Memory.  They describe experiment artifacts and the complete
solution returned by the E3 global-rewrite branch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from research_agent.controller.budget import ControllerBudget
from research_agent.models.base import NonEmptyStr, StrictModel
from research_agent.models.memory import ResearchMemory
from research_agent.models.state import Component, CurrentSolution, ResearchState
from research_agent.models.trajectory import TrajectoryEvent


class E3BudgetSnapshot(StrictModel):
    max_steps: int = Field(ge=0)
    max_searches: int = Field(ge=0)
    steps_used: int = Field(ge=0)
    searches_used: int = Field(ge=0)
    remaining_steps: int = Field(ge=0)
    remaining_searches: int = Field(ge=0)

    @classmethod
    def from_budget(cls, budget: ControllerBudget) -> "E3BudgetSnapshot":
        return cls(
            max_steps=budget.max_steps,
            max_searches=budget.max_searches,
            steps_used=budget.steps_used,
            searches_used=budget.searches_used,
            remaining_steps=budget.remaining_steps,
            remaining_searches=budget.remaining_searches,
        )


class GlobalRevisionComponentDraft(StrictModel):
    id: NonEmptyStr
    description: NonEmptyStr


class GlobalRevisionDraft(StrictModel):
    """One complete solution rewrite returned by the E3-only LLM call."""

    status: Literal["revised", "unable_to_revise"]
    core_idea: str | None = None
    rationale: str | None = None
    components: list[GlobalRevisionComponentDraft] = Field(default_factory=list)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_complete_rewrite(self) -> "GlobalRevisionDraft":
        if self.status == "revised":
            if not self.core_idea or not self.core_idea.strip():
                raise ValueError("revised global solution requires core_idea")
            if not self.rationale or not self.rationale.strip():
                raise ValueError("revised global solution requires rationale")
            if not 2 <= len(self.components) <= 5:
                raise ValueError("revised global solution requires 2-5 components")
            ids = [component.id for component in self.components]
            if len(set(ids)) != len(ids):
                raise ValueError("global solution component ids must be unique")
            if self.failure_reason is not None:
                raise ValueError("revised global solution cannot provide failure_reason")
        else:
            if self.core_idea is not None or self.rationale is not None or self.components:
                raise ValueError("unable_to_revise cannot contain a solution rewrite")
            if not self.failure_reason or not self.failure_reason.strip():
                raise ValueError("unable_to_revise requires failure_reason")
        return self


class E3Checkpoint(StrictModel):
    paper_id: NonEmptyStr
    checkpoint_step: int = Field(ge=0)
    replay_sequence: int = Field(ge=1)
    state: ResearchState
    memory: ResearchMemory
    budget: E3BudgetSnapshot
    trajectory_prefix: list[TrajectoryEvent] = Field(default_factory=list)
    preceding_action: str
    trigger_type: Literal["overlap_repair", "inspiration_strengthening"]


class RevisionCase(StrictModel):
    paper_id: NonEmptyStr
    source_dir: NonEmptyStr
    first_revision_step: int = Field(ge=0)
    first_revision_replay_sequence: int = Field(ge=1)
    trigger_type: Literal["overlap_repair", "inspiration_strengthening"] | None = None
    checkpoint: E3Checkpoint | None = None
    valid: bool
    validation_errors: list[str] = Field(default_factory=list)


class HistoricalLocalBranch(StrictModel):
    paper_id: NonEmptyStr
    trigger_type: Literal["overlap_repair", "inspiration_strengthening"]
    checkpoint_step: int = Field(ge=0)
    solution_before: CurrentSolution
    verification_before: dict[str, Any] | None = None
    inspiration_before: dict[str, Any] | None = None
    selected_inspiration_ref: str | None = None
    affected_units: list[str] = Field(default_factory=list)
    historical_revision_result: dict[str, Any]
    historical_changed_units: list[str] = Field(default_factory=list)
    historical_verification_focus: dict[str, Any] | None = None
    historical_post_revision_outcome: dict[str, Any] | None = None


class ValidationResult(StrictModel):
    paper_id: NonEmptyStr
    valid: bool
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)


def materialize_global_solution(
    current_solution: CurrentSolution,
    draft: GlobalRevisionDraft,
) -> CurrentSolution:
    """Convert a complete E3 rewrite to the unchanged production State shape."""

    if draft.status != "revised":
        raise ValueError("cannot materialize an unsuccessful global revision")
    return CurrentSolution(
        id=current_solution.id,
        core_idea=draft.core_idea or "",
        rationale=draft.rationale or "",
        components=[
            Component(id=item.id, description=item.description, status="modified")
            for item in draft.components
        ],
        status="developed",
        novelty_status="unchecked",
        novelty_summary=None,
        evidence_refs=[],
    )


__all__ = [
    "E3BudgetSnapshot",
    "E3Checkpoint",
    "GlobalRevisionComponentDraft",
    "GlobalRevisionDraft",
    "HistoricalLocalBranch",
    "RevisionCase",
    "ValidationResult",
    "materialize_global_solution",
]
