"""Execution-only contracts for offline trajectory runs and replay."""

from typing import Any, Literal

from pydantic import Field, model_validator

from ..controller.budget import ControllerBudget
from .base import NonEmptyStr, ScientificAction, StrictModel
from .commit import CommitSnapshot, FinalResearchProposal
from .state import ResearchState


REPLAY_MEMORY_CATEGORIES = frozenset(
    {
        "paper_memory",
        "verification_memory",
        "inspiration_memory",
        "revision_memory",
        "failure_memory",
        "decision_log",
    }
)


class TrajectoryReplayRecord(StrictModel):
    """Canonical-after-value journal entry used only for engineering replay."""

    seed_paper_id: NonEmptyStr
    sequence: int = Field(ge=1)
    step: int = Field(ge=0)
    kind: Literal["bootstrap", "action", "internal"]
    action: ScientificAction | None = None
    state_after: ResearchState
    memory_patch: dict[str, Any] = Field(default_factory=dict)
    budget_after: ControllerBudget

    @model_validator(mode="after")
    def validate_memory_categories(self) -> "TrajectoryReplayRecord":
        unknown = set(self.memory_patch) - REPLAY_MEMORY_CATEGORIES
        if unknown:
            raise ValueError(
                "memory_patch contains unsupported categories: "
                + ", ".join(sorted(unknown))
            )
        if self.kind == "action" and self.action is None:
            raise ValueError("action replay records require an action")
        if self.kind != "action" and self.action is not None:
            raise ValueError("bootstrap/internal replay records cannot carry an action")
        return self


class RunBudget(StrictModel):
    max_steps: int = Field(ge=0)
    max_searches: int = Field(ge=0)
    steps_used: int = Field(ge=0)
    searches_used: int = Field(ge=0)
    remaining_steps: int = Field(ge=0)
    remaining_searches: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> "RunBudget":
        if self.steps_used > self.max_steps or self.searches_used > self.max_searches:
            raise ValueError("run budget usage cannot exceed its maximum")
        if self.remaining_steps != self.max_steps - self.steps_used:
            raise ValueError("remaining_steps does not match the budget totals")
        if self.remaining_searches != self.max_searches - self.searches_used:
            raise ValueError("remaining_searches does not match the budget totals")
        return self


class RunManifest(StrictModel):
    """Runner artifact; never part of Research State or Research Memory."""

    paper_id: NonEmptyStr
    status: Literal["committed", "stopped", "failed", "finalization_failed"]
    budget: RunBudget
    trajectory_event_count: int = Field(ge=0)
    replay_record_count: int = Field(ge=0)
    commit_gate_passed: bool = False
    commit_snapshot_written: bool = False
    final_proposal_written: bool = False
    unresolved_issues: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class TrajectoryRunResult(StrictModel):
    """Execution result returned by the offline runner."""

    status: Literal["committed", "stopped", "failed", "finalization_failed"]
    unresolved_issues: list[str] = Field(default_factory=list)
    final_state: ResearchState
    budget: ControllerBudget
    commit_snapshot: CommitSnapshot | None = None
    final_proposal: FinalResearchProposal | None = None
    error_type: str | None = None
    error_message: str | None = None
    output_dir: str | None = None


__all__ = [
    "REPLAY_MEMORY_CATEGORIES",
    "RunBudget",
    "RunManifest",
    "TrajectoryReplayRecord",
    "TrajectoryRunResult",
]
