"""Commit Gate, immutable snapshot, and final proposal contracts."""

from pydantic import Field

from .base import NonEmptyStr, StrictModel
from .gap import CandidateGap
from .state import Component, CurrentSolution, Experiment
from .verification import GapVerificationResult, SolutionVerificationResult


class CommitCheck(StrictModel):
    commit_allowed: bool
    unmet_conditions: list[str] = Field(default_factory=list)


class CommitSnapshot(StrictModel):
    active_gap: CandidateGap
    current_solution: CurrentSolution
    experiments: list[Experiment] = Field(default_factory=list)
    latest_gap_verification: GapVerificationResult
    latest_solution_verification: SolutionVerificationResult
    relevant_evidence_refs: list[str] = Field(default_factory=list)


class FinalizerDraft(StrictModel):
    """Narrative-only Finalizer artifact; committed science is not editable."""

    novelty_positioning: NonEmptyStr
    expected_contributions: list[NonEmptyStr] = Field(min_length=2, max_length=3)


class FinalResearchProposal(StrictModel):
    research_gap: str
    proposed_solution: str
    solution_rationale: str
    method_components: list[Component] = Field(default_factory=list)
    novelty_positioning: str
    experimental_plan: list[Experiment] = Field(default_factory=list)
    expected_contributions: list[str] = Field(default_factory=list)


__all__ = [
    "CommitCheck",
    "CommitSnapshot",
    "FinalizerDraft",
    "FinalResearchProposal",
]
