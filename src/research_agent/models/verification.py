"""Related-work and novelty verification contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .base import NonEmptyStr, StrictModel


class ComparisonRecord(StrictModel):
    paper_ref: NonEmptyStr
    problem_relation: Literal["same", "related", "different"]
    mechanism_relation: Literal["same", "partial", "different", "unclear"]
    shared_elements: list[str] = Field(default_factory=list)
    novelty_relevant_overlaps: list[str] = Field(default_factory=list)
    key_difference: str


class GapVerificationResult(StrictModel):
    status: Literal["valid", "partially_resolved", "resolved"]
    summary: str
    covered_aspects: list[str] = Field(default_factory=list)
    remaining_gap: str | None = None
    related_papers: list[str] = Field(default_factory=list)
    evidence_level: Literal["abstract_only", "full_text"]


class SolutionOverlap(StrictModel):
    component_id: NonEmptyStr | None = None
    overlap_summary: str
    matched_paper_ref: NonEmptyStr


class SolutionVerificationResult(StrictModel):
    novelty_status: Literal["distinct", "partial_overlap", "core_overlap"]
    summary: str
    shared_elements: list[str] = Field(default_factory=list)
    novelty_relevant_overlaps: list[SolutionOverlap] = Field(default_factory=list)
    distinctive_elements: list[str] = Field(default_factory=list)
    key_differences: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)
    evidence_level: Literal["abstract_only", "full_text"]


class SolutionVerificationDraft(StrictModel):
    """Internal one-call output: comparisons plus the canonical final result."""

    comparisons: list[ComparisonRecord] = Field(default_factory=list, max_length=5)
    result: SolutionVerificationResult


class VerificationFocus(StrictModel):
    mode: Literal["full_solution", "changed_units"]
    changed_units: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_changed_units(self) -> "VerificationFocus":
        if self.mode == "full_solution" and self.changed_units:
            raise ValueError("full_solution verification cannot name changed_units")
        if self.mode == "changed_units" and not self.changed_units:
            raise ValueError("changed_units verification requires changed_units")
        return self
