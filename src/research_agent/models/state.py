"""Mutable Research State contracts."""

from typing import Literal

from pydantic import Field

from .base import NonEmptyStr, StrictModel
from .gap import CandidateGap


class CandidateSolution(StrictModel):
    id: NonEmptyStr
    core_idea: str
    rationale: str
    status: Literal["candidate", "selected", "rejected"]


class Component(StrictModel):
    id: NonEmptyStr
    description: str
    status: Literal["ready", "modified", "stale"]
    evidence_refs: list[str] = Field(default_factory=list)


class CurrentSolution(StrictModel):
    id: NonEmptyStr
    core_idea: str
    rationale: str
    components: list[Component] = Field(min_length=2, max_length=5)
    status: Literal["developed"]
    novelty_status: Literal["unchecked", "distinct", "partial_overlap", "core_overlap"]
    novelty_summary: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class Experiment(StrictModel):
    id: NonEmptyStr
    objective: str
    setup: str
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    what_it_validates: str
    validation_criterion: str
    status: Literal["ready", "stale"]


class ResearchState(StrictModel):
    active_gap: CandidateGap | None = None
    candidate_solutions: list[CandidateSolution] = Field(default_factory=list)
    current_solution: CurrentSolution | None = None
    experiments: list[Experiment] = Field(default_factory=list)
