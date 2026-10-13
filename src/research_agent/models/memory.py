"""Trajectory-local Research Memory records."""

from typing import Literal

from pydantic import Field, model_validator

from .base import NonEmptyStr, ScientificAction, StrictModel
from .inspiration import (
    BridgeEntity,
    InspirationCandidate,
    InspirationNeed,
    InspirationResult,
)
from .revision import RevisionResult
from .verification import (
    GapVerificationResult,
    SolutionVerificationResult,
    VerificationFocus,
)


class PaperRecord(StrictModel):
    paper_id: NonEmptyStr
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    source: str
    url: str | None = None


class RelatedPaperPool(StrictModel):
    seed_paper_id: NonEmptyStr
    papers: list[PaperRecord] = Field(default_factory=list)


class InspirationSearchRecord(StrictModel):
    query: str
    bridge_entities: list[BridgeEntity] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)


class SearchRecord(StrictModel):
    """Audit record for an external-paper search and its cache source."""

    provider: NonEmptyStr
    seed_paper_id: NonEmptyStr
    operation: Literal["related_pool_acquisition"]
    returned_paper_refs: list[str] = Field(default_factory=list)
    created_at_step: int
    cache_hit: bool = False
    cache_scope: Literal["none", "trajectory", "global", "global_partial"] = "none"
    cached_operations: list[str] = Field(default_factory=list)
    provider_request_count: int | None = None
    resolved_provider_paper_id: str | None = None
    seed_resolution_candidate_count: int | None = Field(default=None, ge=1)
    seed_resolution_selected_title: str | None = None
    seed_resolution_selected_year: int | None = None
    seed_resolution_selected_venue: str | None = None
    seed_resolution_reason: str | None = None


class VerificationRecord(StrictModel):
    id: NonEmptyStr
    task: Literal["VERIFY_GAP", "VERIFY_SOLUTION"]
    target_ref: NonEmptyStr
    verification_focus: VerificationFocus | None = None
    result: GapVerificationResult | SolutionVerificationResult
    paper_refs: list[str] = Field(default_factory=list)
    created_at_step: int

    @model_validator(mode="after")
    def validate_task_result_focus(self) -> "VerificationRecord":
        if self.task == "VERIFY_GAP":
            if not isinstance(self.result, GapVerificationResult):
                raise ValueError("VERIFY_GAP requires a GapVerificationResult")
            if self.verification_focus is not None:
                raise ValueError("VERIFY_GAP cannot provide verification_focus")
        else:
            if not isinstance(self.result, SolutionVerificationResult):
                raise ValueError("VERIFY_SOLUTION requires a SolutionVerificationResult")
            if self.verification_focus is None:
                raise ValueError("VERIFY_SOLUTION requires verification_focus")
        return self


class InspirationRecord(StrictModel):
    id: NonEmptyStr
    inspiration_need: InspirationNeed
    proposed_entities: list[BridgeEntity] = Field(default_factory=list)
    selected_entities: list[BridgeEntity] = Field(default_factory=list)
    search_query: str
    grounding_status: str
    inspirations: list[InspirationCandidate] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    created_at_step: int


class RevisionRecord(StrictModel):
    id: NonEmptyStr
    revision_reason: str
    affected_units: list[str] = Field(default_factory=list)
    selected_inspiration_ref: str | None = None
    changed_units: list[str] = Field(default_factory=list)
    revision_summary: str
    status: Literal["revised", "unable_to_revise"]
    failure_reason: str | None = None


class FailureRecord(StrictModel):
    type: Literal[
        "rejected_gap",
        "rejected_solution",
        "rejected_mechanism",
        "failed_revision",
    ]
    object_ref: NonEmptyStr
    summary: str
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)


class DecisionRecord(StrictModel):
    step: int
    action: ScientificAction
    reason: str
    result_summary: str


class PaperMemory(StrictModel):
    papers: list[PaperRecord] = Field(default_factory=list)
    related_paper_pool: RelatedPaperPool | None = None
    inspiration_search_records: list[InspirationSearchRecord] = Field(default_factory=list)
    search_records: list[SearchRecord] = Field(default_factory=list)


class ResearchMemory(StrictModel):
    paper_memory: PaperMemory = Field(default_factory=PaperMemory)
    verification_memory: list[VerificationRecord] = Field(default_factory=list)
    inspiration_memory: list[InspirationRecord] = Field(default_factory=list)
    revision_memory: list[RevisionRecord] = Field(default_factory=list)
    failure_memory: list[FailureRecord] = Field(default_factory=list)
    decision_log: list[DecisionRecord] = Field(default_factory=list)
