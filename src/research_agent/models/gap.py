"""Gap discovery contracts and LLM-only gap-management drafts."""

from typing import Literal

from pydantic import Field

from .base import NonEmptyStr, StrictModel


class CandidateGap(StrictModel):
    id: NonEmptyStr
    statement: str
    confidence: Literal["low", "medium", "high"]
    confidence_reason: str
    priority: Literal["low", "medium", "high"]
    priority_reason: str
    status: Literal["candidate", "active", "resolved", "rejected"]
    evidence_refs: list[str] = Field(default_factory=list)


class GapAgenda(StrictModel):
    gaps: list[CandidateGap] = Field(default_factory=list, max_length=5)


class GapCandidateDraft(StrictModel):
    """Legacy combined draft retained for artifact/read compatibility."""

    statement: NonEmptyStr
    confidence: Literal["low", "medium", "high"]
    confidence_reason: NonEmptyStr
    priority: Literal["low", "medium", "high"]
    priority_reason: NonEmptyStr
    evidence_refs: list[NonEmptyStr] = Field(min_length=1)


class GapDiscoveryDraft(StrictModel):
    """Legacy combined Gap Discovery draft."""

    gaps: list[GapCandidateDraft] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
    )


class GapGenerationCandidateDraft(StrictModel):
    """Call-1 candidate: statement and evidence only."""

    statement: NonEmptyStr
    evidence_refs: list[NonEmptyStr] = Field(min_length=1)


class GapGenerationDraft(StrictModel):
    """Full-paper Call-1 output before independent evaluation."""

    gaps: list[GapGenerationCandidateDraft] = Field(default_factory=list, max_length=5)


class GapEvaluationItemDraft(StrictModel):
    """Call-2 evaluation for one code-owned generated gap ID."""

    candidate_id: NonEmptyStr
    confidence: Literal["low", "medium", "high"]
    confidence_reason: NonEmptyStr
    priority: Literal["low", "medium", "high"]
    priority_reason: NonEmptyStr


class GapEvaluationDraft(StrictModel):
    """Independent ranking/evaluation of every generated candidate."""

    evaluations: list[GapEvaluationItemDraft] = Field(default_factory=list, max_length=5)


class GapRefinementDraft(StrictModel):
    refined_statement: NonEmptyStr
    refinement_reason: NonEmptyStr


class GapDiscoveryResult(StrictModel):
    gap_agenda: GapAgenda
    active_gap: CandidateGap | None = None
    next_required_action: Literal["VERIFY_RELATED_WORK"] | None = None
