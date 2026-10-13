"""Execution-only contracts for M13 batch evaluation."""

from typing import Literal

from pydantic import Field

from .base import NonEmptyStr, ScientificAction, StrictModel


class EvaluationGap(StrictModel):
    """A compact, human-auditable representation of one discovered gap."""

    id: NonEmptyStr
    confidence: Literal["low", "medium", "high"]
    priority: Literal["low", "medium", "high"]
    evidence_refs: list[NonEmptyStr] = Field(default_factory=list)


class PaperEvaluation(StrictModel):
    """Aggregated metrics for one isolated paper trajectory."""

    paper_id: NonEmptyStr
    markdown_path: NonEmptyStr
    output_dir: NonEmptyStr
    status: Literal["committed", "stopped", "failed", "finalization_failed"]
    action_sequence: list[ScientificAction] = Field(default_factory=list)
    commit_status: Literal["committed", "not_committed"]
    finalizer_status: Literal["written", "not_written"]

    gap_found: bool
    candidate_gap_count: int = Field(ge=0)
    gaps: list[EvaluationGap] = Field(default_factory=list)

    solution_generated: bool
    candidate_solution_count: int = Field(ge=0)
    selected_solution_components: int | None = Field(default=None, ge=0)
    novelty_status: Literal["distinct", "partial_overlap", "core_overlap"] | None = None

    inspiration_available: bool
    inspiration_triggered: bool
    revision_triggered: bool
    inspiration_found: bool = False
    inspiration_used: bool = False

    llm_calls: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    known_prompt_tokens: int = Field(default=0, ge=0)
    known_completion_tokens: int = Field(default=0, ge=0)
    known_total_tokens: int = Field(default=0, ge=0)
    calls_with_token_usage: int = Field(default=0, ge=0)
    calls_missing_token_usage: int = Field(default=0, ge=0)
    token_metadata_complete: bool
    semantic_scholar_requests: int = Field(ge=0)
    arxiv_requests: int = Field(ge=0)

    replay_verified: bool
    state_consistent: bool
    memory_consistent: bool
    failure_type: str | None = None
    error_message: str | None = None


class EvaluationReport(StrictModel):
    """Categorical/count-based report for one batch evaluation."""

    batch_name: NonEmptyStr
    input_dir: NonEmptyStr
    output_dir: NonEmptyStr
    selection_seed: int | None = None
    selected_paper_ids: list[NonEmptyStr] = Field(default_factory=list)

    paper_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    stopped_count: int = Field(ge=0)
    papers: list[PaperEvaluation] = Field(default_factory=list)

    commit_status_counts: dict[str, int] = Field(default_factory=dict)
    finalizer_status_counts: dict[str, int] = Field(default_factory=dict)
    gap_found_counts: dict[str, int] = Field(default_factory=dict)
    novelty_status_counts: dict[str, int] = Field(default_factory=dict)
    inspiration_available_count: int = Field(ge=0)
    inspiration_triggered_count: int = Field(ge=0)
    inspiration_found_count: int = Field(default=0, ge=0)
    inspiration_used_count: int = Field(ge=0)
    revision_triggered_count: int = Field(ge=0)

    llm_calls_total: int = Field(ge=0)
    prompt_tokens_total: int | None = Field(default=None, ge=0)
    completion_tokens_total: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    known_prompt_tokens: int = Field(default=0, ge=0)
    known_completion_tokens: int = Field(default=0, ge=0)
    known_total_tokens: int = Field(default=0, ge=0)
    calls_with_token_usage: int = Field(default=0, ge=0)
    calls_missing_token_usage: int = Field(default=0, ge=0)
    token_metadata_complete: bool
    semantic_scholar_requests_total: int = Field(ge=0)
    arxiv_requests_total: int = Field(ge=0)

    replay_verified_count: int = Field(ge=0)
    state_consistent_count: int = Field(ge=0)
    memory_consistent_count: int = Field(ge=0)
    failure_type_counts: dict[str, int] = Field(default_factory=dict)


__all__ = ["EvaluationGap", "PaperEvaluation", "EvaluationReport"]
