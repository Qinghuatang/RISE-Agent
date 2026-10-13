"""Workflow-neutral contracts for the Direct-Neutral baseline."""

from typing import Literal

from pydantic import Field

from .base import NonEmptyStr, StrictModel


class DirectNeutralProposal(StrictModel):
    """The deliberately minimal one-shot Direct-Neutral proposal."""

    research_problem: NonEmptyStr
    proposed_method: NonEmptyStr
    technical_rationale: NonEmptyStr
    experimental_plan: NonEmptyStr


class DirectNeutralCallAudit(StrictModel):
    """One logical generation record, including physical transport attempts."""

    logical_call_id: Literal["direct_neutral_generation"]
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    physical_api_attempts: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_chars: int = Field(ge=0)
    success: bool
    error: dict[str, object] | None = None


class DirectNeutralRunManifest(StrictModel):
    """Per-paper execution metadata for Direct-Neutral."""

    paper_id: str
    source_md: str
    baseline_name: Literal["Direct-Neutral"]
    prompt_version: str
    prompt_hash: str
    model: str | None = None
    provider: str | None = None
    temperature: float
    max_tokens: int | None = Field(default=None, ge=1)
    status: Literal["succeeded", "failed"]
    logical_scientific_generation_calls: int = Field(ge=0, le=1)
    physical_api_attempts: int = Field(ge=0)
    input_chars: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None


__all__ = [
    "DirectNeutralCallAudit",
    "DirectNeutralProposal",
    "DirectNeutralRunManifest",
]
