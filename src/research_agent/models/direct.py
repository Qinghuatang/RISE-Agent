"""Strict, workflow-neutral contracts for the Direct LLM baseline."""

from typing import Literal

from pydantic import Field, model_validator

from .base import NonEmptyStr, StrictModel


class DirectMethodComponent(StrictModel):
    """One concrete technical component in a Direct proposal."""

    name: NonEmptyStr
    description: NonEmptyStr


class DirectProposedMethod(StrictModel):
    """The method section of a one-shot Direct proposal."""

    core_idea: NonEmptyStr
    components: list[DirectMethodComponent] = Field(min_length=2, max_length=5)


class DirectExperiment(StrictModel):
    """One experiment required by the Direct baseline contract."""

    objective: NonEmptyStr
    setup: NonEmptyStr
    baselines: list[NonEmptyStr] = Field(min_length=1)
    metrics: list[NonEmptyStr] = Field(min_length=1)
    what_it_validates: NonEmptyStr
    validation_criterion: NonEmptyStr

    @model_validator(mode="after")
    def requires_seed_method_baseline(self) -> "DirectExperiment":
        if "Seed Method" not in self.baselines:
            raise ValueError('baselines must include the literal "Seed Method"')
        return self


class DirectProposal(StrictModel):
    """Canonical one-shot Direct LLM research proposal."""

    research_problem: NonEmptyStr
    proposed_method: DirectProposedMethod
    technical_rationale: NonEmptyStr
    experimental_plan: list[DirectExperiment] = Field(min_length=2, max_length=4)
    expected_contributions: list[NonEmptyStr] = Field(min_length=2, max_length=3)


class DirectLLMCallAudit(StrictModel):
    """Safe metadata for the one Direct generation call."""

    task: Literal["direct_llm_generation"]
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_chars: int = Field(ge=0)
    success: bool
    raw_response_saved: bool = False
    error_type: str | None = None
    error_message: str | None = None


class DirectRunManifest(StrictModel):
    """Execution-only artifact describing one Direct paper run."""

    paper_id: str
    markdown_path: str
    status: Literal["succeeded", "failed"]
    llm_request_count: int = Field(ge=0, le=1)
    input_chars: int = Field(ge=0)
    prompt_sha256: str
    schema_sha256: str
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None


__all__ = [
    "DirectExperiment",
    "DirectLLMCallAudit",
    "DirectMethodComponent",
    "DirectProposal",
    "DirectProposedMethod",
    "DirectRunManifest",
]
