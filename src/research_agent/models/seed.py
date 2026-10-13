"""Contracts extracted from one MinerU-produced seed paper."""

from pydantic import Field

from .base import NonEmptyStr, StrictModel


class ExperimentContext(StrictModel):
    tasks_or_datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    major_baselines: list[str] = Field(default_factory=list)
    experimental_setting: str | None = None


class SeedPaperContext(StrictModel):
    paper_id: NonEmptyStr
    title: str | None = None
    abstract: str | None = None
    research_objective: str | None = None
    core_method: str | None = None
    main_components: list[str] = Field(default_factory=list)
    main_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    method_constraints: list[str] = Field(default_factory=list)
    experiment_context: ExperimentContext
