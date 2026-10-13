"""Intermediate LLM draft and audit contracts for SeedPaperContext ingestion."""

from pydantic import Field, model_validator

from .base import NonEmptyStr, StrictModel
from .seed import SeedPaperContext


class EvidenceText(StrictModel):
    value: str | None = None
    source_section_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_value(self) -> "EvidenceText":
        if self.value is not None and self.value.strip() and not self.source_section_ids:
            raise ValueError("non-empty extracted text requires source_section_ids")
        return self


class EvidenceItem(StrictModel):
    value: NonEmptyStr
    source_section_ids: list[NonEmptyStr] = Field(min_length=1)


class ExperimentContextExtractionDraft(StrictModel):
    tasks_or_datasets: list[EvidenceItem] = Field(default_factory=list)
    metrics: list[EvidenceItem] = Field(default_factory=list)
    major_baselines: list[EvidenceItem] = Field(default_factory=list)
    experimental_setting: EvidenceText = Field(default_factory=EvidenceText)


class SeedContextExtractionDraft(StrictModel):
    """LLM output; final title, abstract and paper_id remain deterministic."""

    paper_id: str | None = None
    title: str | None = None
    abstract: str | None = None
    research_objective: EvidenceText = Field(default_factory=EvidenceText)
    core_method: EvidenceText = Field(default_factory=EvidenceText)
    main_components: list[EvidenceItem] = Field(default_factory=list)
    main_findings: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[EvidenceItem] = Field(default_factory=list)
    method_constraints: list[EvidenceItem] = Field(default_factory=list)
    experiment_context: ExperimentContextExtractionDraft = Field(
        default_factory=ExperimentContextExtractionDraft
    )


class SourceAnchor(StrictModel):
    field_path: NonEmptyStr
    section_id: NonEmptyStr
    section_heading: str
    start_line: int
    end_line: int
    excerpt: str


class LLMCallAudit(StrictModel):
    provider: str
    model: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_chars: int | None = None


class SeedExtractionAudit(StrictModel):
    anchors: list[SourceAnchor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_call_metadata: list[LLMCallAudit] = Field(default_factory=list)


class SeedContextExtraction(StrictModel):
    """Convenient structured result; output files still separate context/audit."""

    context: SeedPaperContext
    audit: SeedExtractionAudit
