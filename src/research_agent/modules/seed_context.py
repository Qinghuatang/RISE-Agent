"""MinerU Markdown to SeedPaperContext extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from ..errors import ConfigurationError, ExtractionError
from ..ingestion.markdown import (
    FullPaperReadingContext,
    build_full_paper_reading_context,
    parse_markdown_file,
    parse_mineru_markdown,
)
from ..llm.client import LLMClient, LLMMessage, LLMRequest, StructuredLLMResponse
from ..models.seed import ExperimentContext, SeedPaperContext
from ..models.seed_extraction import (
    EvidenceItem,
    EvidenceText,
    LLMCallAudit,
    SeedContextExtraction,
    SeedContextExtractionDraft,
    SeedExtractionAudit,
    SourceAnchor,
)
from ..serialization import write_json


@dataclass(frozen=True)
class SeedContextConfig:
    """Output-budget and audit controls for one complete-paper extraction."""

    # Keep enough output headroom for JSON mode while asking the model for
    # concise evidence-grounded summaries rather than copied paper text.
    max_tokens: int = 8_000
    audit_excerpt_chars: int = 240

    def __post_init__(self) -> None:
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("audit_excerpt_chars", self.audit_excerpt_chars),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SeedContextConfig":
        values = os.environ if environ is None else environ

        def parse(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc
            if value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
            return value

        return cls(
            max_tokens=parse("SEED_CONTEXT_MAX_TOKENS", cls.max_tokens),
            audit_excerpt_chars=parse(
                "SEED_CONTEXT_AUDIT_EXCERPT_CHARS", cls.audit_excerpt_chars
            ),
        )


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "seed_context.md"


def _system_prompt() -> str:
    prompt = _prompt_path().read_text(encoding="utf-8")
    schema = json.dumps(
        SeedContextExtractionDraft.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return prompt.replace("{{SEED_CONTEXT_EXTRACTION_DRAFT_JSON_SCHEMA}}", schema)


def _metadata_audit(
    response: StructuredLLMResponse[SeedContextExtractionDraft],
    input_chars: int,
) -> LLMCallAudit:
    metadata = response.metadata
    return LLMCallAudit(
        provider=metadata.provider,
        model=metadata.model,
        request_id=metadata.request_id,
        prompt_tokens=metadata.prompt_tokens,
        completion_tokens=metadata.completion_tokens,
        total_tokens=metadata.total_tokens,
        input_chars=input_chars,
    )


def _draft_source_ids(draft: SeedContextExtractionDraft) -> set[str]:
    payload = draft.model_dump(mode="python")
    source_ids: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "source_section_ids" and isinstance(item, list):
                    source_ids.update(str(source_id) for source_id in item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return source_ids


def _section_excerpt(section, excerpt_limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", section.text).strip()
    return collapsed[:excerpt_limit]


class SeedContextExtractor:
    """LLM-client-neutral, one-call extraction with deterministic evidence checks."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: SeedContextConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SeedContextConfig.from_env()
        self._system_prompt = _system_prompt()

    def parse(self, markdown_text: str, paper_id: str) -> FullPaperReadingContext:
        parsed = parse_mineru_markdown(markdown_text, paper_id=paper_id)
        return build_full_paper_reading_context(parsed)

    def extract_text(
        self,
        markdown_text: str,
        paper_id: str,
    ) -> SeedContextExtraction:
        return self._extract_reading_context(self.parse(markdown_text, paper_id))

    def extract_file(self, path: str | Path) -> SeedContextExtraction:
        result, _ = self.extract_file_with_reading(path)
        return result

    def extract_file_with_reading(
        self, path: str | Path
    ) -> tuple[SeedContextExtraction, FullPaperReadingContext]:
        """Extract once and also return the deterministic full-paper view.

        The reading context is an execution-only handoff for Gap Discovery;
        it is not persisted in SeedPaperContext and does not add an LLM call.
        """

        parsed = parse_markdown_file(path)
        reading = build_full_paper_reading_context(parsed)
        return self._extract_reading_context(reading), reading

    def _extract_reading_context(
        self,
        reading: FullPaperReadingContext,
    ) -> SeedContextExtraction:
        input_text = reading.render()
        response = self._call_full_body(reading, input_text)
        self._validate_draft_sources(reading, response.output)
        metadata = [_metadata_audit(response, len(input_text))]
        return self._materialize(reading, response.output, metadata)

    @staticmethod
    def _validate_draft_sources(
        reading: FullPaperReadingContext,
        draft: SeedContextExtractionDraft,
    ) -> None:
        allowed = set(reading.body_section_ids)
        known = {section.section_id for section in reading.parsed.sections}
        source_ids = _draft_source_ids(draft)
        unknown = source_ids - known
        if unknown:
            raise ExtractionError(
                f"LLM draft references unknown source section(s): {sorted(unknown)}"
            )
        outside_body = source_ids - allowed
        if outside_body:
            raise ExtractionError(
                "LLM draft references a section outside the supplied main-paper body: "
                f"{sorted(outside_body)}"
            )

    def _call_full_body(
        self,
        reading: FullPaperReadingContext,
        input_text: str,
    ) -> StructuredLLMResponse[SeedContextExtractionDraft]:
        user_content = (
            "INPUT PAPER CONTENT\n"
            "Read the complete supplied main-paper body once. The source section index "
            "above the body is the only valid evidence-id vocabulary.\n\n"
            f"{input_text}\n\n"
            "Return exactly one JSON object."
        )
        return self._complete(user_content, reading.paper_id, len(input_text))

    def _complete(
        self,
        user_content: str,
        paper_id: str,
        input_chars: int,
    ) -> StructuredLLMResponse[SeedContextExtractionDraft]:
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=self._system_prompt),
                LLMMessage(
                    role="user",
                    content=f"PAPER_ID (deterministic, do not change): {paper_id}\n\n{user_content}",
                ),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={
                "task": "seed_context_extraction",
                "paper_id": paper_id,
                "input_chars": input_chars,
            },
        )
        return self.llm_client.complete_structured(request, SeedContextExtractionDraft)

    def _materialize(
        self,
        reading: FullPaperReadingContext,
        draft: SeedContextExtractionDraft,
        metadata: list[LLMCallAudit],
    ) -> SeedContextExtraction:
        parsed = reading.parsed
        by_id = {section.section_id: section for section in reading.body_sections}
        allowed_ids = set(reading.body_section_ids)
        warnings = list(reading.warnings)
        if draft.paper_id and draft.paper_id != parsed.paper_id:
            warnings.append("llm_paper_id_ignored")
        if draft.title is not None and parsed.title is None:
            warnings.append("llm_title_ignored_parser_title_missing")
        if draft.abstract is not None and parsed.abstract is None:
            warnings.append("llm_abstract_ignored_parser_abstract_missing")

        anchors: list[SourceAnchor] = []

        def add_anchor(field_path: str, section_id: str) -> None:
            if section_id not in by_id:
                raise ExtractionError(
                    f"field {field_path} references a section outside the supplied main-paper body: {section_id}"
                )
            section = by_id[section_id]
            anchors.append(
                SourceAnchor(
                    field_path=field_path,
                    section_id=section.section_id,
                    section_heading=section.heading,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    excerpt=_section_excerpt(section, self.config.audit_excerpt_chars),
                )
            )

        def evidence_value(field_path: str, evidence: EvidenceText) -> str | None:
            if evidence.value is None or not evidence.value.strip():
                return None
            for section_id in evidence.source_section_ids:
                add_anchor(field_path, section_id)
            return evidence.value.strip()

        def evidence_list(field_path: str, values: list[EvidenceItem]) -> list[str]:
            result: list[str] = []
            for index, item in enumerate(values):
                for section_id in item.source_section_ids:
                    add_anchor(f"{field_path}[{index}]", section_id)
                result.append(item.value.strip())
            return result

        if parsed.title and parsed.title_section_id:
            add_anchor("title", parsed.title_section_id)
        if parsed.abstract and parsed.abstract_section_id:
            add_anchor("abstract", parsed.abstract_section_id)

        experiment = draft.experiment_context
        context = SeedPaperContext(
            paper_id=parsed.paper_id,
            title=parsed.title,
            abstract=parsed.abstract,
            research_objective=evidence_value("research_objective", draft.research_objective),
            core_method=evidence_value("core_method", draft.core_method),
            main_components=evidence_list("main_components", draft.main_components),
            main_findings=evidence_list("main_findings", draft.main_findings),
            limitations=evidence_list("limitations", draft.limitations),
            method_constraints=evidence_list(
                "method_constraints", draft.method_constraints
            ),
            experiment_context=ExperimentContext(
                tasks_or_datasets=evidence_list(
                    "experiment_context.tasks_or_datasets", experiment.tasks_or_datasets
                ),
                metrics=evidence_list("experiment_context.metrics", experiment.metrics),
                major_baselines=evidence_list(
                    "experiment_context.major_baselines", experiment.major_baselines
                ),
                experimental_setting=evidence_value(
                    "experiment_context.experimental_setting", experiment.experimental_setting
                ),
            ),
        )
        return SeedContextExtraction(
            context=context,
            audit=SeedExtractionAudit(
                anchors=anchors,
                warnings=warnings,
                llm_call_metadata=metadata,
            ),
        )


def write_seed_context_outputs(
    result: SeedContextExtraction,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir) / result.context.paper_id
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "seed_context.json"
    audit_path = root / "seed_extraction_audit.json"
    write_json(context_path, "seed_paper_context", result.context)
    write_json(audit_path, "seed_extraction_audit", result.audit)
    return context_path, audit_path
