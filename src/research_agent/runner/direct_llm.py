"""One-shot Direct LLM baseline runner.

This runner intentionally does not construct Research State, a Controller, a
scientific module, or any retrieval provider.  It performs one structured LLM
call per paper and materializes only Direct-baseline artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..errors import ConfigurationError
from ..ingestion.markdown import build_full_paper_reading_context, parse_markdown_file
from ..llm.client import LLMClient, LLMMessage, LLMRequest, StructuredLLMResponse, sanitize_diagnostic_payload
from ..models.direct import DirectLLMCallAudit, DirectProposal, DirectRunManifest
from ..serialization import write_json


DIRECT_PROMPT_VERSION = "direct_llm_v1"


class RawResponseSource(Protocol):
    last_response: Mapping[str, Any] | None


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "direct_llm.md"


def direct_prompt_text() -> str:
    """Render the frozen prompt with the schema generated from DirectProposal."""

    template = _prompt_path().read_text(encoding="utf-8")
    schema = json.dumps(
        DirectProposal.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.replace("{{DIRECT_PROPOSAL_JSON_SCHEMA}}", schema)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_content(raw_response: Mapping[str, Any] | None) -> str | None:
    if not isinstance(raw_response, Mapping):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(raw_response, ensure_ascii=False, indent=2)
    first = choices[0]
    if not isinstance(first, Mapping):
        return json.dumps(raw_response, ensure_ascii=False, indent=2)
    message = first.get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        return json.dumps(raw_response, ensure_ascii=False, indent=2)
    content = message["content"]
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _raw_metadata(raw_response: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_response, Mapping):
        return {}
    usage = raw_response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return {
        "provider": "deepseek",
        "model": raw_response.get("model") if isinstance(raw_response.get("model"), str) else None,
        "request_id": raw_response.get("id") if isinstance(raw_response.get("id"), str) else None,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
        "total_tokens": usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
    }


def _safe_error(error: Exception) -> str:
    value = sanitize_diagnostic_payload(str(error))
    return str(value).replace("\n", " ")[:1_000]


@dataclass(frozen=True)
class DirectRunResult:
    manifest: DirectRunManifest
    proposal: DirectProposal | None


class DirectLLMRunner:
    """Execute one paper with exactly one client call and no retry wrapper."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        raw_response_source: RawResponseSource | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        if temperature < 0:
            raise ConfigurationError("Direct LLM temperature must be non-negative")
        if max_tokens is not None and max_tokens <= 0:
            raise ConfigurationError("Direct LLM max_tokens must be positive")
        self.llm_client = llm_client
        self.raw_response_source = raw_response_source
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, markdown_path: str | Path, output_dir: str | Path) -> DirectRunResult:
        source = Path(markdown_path)
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"Direct output directory already exists: {destination}")
        destination.mkdir(parents=True)

        prompt_template = direct_prompt_text()
        prompt_sha256 = _sha256(prompt_template)
        schema_sha256 = _sha256(
            json.dumps(DirectProposal.model_json_schema(), ensure_ascii=False, sort_keys=True)
        )
        paper_id = source.stem
        input_chars = 0
        raw_saved = False
        proposal: DirectProposal | None = None
        response: StructuredLLMResponse[DirectProposal] | None = None
        error: Exception | None = None
        call_attempted = False

        try:
            parsed = parse_markdown_file(source)
            reading = build_full_paper_reading_context(parsed)
            rendered_prompt = prompt_template.replace("{paper_markdown}", reading.body_text)
            input_chars = len(rendered_prompt)
            request = LLMRequest(
                messages=(LLMMessage(role="user", content=rendered_prompt),),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={
                    "task": "direct_llm_generation",
                    "paper_id": paper_id,
                    "prompt_version": DIRECT_PROMPT_VERSION,
                },
            )
            call_attempted = True
            response = self.llm_client.complete_structured(request, DirectProposal)
            proposal = response.output
            write_json(destination / "direct_proposal.json", "direct_proposal", proposal)
        except Exception as exc:
            error = exc

        raw_response = (
            self.raw_response_source.last_response
            if self.raw_response_source is not None
            else None
        )
        raw_text = _raw_content(raw_response)
        if raw_text is not None:
            (destination / "raw_response.txt").write_text(raw_text, encoding="utf-8")
            raw_saved = True

        response_metadata = response.metadata if response is not None else None
        fallback_metadata = _raw_metadata(raw_response)
        provider = response_metadata.provider if response_metadata else fallback_metadata.get("provider")
        model = response_metadata.model if response_metadata else fallback_metadata.get("model")
        request_id = response_metadata.request_id if response_metadata else fallback_metadata.get("request_id")
        prompt_tokens = response_metadata.prompt_tokens if response_metadata else fallback_metadata.get("prompt_tokens")
        completion_tokens = response_metadata.completion_tokens if response_metadata else fallback_metadata.get("completion_tokens")
        total_tokens = response_metadata.total_tokens if response_metadata else fallback_metadata.get("total_tokens")

        call_audit = DirectLLMCallAudit(
            task="direct_llm_generation",
            provider=provider,
            model=model,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_chars=input_chars,
            success=error is None,
            raw_response_saved=raw_saved,
            error_type=type(error).__name__ if error else None,
            error_message=_safe_error(error) if error else None,
        )
        write_json(destination / "llm_call.json", "direct_llm_call_audit", call_audit)

        manifest = DirectRunManifest(
            paper_id=paper_id,
            markdown_path=str(source),
            status="succeeded" if error is None else "failed",
            llm_request_count=1 if call_attempted else 0,
            input_chars=input_chars,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error_type=type(error).__name__ if error else None,
            error_message=_safe_error(error) if error else None,
        )
        write_json(destination / "run_manifest.json", "direct_run_manifest", manifest)
        return DirectRunResult(manifest=manifest, proposal=proposal)


def run_direct_batch(
    markdown_paths: Sequence[str | Path],
    output_dir: str | Path,
    runner_factory: Callable[[], DirectLLMRunner],
    *,
    batch_name: str = "direct_llm_batch",
) -> dict[str, Any]:
    """Run a fixed list serially and write one batch-level JSON report."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Direct batch output directory already exists: {destination}")
    paths = sorted((Path(path) for path in markdown_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("Direct batch requires at least one Markdown file")
    destination.mkdir(parents=True)
    papers: list[dict[str, Any]] = []
    for source in paths:
        paper_output = destination / source.stem
        try:
            result = runner_factory().run(source, paper_output)
            papers.append(result.manifest.model_dump(mode="json") | {"output_dir": str(paper_output)})
        except Exception as exc:
            papers.append(
                {
                    "paper_id": source.stem,
                    "markdown_path": str(source),
                    "output_dir": str(paper_output),
                    "status": "failed",
                    "llm_request_count": 0,
                    "error_type": type(exc).__name__,
                    "error_message": _safe_error(exc),
                }
            )
    report = {
        "batch_name": batch_name,
        "paper_count": len(paths),
        "succeeded_count": sum(item["status"] == "succeeded" for item in papers),
        "failed_count": sum(item["status"] == "failed" for item in papers),
        "llm_request_count": sum(int(item.get("llm_request_count", 0)) for item in papers),
        "papers": papers,
    }
    write_json(destination / "direct_batch_report.json", "direct_batch_report", report)
    return report


__all__ = [
    "DIRECT_PROMPT_VERSION",
    "DirectLLMRunner",
    "DirectRunResult",
    "direct_prompt_text",
    "run_direct_batch",
]
