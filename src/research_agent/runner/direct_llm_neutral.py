"""Parallel, one-shot Direct-Neutral baseline runner.

This module is intentionally separate from the Direct-Strong runner.  It only
uses the shared Markdown ingestion and provider-neutral LLM adapter; it does
not construct Research State or invoke any Ours workflow component.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..errors import ConfigurationError
from ..ingestion.markdown import build_full_paper_reading_context, parse_markdown_file
from ..llm.client import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
    sanitize_diagnostic_payload,
)
from ..llm.deepseek import JsonTransport, UrllibJsonTransport
from ..models.direct_neutral import (
    DirectNeutralCallAudit,
    DirectNeutralProposal,
    DirectNeutralRunManifest,
)
from ..serialization import write_json, write_jsonl


DIRECT_NEUTRAL_PROMPT_VERSION = "direct_neutral_v1"
DIRECT_NEUTRAL_BASELINE_NAME = "Direct-Neutral"


class RawResponseSource(Protocol):
    last_response: Mapping[str, Any] | None


class CountingJsonTransport(JsonTransport):
    """Count physical provider attempts without exposing request headers."""

    def __init__(self, inner: JsonTransport | None = None) -> None:
        self.inner = inner or UrllibJsonTransport()
        self.attempts = 0
        self.last_response: Mapping[str, Any] | None = None

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        self.attempts += 1
        response = self.inner.post_json(url, headers, payload, timeout)
        self.last_response = response
        return response


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "direct_llm_neutral.md"


def direct_neutral_prompt_text() -> str:
    """Return the frozen Direct-Neutral prompt without scientific additions."""

    return _prompt_path().read_text(encoding="utf-8")


def direct_neutral_prompt_hash() -> str:
    return _sha256(direct_neutral_prompt_text())


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


def _safe_error(error: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": sanitize_diagnostic_payload(str(error))[:1_000],
    }
    for attribute in ("schema_name", "validation_errors", "parsed_payload", "raw_response_excerpt"):
        value = getattr(error, attribute, None)
        if value is not None:
            details[attribute] = sanitize_diagnostic_payload(value)
    return details


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DirectNeutralRunResult:
    manifest: DirectNeutralRunManifest
    proposal: DirectNeutralProposal | None


class DirectNeutralRunner:
    """Execute exactly one logical scientific generation call for one paper."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        raw_response_source: RawResponseSource | None = None,
        physical_attempt_source: CountingJsonTransport | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        if temperature < 0:
            raise ConfigurationError("Direct-Neutral temperature must be non-negative")
        if max_tokens is not None and max_tokens <= 0:
            raise ConfigurationError("Direct-Neutral max_tokens must be positive")
        self.llm_client = llm_client
        self.raw_response_source = raw_response_source
        self.physical_attempt_source = physical_attempt_source
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.provider = provider

    def run(self, markdown_path: str | Path, output_dir: str | Path) -> DirectNeutralRunResult:
        source = Path(markdown_path)
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"Direct-Neutral output directory already exists: {destination}")
        destination.mkdir(parents=True)

        prompt_template = direct_neutral_prompt_text()
        prompt_hash = _sha256(prompt_template)
        paper_id = source.stem
        input_chars = 0
        logical_calls = 0
        proposal: DirectNeutralProposal | None = None
        response: StructuredLLMResponse[DirectNeutralProposal] | None = None
        error: Exception | None = None

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
                    "task": "direct_neutral_generation",
                    "paper_id": paper_id,
                    "prompt_version": DIRECT_NEUTRAL_PROMPT_VERSION,
                },
            )
            logical_calls = 1
            response = self.llm_client.complete_structured(request, DirectNeutralProposal)
            proposal = response.output
            write_json(destination / "final_proposal.json", "direct_neutral_proposal", proposal)
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

        physical_attempts = (
            self.physical_attempt_source.attempts
            if self.physical_attempt_source is not None
            else logical_calls
        )
        response_metadata = response.metadata if response is not None else None
        provider = response_metadata.provider if response_metadata else self.provider
        model = response_metadata.model if response_metadata else self.model
        request_id = response_metadata.request_id if response_metadata else None
        prompt_tokens = response_metadata.prompt_tokens if response_metadata else None
        completion_tokens = response_metadata.completion_tokens if response_metadata else None
        total_tokens = response_metadata.total_tokens if response_metadata else None
        error_payload = _safe_error(error) if error is not None else None

        call_audit = DirectNeutralCallAudit(
            logical_call_id="direct_neutral_generation",
            provider=provider,
            model=model,
            request_id=request_id,
            physical_api_attempts=physical_attempts,
            retry_count=max(physical_attempts - 1, 0),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_chars=input_chars,
            success=error is None,
            error=error_payload,
        )
        write_jsonl(destination / "llm_calls.jsonl", "direct_neutral_llm_call", [call_audit])

        manifest = DirectNeutralRunManifest(
            paper_id=paper_id,
            source_md=str(source),
            baseline_name=DIRECT_NEUTRAL_BASELINE_NAME,
            prompt_version=DIRECT_NEUTRAL_PROMPT_VERSION,
            prompt_hash=prompt_hash,
            model=model,
            provider=provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            status="succeeded" if error is None else "failed",
            logical_scientific_generation_calls=logical_calls,
            physical_api_attempts=physical_attempts,
            input_chars=input_chars,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error_type=type(error).__name__ if error else None,
            error_message=(str(error_payload.get("message")) if error_payload else None),
        )
        write_json(destination / "run_manifest.json", "direct_neutral_run_manifest", manifest)
        if error_payload is not None:
            write_json(destination / "error.json", "direct_neutral_error", error_payload)
        return DirectNeutralRunResult(manifest=manifest, proposal=proposal)


def discover_neutral_inputs(input_dir: str | Path, *, expected_count: int = 40) -> list[Path]:
    """Find and validate the deterministic formal Direct-Neutral cohort."""

    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Direct-Neutral input directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    if len(paths) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} Markdown papers, found {len(paths)}: {directory}"
        )
    for path in paths:
        parsed = parse_markdown_file(path)
        build_full_paper_reading_context(parsed)
    return paths


def _paper_summary(result: DirectNeutralRunResult, output_dir: Path) -> dict[str, Any]:
    return result.manifest.model_dump(mode="json") | {"output_dir": str(output_dir)}


def _unexpected_failure(source: Path, output_dir: Path, error: Exception) -> dict[str, Any]:
    return {
        "paper_id": source.stem,
        "source_md": str(source),
        "baseline_name": DIRECT_NEUTRAL_BASELINE_NAME,
        "status": "failed",
        "logical_scientific_generation_calls": 0,
        "physical_api_attempts": 0,
        "output_dir": str(output_dir),
        "error": _safe_error(error),
    }


def _token_stats(papers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = [
        item
        for item in papers
        if int(item.get("logical_scientific_generation_calls", 0) or 0) > 0
    ]
    known_prompt = sum(
        int(item["prompt_tokens"])
        for item in calls
        if isinstance(item.get("prompt_tokens"), int)
    )
    known_completion = sum(
        int(item["completion_tokens"])
        for item in calls
        if isinstance(item.get("completion_tokens"), int)
    )
    known_total = sum(
        int(item["total_tokens"])
        for item in calls
        if isinstance(item.get("total_tokens"), int)
    )
    complete = sum(
        all(isinstance(item.get(field), int) for field in ("prompt_tokens", "completion_tokens", "total_tokens"))
        for item in calls
    )
    missing = len(calls) - complete
    succeeded = sum(item.get("status") == "succeeded" for item in papers)
    successful_calls = [item for item in calls if item.get("status") == "succeeded"]
    return {
        "prompt_tokens": known_prompt,
        "completion_tokens": known_completion,
        "total_known_tokens": known_total,
        "calls_with_usage": complete,
        "calls_missing_usage": missing,
        "token_metadata_complete": missing == 0,
        "average_tokens_per_successful_paper": (
            sum(
                int(item["total_tokens"])
                for item in successful_calls
                if isinstance(item.get("total_tokens"), int)
            )
            / succeeded
            if succeeded
            else None
        ),
    }


def run_direct_neutral_batch(
    markdown_paths: Sequence[str | Path],
    output_dir: str | Path,
    runner_factory: Callable[[], DirectNeutralRunner],
    *,
    batch_name: str = "direct_llm_neutral_nlp40",
    workers: int = 4,
) -> dict[str, Any]:
    """Run independent papers with bounded concurrency and isolated outputs."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Direct-Neutral batch output already exists: {destination}")
    paths = sorted((Path(path) for path in markdown_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("Direct-Neutral batch requires at least one Markdown file")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Direct-Neutral batch contains duplicate paper IDs")
    destination.mkdir(parents=True)
    started_at = _utc_now()
    started_clock = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="direct-neutral")
    futures: dict[Future[DirectNeutralRunResult], tuple[Path, Path]] = {}
    papers: list[dict[str, Any]] = []
    interrupted = False
    try:
        for source in paths:
            paper_output = destination / source.stem
            futures[executor.submit(runner_factory().run, source, paper_output)] = (source, paper_output)
        for future in as_completed(futures):
            source, paper_output = futures[future]
            try:
                result = future.result()
                papers.append(_paper_summary(result, paper_output))
            except Exception as exc:
                papers.append(_unexpected_failure(source, paper_output, exc))
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    papers.sort(key=lambda item: str(item.get("paper_id", "")))
    report = {
        "baseline_name": DIRECT_NEUTRAL_BASELINE_NAME,
        "batch_name": batch_name,
        "total_papers": len(paths),
        "succeeded": sum(item.get("status") == "succeeded" for item in papers),
        "failed": sum(item.get("status") == "failed" for item in papers),
        "logical_llm_calls": sum(int(item.get("logical_scientific_generation_calls", 0) or 0) for item in papers),
        "physical_api_attempts": sum(int(item.get("physical_api_attempts", 0) or 0) for item in papers),
        "workers": workers,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_clock_seconds": time.monotonic() - started_clock,
        **_token_stats(papers),
        "papers": papers,
    }
    write_json(destination / "evaluation_report.json", "direct_neutral_evaluation_report", report)
    return report


__all__ = [
    "DIRECT_NEUTRAL_BASELINE_NAME",
    "DIRECT_NEUTRAL_PROMPT_VERSION",
    "CountingJsonTransport",
    "DirectNeutralRunner",
    "discover_neutral_inputs",
    "direct_neutral_prompt_hash",
    "direct_neutral_prompt_text",
    "run_direct_neutral_batch",
]
