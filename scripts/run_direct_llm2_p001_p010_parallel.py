#!/usr/bin/env python3
"""Run the direct_llm2 prompt on P001--P010 with paper-level concurrency.

This is an independent one-shot baseline entry point.  It reuses the shared
Markdown ingestion and DeepSeek transport, but keeps the direct_llm2 prompt
and its strict output schema separate from Direct-Neutral.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import (  # noqa: E402
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.llm.client import (  # noqa: E402
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
    sanitize_diagnostic_payload,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.models.direct_llm2 import DirectLLM2Proposal  # noqa: E402
from research_agent.runner.direct_llm_neutral import CountingJsonTransport  # noqa: E402
from research_agent.serialization import write_json, write_jsonl  # noqa: E402


EXPECTED_PAPER_NUMBERS = tuple(range(1, 11))
EXPECTED_PAPER_COUNT = len(EXPECTED_PAPER_NUMBERS)
DEFAULT_WORKERS = 10
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_OUTPUT_DIR = Path("/home/tmh/final_baseline/output_direct2")
DEFAULT_BATCH_NAME = "direct_llm2_p001_p010"
PROMPT_PATH = REPO_ROOT / "src" / "research_agent" / "prompts" / "direct_llm2.md"
PAPER_ID_PATTERN = re.compile(r"(?:^|_)P(?P<number>\d{3})(?:_|$)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _prompt_hash() -> str:
    return _sha256(_prompt_text())


def _paper_number(path: Path) -> int | None:
    match = PAPER_ID_PATTERN.search(path.stem)
    return int(match.group("number")) if match else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen direct_llm2 prompt on P001-P010"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument("--model", default=None, help="optional DeepSeek model override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _schema_text() -> str:
    return json.dumps(
        DirectLLM2Proposal.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _render_prompt(paper_markdown: str) -> str:
    template = _prompt_text()
    return template.replace("{{DIRECT_PROPOSAL_JSON_SCHEMA}}", _schema_text()).replace(
        "{paper_markdown}", paper_markdown
    )


def _validate_inputs(args: argparse.Namespace) -> list[Path]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when provided")
    if not args.batch_name or "/" in args.batch_name or "\\" in args.batch_name:
        raise ValueError("--batch-name must be a simple filename component")
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"direct_llm2 prompt does not exist: {PROMPT_PATH}")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    if args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to use symlink output path: {args.output_dir}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output path: {args.output_dir}"
            )

    by_number: dict[int, Path] = {}
    for path in sorted(args.input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        number = _paper_number(path)
        if number not in EXPECTED_PAPER_NUMBERS:
            continue
        if number in by_number:
            raise ValueError(f"duplicate P{number:03d} Markdown input: {path.name}")
        parsed = parse_markdown_file(path)
        build_full_paper_reading_context(parsed)
        by_number[number] = path.resolve()

    missing = [f"P{number:03d}" for number in EXPECTED_PAPER_NUMBERS if number not in by_number]
    if missing:
        raise ValueError("missing expected inputs: " + ", ".join(missing))
    return [by_number[number] for number in EXPECTED_PAPER_NUMBERS]


def _config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    return config


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
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)


def _error_payload(error: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": sanitize_diagnostic_payload(str(error))[:1_000],
    }
    for attribute in ("schema_name", "validation_errors", "parsed_payload", "raw_response_excerpt"):
        value = getattr(error, attribute, None)
        if value is not None:
            details[attribute] = sanitize_diagnostic_payload(value)
    return details


class DirectLLM2SingleRunner:
    """Run one paper with one logical structured-generation call."""

    def __init__(
        self,
        client: DeepSeekClient,
        transport: CountingJsonTransport,
        *,
        temperature: float,
        max_tokens: int | None,
        model: str | None,
    ) -> None:
        self.client = client
        self.transport = transport
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

    def run(self, source: Path, destination: Path) -> dict[str, Any]:
        if destination.exists():
            raise FileExistsError(f"paper output directory already exists: {destination}")
        destination.mkdir(parents=True)
        prompt_hash = _prompt_hash()
        schema_hash = _sha256(json.dumps(DirectLLM2Proposal.model_json_schema(), sort_keys=True))
        logical_calls = 0
        input_chars = 0
        response: StructuredLLMResponse[dict[str, Any]] | None = None
        error: Exception | None = None
        try:
            parsed = parse_markdown_file(source)
            reading = build_full_paper_reading_context(parsed)
            rendered_prompt = _render_prompt(reading.body_text)
            input_chars = len(rendered_prompt)
            request = LLMRequest(
                messages=(LLMMessage(role="user", content=rendered_prompt),),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={
                    "task": "direct_llm2_generation",
                    "paper_id": source.stem,
                    "prompt_version": "direct_llm2_v2",
                },
            )
            logical_calls = 1
            response = self.client.complete_structured(request, DirectLLM2Proposal)
            write_json(
                destination / "final_proposal.json",
                "direct_llm2_model_output",
                response.output,
            )
        except Exception as exc:
            error = exc

        raw_response = self.transport.last_response
        raw_text = _raw_content(raw_response)
        if raw_text is not None:
            (destination / "raw_response.txt").write_text(raw_text, encoding="utf-8")
            (destination / "final_proposal_raw.txt").write_text(raw_text, encoding="utf-8")

        metadata = response.metadata if response is not None else None
        provider = metadata.provider if metadata else "deepseek"
        model = metadata.model if metadata else self.model
        request_id = metadata.request_id if metadata else None
        prompt_tokens = metadata.prompt_tokens if metadata else None
        completion_tokens = metadata.completion_tokens if metadata else None
        total_tokens = metadata.total_tokens if metadata else None
        physical_attempts = self.transport.attempts
        audit = {
            "logical_call_id": "direct_llm2_generation",
            "provider": provider,
            "model": model,
            "request_id": request_id,
            "physical_api_attempts": physical_attempts,
            "retry_count": max(physical_attempts - 1, 0),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_chars": input_chars,
            "success": error is None,
            "error": _error_payload(error) if error else None,
        }
        write_jsonl(destination / "llm_calls.jsonl", "direct_llm2_llm_call", [audit])
        manifest = {
            "paper_id": source.stem,
            "source_md": str(source),
            "baseline_name": "Direct-LLM2",
            "prompt_version": "direct_llm2_v2",
            "prompt_path": str(PROMPT_PATH),
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "model": model,
            "provider": provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "succeeded" if error is None else "failed",
            "logical_scientific_generation_calls": logical_calls,
            "physical_api_attempts": physical_attempts,
            "input_chars": input_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "format_retry_count": 0,
            "retry_reason": None,
            "llm_audit_record_count": 1,
            "output_validation_mode": "four_non_empty_strings",
            "error_type": type(error).__name__ if error else None,
            "error": _error_payload(error) if error else None,
        }
        write_json(destination / "run_manifest.json", "direct_llm2_run_manifest", manifest)
        if error is not None:
            write_json(destination / "error.json", "direct_llm2_error", _error_payload(error))
        return manifest | {"output_dir": str(destination)}


def _token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [row for row in rows if int(row.get("logical_scientific_generation_calls", 0) or 0) > 0]

    def total(field: str) -> int:
        return sum(
            int(row[field])
            for row in calls
            if isinstance(row.get(field), int) and not isinstance(row.get(field), bool)
        )

    complete = sum(
        all(
            isinstance(row.get(field), int) and not isinstance(row.get(field), bool)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        for row in calls
    )
    succeeded = sum(row.get("status") == "succeeded" for row in rows)
    known_successful_totals = [
        int(row["total_tokens"])
        for row in calls
        if row.get("status") == "succeeded"
        and isinstance(row.get("total_tokens"), int)
        and not isinstance(row.get("total_tokens"), bool)
    ]
    return {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_known_tokens": total("total_tokens"),
        "calls_with_usage": complete,
        "calls_missing_usage": len(calls) - complete,
        "token_metadata_complete": complete == len(calls),
        "average_tokens_per_successful_paper": (
            sum(known_successful_totals) / succeeded if succeeded else None
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        paths = _validate_inputs(args)
        config = _config(args)
        config_ready = True
        try:
            config.validate_for_request()
        except Exception:
            config_ready = False
            if not args.dry_run:
                raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print(
            json.dumps(
                {
                    "baseline_name": "Direct-LLM2",
                    "dry_run": True,
                    "input_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "paper_output_dir": str(args.output_dir / "papers"),
                    "total_papers": len(paths),
                    "workers": args.workers,
                    "model": config.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "prompt_path": str(PROMPT_PATH),
                    "prompt_hash": _prompt_hash(),
                    "schema": "DirectLLM2Proposal",
                    "deepseek_config_ready": config_ready,
                    "papers": [path.name for path in paths],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_root = args.output_dir
    papers_root = output_root / "papers"
    papers_root.mkdir(parents=True)
    started_at = _utc_now()
    started_clock = time.monotonic()
    rows: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], tuple[Path, Path]] = {}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="direct-llm2")
    interrupted = False

    def factory() -> DirectLLM2SingleRunner:
        transport = CountingJsonTransport()
        client = DeepSeekClient(config, transport=transport)
        return DirectLLM2SingleRunner(
            client,
            transport,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=config.model,
        )

    try:
        for path in paths:
            destination = papers_root / path.stem
            futures[executor.submit(factory().run, path, destination)] = (path, destination)
        completed = 0
        for future in as_completed(futures):
            source, destination = futures[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                destination.mkdir(parents=True, exist_ok=True)
                error = _error_payload(exc)
                write_json(destination / "error.json", "direct_llm2_error", error)
                row = {
                    "paper_id": source.stem,
                    "source_md": str(source),
                    "baseline_name": "Direct-LLM2",
                    "status": "failed",
                    "logical_scientific_generation_calls": 0,
                    "physical_api_attempts": 0,
                    "error_type": error["error_type"],
                    "error": error,
                    "output_dir": str(destination),
                }
            rows.append(row)
            print(f"[{completed}/{len(paths)}] {source.stem}: {row['status']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    rows.sort(key=lambda row: str(row.get("paper_id", "")))
    report = {
        "baseline_name": "Direct-LLM2",
        "batch_name": args.batch_name,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_root),
        "paper_output_dir": str(papers_root),
        "paper_range": ["P001", "P010"],
        "total_papers": len(paths),
        "succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "logical_llm_calls": sum(
            int(row.get("logical_scientific_generation_calls", 0) or 0) for row in rows
        ),
        "physical_api_attempts": sum(
            int(row.get("physical_api_attempts", 0) or 0) for row in rows
        ),
        "workers": args.workers,
        "prompt_hash": _prompt_hash(),
        "schema_hash": _sha256(json.dumps(DirectLLM2Proposal.model_json_schema(), sort_keys=True)),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_clock_seconds": time.monotonic() - started_clock,
        **_token_stats(rows),
        "papers": rows,
    }
    write_json(output_root / "evaluation_report.json", "direct_llm2_evaluation_report", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
