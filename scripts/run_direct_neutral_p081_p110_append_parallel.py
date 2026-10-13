#!/usr/bin/env python3
"""Append Direct-Neutral results for P081--P110 to an existing output root.

This entry point intentionally reuses the frozen Direct-Neutral runner and
provider configuration.  Only the batch orchestration is different from the
original 40-paper runner: the existing output root is allowed, while each
new paper is written into ``<output-dir>/papers/<paper-stem>`` and existing
paper directories are protected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import (  # noqa: E402
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.runner.direct_llm_neutral import (  # noqa: E402
    CountingJsonTransport,
    DirectNeutralRunner,
    direct_neutral_prompt_hash,
)
from research_agent.serialization import write_json  # noqa: E402


EXPECTED_PAPER_NUMBERS = tuple(range(81, 111))
EXPECTED_PAPER_COUNT = len(EXPECTED_PAPER_NUMBERS)
DEFAULT_WORKERS = 6
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp3"
DEFAULT_OUTPUT_DIR = Path("/home/tmh/baseline/output_direct")
DEFAULT_BATCH_NAME = "direct_neutral_p081_p110"
PAPER_ID_PATTERN = re.compile(r"(?:^|_)P(?P<number>\d{3})(?:_|$)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paper_number(path: Path) -> int | None:
    match = PAPER_ID_PATTERN.search(path.stem)
    return int(match.group("number")) if match else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append frozen Direct-Neutral results for P081-P110"
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


def _validate_paths(args: argparse.Namespace) -> list[Path]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when provided")
    if not args.batch_name or "/" in args.batch_name or "\\" in args.batch_name:
        raise ValueError("--batch-name must be a simple filename component")

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    output_root = args.output_dir
    if not output_root.is_dir():
        raise FileNotFoundError(
            "append target output directory does not exist; refusing to create a new "
            f"Direct-Neutral root: {output_root}"
        )
    papers_root = output_root / "papers"
    if not papers_root.is_dir():
        raise FileNotFoundError(f"expected existing paper directory: {papers_root}")

    candidates = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    by_number: dict[int, Path] = {}
    for path in candidates:
        number = _paper_number(path)
        if number in EXPECTED_PAPER_NUMBERS:
            if number in by_number:
                raise ValueError(f"duplicate P{number:03d} Markdown input: {path.name}")
            parsed = parse_markdown_file(path)
            build_full_paper_reading_context(parsed)
            by_number[number] = path.resolve()

    missing = [f"P{number:03d}" for number in EXPECTED_PAPER_NUMBERS if number not in by_number]
    if missing:
        raise ValueError(
            f"missing expected P081-P110 Markdown files ({len(missing)}): {', '.join(missing)}"
        )
    paths = [by_number[number] for number in EXPECTED_PAPER_NUMBERS]

    conflicts = [path.stem for path in paths if (papers_root / path.stem).exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite existing paper directories: " + ", ".join(conflicts)
        )
    report_path = output_root / f"evaluation_report_{args.batch_name}.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing append report: {report_path}")
    return paths


def _config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    return config


def _error_payload(error: Exception) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "message": str(error)[:1_000],
    }


def _paper_summary(result: Any, output_dir: Path) -> dict[str, Any]:
    return result.manifest.model_dump(mode="json") | {"output_dir": str(output_dir)}


def _unexpected_failure(source: Path, output_dir: Path, error: Exception) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _error_payload(error)
    write_json(output_dir / "error.json", "direct_neutral_append_error", payload)
    return {
        "paper_id": source.stem,
        "source_md": str(source),
        "baseline_name": "Direct-Neutral",
        "status": "failed",
        "logical_scientific_generation_calls": 0,
        "physical_api_attempts": 0,
        "output_dir": str(output_dir),
        "error_type": payload["error_type"],
        "error_message": payload["message"],
    }


def _token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        row
        for row in rows
        if int(row.get("logical_scientific_generation_calls", 0) or 0) > 0
    ]

    def known_sum(field: str) -> int:
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
    successful_known_totals = [
        int(row["total_tokens"])
        for row in calls
        if row.get("status") == "succeeded"
        and isinstance(row.get("total_tokens"), int)
        and not isinstance(row.get("total_tokens"), bool)
    ]
    return {
        "prompt_tokens": known_sum("prompt_tokens"),
        "completion_tokens": known_sum("completion_tokens"),
        "total_known_tokens": known_sum("total_tokens"),
        "calls_with_usage": complete,
        "calls_missing_usage": len(calls) - complete,
        "token_metadata_complete": complete == len(calls),
        "average_tokens_per_successful_paper": (
            sum(successful_known_totals) / succeeded if succeeded else None
        ),
    }


def _run_append(
    paths: list[Path],
    output_root: Path,
    args: argparse.Namespace,
    config: DeepSeekConfig,
) -> dict[str, Any]:
    papers_root = output_root / "papers"
    started_at = _utc_now()
    started_clock = time.monotonic()
    rows: list[dict[str, Any]] = []
    futures: dict[Future[Any], tuple[Path, Path]] = {}

    def factory() -> DirectNeutralRunner:
        transport = CountingJsonTransport()
        client = DeepSeekClient(config, transport=transport)
        return DirectNeutralRunner(
            client,
            raw_response_source=transport,
            physical_attempt_source=transport,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=config.model,
            provider="deepseek",
        )

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="direct-neutral")
    interrupted = False
    try:
        for path in paths:
            paper_output = papers_root / path.stem
            futures[executor.submit(factory().run, path, paper_output)] = (path, paper_output)
        completed = 0
        for future in as_completed(futures):
            source, paper_output = futures[future]
            completed += 1
            try:
                result = future.result()
                row = _paper_summary(result, paper_output)
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                row = _unexpected_failure(source, paper_output, exc)
            rows.append(row)
            print(
                f"[{completed}/{len(paths)}] {source.stem}: {row.get('status', 'failed')}",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    rows.sort(key=lambda row: str(row.get("paper_id", "")))
    report = {
        "baseline_name": "Direct-Neutral",
        "batch_name": args.batch_name,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_root),
        "paper_output_dir": str(papers_root),
        "paper_range": ["P081", "P110"],
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
        "prompt_hash": direct_neutral_prompt_hash(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_clock_seconds": time.monotonic() - started_clock,
        **_token_stats(rows),
        "papers": rows,
    }
    report_path = output_root / f"evaluation_report_{args.batch_name}.json"
    write_json(report_path, "direct_neutral_append_evaluation_report", report)
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        paths = _validate_paths(args)
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
                    "baseline_name": "Direct-Neutral",
                    "dry_run": True,
                    "input_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "paper_output_dir": str(args.output_dir / "papers"),
                    "total_papers": len(paths),
                    "workers": args.workers,
                    "model": config.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "prompt_hash": direct_neutral_prompt_hash(),
                    "deepseek_config_ready": config_ready,
                    "papers": [path.name for path in paths],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = _run_append(paths, args.output_dir, args, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
