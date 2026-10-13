#!/usr/bin/env python3
"""Run Direct-LLM2 for P011--P110 and assemble P001--P010.

The first ten Direct-LLM2 results are reused from the completed v2 run.  This
script only makes new scientific-generation calls for P011--P110, then copies
the existing P001--P010 paper directories into the new, independent output
root and writes one combined report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from research_agent.ingestion.markdown import (  # noqa: E402
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.models.direct_llm2 import DirectLLM2Proposal  # noqa: E402
from research_agent.runner.direct_llm_neutral import CountingJsonTransport  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402

# Reuse the already-frozen Direct-LLM2 single-paper implementation.  The
# existing P001--P010 entry point is imported only as a library here; its main
# function is never called.
from run_direct_llm2_p001_p010_parallel import (  # noqa: E402
    DirectLLM2SingleRunner,
    _error_payload,
    _prompt_hash,
)


EXPECTED_NEW_NUMBERS = tuple(range(11, 111))
EXPECTED_PRIOR_NUMBERS = tuple(range(1, 11))
EXPECTED_TOTAL_NUMBERS = tuple(range(1, 111))
DEFAULT_WORKERS = 10
DEFAULT_INPUT_DIRS = (REPO_ROOT / "data" / "nlp", REPO_ROOT / "data" / "nlp3")
DEFAULT_PRIOR_OUTPUT_DIR = Path("/home/tmh/final_baseline/output_direct2_v2")
DEFAULT_OUTPUT_DIR = Path("/home/tmh/final_baseline/output_direct2")
DEFAULT_BATCH_NAME = "direct_llm2_p011_p110_assembled"
PROMPT_PATH = REPO_ROOT / "src" / "research_agent" / "prompts" / "direct_llm2.md"
PAPER_ID_PATTERN = re.compile(r"(?:^|_)P(?P<number>\d{3})(?:_|$)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Direct-LLM2 on P011-P110 and assemble P001-P010 results"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="input directory; repeat for multiple roots (defaults to data/nlp and data/nlp3)",
    )
    parser.add_argument("--prior-output-dir", type=Path, default=DEFAULT_PRIOR_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument("--model", default=None, help="optional DeepSeek model override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _paper_number(path: Path) -> int | None:
    match = PAPER_ID_PATTERN.search(path.stem)
    return int(match.group("number")) if match else None


def _read_payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload", value)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object payload in {path}")
    return payload


def _find_inputs(input_dirs: list[Path]) -> dict[int, Path]:
    by_number: dict[int, Path] = {}
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        for path in sorted(input_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".md":
                continue
            number = _paper_number(path)
            if number not in EXPECTED_NEW_NUMBERS:
                continue
            if number in by_number:
                raise ValueError(
                    f"duplicate P{number:03d} Markdown input: {by_number[number]} and {path}"
                )
            parsed = parse_markdown_file(path)
            build_full_paper_reading_context(parsed)
            by_number[number] = path.resolve()
    missing = [f"P{number:03d}" for number in EXPECTED_NEW_NUMBERS if number not in by_number]
    if missing:
        raise ValueError(f"missing P011-P110 Markdown inputs ({len(missing)}): {', '.join(missing)}")
    return by_number


def _prior_paper_dirs(prior_output_dir: Path) -> dict[int, Path]:
    papers_root = prior_output_dir / "papers"
    if not papers_root.is_dir():
        raise FileNotFoundError(f"prior paper directory does not exist: {papers_root}")
    found: dict[int, Path] = {}
    current_prompt_hash = _prompt_hash()
    for paper_dir in sorted(papers_root.iterdir()):
        if not paper_dir.is_dir():
            continue
        manifest_path = paper_dir / "run_manifest.json"
        proposal_path = paper_dir / "final_proposal.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_payload(manifest_path)
        paper_id = str(manifest.get("paper_id", paper_dir.name))
        number = _paper_number(Path(paper_id))
        if number not in EXPECTED_PRIOR_NUMBERS:
            continue
        if number in found:
            raise ValueError(f"duplicate prior P{number:03d} result directories")
        if manifest.get("status") != "succeeded":
            raise ValueError(f"prior P{number:03d} is not succeeded: {manifest.get('status')}")
        if not proposal_path.is_file():
            raise ValueError(f"prior P{number:03d} is missing final_proposal.json")
        if manifest.get("baseline_name") != "Direct-LLM2":
            raise ValueError(f"prior P{number:03d} is not a Direct-LLM2 result")
        if manifest.get("prompt_version") != "direct_llm2_v2":
            raise ValueError(f"prior P{number:03d} has unexpected prompt version")
        if manifest.get("prompt_hash") != current_prompt_hash:
            raise ValueError(f"prior P{number:03d} was generated with a different prompt")
        found[number] = paper_dir
    missing = [f"P{number:03d}" for number in EXPECTED_PRIOR_NUMBERS if number not in found]
    if missing:
        raise ValueError(f"missing prior Direct-LLM2 results: {', '.join(missing)}")
    return found


def _config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    return config


def _validate(args: argparse.Namespace) -> tuple[dict[int, Path], dict[int, Path], DeepSeekConfig]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when provided")
    if not args.batch_name or "/" in args.batch_name or "\\" in args.batch_name:
        raise ValueError("--batch-name must be a simple filename component")
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Direct-LLM2 prompt does not exist: {PROMPT_PATH}")

    input_dirs = list(args.input_dir) if args.input_dir else list(DEFAULT_INPUT_DIRS)
    inputs = _find_inputs(input_dirs)
    prior = _prior_paper_dirs(args.prior_output_dir)

    if args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to use symlink output path: {args.output_dir}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output path: {args.output_dir}")

    config = _config(args)
    config_ready = True
    try:
        config.validate_for_request()
    except Exception:
        config_ready = False
        if not args.dry_run:
            raise
    setattr(args, "config_ready", config_ready)
    setattr(args, "input_dirs", input_dirs)
    return inputs, prior, config


def _unexpected_failure(source: Path, destination: Path, error: Exception) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    detail = _error_payload(error)
    write_json(destination / "error.json", "direct_llm2_error", detail)
    return {
        "paper_id": source.stem,
        "source_md": str(source),
        "baseline_name": "Direct-LLM2",
        "status": "failed",
        "logical_scientific_generation_calls": 0,
        "physical_api_attempts": 0,
        "output_dir": str(destination),
        "error_type": detail["error_type"],
        "error": detail,
    }


def _run_new(
    inputs: dict[int, Path],
    output_root: Path,
    args: argparse.Namespace,
    config: DeepSeekConfig,
) -> list[dict[str, Any]]:
    papers_root = output_root / "papers"
    papers_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], tuple[Path, Path]] = {}
    started_count = len(inputs)

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

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="direct-llm2")
    interrupted = False
    try:
        for number in sorted(inputs):
            source = inputs[number]
            destination = papers_root / source.stem
            futures[executor.submit(factory().run, source, destination)] = (source, destination)
        completed = 0
        for future in as_completed(futures):
            source, destination = futures[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                row = _unexpected_failure(source, destination, exc)
            rows.append(row)
            print(f"[{completed}/{started_count}] {source.stem}: {row.get('status', 'failed')}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return rows


def _copy_prior(
    prior_dirs: dict[int, Path],
    output_root: Path,
) -> list[dict[str, Any]]:
    destination_root = output_root / "papers"
    rows: list[dict[str, Any]] = []
    for number in sorted(prior_dirs):
        source_dir = prior_dirs[number]
        destination_dir = destination_root / source_dir.name
        if destination_dir.exists():
            raise FileExistsError(f"assembly destination already exists: {destination_dir}")
        shutil.copytree(source_dir, destination_dir)
        row = _read_payload(destination_dir / "run_manifest.json")
        row["output_dir"] = str(destination_dir)
        row["assembled_from"] = str(source_dir)
        rows.append(row)
        print(f"[assemble] reused P{number:03d}: {destination_dir.name}", flush=True)
    return rows


def _token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        row for row in rows
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


def main() -> int:
    args = _parser().parse_args()
    try:
        inputs, prior_dirs, config = _validate(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print(json.dumps({
            "baseline_name": "Direct-LLM2",
            "dry_run": True,
            "new_range": ["P011", "P110"],
            "new_papers": len(inputs),
            "reused_range": ["P001", "P010"],
            "reused_papers": len(prior_dirs),
            "total_after_assembly": len(inputs) + len(prior_dirs),
            "input_dirs": [str(path) for path in args.input_dirs],
            "prior_output_dir": str(args.prior_output_dir),
            "output_dir": str(args.output_dir),
            "workers": args.workers,
            "model": config.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_path": str(PROMPT_PATH),
            "prompt_hash": _prompt_hash(),
            "deepseek_config_ready": args.config_ready,
            "new_papers_by_number": {f"P{n:03d}": p.name for n, p in sorted(inputs.items())},
        }, ensure_ascii=False, indent=2))
        return 0

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_clock = time.monotonic()
    new_rows = _run_new(inputs, output_root, args, config)
    prior_rows = _copy_prior(prior_dirs, output_root)
    rows = sorted(new_rows + prior_rows, key=lambda row: str(row.get("paper_id", "")))
    report = {
        "baseline_name": "Direct-LLM2",
        "batch_name": args.batch_name,
        "input_dirs": [str(path) for path in args.input_dirs],
        "prior_output_dir": str(args.prior_output_dir),
        "output_dir": str(output_root),
        "paper_output_dir": str(output_root / "papers"),
        "paper_range": ["P001", "P110"],
        "new_paper_range": ["P011", "P110"],
        "reused_paper_range": ["P001", "P010"],
        "total_papers": len(rows),
        "new_papers": len(new_rows),
        "reused_papers": len(prior_rows),
        "succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "logical_llm_calls": sum(
            int(row.get("logical_scientific_generation_calls", 0) or 0) for row in rows
        ),
        "physical_api_attempts": sum(
            int(row.get("physical_api_attempts", 0) or 0) for row in rows
        ),
        "workers": args.workers,
        "prompt_path": str(PROMPT_PATH),
        "prompt_hash": _prompt_hash(),
        "schema_hash": _sha256(json.dumps(DirectLLM2Proposal.model_json_schema(), sort_keys=True)),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_clock_seconds": time.monotonic() - started_clock,
        **_token_stats(rows),
        "papers": rows,
    }
    write_json(output_root / "evaluation_report.json", "direct_llm2_evaluation_report", report)
    write_json(
        output_root / "assembly_manifest.json",
        "direct_llm2_assembly_manifest",
        {
            "baseline_name": "Direct-LLM2",
            "new_paper_range": ["P011", "P110"],
            "reused_paper_range": ["P001", "P010"],
            "source_prior_output_dir": str(args.prior_output_dir),
            "output_dir": str(output_root),
            "prompt_hash": _prompt_hash(),
            "status": "succeeded" if report["failed"] == 0 else "completed_with_failures",
        },
    )
    print(json.dumps({
        "output_dir": str(output_root),
        "total_papers": report["total_papers"],
        "new_papers": report["new_papers"],
        "reused_papers": report["reused_papers"],
        "succeeded": report["succeeded"],
        "failed": report["failed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
