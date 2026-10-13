#!/usr/bin/env python3
"""Run RISE-Flat for the 72 papers outside the eight-paper pilot.

This is an execution-only wrapper around the existing ``FlatRunner``.  It
does not implement or alter the Flat scientific loop.  Each process owns one
paper runner, while the parent process owns batch discovery, progress output,
and final aggregation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.state_evolution_ablation.flat_runner import FlatRunConfig, FlatRunner
from experiments.state_evolution_ablation.pilot8_config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    PILOT_SEEDS,
)
from research_agent.llm.config import DeepSeekConfig
from research_agent.retrieval.arxiv import ArxivConfig
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig


EXPECTED_INPUT_PAPERS = 80
EXPECTED_REMAINING_PAPERS = 72
PAPER_ID_PATTERN = re.compile(r"(P\d{3})(?:_|$)")
PILOT_PAPER_IDS = frozenset(seed.short_id for seed in PILOT_SEEDS)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_flat_remaining72"


@dataclass(frozen=True)
class PaperSpec:
    short_id: str
    filename: str
    source_path: Path


@dataclass(frozen=True)
class WorkerJob:
    paper: PaperSpec
    output_dir: Path
    semantic_scholar_config: Path
    max_steps: int
    max_searches: int
    allow_overwrite: bool
    overwrite_completed: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RISE-Flat for the 72 non-pilot papers with paper-level parallelism"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="maximum number of paper workers (default: 12)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the output directory and rerun incomplete paper branches",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly rerun existing paper branches and overwrite their artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the 80-paper input and print the 72-paper plan without services",
    )
    parser.add_argument(
        "--allow-real-services",
        action="store_true",
        help="required for a real run because Flat may call external services",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_short_id(path: Path) -> str:
    match = PAPER_ID_PATTERN.search(path.stem)
    if match is None:
        raise SystemExit(f"cannot determine Pxxx identity from Markdown filename: {path.name}")
    return match.group(1)


def discover_remaining_papers(input_dir: Path) -> tuple[PaperSpec, ...]:
    """Discover every input Markdown and exclude exactly the frozen pilot IDs."""

    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    paths = tuple(sorted((path for path in input_dir.glob("*.md") if path.is_file()), key=lambda p: p.name))
    if len(paths) != EXPECTED_INPUT_PAPERS:
        raise SystemExit(
            f"expected exactly {EXPECTED_INPUT_PAPERS} Markdown files in {input_dir}, "
            f"found {len(paths)}"
        )

    seen: dict[str, Path] = {}
    for path in paths:
        short_id = _paper_short_id(path)
        if short_id in seen:
            raise SystemExit(
                f"duplicate paper identity {short_id}: {seen[short_id].name} and {path.name}"
            )
        seen[short_id] = path

    missing_pilots = sorted(PILOT_PAPER_IDS - set(seen))
    if missing_pilots:
        raise SystemExit(
            "the frozen eight-paper pilot IDs are not all present in the input directory: "
            + ", ".join(missing_pilots)
        )

    remaining = tuple(
        PaperSpec(short_id=_paper_short_id(path), filename=path.name, source_path=path)
        for path in paths
        if _paper_short_id(path) not in PILOT_PAPER_IDS
    )
    if len(remaining) != EXPECTED_REMAINING_PAPERS:
        raise SystemExit(
            f"expected exactly {EXPECTED_REMAINING_PAPERS} non-pilot papers, found {len(remaining)}"
        )
    return remaining


def _check_output_policy(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not (resume or overwrite):
        raise SystemExit(
            f"refusing to reuse non-empty output directory: {output_dir}; "
            "choose a new directory, or explicitly use --resume/--overwrite"
        )


def _check_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    deepseek = DeepSeekConfig.from_env()
    semantic = SemanticScholarConfig.from_file(args.semantic_scholar_config)
    arxiv = ArxivConfig.from_env()
    deepseek.validate_for_request()
    semantic.validate_for_request()
    return {
        "provider": "deepseek",
        "model": deepseek.model,
        "deepseek_base_url": deepseek.base_url,
        "semantic_scholar_base_url": semantic.base_url,
        "arxiv_cache_dir": arxiv.cache_dir,
        "semantic_scholar_cache_dir": semantic.cache_dir,
        "api_keys_configured": {
            "deepseek": bool(deepseek.api_key),
            "semantic_scholar": bool(semantic.api_key),
        },
    }


def _metadata_path(paper: PaperSpec, output_dir: Path) -> Path:
    return output_dir / paper.short_id / "flat" / "run_metadata.json"


def _branch_dir(paper: PaperSpec, output_dir: Path) -> Path:
    return output_dir / paper.short_id / "flat"


def _load_status(paper: PaperSpec, output_dir: Path) -> str | None:
    path = _metadata_path(paper, output_dir)
    if not path.is_file():
        return None
    try:
        value = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid"
    return value.get("status") if isinstance(value, dict) else "invalid"


def _branch_complete(paper: PaperSpec, output_dir: Path) -> bool:
    return _load_status(paper, output_dir) in {"committed", "stopped"} and (
        _branch_dir(paper, output_dir) / "final_proposal.md"
    ).is_file()


def _safe_error_text(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()[:1000] or type(exc).__name__


def _run_one(job: WorkerJob) -> dict[str, Any]:
    """Worker entry point; all mutable runner/LLM state is created in this process."""

    paper = job.paper
    branch = _branch_dir(paper, job.output_dir)
    if _branch_complete(paper, job.output_dir) and not job.overwrite_completed:
        return {
            "paper_id": paper.short_id,
            "filename": paper.filename,
            "status": "skipped_completed",
            "metadata_path": str(_metadata_path(paper, job.output_dir)),
        }

    config = FlatRunConfig(
        markdown_path=paper.source_path,
        output_dir=branch,
        semantic_scholar_config=job.semantic_scholar_config,
        max_steps=job.max_steps,
        max_searches=job.max_searches,
        allow_real_services=True,
        allow_overwrite=job.allow_overwrite,
    )
    try:
        metadata = FlatRunner(config).run()
        return {
            "paper_id": paper.short_id,
            "filename": paper.filename,
            "status": str(metadata.get("status", "completed")),
            "metadata_path": str(_metadata_path(paper, job.output_dir)),
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        branch.mkdir(parents=True, exist_ok=True)
        _write_json(
            branch / "error.json",
            {"type": type(exc).__name__, "message": _safe_error_text(exc)},
        )
        return {
            "paper_id": paper.short_id,
            "filename": paper.filename,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": _safe_error_text(exc)},
            "metadata_path": str(_metadata_path(paper, job.output_dir)),
        }


def _read_call_records(branch: Path) -> tuple[list[dict[str, Any]], str | None]:
    path = branch / "llm_calls.jsonl"
    if not path.is_file():
        return [], "llm_calls.jsonl is missing"
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return records, f"llm_calls.jsonl line {line_number} is not an object"
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return records, f"cannot read llm_calls.jsonl: {_safe_error_text(exc)}"
    return records, None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _token_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    known_prompt = sum(value for value in (_int_or_none(item.get("prompt_tokens")) for item in values) if value is not None)
    known_completion = sum(value for value in (_int_or_none(item.get("completion_tokens")) for item in values) if value is not None)
    known_total = sum(value for value in (_int_or_none(item.get("total_tokens")) for item in values) if value is not None)
    complete_calls = sum(
        1
        for item in values
        if _int_or_none(item.get("prompt_tokens")) is not None
        and _int_or_none(item.get("completion_tokens")) is not None
        and _int_or_none(item.get("total_tokens")) is not None
    )
    missing_calls = len(values) - complete_calls
    complete = missing_calls == 0
    return {
        "llm_call_record_count": len(values),
        "known_prompt_tokens": known_prompt,
        "known_completion_tokens": known_completion,
        "known_total_tokens": known_total,
        "calls_with_token_usage": complete_calls,
        "calls_missing_token_usage": missing_calls,
        "token_metadata_complete": complete,
        "prompt_tokens_total": known_prompt if complete else None,
        "completion_tokens_total": known_completion if complete else None,
        "total_tokens": known_total if complete else None,
    }


def _paper_report_row(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paper_id = str(result["paper_id"])
    branch = output_dir / paper_id / "flat"
    metadata_path = branch / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            value = _read_json(metadata_path)
            if isinstance(value, dict):
                metadata = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            metadata = {}
    records, audit_error = _read_call_records(branch)
    row: dict[str, Any] = {
        "paper_id": paper_id,
        "filename": result.get("filename"),
        "status": metadata.get("status", result.get("status", "unknown")),
        "output_dir": str(branch.resolve()),
        "actual_steps": metadata.get("actual_steps"),
        "actual_searches": metadata.get("actual_searches"),
        "failure": metadata.get("failure") or result.get("error"),
    }
    row.update(_token_summary(records))
    if audit_error:
        row["token_audit_error"] = audit_error
    return row


def _write_manifest(
    args: argparse.Namespace,
    papers: tuple[PaperSpec, ...],
    runtime_config: dict[str, Any] | None,
) -> None:
    path = args.output_dir / "batch_manifest.json"
    if path.exists() and (args.resume or args.overwrite):
        return
    _write_json(
        path,
        {
            "experiment": "state_evolution_ablation_flat_remaining72",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system": "flat",
            "input_dir": str(args.input_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "semantic_scholar_config": str(args.semantic_scholar_config.resolve()),
            "excluded_pilot_ids": sorted(PILOT_PAPER_IDS),
            "input_paper_count": EXPECTED_INPUT_PAPERS,
            "selected_paper_count": len(papers),
            "selected_papers": [
                {"paper_id": paper.short_id, "filename": paper.filename}
                for paper in papers
            ],
            "workers": args.workers,
            "max_steps": DEFAULT_MAX_STEPS,
            "max_searches": DEFAULT_MAX_SEARCHES,
            "runtime_config": runtime_config,
            "token_logging": {
                "source": "per-paper flat/llm_calls.jsonl",
                "missing_usage_is_not_imputed_as_zero": True,
                "retry_records_are_counted_as_call_records": True,
            },
        },
    )


def _write_batch_report(
    args: argparse.Namespace,
    papers: tuple[PaperSpec, ...],
    results: dict[str, dict[str, Any]],
    runtime_config: dict[str, Any] | None,
    started_at: str,
) -> dict[str, Any]:
    rows = [_paper_report_row(results[paper.short_id], args.output_dir) for paper in papers]
    rows.sort(key=lambda row: row["paper_id"])
    token_keys = (
        "known_prompt_tokens",
        "known_completion_tokens",
        "known_total_tokens",
        "calls_with_token_usage",
        "calls_missing_token_usage",
        "llm_call_record_count",
    )
    token_totals = {key: sum(int(row.get(key) or 0) for row in rows) for key in token_keys}
    token_totals["token_metadata_complete"] = all(
        bool(row.get("token_metadata_complete")) and "token_audit_error" not in row
        for row in rows
    )
    token_totals["prompt_tokens_total"] = (
        token_totals["known_prompt_tokens"] if token_totals["token_metadata_complete"] else None
    )
    token_totals["completion_tokens_total"] = (
        token_totals["known_completion_tokens"] if token_totals["token_metadata_complete"] else None
    )
    token_totals["total_tokens"] = (
        token_totals["known_total_tokens"] if token_totals["token_metadata_complete"] else None
    )
    statuses = [str(row.get("status")) for row in rows]
    report = {
        "experiment": "state_evolution_ablation_flat_remaining72",
        "system": "flat",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "workers": args.workers,
        "max_steps": DEFAULT_MAX_STEPS,
        "max_searches": DEFAULT_MAX_SEARCHES,
        "total_papers": len(papers),
        "committed": statuses.count("committed"),
        "stopped": statuses.count("stopped"),
        "failed": statuses.count("failed"),
        "skipped_completed": statuses.count("skipped_completed"),
        "runtime_config": runtime_config,
        "token_usage": token_totals,
        "token_logging_note": (
            "Known token fields are recomputed from each paper's llm_calls.jsonl. "
            "Missing provider usage is not treated as zero; exact total fields are "
            "null unless every recorded call has prompt, completion, and total usage."
        ),
        "papers": rows,
    }
    _write_json(args.output_dir / "batch_report.json", report)
    return report


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("use either --resume or --overwrite, not both")

    papers = discover_remaining_papers(args.input_dir)
    _check_output_policy(args.output_dir, resume=args.resume, overwrite=args.overwrite)

    plan = {
        "dry_run": args.dry_run,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "workers": args.workers,
        "max_steps": DEFAULT_MAX_STEPS,
        "max_searches": DEFAULT_MAX_SEARCHES,
        "excluded_pilot_ids": sorted(PILOT_PAPER_IDS),
        "paper_count": len(papers),
        "papers": [{"paper_id": paper.short_id, "filename": paper.filename} for paper in papers],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_real_services:
        raise SystemExit("real Flat execution requires --allow-real-services")
    try:
        runtime_config = _check_runtime_config(args)
    except Exception as exc:
        raise SystemExit(f"real provider configuration is invalid: {_safe_error_text(exc)}") from exc

    started_at = datetime.now(timezone.utc).isoformat()
    _write_manifest(args, papers, runtime_config)

    # Refuse incomplete branches unless the user explicitly selected resume or
    # overwrite.  Completed branches are skipped by workers on --resume.
    for paper in papers:
        branch = _branch_dir(paper, args.output_dir)
        if branch.exists() and not _branch_complete(paper, args.output_dir) and not (
            args.resume or args.overwrite
        ):
            raise SystemExit(
                f"incomplete Flat branch exists: {branch}; use --resume or --overwrite"
            )

    allow_overwrite = args.overwrite or args.resume
    jobs = [
        WorkerJob(
            paper=paper,
            output_dir=args.output_dir,
            semantic_scholar_config=args.semantic_scholar_config,
            max_steps=DEFAULT_MAX_STEPS,
            max_searches=DEFAULT_MAX_SEARCHES,
            allow_overwrite=allow_overwrite,
            overwrite_completed=args.overwrite,
        )
        for paper in papers
    ]
    results: dict[str, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(max_workers=args.workers)
    future_to_job = {}
    try:
        for job in jobs:
            future_to_job[executor.submit(_run_one, job)] = job
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                branch = _branch_dir(job.paper, args.output_dir)
                branch.mkdir(parents=True, exist_ok=True)
                error = {"type": type(exc).__name__, "message": _safe_error_text(exc)}
                _write_json(branch / "error.json", error)
                result = {
                    "paper_id": job.paper.short_id,
                    "filename": job.paper.filename,
                    "status": "failed",
                    "error": error,
                }
            results[job.paper.short_id] = result
            completed += 1
            print(
                f"[Flat remaining72 {completed}/{len(jobs)}] "
                f"{job.paper.short_id}: {result.get('status', 'unknown')}",
                flush=True,
            )
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted by user; completed paper artifacts were retained.", file=sys.stderr)
        raise
    else:
        executor.shutdown(wait=True)

    report = _write_batch_report(args, papers, results, runtime_config, started_at)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "batch_report": str((args.output_dir / "batch_report.json").resolve()),
                "total_papers": report["total_papers"],
                "committed": report["committed"],
                "stopped": report["stopped"],
                "failed": report["failed"],
                "known_total_tokens": report["token_usage"]["known_total_tokens"],
                "calls_missing_token_usage": report["token_usage"]["calls_missing_token_usage"],
                "token_metadata_complete": report["token_usage"]["token_metadata_complete"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
