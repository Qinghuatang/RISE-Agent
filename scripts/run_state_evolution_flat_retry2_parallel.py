#!/usr/bin/env python3
"""Retry the remaining failed/stopped Flat papers from the retry15 run."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.state_evolution_ablation.pilot8_config import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
)
from scripts.run_state_evolution_flat_remaining72_parallel import (  # noqa: E402
    PaperSpec,
    WorkerJob,
    _branch_complete,
    _branch_dir,
    _check_output_policy,
    _check_runtime_config,
    _paper_report_row,
    _run_one,
    _safe_error_text,
    _write_json,
)


DEFAULT_SOURCE_RUN = REPO_ROOT / "outputs" / "state_evolution_ablation_flat_retry15"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_flat_retry2"
EXPECTED_RETRY_PAPERS = 2
RETRY_STATUSES = frozenset({"failed", "stopped"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry the two remaining failed/stopped Flat papers"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="maximum number of paper workers (default: 2)",
    )
    parser.add_argument("--resume", action="store_true", help="rerun incomplete branches in an existing retry directory")
    parser.add_argument("--overwrite", action="store_true", help="explicitly rerun completed retry branches")
    parser.add_argument("--dry-run", action="store_true", help="show the exact retry set without calling services")
    parser.add_argument("--allow-real-services", action="store_true", help="required for a real retry run")
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON file {path}: {_safe_error_text(exc)}") from exc


def discover_retry_papers(source_run: Path, input_dir: Path) -> tuple[PaperSpec, ...]:
    report_path = source_run / "batch_report.json"
    if not report_path.is_file():
        raise SystemExit(f"source batch report does not exist: {report_path}")
    report = _read_json(report_path)
    rows = report.get("papers") if isinstance(report, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"source batch report has no papers list: {report_path}")
    selected = [
        row for row in rows
        if isinstance(row, dict) and row.get("status") in RETRY_STATUSES
    ]
    if len(selected) != EXPECTED_RETRY_PAPERS:
        counts = {
            status: sum(row.get("status") == status for row in selected)
            for status in sorted(RETRY_STATUSES)
        }
        raise SystemExit(
            f"expected exactly {EXPECTED_RETRY_PAPERS} failed/stopped papers in {report_path}, "
            f"found {len(selected)} ({counts})"
        )

    papers: list[PaperSpec] = []
    seen: set[str] = set()
    for row in selected:
        paper_id = row.get("paper_id")
        filename = row.get("filename")
        if not isinstance(paper_id, str) or not paper_id:
            raise SystemExit(f"retry row has invalid paper_id: {row!r}")
        if not isinstance(filename, str) or not filename:
            raise SystemExit(f"retry row has invalid filename for {paper_id}: {row!r}")
        if paper_id in seen:
            raise SystemExit(f"duplicate retry paper_id in source report: {paper_id}")
        seen.add(paper_id)
        source_path = input_dir / filename
        if not source_path.is_file():
            raise SystemExit(f"retry Markdown file does not exist: {source_path}")
        papers.append(PaperSpec(paper_id, filename, source_path))
    papers.sort(key=lambda paper: (paper.short_id, paper.filename))
    return tuple(papers)


def _write_manifest(args: argparse.Namespace, papers: tuple[PaperSpec, ...], runtime_config: dict[str, Any] | None) -> None:
    path = args.output_dir / "batch_manifest.json"
    if path.exists() and (args.resume or args.overwrite):
        return
    _write_json(
        path,
        {
            "experiment": "state_evolution_ablation_flat_retry2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system": "flat",
            "source_run": str(args.source_run.resolve()),
            "input_dir": str(args.input_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "semantic_scholar_config": str(args.semantic_scholar_config.resolve()),
            "selection_rule": "source batch status is failed or stopped",
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
    runtime_config: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    rows = [_paper_report_row(results[paper.short_id], args.output_dir) for paper in papers]
    rows.sort(key=lambda row: row["paper_id"])
    token_fields = (
        "known_prompt_tokens",
        "known_completion_tokens",
        "known_total_tokens",
        "calls_with_token_usage",
        "calls_missing_token_usage",
        "llm_call_record_count",
    )
    token_usage = {field: sum(int(row.get(field) or 0) for row in rows) for field in token_fields}
    token_usage["token_metadata_complete"] = all(
        bool(row.get("token_metadata_complete")) and "token_audit_error" not in row
        for row in rows
    )
    token_usage["prompt_tokens_total"] = (
        token_usage["known_prompt_tokens"] if token_usage["token_metadata_complete"] else None
    )
    token_usage["completion_tokens_total"] = (
        token_usage["known_completion_tokens"] if token_usage["token_metadata_complete"] else None
    )
    token_usage["total_tokens"] = (
        token_usage["known_total_tokens"] if token_usage["token_metadata_complete"] else None
    )
    statuses = [str(row.get("status")) for row in rows]
    report = {
        "experiment": "state_evolution_ablation_flat_retry2",
        "system": "flat",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.source_run.resolve()),
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
        "token_usage": token_usage,
        "token_logging_note": (
            "Token totals are recomputed from each paper's llm_calls.jsonl. "
            "Missing provider usage is not treated as zero, and exact aggregate "
            "totals are null unless every recorded call has complete usage."
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

    papers = discover_retry_papers(args.source_run, args.input_dir)
    _check_output_policy(args.output_dir, resume=args.resume, overwrite=args.overwrite)
    plan = {
        "dry_run": args.dry_run,
        "source_run": str(args.source_run.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "workers": args.workers,
        "max_steps": DEFAULT_MAX_STEPS,
        "max_searches": DEFAULT_MAX_SEARCHES,
        "paper_count": len(papers),
        "papers": [{"paper_id": paper.short_id, "filename": paper.filename} for paper in papers],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if not args.allow_real_services:
        raise SystemExit("real Flat retry requires --allow-real-services")
    try:
        runtime_config = _check_runtime_config(args)
    except Exception as exc:
        raise SystemExit(f"real provider configuration is invalid: {_safe_error_text(exc)}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args, papers, runtime_config)
    for paper in papers:
        branch = _branch_dir(paper, args.output_dir)
        if branch.exists() and not _branch_complete(paper, args.output_dir) and not (
            args.resume or args.overwrite
        ):
            raise SystemExit(
                f"incomplete retry branch exists: {branch}; use --resume or --overwrite"
            )

    started_at = datetime.now(timezone.utc).isoformat()
    allow_overwrite = args.resume or args.overwrite
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
                f"[Flat retry2 {completed}/{len(jobs)}] "
                f"{job.paper.short_id}: {result.get('status', 'unknown')}",
                flush=True,
            )
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted by user; completed retry artifacts were retained.", file=sys.stderr)
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
