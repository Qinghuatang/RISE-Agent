#!/usr/bin/env python3
"""Retry the six papers without a final proposal in output_final_30.

This wrapper reuses the existing per-paper pilot and paper-level parallel
runner.  It only discovers the retry cohort and adds simple progress output;
it does not change the scientific workflow or its budgets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_remaining_nlp30_parallel import (  # noqa: E402
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    MAX_SEARCHES,
    MAX_STEPS,
    ProgressSelectedPaperRunner,
    _preflight_config,
)
from research_agent.runner.batch_evaluation import (  # noqa: E402
    BatchEvaluationConfig,
    BatchEvaluationError,
    make_real_paper_executor,
)
from research_agent.runner.parallel_batch import ParallelBatchConfig  # noqa: E402


EXPECTED_INPUT_MARKDOWNS = 80
EXPECTED_SOURCE_PAPERS = 30
EXPECTED_RETRY_PAPERS = 6
DEFAULT_WORKERS = 6

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_SOURCE_RUN = REPO_ROOT / "outputs" / "output_final_30"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final_6"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry exactly the six incomplete papers from output_final_30"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default="ours_retry_output_final_6")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _report_payload(source_run: Path) -> dict[str, Any]:
    report_path = source_run / "evaluation_report.json"
    if not report_path.is_file():
        raise BatchEvaluationError(f"missing source evaluation report: {report_path}")
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvaluationError(f"cannot read source evaluation report: {report_path}") from exc
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise BatchEvaluationError(f"invalid source evaluation report: {report_path}")
    if payload.get("paper_count") != EXPECTED_SOURCE_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_SOURCE_PAPERS} papers in source report, "
            f"found {payload.get('paper_count')}"
        )
    return payload


def _discover_retry_papers(args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
    if not args.input_dir.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {args.input_dir}")
    source_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )
    if len(source_paths) != EXPECTED_INPUT_MARKDOWNS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_INPUT_MARKDOWNS} Markdown files, "
            f"found {len(source_paths)}: {args.input_dir}"
        )
    source_by_id = {path.stem: path.resolve() for path in source_paths}
    if len(source_by_id) != len(source_paths):
        raise BatchEvaluationError("input Markdown files contain duplicate filename stems")

    payload = _report_payload(args.source_run)
    retry: list[Path] = []
    seen: set[str] = set()
    for row in payload["papers"]:
        if not isinstance(row, dict) or not isinstance(row.get("paper_id"), str):
            raise BatchEvaluationError("source report contains malformed paper row")
        paper_id = row["paper_id"]
        if paper_id in seen:
            raise BatchEvaluationError(f"duplicate paper_id in source report: {paper_id}")
        seen.add(paper_id)
        source = source_by_id.get(paper_id)
        if source is None:
            raise BatchEvaluationError(f"source Markdown missing for paper: {paper_id}")
        final_proposal = args.source_run / paper_id / "final_research_proposal.json"
        complete = row.get("status") == "committed" and final_proposal.is_file()
        if not complete:
            retry.append(source)

    retry.sort(key=lambda path: path.name)
    if len(seen) != EXPECTED_SOURCE_PAPERS:
        raise BatchEvaluationError(
            f"source report contains {len(seen)} unique papers, expected {EXPECTED_SOURCE_PAPERS}"
        )
    if len(retry) != EXPECTED_RETRY_PAPERS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_RETRY_PAPERS} incomplete papers, found {len(retry)}"
        )
    return retry, {
        "input_markdown_count": len(source_paths),
        "source_run_paper_count": len(seen),
        "retry_paper_count": len(retry),
        "source_run": str(args.source_run),
        "retry_papers": [path.name for path in retry],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    papers, selection_info = _discover_retry_papers(args)
    config_info = _preflight_config(args, require_services=not args.dry_run)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "batch_name": args.batch_name,
                    "input_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "workers": args.workers,
                    "max_steps": MAX_STEPS,
                    "max_searches": MAX_SEARCHES,
                    "config": config_info,
                    **selection_info,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=None,
        allow_overwrite=False,
    )
    executor = make_real_paper_executor(
        semantic_scholar_config=args.semantic_scholar_config,
        max_steps=MAX_STEPS,
        max_searches=MAX_SEARCHES,
        allow_real_services=True,
        allow_overwrite=False,
    )
    runner = ProgressSelectedPaperRunner(
        batch_config,
        executor,
        parallel_config=ParallelBatchConfig(
            workers=args.workers,
            expected_papers=EXPECTED_RETRY_PAPERS,
        ),
        selected_papers=papers,
    )
    report = runner.run()
    print(
        json.dumps(
            {
                "batch_name": args.batch_name,
                "output_dir": str(args.output_dir),
                "paper_count": report.paper_count,
                "completed_count": report.completed_count,
                "failed_count": report.failed_count,
                "stopped_count": report.stopped_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
