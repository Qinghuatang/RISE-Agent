#!/usr/bin/env python3
"""Replace incomplete papers in output_final_6 with a fresh retry.

The script reuses the existing per-paper pilot and paper-level parallel
runner.  It discovers incomplete papers from the existing six-paper report,
removes only their exact per-paper directories, runs those papers, and then
restores a six-paper aggregate report containing both retained successes and
new retry results.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from research_agent.models.evaluation import PaperEvaluation  # noqa: E402
from research_agent.runner.batch_evaluation import (  # noqa: E402
    BatchEvaluationConfig,
    BatchEvaluationError,
    build_evaluation_report,
    make_real_paper_executor,
)
from research_agent.runner.parallel_batch import ParallelBatchConfig  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402


EXPECTED_INPUT_MARKDOWNS = 80
EXPECTED_SOURCE_PAPERS = 6
EXPECTED_RETRY_PAPERS = 3
DEFAULT_WORKERS = 3

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_SOURCE_RUN = REPO_ROOT / "outputs" / "output_final_6"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final_6"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry the three incomplete papers in output_final_6"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default="ours_retry_output_final_6_replace")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_report(source_run: Path) -> dict[str, Any]:
    report_path = source_run / "evaluation_report.json"
    if not report_path.is_file():
        raise BatchEvaluationError(f"missing evaluation report: {report_path}")
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvaluationError(f"cannot read evaluation report: {report_path}") from exc
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise BatchEvaluationError(f"invalid evaluation report: {report_path}")
    if payload.get("paper_count") != EXPECTED_SOURCE_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_SOURCE_PAPERS} papers in report, "
            f"found {payload.get('paper_count')}"
        )
    return payload


def _discover_retry(
    args: argparse.Namespace,
) -> tuple[list[Path], list[PaperEvaluation], dict[str, Any]]:
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

    payload = _read_report(args.source_run)
    retry: list[Path] = []
    retained: list[PaperEvaluation] = []
    seen: set[str] = set()
    for row in payload["papers"]:
        if not isinstance(row, dict) or not isinstance(row.get("paper_id"), str):
            raise BatchEvaluationError("evaluation report contains malformed paper row")
        paper_id = row["paper_id"]
        if paper_id in seen:
            raise BatchEvaluationError(f"duplicate paper_id in evaluation report: {paper_id}")
        seen.add(paper_id)
        markdown_path = source_by_id.get(paper_id)
        if markdown_path is None:
            raise BatchEvaluationError(f"source Markdown missing for paper: {paper_id}")
        paper_dir = args.source_run / paper_id
        final_proposal = paper_dir / "final_research_proposal.json"
        complete = row.get("status") == "committed" and final_proposal.is_file()
        if complete:
            try:
                retained.append(PaperEvaluation.model_validate(row))
            except Exception as exc:
                raise BatchEvaluationError(
                    f"retained paper row is not a valid PaperEvaluation: {paper_id}"
                ) from exc
        else:
            retry.append(markdown_path)

    retry.sort(key=lambda path: path.name)
    if len(seen) != EXPECTED_SOURCE_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_SOURCE_PAPERS} unique papers in source report, found {len(seen)}"
        )
    if len(retained) != EXPECTED_SOURCE_PAPERS - EXPECTED_RETRY_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_SOURCE_PAPERS - EXPECTED_RETRY_PAPERS} retained successes, "
            f"found {len(retained)}"
        )
    if len(retry) != EXPECTED_RETRY_PAPERS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_RETRY_PAPERS} incomplete papers, found {len(retry)}"
        )
    return retry, retained, {
        "input_markdown_count": len(source_paths),
        "source_report_paper_count": len(seen),
        "retained_success_count": len(retained),
        "retry_paper_count": len(retry),
        "source_run": str(args.source_run),
        "retry_papers": [path.name for path in retry],
    }


def _remove_only_retry_dirs(source_run: Path, papers: list[Path]) -> None:
    for markdown_path in papers:
        paper_dir = source_run / markdown_path.stem
        if not paper_dir.exists() and not paper_dir.is_symlink():
            # A previous interrupted/manual cleanup may already have removed
            # the incomplete artifact.  This is safe: the report has already
            # identified this paper as needing replacement, and the runner
            # will create a fresh directory for it.
            print(
                f"[REPLACE] previous incomplete artifacts already absent: {paper_dir.name}",
                flush=True,
            )
            continue
        if not paper_dir.is_dir() or paper_dir.is_symlink():
            raise BatchEvaluationError(f"retry paper directory does not exist: {paper_dir}")
        print(f"[REPLACE] removing previous incomplete artifacts: {paper_dir.name}", flush=True)
        shutil.rmtree(paper_dir)


def _assert_retry_dirs_absent(output_dir: Path, papers: list[Path]) -> None:
    """Keep the existing batch root but forbid per-paper overwrites."""

    existing = [output_dir / paper.stem for paper in papers if (output_dir / paper.stem).exists()]
    if existing:
        raise BatchEvaluationError(
            "retry paper output unexpectedly exists after cleanup: "
            + ", ".join(map(str, existing))
        )


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.source_run.resolve() != args.output_dir.resolve():
        raise SystemExit("--source-run and --output-dir must be the same output_final_6 directory")
    if not args.output_dir.is_dir():
        raise SystemExit(f"output directory does not exist: {args.output_dir}")

    retry_papers, retained, selection_info = _discover_retry(args)
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

    _remove_only_retry_dirs(args.output_dir, retry_papers)
    _assert_retry_dirs_absent(args.output_dir, retry_papers)
    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=None,
        # The batch root already contains retained successful papers.  The
        # exact retry directories were removed and checked above, so allowing
        # the existing root does not permit overwriting retained papers.
        allow_overwrite=True,
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
        selected_papers=retry_papers,
    )
    retry_report = runner.run()
    all_papers = retained + list(retry_report.papers)
    all_papers.sort(key=lambda paper: paper.paper_id)
    merged_report = build_evaluation_report(
        batch_config,
        all_papers,
        [paper.paper_id for paper in all_papers],
    )
    write_json(args.output_dir / "evaluation_report.json", "evaluation_report", merged_report)
    print(
        json.dumps(
            {
                "batch_name": args.batch_name,
                "output_dir": str(args.output_dir),
                "paper_count": merged_report.paper_count,
                "completed_count": merged_report.completed_count,
                "failed_count": merged_report.failed_count,
                "stopped_count": merged_report.stopped_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
