#!/usr/bin/env python3
"""Retry non-budget failures from a completed paper-level Ours batch.

The previous evaluation report is the source of the retry cohort. Papers that
stopped normally because the scientific budget was exhausted are deliberately
excluded. Every selected paper is executed through the existing per-paper
executor and the existing paper-level parallel batch runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.llm.config import DeepSeekConfig
from research_agent.retrieval.arxiv import ArxivConfig
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig
from research_agent.runner.batch_evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationError,
    make_real_paper_executor,
)
from research_agent.runner.parallel_batch import (
    ParallelBatchConfig,
    ParallelBatchEvaluationRunner,
)


MAX_STEPS = 12
MAX_SEARCHES = 2
DEFAULT_PREVIOUS_OUTPUT = REPO_ROOT / "outputs" / "ours_parallel_nlp40_workers8_2"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ours_parallel_nlp40_retry_non_budget"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retry failed papers from a previous Ours batch, excluding normal "
            "budget-exhausted stops"
        )
    )
    parser.add_argument(
        "--previous-output-dir",
        type=Path,
        default=DEFAULT_PREVIOUS_OUTPUT,
        help="completed batch directory containing evaluation_report.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nlp",
        help="directory containing the original Markdown papers",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-name", default="ours_parallel_nlp40_retry_non_budget")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic_scholar.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_report(path: Path) -> dict[str, Any]:
    report_path = path / "evaluation_report.json"
    if not report_path.is_file():
        raise BatchEvaluationError(f"previous evaluation report does not exist: {report_path}")
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvaluationError(f"cannot read previous evaluation report: {report_path}") from exc
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise BatchEvaluationError(f"invalid evaluation report payload: {report_path}")
    return payload


def _retry_selection(
    report: dict[str, Any],
    input_dir: Path,
) -> tuple[list[Path], list[str], list[str]]:
    papers = report["papers"]
    selected: list[tuple[str, Path]] = []
    excluded_budget: list[str] = []
    for item in papers:
        if not isinstance(item, dict):
            raise BatchEvaluationError("previous evaluation report contains a malformed paper row")
        paper_id = item.get("paper_id")
        status = item.get("status")
        failure_type = item.get("failure_type")
        markdown_path = item.get("markdown_path")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise BatchEvaluationError("previous evaluation report contains a paper without paper_id")
        if status == "stopped" and failure_type == "budget_exhausted":
            excluded_budget.append(paper_id)
            continue
        if status == "committed":
            continue
        if not isinstance(markdown_path, str) or not markdown_path:
            raise BatchEvaluationError(f"missing markdown_path for retry paper: {paper_id}")
        source = Path(markdown_path)
        if not source.is_file():
            source = input_dir / Path(markdown_path).name
        if not source.is_file() or source.suffix.lower() != ".md":
            raise BatchEvaluationError(
                f"Markdown source for retry paper is missing: {paper_id}: {markdown_path}"
            )
        selected.append((paper_id, source.resolve()))

    selected.sort(key=lambda item: (item[0], item[1].name))
    paper_ids = [paper_id for paper_id, _ in selected]
    paths = [path for _, path in selected]
    if len(set(paper_ids)) != len(paper_ids):
        raise BatchEvaluationError("previous evaluation report contains duplicate retry paper IDs")
    return paths, paper_ids, sorted(excluded_budget)


def _preflight_config(args: argparse.Namespace, *, require_services: bool) -> dict[str, object]:
    if not args.input_dir.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {args.input_dir}")
    if not args.semantic_scholar_config.is_file():
        raise BatchEvaluationError(
            f"Semantic Scholar config does not exist: {args.semantic_scholar_config}"
        )
    deepseek = DeepSeekConfig.from_env()
    semantic = SemanticScholarConfig.from_file(args.semantic_scholar_config)
    arxiv = ArxivConfig.from_env()
    if require_services:
        deepseek.validate_for_request()
        semantic.validate_for_request()
    return {
        "deepseek_configured": bool(deepseek.api_key and deepseek.base_url and deepseek.model),
        "deepseek_model": deepseek.model,
        "semantic_scholar_configured": bool(semantic.base_url),
        "arxiv_cache_dir": arxiv.cache_dir,
        "semantic_scholar_cache_dir": semantic.cache_dir,
    }


class _SelectedPaperRunner(ParallelBatchEvaluationRunner):
    """Use report-selected original paths without staging or copying data."""

    def __init__(self, *args: Any, selected_paths: list[Path], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._selected_paths = selected_paths

    def discover_papers(self) -> list[Path]:
        expected = self.parallel_config.expected_papers
        if expected is not None and len(self._selected_paths) != expected:
            raise BatchEvaluationError(
                f"expected {expected} retry papers, found {len(self._selected_paths)}"
            )
        return list(self._selected_paths)


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    report = _load_report(args.previous_output_dir)
    selected_paths, selected_ids, excluded_budget = _retry_selection(report, args.input_dir)
    if not selected_paths:
        raise SystemExit("No non-budget failed papers need retry")
    config_info = _preflight_config(args, require_services=not args.dry_run)

    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=None,
        allow_overwrite=False,
    )
    parallel_config = ParallelBatchConfig(
        workers=args.workers,
        expected_papers=len(selected_paths),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "previous_output_dir": str(args.previous_output_dir),
                    "input_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "batch_name": args.batch_name,
                    "workers": args.workers,
                    "max_steps": MAX_STEPS,
                    "max_searches": MAX_SEARCHES,
                    "retry_count": len(selected_ids),
                    "retry_papers": selected_ids,
                    "excluded_budget_exhausted": excluded_budget,
                    "config": config_info,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    executor = make_real_paper_executor(
        semantic_scholar_config=args.semantic_scholar_config,
        max_steps=MAX_STEPS,
        max_searches=MAX_SEARCHES,
        allow_real_services=True,
        allow_overwrite=False,
    )
    runner = _SelectedPaperRunner(
        batch_config,
        executor,
        parallel_config,
        selected_paths=selected_paths,
    )
    result = runner.run()
    selection_artifact = {
        "previous_output_dir": str(args.previous_output_dir),
        "excluded_budget_exhausted": excluded_budget,
        "retry_papers": selected_ids,
        "max_steps": MAX_STEPS,
        "max_searches": MAX_SEARCHES,
        "workers": args.workers,
    }
    (args.output_dir / "retry_selection.json").write_text(
        json.dumps(selection_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
