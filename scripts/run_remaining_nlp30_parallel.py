#!/usr/bin/env python3
"""Run the 30 papers not present in the completed Ours result set.

This is an execution-only wrapper around the existing per-paper pilot and
paper-level parallel runner.  It does not implement or alter any scientific
workflow logic.  The completed set is read from ``output_final/papers`` and
the remaining Markdown files are selected by exact filename stem.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.retrieval.arxiv import ArxivConfig  # noqa: E402
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig  # noqa: E402
from research_agent.runner.batch_evaluation import (  # noqa: E402
    BatchEvaluationConfig,
    BatchEvaluationError,
    PaperEvaluation,
    make_real_paper_executor,
)
from research_agent.runner.parallel_batch import (  # noqa: E402
    ParallelBatchConfig,
    ParallelBatchEvaluationRunner,
)


EXPECTED_INPUT_MARKDOWNS = 80
EXPECTED_COMPLETED_PAPERS = 50
EXPECTED_REMAINING_PAPERS = EXPECTED_INPUT_MARKDOWNS - EXPECTED_COMPLETED_PAPERS
DEFAULT_WORKERS = 6
MAX_STEPS = 12
MAX_SEARCHES = 2

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_COMPLETED_DIR = REPO_ROOT / "outputs" / "output_final" / "papers"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final_30"
DEFAULT_SEMANTIC_SCHOLAR_CONFIG = REPO_ROOT / "config" / "semantic_scholar.json"


class SelectedPaperRunner(ParallelBatchEvaluationRunner):
    """Reuse the existing runner while supplying an exact remaining cohort."""

    def __init__(
        self,
        *args: Any,
        selected_papers: list[Path],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._selected_papers = list(selected_papers)

    def discover_papers(self) -> list[Path]:
        expected = self.parallel_config.expected_papers
        if expected is not None and len(self._selected_papers) != expected:
            raise BatchEvaluationError(
                f"expected exactly {expected} selected papers, "
                f"found {len(self._selected_papers)}"
            )
        return list(self._selected_papers)


class ProgressSelectedPaperRunner(SelectedPaperRunner):
    """Add minimal thread-safe start/completion progress reporting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._progress_lock = threading.Lock()
        self._started = 0
        self._completed = 0
        self._total = len(self._selected_papers)

    def _run_one(
        self,
        markdown_path: Path,
        paper_id: str,
        paper_output: Path,
    ) -> PaperEvaluation:
        with self._progress_lock:
            self._started += 1
            started = self._started
            print(
                f"[START {started}/{self._total}] {paper_id}",
                flush=True,
            )
        result = super()._run_one(markdown_path, paper_id, paper_output)
        with self._progress_lock:
            self._completed += 1
            if result.status == "failed":
                detail = result.failure_type or "workflow_failure"
                if result.error_message:
                    detail += f": {result.error_message[:240]}"
                print(
                    f"[FAILED {self._completed}/{self._total}] {paper_id} - {detail}",
                    flush=True,
                )
            elif result.status == "stopped":
                detail = result.failure_type or "stopped"
                print(
                    f"[STOPPED {self._completed}/{self._total}] {paper_id} - {detail}",
                    flush=True,
                )
            else:
                print(
                    f"[SUCCESS {self._completed}/{self._total}] {paper_id}",
                    flush=True,
                )
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 30 remaining Ours papers with six paper-level workers"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--completed-dir", type=Path, default=DEFAULT_COMPLETED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default="ours_remaining_nlp30")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _completed_paper_ids(completed_dir: Path, source_ids: set[str]) -> set[str]:
    if not completed_dir.is_dir():
        raise BatchEvaluationError(f"completed results directory does not exist: {completed_dir}")
    paper_dirs = sorted(path for path in completed_dir.iterdir() if path.is_dir())
    if len(paper_dirs) != EXPECTED_COMPLETED_PAPERS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_COMPLETED_PAPERS} completed paper directories, "
            f"found {len(paper_dirs)}: {completed_dir}"
        )
    completed_ids = {path.name for path in paper_dirs}
    unknown = sorted(completed_ids - source_ids)
    if unknown:
        raise BatchEvaluationError(
            "completed results contain paper directories absent from input Markdown: "
            + ", ".join(unknown)
        )
    incomplete = sorted(
        paper_id
        for paper_id in completed_ids
        if not (completed_dir / paper_id / "final_research_proposal.json").is_file()
    )
    if incomplete:
        raise BatchEvaluationError(
            "completed paper directories without final_research_proposal.json: "
            + ", ".join(incomplete)
        )
    return completed_ids


def _discover_remaining(args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
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
    completed_ids = _completed_paper_ids(args.completed_dir, set(source_by_id))
    remaining = [
        source_by_id[paper_id]
        for paper_id in sorted(set(source_by_id) - completed_ids)
    ]
    if len(remaining) != EXPECTED_REMAINING_PAPERS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_REMAINING_PAPERS} remaining papers, "
            f"found {len(remaining)}"
        )
    info = {
        "input_markdown_count": len(source_paths),
        "completed_paper_count": len(completed_ids),
        "remaining_paper_count": len(remaining),
        "completed_dir": str(args.completed_dir),
        "remaining_papers": [path.name for path in remaining],
    }
    return remaining, info


def _preflight_config(args: argparse.Namespace, *, require_services: bool) -> dict[str, Any]:
    prompt_dir = REPO_ROOT / "src" / "research_agent" / "prompts"
    if not prompt_dir.is_dir():
        raise BatchEvaluationError(f"prompt directory does not exist: {prompt_dir}")
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
        "semantic_scholar_cache_dir": semantic.cache_dir,
        "arxiv_cache_dir": arxiv.cache_dir,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    papers, selection_info = _discover_remaining(args)
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
            expected_papers=EXPECTED_REMAINING_PAPERS,
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
