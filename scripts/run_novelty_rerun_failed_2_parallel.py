#!/usr/bin/env python3
"""Retry P034 and P053 with the complete RISE workflow.

This wrapper only selects the two incomplete papers and delegates execution
to the existing real-paper executor and paper-level parallel runner.  It does
not import or call the RISE-Flat ablation runner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import parse_markdown_file
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
from research_agent.runner.pilot import RealPilotConfig


RETRY_PAPER_IDS = ("P034", "P053")
EXPECTED_PAPERS = len(RETRY_PAPER_IDS)
MAX_STEPS = int(RealPilotConfig.__dataclass_fields__["max_steps"].default)
MAX_SEARCHES = int(RealPilotConfig.__dataclass_fields__["max_searches"].default)
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_SOURCE_OUTPUT_DIR = Path("/home/tmh/baseline/output_novelty_rerun_6")
DEFAULT_OUTPUT_DIR = Path("/home/tmh/baseline/output_novelty_rerun_2")
DEFAULT_BATCH_NAME = "ours_novelty_rerun_failed_2"
DEFAULT_WORKERS = 2
PAPER_ID_RE = re.compile(r"(?<![A-Za-z0-9])P\d{3}(?!\d)")


class SelectedPaperParallelRunner(ParallelBatchEvaluationRunner):
    """Use the shared batch orchestration with an exact retry cohort."""

    def __init__(self, *args, selected_papers: list[Path], **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_papers = selected_papers

    def discover_papers(self) -> list[Path]:
        return list(self._selected_papers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry P034 and P053 with the complete RISE workflow in parallel"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-output-dir", type=Path, default=DEFAULT_SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic_scholar.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _selected_papers(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {input_dir}")

    candidates: dict[str, list[Path]] = {paper_id: [] for paper_id in RETRY_PAPER_IDS}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        matches = [paper_id for paper_id in PAPER_ID_RE.findall(path.name) if paper_id in candidates]
        if not matches:
            continue
        if len(matches) != 1:
            raise BatchEvaluationError(
                f"ambiguous retry paper identity in filename: {path.name}"
            )
        # Validate with the same Markdown ingestion used by the real workflow.
        parse_markdown_file(path)
        candidates[matches[0]].append(path)

    duplicate = {
        paper_id: [path.name for path in paths]
        for paper_id, paths in candidates.items()
        if len(paths) > 1
    }
    if duplicate:
        raise BatchEvaluationError(f"multiple Markdown files resolve to retry IDs: {duplicate}")

    missing = [paper_id for paper_id, paths in candidates.items() if not paths]
    if missing:
        raise BatchEvaluationError(
            "retry paper Markdown file(s) are missing: " + ", ".join(missing)
        )

    selected = [candidates[paper_id][0] for paper_id in RETRY_PAPER_IDS]
    if len(selected) != EXPECTED_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_PAPERS} retry papers, found {len(selected)}"
        )
    return sorted(selected, key=lambda path: path.name)


def _source_dirs(source_output_dir: Path) -> dict[str, list[Path]]:
    if not source_output_dir.is_dir():
        raise BatchEvaluationError(
            f"source output directory does not exist: {source_output_dir}"
        )
    candidates: dict[str, list[Path]] = {paper_id: [] for paper_id in RETRY_PAPER_IDS}
    for path in source_output_dir.iterdir():
        if not path.is_dir():
            continue
        matches = [paper_id for paper_id in PAPER_ID_RE.findall(path.name) if paper_id in candidates]
        if len(matches) == 1:
            candidates[matches[0]].append(path)
    return candidates


def _validate_source_failures(source_output_dir: Path) -> dict[str, str]:
    dirs = _source_dirs(source_output_dir)
    missing = [paper_id for paper_id, paths in dirs.items() if not paths]
    duplicates = {
        paper_id: [str(path) for path in paths]
        for paper_id, paths in dirs.items()
        if len(paths) > 1
    }
    if missing or duplicates:
        details = []
        if missing:
            details.append("missing source paper directories: " + ", ".join(missing))
        if duplicates:
            details.append(f"duplicate source paper directories: {duplicates}")
        raise BatchEvaluationError("; ".join(details))

    result: dict[str, str] = {}
    for paper_id in RETRY_PAPER_IDS:
        paper_dir = dirs[paper_id][0]
        final_files = [
            paper_dir / "final_research_proposal.json",
            paper_dir / "commit_snapshot.json",
        ]
        present = [str(path) for path in final_files if path.is_file()]
        if present:
            raise BatchEvaluationError(
                f"source paper {paper_id} already has final artifact(s); refusing retry: "
                + ", ".join(present)
            )
        result[paper_id] = str(paper_dir)
    return result


def _preflight_config(args: argparse.Namespace, *, require_services: bool) -> dict[str, object]:
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
    if args.workers > EXPECTED_PAPERS:
        raise SystemExit(f"--workers must be at most {EXPECTED_PAPERS}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    source_dirs = _validate_source_failures(args.source_output_dir)
    papers = _selected_papers(args.input_dir)
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
        expected_papers=EXPECTED_PAPERS,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "workflow": "complete_rise",
                    "flat_runner_used": False,
                    "batch_name": args.batch_name,
                    "input_dir": str(args.input_dir.resolve()),
                    "source_output_dir": str(args.source_output_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "workers": args.workers,
                    "max_steps": MAX_STEPS,
                    "max_searches": MAX_SEARCHES,
                    "paper_count": len(papers),
                    "retry_paper_ids": list(RETRY_PAPER_IDS),
                    "source_paper_dirs": source_dirs,
                    "config": config_info,
                    "papers": [path.name for path in papers],
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
    runner = SelectedPaperParallelRunner(
        batch_config,
        executor,
        parallel_config=parallel_config,
        selected_papers=papers,
    )
    report = runner.run()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
