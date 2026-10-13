#!/usr/bin/env python3
"""Run the frozen Ours workflow for the selected three development papers.

This is an execution-only wrapper.  It reuses the existing paper-level
parallel runner and the existing real-paper executor; it does not implement
or alter any scientific workflow logic.
"""

from __future__ import annotations

import argparse
import json
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


MAX_STEPS = 12
MAX_SEARCHES = 2
DEFAULT_WORKERS = 3
SELECTED_FILENAMES = (
    "MinerU_markdown_P006_Efficient_Tool_Use_with_Chain-of-Abstraction_Reasoning_2091868130714607622.md",
    "MinerU_markdown_P022_Speculative_Diffusion_Decoding_2091868057381396484.md",
    "MinerU_markdown_P040_Temporal_Relation_Extraction_GraphTREx_2091868057381396499.md",
)


class SelectedPaperParallelRunner(ParallelBatchEvaluationRunner):
    """Apply the existing parallel runner to a fixed, validated paper list."""

    def __init__(self, *args, selected_papers: list[Path], **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_papers = selected_papers

    def discover_papers(self) -> list[Path]:
        # The fixed cohort is selected by exact filename, then kept in stable
        # filename order.  All execution and artifact handling remains in the
        # parent ParallelBatchEvaluationRunner.
        return list(self._selected_papers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Ours workflow for P006, P022, and P040 in parallel"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nlp",
        help="directory containing the selected Markdown papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "ours_selected_p006_p022_p040_parallel_v1",
        help="new batch output directory; existing paths are rejected",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--batch-name",
        default="ours_selected_p006_p022_p040_parallel_v1",
    )
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

    papers = [input_dir / filename for filename in SELECTED_FILENAMES]
    missing = [path.name for path in papers if not path.is_file()]
    if missing:
        raise BatchEvaluationError(
            "selected Markdown file(s) are missing: " + ", ".join(missing)
        )

    # Validate identity through the same deterministic parser used by the
    # scientific runner.  Do not infer a different paper from a fuzzy title.
    parsed = [(path, parse_markdown_file(path)) for path in papers]
    paper_ids = [context.paper_id for _, context in parsed]
    if len(set(paper_ids)) != len(paper_ids):
        raise BatchEvaluationError("selected Markdown files have duplicate paper IDs")
    if len(papers) != 3:
        raise BatchEvaluationError(f"expected exactly 3 selected papers, found {len(papers)}")
    return sorted(papers, key=lambda path: path.name)


def _preflight_config(
    args: argparse.Namespace,
    *,
    require_services: bool,
) -> dict[str, object]:
    prompt_dir = REPO_ROOT / "src" / "research_agent" / "prompts"
    if not prompt_dir.is_dir():
        raise BatchEvaluationError(f"prompt directory does not exist: {prompt_dir}")
    if not args.semantic_scholar_config.is_file():
        raise BatchEvaluationError(
            f"Semantic Scholar config does not exist: {args.semantic_scholar_config}"
        )

    deepseek = DeepSeekConfig.from_env()
    semantic = SemanticScholarConfig.from_file(args.semantic_scholar_config)
    # ArxivConfig parsing is local and does not contact arXiv.
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
    if args.workers > len(SELECTED_FILENAMES):
        raise SystemExit(
            f"--workers must be at most {len(SELECTED_FILENAMES)} for this three-paper batch"
        )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

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
        expected_papers=len(SELECTED_FILENAMES),
    )

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
                    "paper_count": len(papers),
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
