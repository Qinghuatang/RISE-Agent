#!/usr/bin/env python3
"""Paper-level parallel runner for the frozen Ours 40-paper workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


EXPECTED_PAPERS = 40
MAX_STEPS = 12
MAX_SEARCHES = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the existing Ours workflow with paper-level parallelism"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nlp",
        help="directory containing exactly 40 Markdown papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "ours_parallel_nlp40",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-name", default="ours_parallel_nlp40")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic_scholar.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


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
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    config_info = _preflight_config(args, require_services=not args.dry_run)
    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=1301,
        allow_overwrite=False,
    )
    parallel = ParallelBatchEvaluationRunner(
        batch_config,
        executor=lambda *_: None,
        parallel_config=ParallelBatchConfig(
            workers=args.workers,
            expected_papers=EXPECTED_PAPERS,
        ),
    )
    papers = parallel.discover_papers()

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
    runner = ParallelBatchEvaluationRunner(
        batch_config,
        executor,
        parallel_config=ParallelBatchConfig(
            workers=args.workers,
            expected_papers=EXPECTED_PAPERS,
        ),
    )
    report = runner.run()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
