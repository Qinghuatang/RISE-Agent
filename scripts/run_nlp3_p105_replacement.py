#!/usr/bin/env python3
"""Run the replacement P105 with the complete RISE workflow."""

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


TARGET_PAPER_ID = "P105"
MAX_STEPS = int(RealPilotConfig.__dataclass_fields__["max_steps"].default)
MAX_SEARCHES = int(RealPilotConfig.__dataclass_fields__["max_searches"].default)
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp3"
DEFAULT_OUTPUT_DIR = Path("/home/tmh/baseline/output_rise_nlp3_p105_replacement")
DEFAULT_BATCH_NAME = "rise_nlp3_p105_replacement"
PAPER_ID_RE = re.compile(r"(?<![A-Za-z0-9])P\d{3}(?!\d)")


class SelectedPaperRunner(ParallelBatchEvaluationRunner):
    """Run the existing batch runner over the replacement P105 only."""

    def __init__(self, *args, selected_paper: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_paper = selected_paper

    def discover_papers(self) -> list[Path]:
        return [self._selected_paper]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the replacement P105 with the complete RISE workflow"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic_scholar.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _find_replacement_p105(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {input_dir}")

    matches: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        ids = PAPER_ID_RE.findall(path.name)
        if TARGET_PAPER_ID in ids:
            if ids.count(TARGET_PAPER_ID) != 1 or len(ids) != 1:
                raise BatchEvaluationError(
                    f"ambiguous P105 identity in Markdown filename: {path.name}"
                )
            matches.append(path)

    if not matches:
        raise BatchEvaluationError(f"replacement P105 Markdown file is missing from: {input_dir}")
    if len(matches) != 1:
        raise BatchEvaluationError(
            "multiple Markdown files resolve to P105: "
            + ", ".join(path.name for path in matches)
        )

    # Validate with the same ingestion parser used by the real workflow.
    parse_markdown_file(matches[0])
    return matches[0]


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
    if args.workers != 1:
        raise SystemExit("this single-paper runner must use --workers 1")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    paper = _find_replacement_p105(args.input_dir)
    config_info = _preflight_config(args, require_services=not args.dry_run)
    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=None,
        allow_overwrite=False,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "workflow": "complete_rise",
                    "flat_runner_used": False,
                    "paper_id": TARGET_PAPER_ID,
                    "paper": paper.name,
                    "input_dir": str(args.input_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "workers": args.workers,
                    "max_steps": MAX_STEPS,
                    "max_searches": MAX_SEARCHES,
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
    runner = SelectedPaperRunner(
        batch_config,
        executor,
        parallel_config=ParallelBatchConfig(workers=1, expected_papers=1),
        selected_paper=paper,
    )
    report = runner.run()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
