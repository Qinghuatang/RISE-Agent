#!/usr/bin/env python3
"""Run the existing Ours workflow on one half of the confirmed 64-paper cohort.

This is an execution-only wrapper around the existing paper-level parallel
runner.  It does not implement or alter any scientific workflow logic.  The
cohort is read from the already prepared paired-results manifest, mapped to one
source Markdown directory containing the 80 files, sorted by Markdown
filename, and divided into two equal 32-paper partitions selected by
``--shard-index`` (0 or 1).
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


EXPECTED_TOTAL_PAPERS = 64
EXPECTED_INPUT_MARKDOWNS = 80
SHARD_COUNT = 2
PAPERS_PER_SHARD = EXPECTED_TOTAL_PAPERS // SHARD_COUNT
DEFAULT_WORKERS = 8
MAX_STEPS = 12
MAX_SEARCHES = 2

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_COHORT_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "blind_review_ours_vs_direct_all_results_v1"
    / "paired_results.json"
)
DEFAULT_SEMANTIC_SCHOLAR_CONFIG = REPO_ROOT / "config" / "semantic_scholar.json"


class SelectedPaperParallelRunner(ParallelBatchEvaluationRunner):
    """Reuse the existing runner while supplying an explicit paper cohort."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one 32-paper partition of the confirmed 64-paper Ours cohort "
            "with paper-level parallelism"
        )
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        choices=range(SHARD_COUNT),
        required=True,
        help="stable partition index: 0 or 1",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory containing the 80 source Markdown files",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=DEFAULT_COHORT_MANIFEST,
        help="JSON list containing the confirmed 64 complete paired paper records",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new output directory; existing paths are rejected",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default="ours_paired64")
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_output_dir(shard_index: int) -> Path:
    # The labels distinguish the two independent destinations without
    # encoding temporal wording in per-paper scientific artifacts.
    label = ("a", "b")[shard_index]
    return REPO_ROOT / "outputs" / f"ours_paired64_segment_{label}"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BatchEvaluationError(f"cohort manifest does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvaluationError(f"cannot read cohort manifest: {path}") from exc
    if not isinstance(document, list):
        raise BatchEvaluationError(
            "cohort manifest must be a JSON list of paired paper records"
        )
    return [
        item
        for item in document
        if isinstance(item, dict)
    ]


def _discover_confirmed_cohort(
    input_dir: Path,
    cohort_manifest: Path,
) -> list[Path]:
    if not input_dir.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {input_dir}")

    manifest_rows = _load_manifest(cohort_manifest)
    if len(manifest_rows) != EXPECTED_TOTAL_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_TOTAL_PAPERS} paired paper records, "
            f"found {len(manifest_rows)} in {cohort_manifest}"
        )

    source_by_stem: dict[str, Path] = {}
    source_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    ]
    if len(source_paths) != EXPECTED_INPUT_MARKDOWNS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_INPUT_MARKDOWNS} Markdown files, "
            f"found {len(source_paths)} in {input_dir}"
        )
    for path in source_paths:
        source_by_stem[path.stem] = path.resolve()
    selected: list[Path] = []
    seen_ids: set[str] = set()
    for row in manifest_rows:
        paper_id = row.get("paper_id")
        status = row.get("status")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise BatchEvaluationError("cohort manifest contains a record without paper_id")
        if status != "paired_complete":
            raise BatchEvaluationError(
                f"cohort manifest record is not paired_complete: {paper_id}"
            )
        if paper_id in seen_ids:
            raise BatchEvaluationError(f"duplicate paper_id in cohort manifest: {paper_id}")
        seen_ids.add(paper_id)
        source = source_by_stem.get(paper_id)
        if source is None:
            raise BatchEvaluationError(
                f"Markdown source for paired paper is missing: {paper_id}"
            )
        selected.append(source)

    if len(selected) != EXPECTED_TOTAL_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_TOTAL_PAPERS} unique Markdown sources, found {len(selected)}"
        )
    return sorted(selected, key=lambda path: path.name)


def _select_shard(papers: list[Path], shard_index: int) -> list[Path]:
    if len(papers) != EXPECTED_TOTAL_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_TOTAL_PAPERS} papers before partitioning, found {len(papers)}"
        )
    start = shard_index * PAPERS_PER_SHARD
    stop = start + PAPERS_PER_SHARD
    selected = papers[start:stop]
    if len(selected) != PAPERS_PER_SHARD:
        raise BatchEvaluationError(
            f"partition {shard_index} contains {len(selected)} papers, "
            f"expected {PAPERS_PER_SHARD}"
        )
    return selected


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
    arxiv = ArxivConfig.from_env()
    if require_services:
        deepseek.validate_for_request()
        semantic.validate_for_request()
    return {
        "deepseek_configured": bool(
            deepseek.api_key and deepseek.base_url and deepseek.model
        ),
        "deepseek_model": deepseek.model,
        "semantic_scholar_configured": bool(semantic.base_url),
        "semantic_scholar_cache_dir": semantic.cache_dir,
        "arxiv_cache_dir": arxiv.cache_dir,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    output_dir = args.output_dir or _default_output_dir(args.shard_index)
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {output_dir}")

    all_papers = _discover_confirmed_cohort(args.input_dir, args.cohort_manifest)
    papers = _select_shard(all_papers, args.shard_index)
    config_info = _preflight_config(args, require_services=not args.dry_run)

    batch_config = BatchEvaluationConfig(
        input_dir=args.input_dir,
        output_dir=output_dir,
        batch_name=args.batch_name,
        max_papers=None,
        selection_seed=None,
        allow_overwrite=False,
    )
    parallel_config = ParallelBatchConfig(
        workers=args.workers,
        expected_papers=PAPERS_PER_SHARD,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "batch_name": args.batch_name,
                    "input_dir": str(args.input_dir),
                    "cohort_manifest": str(args.cohort_manifest),
                    "output_dir": str(output_dir),
                    "workers": args.workers,
                    "paper_count": len(papers),
                    "input_markdown_count": EXPECTED_INPUT_MARKDOWNS,
                    "total_confirmed_papers": len(all_papers),
                    "max_steps": MAX_STEPS,
                    "max_searches": MAX_SEARCHES,
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
