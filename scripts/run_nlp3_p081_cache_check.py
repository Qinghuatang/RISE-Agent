#!/usr/bin/env python3
"""Run the complete RISE workflow for P081 and report retrieval cache use.

The scientific execution is delegated to the existing real-paper executor.
This wrapper only selects P081 from ``data/nlp3`` and adds a small, read-only
summary of the existing Semantic Scholar provider audit after the run.
It deliberately does not import or call the RISE-Flat ablation runner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

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


TARGET_PAPER_ID = "P081"
MAX_STEPS = int(RealPilotConfig.__dataclass_fields__["max_steps"].default)
MAX_SEARCHES = int(RealPilotConfig.__dataclass_fields__["max_searches"].default)
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp3"
DEFAULT_OUTPUT_DIR = Path("/home/tmh/baseline/output_nlp3_p081")
DEFAULT_BATCH_NAME = "ours_nlp3_p081_cache_check"
PAPER_ID_RE = re.compile(r"(?<![A-Za-z0-9])P\d{3}(?!\d)")


class SelectedPaperRunner(ParallelBatchEvaluationRunner):
    """Run the existing paper-level runner over exactly one selected paper."""

    def __init__(self, *args, selected_paper: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_paper = selected_paper

    def discover_papers(self) -> list[Path]:
        return [self._selected_paper]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run complete RISE for P081 and inspect Semantic Scholar cache usage"
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


def _find_p081(input_dir: Path) -> Path:
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
                    f"ambiguous P081 identity in Markdown filename: {path.name}"
                )
            matches.append(path)

    if not matches:
        raise BatchEvaluationError(f"P081 Markdown file is missing from: {input_dir}")
    if len(matches) != 1:
        raise BatchEvaluationError(
            "multiple Markdown files resolve to P081: "
            + ", ".join(path.name for path in matches)
        )

    # Use the same ingestion parser as the real workflow for local validation.
    parse_markdown_file(matches[0])
    return matches[0]


def _preflight_config(args: argparse.Namespace, *, require_services: bool) -> dict[str, Any]:
    if not args.semantic_scholar_config.is_file():
        raise BatchEvaluationError(
            f"Semantic Scholar config does not exist: {args.semantic_scholar_config}"
        )

    prompt_dir = REPO_ROOT / "src" / "research_agent" / "prompts"
    if not prompt_dir.is_dir():
        raise BatchEvaluationError(f"prompt directory does not exist: {prompt_dir}")

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


def _read_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("payload"), dict):
        return data["payload"]
    return data if isinstance(data, dict) else {}


def _write_cache_check(output_dir: Path, paper_output: Path) -> dict[str, Any]:
    audit_path = paper_output / "provider_audit.json"
    if not audit_path.is_file():
        result: dict[str, Any] = {
            "status": "unavailable",
            "reason": "provider_audit.json was not produced",
        }
    else:
        audit = _read_payload(audit_path)
        semantic = audit.get("semantic_scholar", {})
        result = {
            "status": "observed",
            "cache_scope": semantic.get("cache_scope", "unknown"),
            "global_cache_operations": semantic.get("global_cache_operations", []),
            "http_request_count": semantic.get("http_request_count"),
            "retry_count": semantic.get("retry_count"),
            "resolved_seed_provider_id": semantic.get("resolved_seed_provider_id"),
            "seed_resolution": semantic.get("seed_resolution", {}),
            "related_pool_size": semantic.get("related_pool_size"),
            "cache_dir": semantic.get("cache_dir"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_scholar_cache_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.workers != 1:
        raise SystemExit("this single-paper diagnostic must use --workers 1")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    paper = _find_p081(args.input_dir)
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
                    "cache_check": (
                        "runtime provider_audit will report cache_scope and HTTP requests"
                    ),
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
    paper_output = args.output_dir / parse_markdown_file(paper).paper_id
    cache_check = _write_cache_check(args.output_dir, paper_output)
    print(
        json.dumps(
            {
                "status": report.papers[0].status if report.papers else "unknown",
                "paper": paper.name,
                "output_dir": str(args.output_dir.resolve()),
                "evaluation_report": str(args.output_dir / "evaluation_report.json"),
                "cache_check": cache_check,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
