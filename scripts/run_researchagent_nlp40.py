#!/usr/bin/env python3
"""Run the official ResearchAgent as a 40-paper baseline.

The command requires ``--allow-real-services`` for an actual run.  Without it,
use ``--dry-run`` for deterministic input/config checks only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.baselines.researchagent import (  # noqa: E402
    ResearchAgentBaselineRunner,
    ResearchAgentDeepSeekClient,
    ResearchAgentSemanticScholarAdapter,
    load_seed_markdown,
    run_researchagent_batch,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig  # noqa: E402


EXPECTED_PAPERS = 40


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official ResearchAgent 40-paper baseline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nlp",
        help="directory containing the 40 Seed Paper Markdown files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "researchagent_nlp40",
    )
    parser.add_argument("--batch-name", default="researchagent_nlp40")
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("/home/tmh/baseline/ResearchAgent"),
    )
    parser.add_argument(
        "--knowledge-path",
        type=Path,
        default=Path("/home/tmh/baseline/ResearchAgent/data/knowledge.jsonl"),
    )
    parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic_scholar.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-real-services",
        action="store_true",
        help="required to run DeepSeek and Semantic Scholar; never implied",
    )
    return parser


def _paper_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"),
        key=lambda path: path.name,
    )
    if len(paths) != EXPECTED_PAPERS:
        raise SystemExit(f"expected exactly {EXPECTED_PAPERS} Markdown files, found {len(paths)}: {input_dir}")
    return paths


def _check_static(args: argparse.Namespace, paths: list[Path]) -> list[dict[str, str]]:
    if not args.official_repo.is_dir() or not (args.official_repo / "code").is_dir():
        raise SystemExit(f"official ResearchAgent repository/code not found: {args.official_repo}")
    if not args.knowledge_path.is_file():
        raise SystemExit(f"official knowledge store not found: {args.knowledge_path}")
    if not args.semantic_scholar_config.is_file():
        raise SystemExit(f"Semantic Scholar config not found: {args.semantic_scholar_config}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output path: {args.output_dir}")
    seed_inputs = []
    for path in paths:
        try:
            seed = load_seed_markdown(path)
        except Exception as exc:
            raise SystemExit(f"cannot deterministically extract title/abstract from {path}: {exc}") from exc
        seed_inputs.append({"paper_id": seed.paper_id, "title": seed.title})
    return seed_inputs


def _load_official_knowledge_store(official_repo: Path, knowledge_path: Path):
    code_root = str(official_repo / "code")
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from knowledge.store import KnowledgeStore

    return KnowledgeStore(str(knowledge_path))


def main() -> int:
    args = _parser().parse_args()
    paths = _paper_files(args.input_dir)
    seed_inputs = _check_static(args, paths)
    deepseek_config = DeepSeekConfig.from_env()
    semantic_config = SemanticScholarConfig.from_file(args.semantic_scholar_config)

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "paper_count": len(paths),
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "official_repo": str(args.official_repo),
            "knowledge_path": str(args.knowledge_path),
            "semantic_scholar_config": str(args.semantic_scholar_config),
            "papers": seed_inputs,
            "deepseek_configured": all((deepseek_config.api_key, deepseek_config.base_url, deepseek_config.model)),
            "semantic_scholar_base_url_configured": bool(semantic_config.base_url),
            "llm_calls_started": 0,
            "network_calls_started": 0,
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.allow_real_services:
        raise SystemExit("refusing to call external services; pass --allow-real-services for an actual run")
    deepseek_config.validate_for_request()
    semantic_config.validate_for_request()
    knowledge_store = _load_official_knowledge_store(args.official_repo, args.knowledge_path)

    def factory() -> ResearchAgentBaselineRunner:
        return ResearchAgentBaselineRunner(
            official_repo=args.official_repo,
            llm_client=ResearchAgentDeepSeekClient(deepseek_config),
            semantic_scholar=ResearchAgentSemanticScholarAdapter(semantic_config),
            knowledge_store=knowledge_store,
            iterations=3,
        )

    report = run_researchagent_batch(
        paths,
        args.output_dir,
        factory,
        batch_name=args.batch_name,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
