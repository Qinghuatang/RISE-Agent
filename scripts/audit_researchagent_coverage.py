#!/usr/bin/env python3
"""Audit ResearchAgent Seed/reference/entity coverage without any LLM calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.baselines.researchagent import (  # noqa: E402
    ResearchAgentSemanticScholarAdapter,
    load_seed_markdown,
)
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ResearchAgent retrieval/knowledge coverage")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "nlp")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "researchagent_coverage_audit.json",
    )
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required for Semantic Scholar requests; never implied",
    )
    return parser


def _files(input_dir: Path, expected_count: int) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"),
        key=lambda path: path.name,
    )
    if len(paths) != expected_count:
        raise SystemExit(f"expected exactly {expected_count} Markdown files, found {len(paths)}: {input_dir}")
    return paths


def _knowledge_store(official_repo: Path, knowledge_path: Path):
    code_root = str(official_repo / "code")
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from knowledge.store import KnowledgeStore

    return KnowledgeStore(str(knowledge_path))


def _static_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        seed = load_seed_markdown(path)
        rows.append({
            "paper_id": seed.paper_id,
            "seed_resolved": None,
            "reference_count": None,
            "knowledge_covered_reference_count": None,
            "entity_count": None,
            "status": "not_run_dry_run",
        })
    return rows


def main() -> int:
    args = _parser().parse_args()
    paths = _files(args.input_dir, args.expected_count)
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing audit output: {args.output}")
    if not args.knowledge_path.is_file():
        raise SystemExit(f"knowledge store not found: {args.knowledge_path}")
    if not args.semantic_scholar_config.is_file():
        raise SystemExit(f"Semantic Scholar config not found: {args.semantic_scholar_config}")

    if args.dry_run:
        rows = _static_rows(paths)
        report = {
            "audit_mode": "dry_run",
            "paper_count": len(paths),
            "llm_calls": 0,
            "network_calls_started": 0,
            "papers": rows,
        }
        write_json(args.output, "researchagent_coverage_audit", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if not args.allow_network:
        raise SystemExit("refusing to call Semantic Scholar; pass --allow-network for an actual audit")
    config = SemanticScholarConfig.from_file(args.semantic_scholar_config)
    config.validate_for_request()
    store = _knowledge_store(Path("/home/tmh/baseline/ResearchAgent"), args.knowledge_path)
    rows: list[dict[str, Any]] = []
    for path in paths:
        seed = load_seed_markdown(path)
        adapter = ResearchAgentSemanticScholarAdapter(config)
        row: dict[str, Any] = {"paper_id": seed.paper_id, "status": "failed"}
        try:
            resolved = adapter.resolve_seed_paper(seed.title, seed.abstract)
            references = adapter.get_relevant_references(resolved, top_k=10)
            covered = sum(
                1
                for reference in references
                if isinstance(reference.get("corpusId"), int)
                and reference["corpusId"] in store.paper2entities
            )
            entity_ids = [resolved["corpusId"]] + [
                reference["corpusId"]
                for reference in references
                if isinstance(reference.get("corpusId"), int)
            ]
            entities = store.get_relevant_entities(entity_ids)
            row.update({
                "seed_resolved": True,
                "resolved_provider_paper_id": resolved.get("paperId"),
                "reference_count": len(references),
                "knowledge_covered_reference_count": covered,
                "entity_count": len(entities),
                "status": "succeeded",
            })
        except Exception as exc:
            row.update({"seed_resolved": False, "error_type": type(exc).__name__, "error_message": str(exc)[:1_000]})
        row.update({
            "http_attempt_count": adapter.http_attempt_count,
            "cache_hits": adapter.cache_hits,
            "cache_hit_operations": adapter.cache_hit_operations,
            "seed_resolution": adapter.last_resolution,
        })
        rows.append(row)

    report = {
        "audit_mode": "network",
        "paper_count": len(paths),
        "llm_calls": 0,
        "network_calls_started": sum(row.get("http_attempt_count", 0) for row in rows),
        "succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "papers": rows,
    }
    write_json(args.output, "researchagent_coverage_audit", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
