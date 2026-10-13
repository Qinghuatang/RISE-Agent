#!/usr/bin/env python3
"""Run E3's final unified re-verification without rerunning revisions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.e3_state_counterfactual.unified import (  # noqa: E402
    discover_unified_cases,
    run_unified_case,
    write_unified_manifest,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.modules.related_work_verification import VerificationConfig  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_E3_OUTPUT = REPO_ROOT / "outputs" / "e3_state_counterfactual_global"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "e3_final_analysis"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E3 unified full-solution re-verification"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--e3-output-dir", type=Path, default=DEFAULT_E3_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-id", help="full paper_id or short id such as P002")
    parser.add_argument("--all", action="store_true", help="run all valid E3 cases")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and list cases only")
    return parser


def _short_id(paper_id: str) -> str | None:
    match = re.search(r"(?:^|_)P(\d{3})(?:_|$)", paper_id)
    return f"P{match.group(1)}" if match else None


def _select_case(cases, requested: str):
    exact = [case for case in cases if case.paper_id == requested]
    if len(exact) == 1:
        return exact[0]
    short = [case for case in cases if _short_id(case.paper_id) == requested]
    if len(short) == 1:
        return short[0]
    raise SystemExit(f"unknown or ambiguous paper id: {requested}")


def main() -> int:
    args = _parser().parse_args()
    if args.paper_id and args.all:
        raise SystemExit("choose either --paper-id or --all")
    if not args.paper_id and not args.all:
        raise SystemExit("execution requires --paper-id or --all; use --dry-run to audit")

    discovery = discover_unified_cases(args.source_dir, args.e3_output_dir)
    print(
        json.dumps(
            {
                "discovered_revision_cases": discovery.discovered_revision_cases,
                "valid_unified_cases": len(discovery.cases),
                "invalid_cases": discovery.errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if discovery.errors:
        raise SystemExit("unified verification preflight failed; inspect invalid_cases above")
    cases = discovery.cases
    if args.paper_id:
        cases = [_select_case(cases, args.paper_id)]
    if args.dry_run:
        print(json.dumps({"selected_cases": [case.paper_id for case in cases]}, ensure_ascii=False, indent=2))
        return 0

    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    e3_output_dir = args.e3_output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise SystemExit("output-dir must not be inside or equal to the RISE source directory")
    if output_dir == e3_output_dir or e3_output_dir in output_dir.parents:
        raise SystemExit("output-dir must be separate from the existing E3 output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    deepseek = DeepSeekConfig.from_env()
    deepseek.validate_for_request()
    verification_config = VerificationConfig.from_env()
    write_unified_manifest(
        output_dir,
        source_dir=source_dir,
        e3_output_dir=e3_output_dir,
        deepseek=deepseek,
        verification_config=verification_config,
        case_count=len(cases),
    )

    results = []
    for case in cases:
        print(f"[E3-unified] {case.paper_id}: targeted + global full verification")
        result = run_unified_case(
            case,
            output_dir,
            deepseek=deepseek,
            verification_config=verification_config,
            overwrite=args.overwrite,
        )
        print(f"[E3-unified] {case.paper_id}: {result['status']}")
        results.append(result)
    write_json(
        output_dir / "execution_summary.json",
        "e3_unified_reverification_summary",
        {
            "case_count": len(cases),
            "branch_count": len(cases) * 2,
            "completed": sum(item.get("status") == "completed" for item in results),
            "failed": sum(item.get("status") == "failed" for item in results),
            "results": sorted(results, key=lambda item: item.get("paper_id", "")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
