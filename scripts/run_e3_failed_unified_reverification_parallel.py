#!/usr/bin/env python3
"""Retry incomplete E3 unified verification cases with paper-level parallelism.

No revision, inspiration, or retrieval is executed here.  Existing successful
Targeted/Global verification artifacts are reused by the original E3 runner.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.e3_state_counterfactual.parallel_retry import (  # noqa: E402
    run_incomplete_cases_parallel,
    select_incomplete_cases,
)
from experiments.e3_state_counterfactual.unified import (  # noqa: E402
    discover_unified_cases,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402
from research_agent.modules.related_work_verification import VerificationConfig  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_E3_OUTPUT = REPO_ROOT / "outputs" / "e3_state_counterfactual_global"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "e3_final_analysis"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry incomplete E3 unified verification cases in parallel"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--e3-output-dir", type=Path, default=DEFAULT_E3_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all-failed", action="store_true")
    selector.add_argument("--paper-id", help="full paper_id or short id such as P002")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="optional override; otherwise use DEEPSEEK_TIMEOUT_SECONDS/default",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="optional transport retry override; otherwise use environment/default",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover and list incomplete cases without calling DeepSeek",
    )
    return parser


def _short_id(paper_id: str) -> str | None:
    match = re.search(r"(?:^|_)P(\d{3})(?:_|$)", paper_id)
    return f"P{match.group(1)}" if match else None


def _select_short_case(cases, requested: str):
    exact = [case for case in cases if case.paper_id == requested]
    if len(exact) == 1:
        return exact[0]
    short = [case for case in cases if _short_id(case.paper_id) == requested]
    if len(short) == 1:
        return short[0]
    raise SystemExit(f"unknown or ambiguous paper id: {requested}")


def _validate_paths(source_dir: Path, e3_output_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    e3_output = e3_output_dir.resolve()
    output = output_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    if not e3_output.is_dir():
        raise SystemExit(f"E3 input directory does not exist: {e3_output}")
    if not output.is_dir():
        raise SystemExit(
            "retry output directory must already exist; refusing to create a new E3 run: "
            f" {output}"
        )
    if output == source or source in output.parents:
        raise SystemExit("output-dir must not be inside or equal to the RISE source directory")
    if output == e3_output or e3_output in output.parents:
        raise SystemExit("output-dir must be separate from the existing E3 output directory")
    if not (output / "unified_verification").is_dir():
        raise SystemExit(f"not an E3 unified output directory: {output}")


def _build_config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    overrides = {}
    if args.timeout_seconds is not None:
        if args.timeout_seconds <= 0:
            raise SystemExit("--timeout-seconds must be positive")
        overrides["timeout_seconds"] = args.timeout_seconds
    if args.max_retries is not None:
        if args.max_retries < 0:
            raise SystemExit("--max-retries must be non-negative")
        overrides["max_retries"] = args.max_retries
    if overrides:
        config = dataclasses.replace(config, **overrides)
    config.validate_for_request()
    return config


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    _validate_paths(args.source_dir, args.e3_output_dir, args.output_dir)

    discovery = discover_unified_cases(args.source_dir, args.e3_output_dir)
    if discovery.errors:
        raise SystemExit(
            "E3 unified preflight failed: "
            + json.dumps(discovery.errors, ensure_ascii=False)
        )
    incomplete = select_incomplete_cases(discovery.cases, args.output_dir)
    if args.paper_id:
        case = _select_short_case(discovery.cases, args.paper_id)
        if case not in incomplete:
            raise SystemExit(f"{args.paper_id} is already complete; no retry scheduled")
        incomplete = [case]

    print(
        json.dumps(
            {
                "discovered_revision_cases": discovery.discovered_revision_cases,
                "valid_unified_cases": len(discovery.cases),
                "incomplete_case_count": len(incomplete),
                "incomplete_cases": [case.paper_id for case in incomplete],
                "workers": args.workers,
                "output_dir": str(args.output_dir.resolve()),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if not incomplete:
        print("No incomplete E3 cases; no LLM calls made.")
        return 0

    deepseek = _build_config(args)
    verification_config = VerificationConfig.from_env()
    output_dir = args.output_dir.resolve()
    write_json(
        output_dir / "retry_execution_manifest.json",
        "e3_unified_retry_manifest",
        {
            "experiment": "E3_unified_reverification_retry",
            "source_dir": str(args.source_dir.resolve()),
            "e3_output_dir": str(args.e3_output_dir.resolve()),
            "output_dir": str(output_dir),
            "case_count": len(incomplete),
            "branch_count": len(incomplete) * 2,
            "workers": args.workers,
            "resume_policy": "skip_existing_successful_branches",
            "verification_focus": "full_solution",
            "new_related_paper_retrievals": 0,
            "verification_config": {
                "temperature": 0.0,
                "max_tokens": verification_config.max_tokens,
            },
            "deepseek": deepseek.public_dict(),
        },
    )

    def progress(result: dict, completed: int, total: int) -> None:
        print(
            f"[E3-retry] {completed}/{total} "
            f"{result.get('paper_id', '<unknown>')}: {result.get('status')}"
        )

    results = run_incomplete_cases_parallel(
        incomplete,
        output_dir,
        deepseek=deepseek,
        verification_config=verification_config,
        workers=args.workers,
        progress=progress,
    )
    write_json(
        output_dir / "retry_execution_summary.json",
        "e3_unified_retry_summary",
        {
            "case_count": len(incomplete),
            "branch_count": len(incomplete) * 2,
            "completed": sum(item.get("status") == "completed" for item in results),
            "failed": sum(item.get("status") == "failed" for item in results),
            "results": results,
        },
    )
    return 0 if all(item.get("status") == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
