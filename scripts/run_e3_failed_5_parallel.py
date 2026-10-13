#!/usr/bin/env python3
"""Retry the currently failed E3 cases with bounded paper-level concurrency.

This wrapper intentionally reuses the original E3 runner.  Completed cases are
passed through to the runner and skipped by its resume logic; only cases whose
metadata is currently ``failed`` can issue new LLM requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.e3_state_counterfactual.checkpoint import discover_revision_cases  # noqa: E402
from experiments.e3_state_counterfactual.runner import E3RunConfig, run_cases  # noqa: E402
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.llm.recording import RecordingLLMClient  # noqa: E402
from research_agent.serialization import read_json  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "e3_state_counterfactual_global"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry failed E3 cases with paper-level parallelism"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--expected-failures", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the failed cases without loading DeepSeek or running them",
    )
    return parser


def _status(output_dir: Path, paper_id: str) -> str | None:
    metadata_path = output_dir / "cases" / paper_id / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = read_json(metadata_path, expected_schema="e3_run_metadata")["payload"]
    except Exception:
        return None
    return payload.get("status") if isinstance(payload, dict) else None


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.expected_failures < 0:
        raise SystemExit("--expected-failures must be non-negative")

    cases = discover_revision_cases(args.source_dir)
    failed_ids = [
        case.paper_id
        for case in cases
        if _status(args.output_dir, case.paper_id) == "failed"
    ]
    if len(failed_ids) != args.expected_failures:
        raise SystemExit(
            f"expected {args.expected_failures} failed E3 cases, found {len(failed_ids)}: "
            + ", ".join(failed_ids)
        )

    print(
        json.dumps(
            {
                "failed_case_count": len(failed_ids),
                "failed_cases": failed_ids,
                "workers": args.workers,
                "source_dir": str(args.source_dir),
                "output_dir": str(args.output_dir),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    deepseek = DeepSeekConfig.from_env()
    deepseek.validate_for_request()
    config = E3RunConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=False,
    )
    results = run_cases(
        config,
        cases,
        deepseek=deepseek,
        workers=args.workers,
        llm_factory=lambda: RecordingLLMClient(
            DeepSeekClient(deepseek),
            unbounded_output=True,
        ),
    )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
