#!/usr/bin/env python3
"""Run or validate E3: localized historical revision vs global rewrite.

By default this command only becomes an LLM/network execution when ``--all``
or ``--paper-id`` is supplied without ``--validate-only``.  The source RISE
outputs are read-only; all E3 artifacts go to a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.e3_state_counterfactual.checkpoint import discover_revision_cases  # noqa: E402
from experiments.e3_state_counterfactual.runner import (  # noqa: E402
    E3RunConfig,
    build_real_llm,
    run_cases,
    validate_only,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.llm.recording import RecordingLLMClient  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "e3_state_counterfactual_global"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E3 state-evolution counterfactual experiment harness"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="read-only completed RISE per-paper output directory",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="discover cases and validate checkpoints without LLM/network calls",
    )
    parser.add_argument("--paper-id", help="full paper_id or short id such as P002")
    parser.add_argument("--all", action="store_true", help="run all valid revision cases")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="maximum number of paper-level workers (default: 8)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.paper_id and args.all:
        raise SystemExit("choose either --paper-id or --all, not both")
    if not args.validate_only and not args.paper_id and not args.all:
        raise SystemExit("execution requires --paper-id or --all; use --validate-only for audit")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    random.seed(args.seed)
    config = E3RunConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    if args.validate_only:
        cases = validate_only(config)
        print(
            json.dumps(
                {
                    "validate_only": True,
                    "source_dir": str(args.source_dir),
                    "output_dir": str(args.output_dir),
                    "revision_cases": len(cases),
                    "valid_cases": sum(case.valid for case in cases),
                    "overlap_repair": sum(
                        case.valid and case.trigger_type == "overlap_repair"
                        for case in cases
                    ),
                    "inspiration_strengthening": sum(
                        case.valid and case.trigger_type == "inspiration_strengthening"
                        for case in cases
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    cases = discover_revision_cases(args.source_dir)
    if args.paper_id:
        from experiments.e3_state_counterfactual.runner import _select_case

        cases = [_select_case(cases, args.paper_id)]
    # Create a fresh recording client for every paper.  This keeps the two
    # calls and their audit records isolated per counterfactual case.
    deepseek = DeepSeekConfig.from_env()
    deepseek.validate_for_request()
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
