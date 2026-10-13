#!/usr/bin/env python3
"""Analyze E3 unified paired verification artifacts without any LLM calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.e3_state_counterfactual.final_analysis import analyze_final  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_E3_OUTPUT = REPO_ROOT / "outputs" / "e3_state_counterfactual_global"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "e3_final_analysis"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze final E3 paired verification")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--e3-output-dir", type=Path, default=DEFAULT_E3_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze_final(args.source_dir, args.e3_output_dir, args.output_dir)
    print(
        {
            "valid": result["validation"]["valid"],
            "case_count": result["summary"]["case_count"],
            "branch_count": result["summary"]["branch_count"],
            "failure_case_count": result["failure_case_count"],
            "stochasticity_changes": result["stochasticity"]["changed_outcome_count"],
            "output_dir": str(args.output_dir),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
