#!/usr/bin/env python3
"""Offline statistical analysis for the three Full-vs-Flat blind judges.

The three result files contain signed A-vs-B scores.  The fixed source key is
used only to interpret that existing orientation as Full-minus-Flat; source
scores are never edited.  No LLM or external service is used.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import random
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLIND_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval_final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_statistics"
EXPECTED_CASES = 80
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
RESULT_FILES = ("result1.jsonl", "result2.jsonl", "result3.jsonl")
DIMENSIONS = (
    "research_problem_quality",
    "novelty",
    "technical_mechanism_quality",
    "soundness_feasibility",
    "experimental_quality",
    "scientific_significance",
    "overall_proposal_quality",
)
DISPLAY_NAMES = {
    "research_problem_quality": "Research Problem Quality",
    "novelty": "Novelty",
    "technical_mechanism_quality": "Technical Mechanism Quality",
    "soundness_feasibility": "Soundness & Feasibility",
    "experimental_quality": "Experimental Quality",
    "scientific_significance": "Scientific Significance",
    "overall_proposal_quality": "Overall Proposal Quality",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze three Full-vs-Flat blind evaluations")
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_result(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise RuntimeError(f"{path}:{line_number} has invalid case_id")
            if case_id in rows:
                raise RuntimeError(f"{path}: duplicate case_id {case_id}")
            values: dict[str, int] = {}
            for dimension in DIMENSIONS:
                value = row.get(dimension)
                if not isinstance(value, int) or isinstance(value, bool) or value not in {-2, -1, 0, 1, 2}:
                    raise RuntimeError(
                        f"{path}:{line_number}:{dimension} must be an integer in -2..2, got {value!r}"
                    )
                values[dimension] = value
            rows[case_id] = values
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"{path} must contain exactly {EXPECTED_CASES} cases, found {len(rows)}")
    return rows


def _load_source_key(path: Path) -> dict[str, dict[str, str]]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"source key is not an object: {path}")
    if len(value) != EXPECTED_CASES:
        raise RuntimeError(f"source key must contain {EXPECTED_CASES} cases, found {len(value)}")
    result: dict[str, dict[str, str]] = {}
    for case_id, mapping in value.items():
        if not isinstance(mapping, dict):
            raise RuntimeError(f"source key mapping is not an object: {case_id}")
        proposal_a = mapping.get("proposal_a")
        proposal_b = mapping.get("proposal_b")
        if {proposal_a, proposal_b} != {"full", "flat"}:
            raise RuntimeError(f"invalid fixed A/B source mapping: {case_id}")
        result[case_id] = {"proposal_a": proposal_a, "proposal_b": proposal_b}
    return result


def _validate_case_ids(result_sets: dict[str, dict[str, dict[str, int]]], source_key: dict[str, dict[str, str]]) -> dict[str, Any]:
    expected = {f"B2_C{index:03d}" for index in range(1, EXPECTED_CASES + 1)}
    checks: dict[str, Any] = {
        "result_files": {},
        "all_case_id_sets_identical": True,
        "source_key_case_ids_complete": set(source_key) == expected,
        "expected_case_count": EXPECTED_CASES,
    }
    reference_ids: set[str] | None = None
    for name, rows in result_sets.items():
        ids = set(rows)
        checks["result_files"][name] = {
            "case_count": len(rows),
            "complete_expected_coverage": ids == expected,
            "duplicate_case_ids": False,
            "missing_case_ids": sorted(expected - ids),
            "extra_case_ids": sorted(ids - expected),
        }
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            checks["all_case_id_sets_identical"] = False
    checks["all_score_values_validated"] = True
    checks["passed"] = all(
        item["case_count"] == EXPECTED_CASES and item["complete_expected_coverage"]
        for item in checks["result_files"].values()
    ) and checks["all_case_id_sets_identical"] and checks["source_key_case_ids_complete"]
    return checks


def _full_minus_flat_scores(
    result_sets: dict[str, dict[str, dict[str, int]]], source_key: dict[str, dict[str, str]]
) -> dict[str, dict[str, list[float]]]:
    """Return judge -> dimension -> 80 fixed-orientation Full-minus-Flat values."""
    converted: dict[str, dict[str, list[float]]] = {}
    for judge, rows in result_sets.items():
        converted[judge] = {dimension: [] for dimension in DIMENSIONS}
        for case_id in sorted(rows):
            # A positive raw score means Proposal A was preferred.  Flip only
            # when the already-fixed source key says Full was Proposal B.
            orientation = 1 if source_key[case_id]["proposal_a"] == "full" else -1
            for dimension in DIMENSIONS:
                converted[judge][dimension].append(float(orientation * rows[case_id][dimension]))
    return converted


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[index] for index in rng.choices(range(len(values)), k=len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _sign_test_p(full_wins: int, flat_wins: int) -> float | None:
    non_ties = full_wins + flat_wins
    if non_ties == 0:
        return None
    smaller_tail = min(full_wins, flat_wins)
    lower_tail = sum(math.comb(non_ties, i) for i in range(smaller_tail + 1)) / (2**non_ties)
    return min(1.0, 2 * lower_tail)


def _metric(values: list[float], ci: tuple[float, float]) -> dict[str, Any]:
    full_wins = sum(value > 0 for value in values)
    flat_wins = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    return {
        "panel_mean": sum(values) / len(values),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "full_wins": full_wins,
        "flat_wins": flat_wins,
        "ties": ties,
        "sign_test_p": _sign_test_p(full_wins, flat_wins),
        "seed_count": len(values),
    }


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        # Keep small exact-test p-values nonzero. Presentation rounding is
        # handled only in summary.md; JSON statistics retain float precision.
        return value
    if isinstance(value, dict):
        return {key: _round_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_value(item) for item in value]
    return value


def _write_summary(
    path: Path,
    statistics: dict[str, Any],
    validation: dict[str, Any],
    diagnostic_status: str,
) -> None:
    overall = statistics["dimensions"]["overall_proposal_quality"]
    lines = [
        "# RISE-Full vs. RISE-Flat: Three-Judge Paired Statistics",
        "",
        "This is an offline descriptive/inferential analysis of 80 paired seeds.",
        "The fixed source key is used only to decode the existing A/B orientation; original scores are unchanged.",
        "",
        "## Primary endpoint",
        "",
        f"Overall Proposal Quality: Full − Flat = **{overall['panel_mean']:+.4f}**",
        f"95% bootstrap CI = **[{overall['bootstrap_ci_low']:+.4f}, {overall['bootstrap_ci_high']:+.4f}]**",
        f"Seed-level W/T/L = **{overall['full_wins']}/{overall['ties']}/{overall['flat_wins']}**",
        f"Exact two-sided sign-test p-value = **{overall['sign_test_p']:.6g}**",
        "",
        "## Dimension-level panel results",
        "",
        "| Dimension | Full−Flat panel mean | 95% bootstrap CI | Full / Flat / Tie | Sign-test p |",
        "|---|---:|---:|---:|---:|",
    ]
    for dimension in DIMENSIONS:
        item = statistics["dimensions"][dimension]
        lines.append(
            f"| {DISPLAY_NAMES[dimension]} | {item['panel_mean']:+.4f} | "
            f"[{item['bootstrap_ci_low']:+.4f}, {item['bootstrap_ci_high']:+.4f}] | "
            f"{item['full_wins']} / {item['flat_wins']} / {item['ties']} | {item['sign_test_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Judge-level descriptive means",
        "",
        "| Dimension | Judge 1 | Judge 2 | Judge 3 | Panel Mean | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dimension in DIMENSIONS:
        item = statistics["dimensions"][dimension]
        lines.append(
            f"| {DISPLAY_NAMES[dimension]} | {item['judge1_mean']:+.4f} | {item['judge2_mean']:+.4f} | "
            f"{item['judge3_mean']:+.4f} | {item['panel_mean']:+.4f} | "
            f"[{item['bootstrap_ci_low']:+.4f}, {item['bootstrap_ci_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Diagnostic status",
        "",
        f"Canonical/Adaptive trajectory-stratified diagnostic: {diagnostic_status}.",
        "The three judges are not treated as independent statistical samples; bootstrap and sign tests use 80 seed-level panel scores.",
        "",
        f"Data validation passed: **{validation['passed']}**.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing statistics directory: {args.output_dir}")

    result_paths = {name: args.blind_dir / "result" / name for name in RESULT_FILES}
    missing = [str(path) for path in result_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing result files: " + ", ".join(missing))
    result_sets = {name: _load_result(path) for name, path in result_paths.items()}
    source_key = _load_source_key(args.blind_dir / "source_key.json")
    validation = _validate_case_ids(result_sets, source_key)
    if not validation["passed"]:
        raise SystemExit("input validation failed; no statistics were generated")

    converted = _full_minus_flat_scores(result_sets, source_key)
    rng = random.Random(BOOTSTRAP_SEED)
    dimensions: dict[str, dict[str, Any]] = {}
    seed_scores: list[dict[str, Any]] = []
    for index, case_id in enumerate(sorted(source_key)):
        seed_scores.append({"case_id": case_id})
    for dimension in DIMENSIONS:
        panel_values = [
            sum(converted[judge][dimension][index] for judge in RESULT_FILES) / len(RESULT_FILES)
            for index in range(EXPECTED_CASES)
        ]
        ci = _bootstrap_ci(panel_values, rng)
        item = _metric(panel_values, ci)
        for judge_index, judge in enumerate(RESULT_FILES, 1):
            item[f"judge{judge_index}_mean"] = sum(converted[judge][dimension]) / EXPECTED_CASES
        dimensions[dimension] = item
        for index, value in enumerate(panel_values):
            seed_scores[index][dimension] = value

    statistics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_files": {name: str(path.resolve()) for name, path in result_paths.items()},
        "source_key_used_only_for_fixed_ab_orientation": True,
        "statistical_unit": "80 seed-level panel scores",
        "judge_count": 3,
        "seed_count": EXPECTED_CASES,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile bootstrap over 80 seed-level panel means",
        },
        "primary_endpoint": "overall_proposal_quality",
        "dimensions": dimensions,
    }
    statistics = _round_value(statistics)
    for row in seed_scores:
        for dimension in DIMENSIONS:
            row[dimension] = round(row[dimension], 10)
    summary_path = args.output_dir / "summary.md"

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "seed_level_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in seed_scores:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(args.output_dir / "full_vs_flat_statistics.json", {"validation": validation, **statistics})
    _write_summary(summary_path, statistics, validation, "skipped (no reliable trajectory classification was used)")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "validation_passed": validation["passed"],
        "seed_count": EXPECTED_CASES,
        "primary_panel_mean": statistics["dimensions"]["overall_proposal_quality"]["panel_mean"],
        "primary_ci": [
            statistics["dimensions"]["overall_proposal_quality"]["bootstrap_ci_low"],
            statistics["dimensions"]["overall_proposal_quality"]["bootstrap_ci_high"],
        ],
        "primary_wtl": {
            "full": statistics["dimensions"]["overall_proposal_quality"]["full_wins"],
            "flat": statistics["dimensions"]["overall_proposal_quality"]["flat_wins"],
            "tie": statistics["dimensions"]["overall_proposal_quality"]["ties"],
        },
        "primary_sign_test_p": statistics["dimensions"]["overall_proposal_quality"]["sign_test_p"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
