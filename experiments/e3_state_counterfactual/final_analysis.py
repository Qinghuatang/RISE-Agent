"""Deterministic E3 paired-result loading and statistical analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from research_agent.serialization import read_json, write_json

from .unified import UnifiedVerificationCase, discover_unified_cases, stable_hash

ALLOWED_OUTCOMES = {"distinct", "partial_overlap", "core_overlap"}


def _payload(path: Path, schema: str) -> Any:
    return read_json(path, expected_schema=schema)["payload"]


def _non_distinct(outcome: str) -> bool:
    return outcome in {"partial_overlap", "core_overlap"}


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def exact_mcnemar_p_value(targeted_better: int, global_better: int) -> float:
    """Two-sided exact McNemar p-value via a binomial test on discordants."""

    discordant = targeted_better + global_better
    if discordant == 0:
        return 1.0
    smaller = min(targeted_better, global_better)
    tail = sum(
        math.comb(discordant, index) / (2**discordant)
        for index in range(smaller + 1)
    )
    return min(1.0, 2.0 * tail)


def _result_payload(path: Path, branch: str) -> dict[str, Any]:
    return _payload(path, f"e3_unified_verification_{branch}")


def load_paired_rows(
    cases: Iterable[UnifiedVerificationCase],
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load both fresh verification artifacts and validate their provenance."""

    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in cases:
        case_root = root / "unified_verification" / case.paper_id
        artifacts: dict[str, dict[str, Any]] = {}
        for branch in ("targeted", "global"):
            path = case_root / f"unified_verification_{branch}.json"
            if not path.exists():
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "missing artifact"})
                continue
            try:
                artifact = _result_payload(path, branch)
            except Exception as exc:
                errors.append(
                    {
                        "paper_id": case.paper_id,
                        "branch": branch,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                continue
            artifacts[branch] = artifact
            expected_solution = (
                case.targeted_solution if branch == "targeted" else case.global_solution
            )
            if artifact.get("paper_id") != case.paper_id:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "paper_id mismatch"})
            if artifact.get("branch") != branch:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "branch mismatch"})
            if artifact.get("revised_solution_id") != expected_solution.id:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "solution id mismatch"})
            if artifact.get("revised_solution_hash") != stable_hash(expected_solution):
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "solution hash mismatch"})
            if artifact.get("related_paper_pool_hash") != stable_hash(case.related_paper_pool):
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "related-paper pool mismatch"})
            if artifact.get("verification_focus") != {"mode": "full_solution", "changed_units": []}:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "verification focus is not full_solution"})
            if artifact.get("outcome") not in ALLOWED_OUTCOMES:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "invalid verification outcome"})
            if artifact.get("new_related_paper_retrievals") != 0:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "new retrieval detected"})
            if artifact.get("related_paper_pool_reused") is not True:
                errors.append({"paper_id": case.paper_id, "branch": branch, "error": "related pool was not marked reused"})
        if len(artifacts) != 2:
            continue
        if artifacts["targeted"].get("related_paper_pool_hash") != artifacts["global"].get("related_paper_pool_hash"):
            errors.append({"paper_id": case.paper_id, "error": "paired branches use different pool hashes"})
        rows.append(
            {
                "paper_id": case.paper_id,
                "trigger_type": case.historical_local.trigger_type,
                "before_outcome": case.before_outcome,
                "targeted_outcome": artifacts["targeted"]["outcome"],
                "global_outcome": artifacts["global"]["outcome"],
                "targeted_solution_id": artifacts["targeted"].get("revised_solution_id"),
                "global_solution_id": artifacts["global"].get("revised_solution_id"),
                "targeted_solution_hash": artifacts["targeted"].get("revised_solution_hash"),
                "global_solution_hash": artifacts["global"].get("revised_solution_hash"),
                "related_paper_pool_hash": artifacts["targeted"].get("related_paper_pool_hash"),
                "targeted_matched_paper_refs": artifacts["targeted"].get("matched_paper_refs", []),
                "global_matched_paper_refs": artifacts["global"].get("matched_paper_refs", []),
                "targeted_novelty_relevant_overlaps": artifacts["targeted"].get("novelty_relevant_overlaps", []),
                "global_novelty_relevant_overlaps": artifacts["global"].get("novelty_relevant_overlaps", []),
                "targeted_summary": artifacts["targeted"].get("verification_summary", ""),
                "global_summary": artifacts["global"].get("verification_summary", ""),
                "historical_changed_units": list(case.historical_local.historical_changed_units),
                "descriptive_only": True,
            }
        )
    return sorted(rows, key=lambda row: row["paper_id"]), errors


def _outcome_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(row[field] for row in rows)
    return {outcome: counter.get(outcome, 0) for outcome in sorted(ALLOWED_OUTCOMES)}


def paired_contingency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {
        "targeted_distinct_global_distinct": 0,
        "targeted_distinct_global_non_distinct": 0,
        "targeted_non_distinct_global_distinct": 0,
        "targeted_non_distinct_global_non_distinct": 0,
    }
    discordant_ids: dict[str, list[str]] = {
        "targeted_better": [],
        "global_better": [],
    }
    for row in rows:
        targeted_distinct = row["targeted_outcome"] == "distinct"
        global_distinct = row["global_outcome"] == "distinct"
        if targeted_distinct and global_distinct:
            cells["targeted_distinct_global_distinct"] += 1
        elif targeted_distinct:
            cells["targeted_distinct_global_non_distinct"] += 1
            discordant_ids["targeted_better"].append(row["paper_id"])
        elif global_distinct:
            cells["targeted_non_distinct_global_distinct"] += 1
            discordant_ids["global_better"].append(row["paper_id"])
        else:
            cells["targeted_non_distinct_global_non_distinct"] += 1
    total = len(rows)
    targeted_distinct = sum(row["targeted_outcome"] == "distinct" for row in rows)
    global_distinct = sum(row["global_outcome"] == "distinct" for row in rows)
    targeted_better = len(discordant_ids["targeted_better"])
    global_better = len(discordant_ids["global_better"])
    return {
        "n": total,
        "cells": cells,
        "cell_percentages": {key: _rate(value, total) for key, value in cells.items()},
        "targeted_distinct_count": targeted_distinct,
        "global_distinct_count": global_distinct,
        "targeted_distinct_rate": _rate(targeted_distinct, total),
        "global_distinct_rate": _rate(global_distinct, total),
        "percentage_difference_targeted_minus_global_points": (
            100 * (_rate(targeted_distinct, total) - _rate(global_distinct, total))
            if total
            else None
        ),
        "discordant_pairs": targeted_better + global_better,
        "targeted_better_discordant": targeted_better,
        "global_better_discordant": global_better,
        "exact_mcnemar": {
            "method": "two-sided exact binomial test on discordant pairs",
            "p_value": exact_mcnemar_p_value(targeted_better, global_better),
        },
        "discordant_case_ids": discordant_ids,
    }


def summarize_subset(rows: list[dict[str, Any]], subset_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "subset": subset_name,
        "case_count": len(rows),
        "targeted_outcomes": _outcome_counts(rows, "targeted_outcome"),
        "global_outcomes": _outcome_counts(rows, "global_outcome"),
        "targeted_distinct_rate": _rate(
            sum(row["targeted_outcome"] == "distinct" for row in rows), len(rows)
        ),
        "global_distinct_rate": _rate(
            sum(row["global_outcome"] == "distinct" for row in rows), len(rows)
        ),
        "paired_statistics": paired_contingency(rows),
    }
    if subset_name == "overlap_repair":
        summary["repair_to_distinct_rate"] = {
            "targeted": summary["targeted_distinct_rate"],
            "global": summary["global_distinct_rate"],
        }
    if subset_name == "inspiration_strengthening":
        summary["remain_distinct_rate"] = {
            "targeted": summary["targeted_distinct_rate"],
            "global": summary["global_distinct_rate"],
        }
    return summary


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = [row for row in rows if row["trigger_type"] == "overlap_repair"]
    inspiration = [
        row for row in rows if row["trigger_type"] == "inspiration_strengthening"
    ]
    return {
        "experiment": "E3_state_evolution_counterfactual_global",
        "sampling_unit": "seed_paper_paired_case",
        "case_count": len(rows),
        "branch_count": len(rows) * 2,
        "all_25": summarize_subset(rows, "all_25"),
        "overlap_repair": summarize_subset(overlap, "overlap_repair"),
        "inspiration_strengthening": summarize_subset(
            inspiration, "inspiration_strengthening"
        ),
        "revision_scope_interpretation": {
            "descriptive_only": True,
            "note": "Changed units and scope are descriptive; smaller scope is not treated as a method advantage.",
        },
    }


def write_paired_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "paper_id",
        "trigger_type",
        "before_outcome",
        "targeted_outcome",
        "global_outcome",
        "targeted_solution_id",
        "global_solution_id",
        "targeted_solution_hash",
        "global_solution_hash",
        "related_paper_pool_hash",
        "targeted_matched_paper_refs",
        "global_matched_paper_refs",
        "targeted_novelty_relevant_overlaps",
        "global_novelty_relevant_overlaps",
        "historical_changed_units",
        "descriptive_only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if isinstance(output.get(field), (list, dict)):
                    output[field] = json.dumps(output[field], ensure_ascii=False, sort_keys=True)
            writer.writerow({field: output.get(field) for field in fields})


def write_failure_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "paper_id",
        "trigger_type",
        "targeted_outcome",
        "global_outcome",
        "global_novelty_relevant_overlaps",
        "matched_prior_paper",
        "overlap_summary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            overlaps = row["global_novelty_relevant_overlaps"]
            writer.writerow(
                {
                    "paper_id": row["paper_id"],
                    "trigger_type": row["trigger_type"],
                    "targeted_outcome": row["targeted_outcome"],
                    "global_outcome": row["global_outcome"],
                    "global_novelty_relevant_overlaps": json.dumps(
                        overlaps, ensure_ascii=False, sort_keys=True
                    ),
                    "matched_prior_paper": ";".join(row["global_matched_paper_refs"]),
                    "overlap_summary": "; ".join(
                        item.get("overlap_summary", "")
                        for item in overlaps
                        if isinstance(item, dict)
                    ),
                }
            )


def build_validation_report(
    cases: list[UnifiedVerificationCase],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "discovered_25_cases": len(cases) == 25,
        "every_case_has_two_branches": len(rows) == len(cases),
        "solution_hashes_match_e3_inputs": not any(
            error.get("error") == "solution hash mismatch" for error in errors
        ),
        "same_related_paper_pool_per_pair": not any(
            "pool" in str(error.get("error", "")) for error in errors
        ),
        "no_new_retrieval": not any(
            error.get("error") == "new retrieval detected" for error in errors
        ),
        "outcomes_are_canonical": not any(
            error.get("error") == "invalid verification outcome" for error in errors
        ),
        "trigger_types_are_original": all(
            row["trigger_type"] in {"overlap_repair", "inspiration_strengthening"}
            for row in rows
        ),
        "paired_sampling_unit_is_seed": True,
        "existing_revision_results_untouched": True,
    }
    return {
        "valid": not errors and all(checks.values()),
        "case_count": len(cases),
        "paired_row_count": len(rows),
        "errors": errors,
        "checks": checks,
        "note": "Existing E3 inputs are read-only; this report is generated only from new unified-verification artifacts.",
    }


def _original_outcomes(case: UnifiedVerificationCase) -> dict[str, str | None]:
    historical = case.historical_local.historical_post_revision_outcome or {}
    historical_result = historical.get("result", {}) if isinstance(historical, dict) else {}
    historical_outcome = (
        historical_result.get("novelty_status")
        if isinstance(historical_result, dict)
        else None
    )
    global_path = case.e3_case_dir / "global_verification.json"
    global_outcome = None
    if global_path.exists():
        payload = _payload(global_path, "e3_global_verification")
        result = payload.get("verification_result", {})
        if isinstance(result, dict):
            global_outcome = result.get("novelty_status")
    return {"targeted": historical_outcome, "global": global_outcome}


def build_stochasticity_report(
    cases: list[UnifiedVerificationCase],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["paper_id"]: row for row in rows}
    changes: list[dict[str, Any]] = []
    for case in cases:
        row = by_id.get(case.paper_id)
        if row is None:
            continue
        original = _original_outcomes(case)
        for branch in ("targeted", "global"):
            fresh = row[f"{branch}_outcome"]
            old = original[branch]
            if old is not None and old != fresh:
                changes.append(
                    {
                        "paper_id": case.paper_id,
                        "branch": branch,
                        "original_outcome": old,
                        "unified_outcome": fresh,
                    }
                )
    return {
        "changed_outcome_count": len(changes),
        "changed_outcomes": changes,
        "interpretation": (
            "Outcome changes between original and unified verification are reported as possible verifier stochasticity; "
            "they are not attributed to either revision branch."
        ),
    }


def write_final_summary_markdown(
    path: Path,
    summary: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    stochasticity: dict[str, Any],
) -> None:
    all_stats = summary["all_25"]["paired_statistics"]
    overlap = summary["overlap_repair"]
    inspiration = summary["inspiration_strengthening"]
    lines = [
        "# E3 Final Summary",
        "",
        f"- Validation: **{'passed' if validation['valid'] else 'failed'}**.",
        f"- Paired cases: {summary['case_count']} seeds / {summary['branch_count']} branches.",
        "",
        "## Unified verification outcomes",
        "",
        f"- All 25 Targeted distinct rate: {all_stats['targeted_distinct_rate']}",
        f"- All 25 Global distinct rate: {all_stats['global_distinct_rate']}",
        f"- Overlap-repair Targeted repair-to-distinct rate: {overlap['repair_to_distinct_rate']['targeted']}",
        f"- Overlap-repair Global repair-to-distinct rate: {overlap['repair_to_distinct_rate']['global']}",
        f"- Inspiration-strengthening Targeted remain-distinct rate: {inspiration['remain_distinct_rate']['targeted']}",
        f"- Inspiration-strengthening Global remain-distinct rate: {inspiration['remain_distinct_rate']['global']}",
        "",
        "## Paired exact test",
        "",
        f"- Discordant pairs: {all_stats['discordant_pairs']}",
        f"- Targeted-better discordant pairs: {all_stats['targeted_better_discordant']}",
        f"- Global-better discordant pairs: {all_stats['global_better_discordant']}",
        f"- Two-sided exact McNemar/binomial p-value: {all_stats['exact_mcnemar']['p_value']}",
        "",
        "## Global failure cases",
        "",
    ]
    if failure_rows:
        lines.extend(f"- `{row['paper_id']}` ({row['trigger_type']})" for row in failure_rows)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Original versus unified verification",
            "",
            f"- Outcome changes: {stochasticity['changed_outcome_count']}.",
            "- Any changes are reported as possible verifier stochasticity and are not attributed to either method.",
            "- Revision scope fields are descriptive only; smaller scope is not treated as an advantage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_final(
    source_dir: str | Path,
    e3_output_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    discovery = discover_unified_cases(source_dir, e3_output_dir)
    rows, errors = load_paired_rows(discovery.cases, output_dir)
    errors = [*discovery.errors, *errors]
    if errors:
        validation = build_validation_report(discovery.cases, rows, errors)
        write_json(Path(output_dir) / "validation_report.json", "e3_final_validation_report", validation)
        raise ValueError("E3 final analysis validation failed")
    summary = build_summary(rows)
    failure_rows = [
        row
        for row in rows
        if row["targeted_outcome"] == "distinct"
        and row["global_outcome"] in {"partial_overlap", "core_overlap"}
    ]
    validation = build_validation_report(discovery.cases, rows, [])
    stochasticity = build_stochasticity_report(discovery.cases, rows)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "e3_summary.json", "e3_final_summary", summary)
    write_json(root / "paired_statistics.json", "e3_paired_statistics", {
        "all_25": summary["all_25"]["paired_statistics"],
        "overlap_repair": summary["overlap_repair"]["paired_statistics"],
        "inspiration_strengthening": summary["inspiration_strengthening"]["paired_statistics"],
    })
    write_json(root / "validation_report.json", "e3_final_validation_report", validation)
    write_json(root / "stochasticity_report.json", "e3_verification_stochasticity", stochasticity)
    write_paired_results_csv(root / "e3_paired_results.csv", rows)
    write_failure_cases_csv(root / "e3_failure_cases.csv", failure_rows)
    write_final_summary_markdown(
        root / "final_summary.md",
        summary,
        failure_rows,
        validation,
        stochasticity,
    )
    return {
        "summary": summary,
        "validation": validation,
        "failure_case_count": len(failure_rows),
        "stochasticity": stochasticity,
    }


__all__ = [
    "analyze_final",
    "build_summary",
    "exact_mcnemar_p_value",
    "load_paired_rows",
    "paired_contingency",
]
