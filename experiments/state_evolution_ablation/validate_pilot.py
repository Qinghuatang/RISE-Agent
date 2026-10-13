#!/usr/bin/env python3
"""Offline validation for the state-evolution ablation pilot outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.state_evolution_ablation.full_reference import FullReferenceError, inspect_full_reference
from experiments.state_evolution_ablation.pilot8_config import (
    DEFAULT_FULL_OUTPUT_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    PILOT_SEEDS,
    resolve_seed_paths,
)


FORBIDDEN_FLAT_KEYS = {
    "active_gap",
    "current_solution",
    "components",
    "experiments",
    "gap_status",
    "novelty_state",
    "affected_units",
    "changed_units",
    "stale_dependencies",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        found.update(str(key) for key in value)
        for item in value.values():
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def _load_status(path: Path) -> str | None:
    try:
        value = _json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid"
    return value.get("status") if isinstance(value, dict) else "invalid"


def _validate_flat(branch: Path) -> list[str]:
    errors: list[str] = []
    metadata_path = branch / "run_metadata.json"
    for filename in ("run_metadata.json", "action_history.json", "evidence_history.json", "llm_calls.jsonl"):
        if not (branch / filename).is_file():
            errors.append(f"missing Flat artifact: {branch / filename}")
    if errors:
        return errors
    try:
        metadata = _json(metadata_path)
        actions = _json(branch / "action_history.json")
        evidence = _json(branch / "evidence_history.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid Flat artifact JSON in {branch}: {exc}"]
    if not isinstance(metadata, dict):
        errors.append("Flat run_metadata is not an object")
        return errors
    if metadata.get("system") != "flat":
        errors.append("Flat metadata system is not 'flat'")
    if metadata.get("max_steps") != DEFAULT_MAX_STEPS or metadata.get("max_searches") != DEFAULT_MAX_SEARCHES:
        errors.append("Flat budget is incompatible with current Full defaults")
    if not isinstance(actions, list):
        errors.append("Flat action_history is not a list")
        actions = []
    if not isinstance(evidence, list):
        errors.append("Flat evidence_history is not a list")
        evidence = []
    if len(actions) > DEFAULT_MAX_STEPS:
        errors.append("Flat trajectory exceeds max_steps")
    if metadata.get("actual_steps") != len(actions):
        errors.append("Flat actual_steps disagrees with action_history")
    if metadata.get("actual_searches", 0) > DEFAULT_MAX_SEARCHES:
        errors.append("Flat actual_searches exceeds max_searches")
    if metadata.get("status") in {"committed", "stopped"} and not (branch / "final_proposal.md").is_file():
        errors.append("completed Flat branch has no final_proposal.md")
    if not isinstance(actions, list) or not all(isinstance(item, dict) for item in actions):
        errors.append("Flat action_history contains non-object entries")
    else:
        actions_seen = [item.get("action") for item in actions]
        invalid_actions = sorted(set(actions_seen) - {"VERIFY", "SEEK_INSPIRATION", "REFINE_PROPOSAL", "DESIGN_EXPERIMENTS", "COMMIT"})
        if invalid_actions:
            errors.append(f"Flat action_history contains invalid actions: {invalid_actions}")
        for left, right in zip(actions, actions[1:]):
            if left.get("step", 0) + 1 != right.get("step", 0):
                errors.append("Flat trajectory step numbering is not contiguous")
                break
        for index in range(len(actions) - 2):
            if actions[index].get("action") == actions[index + 1].get("action") == actions[index + 2].get("action") and actions[index].get("action") != "REFINE_PROPOSAL":
                errors.append("Flat trajectory may contain an obvious three-action loop")
                break
    found: set[str] = set()
    for filename in ("run_metadata.json", "action_history.json", "evidence_history.json", "provider_audit.json"):
        path = branch / filename
        if path.is_file():
            try:
                _walk_keys(_json(path), found)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
    forbidden = sorted(found & FORBIDDEN_FLAT_KEYS)
    if forbidden:
        errors.append(f"Flat artifacts persist RISE-only fields: {forbidden}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Full-reference/Flat pilot offline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--full-output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--semantic-scholar-config", type=Path, default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG)
    args = parser.parse_args()
    paths = resolve_seed_paths(args.input_dir)
    report: dict[str, Any] = {
        "experiment": "state_evolution_ablation_pilot8",
        "output_dir": str(args.output_dir.resolve()),
        "expected_seed_count": 8,
        "seed_count": len(PILOT_SEEDS),
        "full_reference": {"available": 0, "invalid": 0, "api_calls_in_this_experiment": 0},
        "flat": {"available": 0, "complete": 0, "failed_or_incomplete": 0},
        "checks": [],
        "errors": [],
    }
    errors: list[str] = []
    if len(PILOT_SEEDS) != 8:
        errors.append(f"pilot config contains {len(PILOT_SEEDS)} seeds, expected 8")
    if not args.input_dir.is_dir():
        errors.append(f"input directory missing: {args.input_dir}")
    if not args.full_output_dir.is_dir():
        errors.append(f"historical Full directory missing: {args.full_output_dir}")
    if not args.semantic_scholar_config.is_file():
        errors.append(f"Semantic Scholar config missing: {args.semantic_scholar_config}")
    if args.output_dir.is_dir():
        for seed in PILOT_SEEDS:
            branch_full = args.output_dir / seed.short_id / "full"
            branch_flat = args.output_dir / seed.short_id / "flat"
            try:
                reference = inspect_full_reference(seed, args.full_output_dir)
                report["full_reference"]["available"] += 1
                report["full_reference"]["api_calls_in_this_experiment"] += int(reference["api_calls_in_this_experiment"])
            except FullReferenceError as exc:
                report["full_reference"]["invalid"] += 1
                errors.append(f"{seed.short_id} Full reference: {exc}")
            if branch_full.is_dir() and (branch_full / "full_reference.json").is_file():
                try:
                    stored = _json(branch_full / "full_reference.json")
                    if stored.get("paper_id") != seed.paper_id or stored.get("read_only") is not True:
                        errors.append(f"{seed.short_id} stored Full reference is incompatible")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(f"{seed.short_id} stored Full reference is invalid: {exc}")
            flat_status = _load_status(branch_flat / "run_metadata.json")
            if branch_flat.exists():
                report["flat"]["available"] += 1
                flat_errors = _validate_flat(branch_flat)
                if flat_errors:
                    report["flat"]["failed_or_incomplete"] += 1
                    errors.extend(f"{seed.short_id} Flat: {item}" for item in flat_errors)
                else:
                    report["flat"]["complete"] += 1
                if flat_status == "failed":
                    report["flat"]["failed_or_incomplete"] += 1
    else:
        errors.append(f"pilot output directory missing: {args.output_dir}")
    report["checks"] = {
        "seed_files_present": [short_id for short_id, path in paths.items() if path.is_file()],
        "seed_files_missing": [short_id for short_id, path in paths.items() if not path.is_file()],
        "full_reference_coverage": report["full_reference"]["available"] == 8,
        "flat_complete_coverage": report["flat"]["complete"] == 8,
        "budget_limits": {"max_steps": DEFAULT_MAX_STEPS, "max_searches": DEFAULT_MAX_SEARCHES},
        "flat_typed_rise_state_detected": any("RISE-only fields" in item for item in errors),
    }
    report["errors"] = errors
    report["valid"] = not errors
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plain = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "validation_report.json").write_text(_plain, encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "full_references": report["full_reference"], "flat": report["flat"], "errors": len(errors)}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

