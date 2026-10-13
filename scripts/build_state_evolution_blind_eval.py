#!/usr/bin/env python3
"""Build a strictly anonymous Full-vs-Flat paired evaluation dataset.

Only deterministic parsing, formatting, shuffling, and validation are used.
No LLM or external service is called, and source artifacts are never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import parse_markdown_file  # noqa: E402


EXPECTED_PAPERS = 80
EXPECTED_CASES = 80
RANDOM_SEED = 20260910
PAPER_ID_RE = re.compile(r"(P\d{3})(?:_|$)")
DEFAULT_INPUT_DIR = Path("/home/tmh/research-agent/data/nlp")
DEFAULT_ASSEMBLED_DIR = Path("/home/tmh/baseline/state_evolution_ablation")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval"

SOURCE_LEAK_TOKENS = (
    "rise-full",
    "rise-flat",
    "flat runner",
    "source_method",
    "source_run",
    "output_path",
    "/home/tmh/",
    "outputs/",
)
FORBIDDEN_BLIND_KEYS = {
    "source_method",
    "source_run",
    "source_label",
    "filename",
    "source_path",
    "output_path",
    "trajectory_path",
    "metadata",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build anonymous Full-vs-Flat blind pairs")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--assembled-dir", type=Path, default=DEFAULT_ASSEMBLED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} is missing or empty")
    return value


def _paper_id(path: Path) -> str:
    match = PAPER_ID_RE.search(path.stem)
    if match is None:
        raise RuntimeError(f"cannot determine Pxxx ID from {path.name}")
    return match.group(1)


def _load_inputs(input_dir: Path) -> dict[str, dict[str, str]]:
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md")
    if len(paths) != EXPECTED_PAPERS:
        raise RuntimeError(f"expected {EXPECTED_PAPERS} Markdown files, found {len(paths)}")
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        short_id = _paper_id(path)
        if short_id in result:
            raise RuntimeError(f"duplicate paper ID: {short_id}")
        parsed = parse_markdown_file(path)
        title = _nonempty_string(parsed.title, f"{short_id} title")
        abstract = _nonempty_string(parsed.abstract, f"{short_id} abstract")
        result[short_id] = {"title": title, "abstract": abstract, "path": str(path.resolve())}
    return result


def _payload(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    payload = value.get("payload", value) if isinstance(value, dict) else value
    if not isinstance(payload, dict):
        raise RuntimeError(f"proposal payload is not an object: {path}")
    return payload


def _assert_committed(metadata_path: Path, *, status_field: str = "status") -> dict[str, Any]:
    value = _payload(metadata_path)
    if value.get(status_field) != "committed":
        raise RuntimeError(f"result is not committed: {metadata_path} ({value.get(status_field)!r})")
    return value


def _format_experiments(experiments: Any) -> str:
    if not isinstance(experiments, list) or not experiments:
        raise RuntimeError("Full experimental_plan is missing or empty")
    blocks: list[str] = []
    for index, experiment in enumerate(experiments, 1):
        if not isinstance(experiment, dict):
            raise RuntimeError(f"Full experimental_plan item {index} is not an object")
        fields = [
            ("Objective", experiment.get("objective")),
            ("Setup", experiment.get("setup")),
            ("Baselines", experiment.get("baselines")),
            ("Metrics", experiment.get("metrics")),
            ("What It Validates", experiment.get("what_it_validates")),
            ("Validation Criterion", experiment.get("validation_criterion")),
        ]
        lines = [f"Experiment {index}"]
        for label, value in fields:
            if isinstance(value, list):
                rendered = "; ".join(_nonempty_string(item, f"experiment {index} {label}") for item in value)
            else:
                rendered = _nonempty_string(value, f"experiment {index} {label}")
            lines.append(f"{label}: {rendered}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _full_proposal(assembled_dir: Path, short_id: str) -> dict[str, str]:
    branch = assembled_dir / "rise" / short_id
    _assert_committed(branch / "pilot_manifest.json")
    payload = _payload(branch / "final_research_proposal.json")
    research_problem = _nonempty_string(payload.get("research_gap"), f"{short_id} Full research_gap")
    proposed_solution = _nonempty_string(payload.get("proposed_solution"), f"{short_id} Full proposed_solution")
    rationale = _nonempty_string(payload.get("solution_rationale"), f"{short_id} Full solution_rationale")
    components = payload.get("method_components")
    if not isinstance(components, list) or not components:
        raise RuntimeError(f"{short_id} Full method_components is missing or empty")
    component_lines: list[str] = []
    for index, component in enumerate(components, 1):
        if not isinstance(component, dict):
            raise RuntimeError(f"{short_id} Full component {index} is not an object")
        component_id = _nonempty_string(component.get("id"), f"{short_id} Full component {index} id")
        description = _nonempty_string(component.get("description"), f"{short_id} Full component {index} description")
        component_lines.append(f"{component_id}: {description}")
    experimental_plan = _format_experiments(payload.get("experimental_plan"))
    return {
        "research_problem": research_problem,
        "proposed_method": (
            f"Proposed Solution:\n{proposed_solution}\n\n"
            f"Technical Rationale:\n{rationale}\n\n"
            "Method Components:\n" + "\n".join(component_lines)
        ),
        "experimental_plan": experimental_plan,
    }


def _flat_proposal(assembled_dir: Path, short_id: str) -> dict[str, str]:
    branch = assembled_dir / "flat" / short_id
    _assert_committed(branch / "run_metadata.json")
    payload = _payload(branch / "final_proposal.json")
    return {
        "research_problem": _nonempty_string(payload.get("research_problem"), f"{short_id} Flat research_problem"),
        "proposed_method": _nonempty_string(payload.get("proposed_method"), f"{short_id} Flat proposed_method"),
        "experimental_plan": _nonempty_string(payload.get("experimental_plan"), f"{short_id} Flat experimental_plan"),
    }


def _word_count(proposal: dict[str, str]) -> int:
    return sum(len(re.findall(r"\S+", proposal[field])) for field in proposal)


def _walk_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _blind_leak_audit(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    leaked_keys: list[str] = []
    leaked_values: list[dict[str, str]] = []
    for pair in pairs:
        for key in pair:
            if key in FORBIDDEN_BLIND_KEYS:
                leaked_keys.append(key)
        for path, value in _walk_strings(pair):
            lowered = value.lower()
            for token in SOURCE_LEAK_TOKENS:
                if token in lowered:
                    leaked_values.append({"case_id": pair.get("case_id", ""), "path": path, "token": token})
    return {
        "leak_found": bool(leaked_keys or leaked_values),
        "forbidden_keys": sorted(set(leaked_keys)),
        "forbidden_value_hits": leaked_values,
        "research_state_requires_manual_review": any(
            "research state" in value.lower()
            for pair in pairs
            for _, value in _walk_strings(pair)
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing blind output directory: {args.output_dir}")

    inputs = _load_inputs(args.input_dir)
    short_ids = sorted(inputs)
    full = {short_id: _full_proposal(args.assembled_dir, short_id) for short_id in short_ids}
    flat = {short_id: _flat_proposal(args.assembled_dir, short_id) for short_id in short_ids}

    rng = random.Random(RANDOM_SEED)
    shuffled_ids = list(short_ids)
    rng.shuffle(shuffled_ids)
    orientations = ["full_a"] * 40 + ["flat_a"] * 40
    rng.shuffle(orientations)

    pairs: list[dict[str, Any]] = []
    source_key: dict[str, dict[str, str]] = {}
    for index, (short_id, orientation) in enumerate(zip(shuffled_ids, orientations), 1):
        case_id = f"B2_C{index:03d}"
        if orientation == "full_a":
            proposal_a, proposal_b = full[short_id], flat[short_id]
            source_a, source_b = "full", "flat"
        else:
            proposal_a, proposal_b = flat[short_id], full[short_id]
            source_a, source_b = "flat", "full"
        pairs.append({
            "case_id": case_id,
            "seed_title": inputs[short_id]["title"],
            "seed_abstract": inputs[short_id]["abstract"],
            "proposal_a": proposal_a,
            "proposal_b": proposal_b,
        })
        source_key[case_id] = {
            "seed_id": short_id,
            "proposal_a": source_a,
            "proposal_b": source_b,
        }

    leak_audit = _blind_leak_audit(pairs)
    full_as = sum(value["proposal_a"] == "full" for value in source_key.values())
    flat_as = sum(value["proposal_a"] == "flat" for value in source_key.values())
    full_words = [_word_count(full[short_id]) for short_id in short_ids]
    flat_words = [_word_count(flat[short_id]) for short_id in short_ids]
    invalid_cases: list[str] = []
    if len(pairs) != EXPECTED_CASES:
        invalid_cases.append(f"case_count={len(pairs)}")
    if len({pair["case_id"] for pair in pairs}) != EXPECTED_CASES:
        invalid_cases.append("duplicate_case_ids")
    if len({source_key[case_id]["seed_id"] for case_id in source_key}) != EXPECTED_PAPERS:
        invalid_cases.append("duplicate_or_missing_seed_ids")
    if full_as != 40 or flat_as != 40:
        invalid_cases.append(f"orientation_balance={full_as}/{flat_as}")
    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "paper_count": EXPECTED_PAPERS,
        "case_count": len(pairs),
        "case_id_coverage": {
            "expected": [f"B2_C{index:03d}" for index in range(1, EXPECTED_CASES + 1)],
            "actual_count": len(pairs),
            "all_present": {pair["case_id"] for pair in pairs} == {f"B2_C{index:03d}" for index in range(1, EXPECTED_CASES + 1)},
        },
        "unique_seed_count": len({source_key[case_id]["seed_id"] for case_id in source_key}),
        "full_flat_final_proposal_counts": {"full": len(full), "flat": len(flat)},
        "all_seed_pairs_complete": all(
            case_id in source_key and source_key[case_id]["proposal_a"] != source_key[case_id]["proposal_b"]
            for case_id in source_key
        ),
        "orientation_balance": {"full_as_a": full_as, "full_as_b": 80 - full_as, "flat_as_a": flat_as, "flat_as_b": 80 - flat_as},
        "empty_required_fields": [],
        "blind_source_label_leakage": leak_audit,
        "proposal_field_structure_identical": all(
            set(pair["proposal_a"]) == {"research_problem", "proposed_method", "experimental_plan"}
            and set(pair["proposal_b"]) == {"research_problem", "proposed_method", "experimental_plan"}
            for pair in pairs
        ),
        "invalid_cases": invalid_cases,
        "passed": not invalid_cases and not leak_audit["leak_found"],
    }
    summary = {
        "total_cases": len(pairs),
        "full_as_a": full_as,
        "full_as_b": 80 - full_as,
        "flat_as_a": flat_as,
        "flat_as_b": 80 - flat_as,
        "missing_cases": [],
        "invalid_cases": invalid_cases,
        "average_word_count": {"full": sum(full_words) / len(full_words), "flat": sum(flat_words) / len(flat_words)},
        "full_proposal_word_count": {"average": sum(full_words) / len(full_words), "min": min(full_words), "max": max(full_words)},
        "flat_proposal_word_count": {"average": sum(flat_words) / len(flat_words), "min": min(flat_words), "max": max(flat_words)},
        "word_count_definition": "whitespace-separated tokens across the three normalized proposal fields",
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "blind_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(args.output_dir / "source_key.json", source_key)
    _write_json(args.output_dir / "validation_report.json", validation)
    _write_json(args.output_dir / "dataset_summary.json", summary)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(pairs),
        "full_proposals": len(full),
        "flat_proposals": len(flat),
        "full_as_a": full_as,
        "full_as_b": 80 - full_as,
        "flat_as_a": flat_as,
        "flat_as_b": 80 - flat_as,
        "validation_passed": validation["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
