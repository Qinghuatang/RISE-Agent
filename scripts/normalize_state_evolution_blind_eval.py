#!/usr/bin/env python3
"""Create the final content-preserving Full-vs-Flat blind dataset.

This script deliberately does not call an LLM or inspect any scientific
content semantically.  It only removes explicit template/schema labels,
normalizes the outer field shape, deterministically reorders cases, and
validates the resulting blind artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval_normalized"
EXPECTED_CASES = 80
BLOCK_SIZE = 20
RANDOM_SEED = 20260910
REQUIRED_PROPOSAL_FIELDS = {
    "research_problem",
    "proposed_method",
    "experimental_plan",
}


_TEMPLATE_LABELS = (
    "Proposed Solution",
    "Technical Rationale",
    "Method Components",
    "Objective",
    "Setup",
    "Baselines",
    "Metrics",
    "What It Validates",
    "Validation Criterion",
)
_TEMPLATE_LABEL_RE = re.compile(
    r"^(?P<prefix>\s*(?:[-*]\s*)?)"
    r"(?:\*\*|__|`)?\s*"
    r"(?P<label>Proposed Solution|Technical Rationale|Method Components|"
    r"Objective|Setup|Baselines|Metrics|What It Validates|Validation Criterion)"
    r"\s*(?:\*\*|__|`)?\s*[:：]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_COMPONENT_LABEL_RE = re.compile(
    r"^(?P<prefix>\s*(?:[-*]\s*)?)"
    r"(?P<label>C\d+|Component\s+\d+)\s*[:：]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_INLINE_COMPONENT_LABEL_RE = re.compile(r"\bC\d+\s*[:：]\s*", re.IGNORECASE)
_EXPERIMENT_LABEL_RE = re.compile(
    r"^(?P<prefix>\s*(?:[-*]\s*)?)"
    r"Experiment\s+\d+\s*(?::|：)?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_SNAKE_SCHEMA_REPLACEMENTS = {
    "core_solution": "proposed method",
    "proposed_solution": "proposed method",
    "solution_rationale": "technical rationale",
    "method_components": "method components",
    "experimental_plan": "experimental plan",
    "research_gap": "research problem",
    "what_it_validates": "what it validates",
    "validation_criterion": "validation criterion",
}
_SNAKE_SCHEMA_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _SNAKE_SCHEMA_REPLACEMENTS)) + r")\b",
    re.IGNORECASE,
)
_SEED_METHOD_RE = re.compile(r"\bSeed Method\b", re.IGNORECASE)

_FORBIDDEN_BLIND_KEYS = {
    "source_method",
    "source_run",
    "source_label",
    "filename",
    "source_path",
    "output_path",
    "trajectory_path",
    "metadata",
}
_SOURCE_IDENTITY_RE = re.compile(
    r"\b(?:rise\s*[-_]\s*(?:full|flat)|rise[- ]full|rise[- ]flat|"
    r"flat\s+runner|full\s+runner)\b",
    re.IGNORECASE,
)
_PATH_LEAK_TOKENS = ("/home/tmh/", "outputs/", "trajectory_path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize and rebalance the anonymous Full-vs-Flat blind dataset"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} is missing or empty")
    return value


def _load_pairs(input_dir: Path) -> dict[str, dict[str, Any]]:
    path = input_dir / "blind_pairs.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing input pair file: {path}")
    pairs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            case_id = _nonempty_string(pair.get("case_id"), f"input line {line_number} case_id")
            if case_id in pairs:
                raise RuntimeError(f"duplicate input case_id: {case_id}")
            pairs[case_id] = pair
    if len(pairs) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} input pairs, found {len(pairs)}")
    return pairs


def _load_source_key(input_dir: Path) -> dict[str, dict[str, str]]:
    value = _read_json(input_dir / "source_key.json")
    if not isinstance(value, dict) or len(value) != EXPECTED_CASES:
        raise RuntimeError("source_key.json must contain exactly 80 case mappings")
    result: dict[str, dict[str, str]] = {}
    for case_id, mapping in value.items():
        if not isinstance(mapping, dict):
            raise RuntimeError(f"source mapping is not an object: {case_id}")
        seed_id = _nonempty_string(mapping.get("seed_id"), f"{case_id} seed_id")
        proposal_a = _nonempty_string(mapping.get("proposal_a"), f"{case_id} proposal_a")
        proposal_b = _nonempty_string(mapping.get("proposal_b"), f"{case_id} proposal_b")
        if {proposal_a, proposal_b} != {"full", "flat"}:
            raise RuntimeError(f"{case_id} must map exactly one full and one flat proposal")
        result[case_id] = {
            "seed_id": seed_id,
            "proposal_a": proposal_a,
            "proposal_b": proposal_b,
        }
    if len({mapping["seed_id"] for mapping in result.values()}) != EXPECTED_CASES:
        raise RuntimeError("source_key.json does not cover 80 unique seeds")
    return result


def _input_validation(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "validation_report.json"
    if not path.is_file():
        return {"available": False, "passed": None}
    report = _read_json(path)
    return {
        "available": True,
        "passed": report.get("passed") is True,
        "reported_case_count": report.get("case_count"),
        "reported_unique_seed_count": report.get("unique_seed_count"),
        "reported_full_flat_counts": report.get("full_flat_final_proposal_counts"),
    }


def _normalize_proposal(proposal: Any) -> tuple[dict[str, str], Counter[str]]:
    if not isinstance(proposal, dict):
        raise RuntimeError("proposal is not an object")
    if set(proposal) != REQUIRED_PROPOSAL_FIELDS:
        raise RuntimeError(f"proposal fields are not exactly {sorted(REQUIRED_PROPOSAL_FIELDS)}")

    stats: Counter[str] = Counter()
    normalized: dict[str, str] = {}
    for field in REQUIRED_PROPOSAL_FIELDS:
        text = _nonempty_string(proposal[field], f"proposal.{field}")
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            template_match = _TEMPLATE_LABEL_RE.match(line)
            if template_match:
                stats["template_labels_seen"] += 1
                stats[f"label:{template_match.group('label').lower()}"] += 1
                rest = template_match.group("rest").strip()
                if not rest:
                    stats["template_heading_lines_removed"] += 1
                    continue
                line = template_match.group("prefix") + rest

            experiment_match = _EXPERIMENT_LABEL_RE.match(line)
            if experiment_match:
                stats["experiment_labels_seen"] += 1
                rest = experiment_match.group("rest").strip()
                if not rest:
                    stats["experiment_heading_lines_removed"] += 1
                    continue
                line = experiment_match.group("prefix") + rest

            # Some generated lines contain two nested labels, e.g.
            # ``C1: Component 1: description``. Remove both labels while
            # retaining the description.
            component_match = _COMPONENT_LABEL_RE.match(line)
            while component_match:
                stats["component_labels_seen"] += 1
                stats["component_number_labels_removed"] += 1
                line = component_match.group("prefix") + component_match.group("rest").strip()
                component_match = _COMPONENT_LABEL_RE.match(line)

            inline_component_line, inline_count = _INLINE_COMPONENT_LABEL_RE.subn("", line)
            if inline_count:
                stats["inline_component_number_labels_removed"] += inline_count
            line = inline_component_line

            replaced_line = _SNAKE_SCHEMA_RE.sub(
                lambda match: _SNAKE_SCHEMA_REPLACEMENTS[match.group(0).lower()],
                line,
            )
            duplicate_neutral_label, duplicate_count = re.subn(
                r"\bproposed\s+proposed\s+method\b",
                "proposed method",
                replaced_line,
                flags=re.IGNORECASE,
            )
            if duplicate_count:
                stats["duplicate_neutral_labels_collapsed"] += duplicate_count
            replaced_line = duplicate_neutral_label
            if replaced_line != line:
                stats["schema_tokens_replaced"] += 1
            line = replaced_line

            replaced_seed_method = _SEED_METHOD_RE.sub("method in the seed paper", line)
            if replaced_seed_method != line:
                stats["seed_method_labels_replaced"] += 1
            lines.append(replaced_seed_method)

        # Whitespace-only normalization is limited to removing surplus empty
        # lines left behind by deleted headings. No content lines are merged.
        compact_lines: list[str] = []
        blank_run = 0
        for line in lines:
            if not line.strip():
                blank_run += 1
                if blank_run > 2:
                    stats["surplus_blank_lines_removed"] += 1
                    continue
            else:
                blank_run = 0
            compact_lines.append(line)
        normalized_text = "\n".join(compact_lines).strip()
        normalized[field] = _nonempty_string(normalized_text, f"normalized proposal.{field}")
    return normalized, stats


def _word_count(proposal: dict[str, str]) -> int:
    return sum(len(re.findall(r"\S+", proposal[field])) for field in REQUIRED_PROPOSAL_FIELDS)


def _walk_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _leakage_audit(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_keys: list[str] = []
    identity_hits: list[dict[str, str]] = []
    path_hits: list[dict[str, str]] = []
    generic_full_flat_counts: Counter[str] = Counter()
    generic_re = re.compile(r"\b(?:full|flat)\b", re.IGNORECASE)
    for pair in pairs:
        for key in pair:
            if key in _FORBIDDEN_BLIND_KEYS:
                forbidden_keys.append(key)
        for path, value in _walk_strings(pair):
            if path.endswith(".case_id"):
                continue
            for match in _SOURCE_IDENTITY_RE.finditer(value):
                identity_hits.append({
                    "case_id": pair.get("case_id", ""),
                    "path": path,
                    "text": match.group(0),
                })
            lowered = value.lower()
            for token in _PATH_LEAK_TOKENS:
                if token in lowered:
                    path_hits.append({
                        "case_id": pair.get("case_id", ""),
                        "path": path,
                        "token": token,
                    })
            for match in generic_re.finditer(value):
                generic_full_flat_counts[match.group(0).lower()] += 1
    return {
        "leak_found": bool(forbidden_keys or identity_hits or path_hits),
        "forbidden_keys": sorted(set(forbidden_keys)),
        "source_identity_hits": identity_hits,
        "path_hits": path_hits,
        # Bare words can be legitimate scientific prose (e.g. "full model").
        # They are reported for review but do not fail the source-label audit.
        "generic_full_flat_terms_for_review": dict(sorted(generic_full_flat_counts.items())),
        "research_state_requires_manual_review": any(
            "research state" in value.lower()
            for pair in pairs
            for _, value in _walk_strings(pair)
        ),
    }


def _format_fingerprint_audit(pairs: list[dict[str, Any]]) -> dict[str, int]:
    marker_patterns = {
        "Proposed Solution:": re.compile(r"^\s*Proposed Solution\s*:", re.IGNORECASE | re.MULTILINE),
        "Technical Rationale:": re.compile(r"^\s*Technical Rationale\s*:", re.IGNORECASE | re.MULTILINE),
        "Method Components:": re.compile(r"^\s*Method Components\s*:", re.IGNORECASE | re.MULTILINE),
        "Objective:": re.compile(r"^\s*Objective\s*:", re.IGNORECASE | re.MULTILINE),
        "Setup:": re.compile(r"^\s*Setup\s*:", re.IGNORECASE | re.MULTILINE),
        "Baselines:": re.compile(r"^\s*Baselines\s*:", re.IGNORECASE | re.MULTILINE),
        "Metrics:": re.compile(r"^\s*Metrics\s*:", re.IGNORECASE | re.MULTILINE),
        "Validation Criterion:": re.compile(r"^\s*Validation Criterion\s*:", re.IGNORECASE | re.MULTILINE),
        "core_solution": re.compile(r"\bcore_solution\b", re.IGNORECASE),
        "component_number_label": re.compile(r"^\s*(?:[-*]\s*)?(?:C\d+|Component\s+\d+)\s*:", re.IGNORECASE | re.MULTILINE),
        "experiment_number_label": re.compile(r"^\s*Experiment\s+\d+\s*:?", re.IGNORECASE | re.MULTILINE),
        "What It Validates:": re.compile(r"^\s*What It Validates\s*:", re.IGNORECASE | re.MULTILINE),
        "Seed Method": re.compile(r"\bSeed Method\b", re.IGNORECASE),
    }
    counts = {name: 0 for name in marker_patterns}
    for pair in pairs:
        for field in ("research_problem", "proposed_method", "experimental_plan"):
            text_a = pair["proposal_a"][field]
            text_b = pair["proposal_b"][field]
            for name, pattern in marker_patterns.items():
                counts[name] += len(pattern.findall(text_a)) + len(pattern.findall(text_b))
    return counts


def _stats(values: list[int]) -> dict[str, float | int]:
    return {
        "average": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")

    pairs_by_case = _load_pairs(args.input_dir)
    source_key = _load_source_key(args.input_dir)
    input_validation = _input_validation(args.input_dir)
    expected_case_ids = {f"B2_C{index:03d}" for index in range(1, EXPECTED_CASES + 1)}
    if set(pairs_by_case) != expected_case_ids or set(source_key) != expected_case_ids:
        raise RuntimeError("input case IDs do not exactly cover B2_C001-B2_C080")

    by_seed: dict[str, dict[str, Any]] = {}
    before_words: dict[str, int] = {}
    after_words: dict[str, int] = {}
    normalization_counts: Counter[str] = Counter()
    for case_id in sorted(source_key):
        pair = pairs_by_case[case_id]
        mapping = source_key[case_id]
        proposal_by_method = {
            mapping["proposal_a"]: pair["proposal_a"],
            mapping["proposal_b"]: pair["proposal_b"],
        }
        seed_id = mapping["seed_id"]
        if seed_id in by_seed:
            raise RuntimeError(f"duplicate seed in input: {seed_id}")
        normalized_by_method: dict[str, dict[str, str]] = {}
        for method in ("full", "flat"):
            original = proposal_by_method[method]
            before_words[f"{seed_id}:{method}"] = _word_count(original)
            normalized, stats = _normalize_proposal(original)
            normalized_by_method[method] = normalized
            after_words[f"{seed_id}:{method}"] = _word_count(normalized)
            normalization_counts.update({f"{method}.{key}": value for key, value in stats.items()})
        by_seed[seed_id] = {
            "seed_title": _nonempty_string(pair.get("seed_title"), f"{case_id} seed_title"),
            "seed_abstract": _nonempty_string(pair.get("seed_abstract"), f"{case_id} seed_abstract"),
            "full": normalized_by_method["full"],
            "flat": normalized_by_method["flat"],
        }

    rng = random.Random(RANDOM_SEED)
    seed_ids = sorted(by_seed)
    rng.shuffle(seed_ids)
    orientations: list[str] = []
    block_distributions: list[dict[str, Any]] = []
    for block_start in range(0, EXPECTED_CASES, BLOCK_SIZE):
        block_orientations = ["full_a"] * 10 + ["flat_a"] * 10
        rng.shuffle(block_orientations)
        orientations.extend(block_orientations)
        block_distributions.append({
            "case_ids": f"B2_C{block_start + 1:03d}-B2_C{block_start + BLOCK_SIZE:03d}",
            "full_as_a": block_orientations.count("full_a"),
            "flat_as_a": block_orientations.count("flat_a"),
        })

    pairs: list[dict[str, Any]] = []
    output_source_key: dict[str, dict[str, str]] = {}
    for index, (seed_id, orientation) in enumerate(zip(seed_ids, orientations), 1):
        case_id = f"B2_C{index:03d}"
        record = by_seed[seed_id]
        if orientation == "full_a":
            proposal_a, proposal_b = record["full"], record["flat"]
            source_a, source_b = "full", "flat"
        else:
            proposal_a, proposal_b = record["flat"], record["full"]
            source_a, source_b = "flat", "full"
        pairs.append({
            "case_id": case_id,
            "seed_title": record["seed_title"],
            "seed_abstract": record["seed_abstract"],
            "proposal_a": proposal_a,
            "proposal_b": proposal_b,
        })
        output_source_key[case_id] = {
            "seed_id": seed_id,
            "proposal_a": source_a,
            "proposal_b": source_b,
        }

    full_as_a = sum(value["proposal_a"] == "full" for value in output_source_key.values())
    flat_as_a = sum(value["proposal_a"] == "flat" for value in output_source_key.values())
    full_before = [before_words[f"{seed_id}:full"] for seed_id in by_seed]
    flat_before = [before_words[f"{seed_id}:flat"] for seed_id in by_seed]
    full_after = [after_words[f"{seed_id}:full"] for seed_id in by_seed]
    flat_after = [after_words[f"{seed_id}:flat"] for seed_id in by_seed]

    leakage = _leakage_audit(pairs)
    fingerprint_counts = _format_fingerprint_audit(pairs)
    empty_fields = [
        {"case_id": pair["case_id"], "side": side, "field": field}
        for pair in pairs
        for side in ("proposal_a", "proposal_b")
        for field in REQUIRED_PROPOSAL_FIELDS
        if not isinstance(pair[side].get(field), str) or not pair[side][field].strip()
    ]
    invalid_cases: list[str] = []
    if len(pairs) != EXPECTED_CASES:
        invalid_cases.append(f"case_count={len(pairs)}")
    if {pair["case_id"] for pair in pairs} != expected_case_ids:
        invalid_cases.append("case_id_coverage")
    if len({mapping["seed_id"] for mapping in output_source_key.values()}) != EXPECTED_CASES:
        invalid_cases.append("unique_seed_count")
    if full_as_a != 40 or flat_as_a != 40:
        invalid_cases.append("global_orientation_balance")
    if any(item["full_as_a"] != 10 or item["flat_as_a"] != 10 for item in block_distributions):
        invalid_cases.append("block_orientation_balance")
    if empty_fields:
        invalid_cases.append("empty_fields")
    if leakage["leak_found"]:
        invalid_cases.append("source_leakage")
    if any(set(pair["proposal_a"]) != REQUIRED_PROPOSAL_FIELDS or set(pair["proposal_b"]) != REQUIRED_PROPOSAL_FIELDS for pair in pairs):
        invalid_cases.append("proposal_field_structure")
    if any(count > 0 for count in fingerprint_counts.values()):
        invalid_cases.append("remaining_format_fingerprint")

    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "random_seed": RANDOM_SEED,
        "passed": not invalid_cases,
        "paper_count": EXPECTED_CASES,
        "case_count": len(pairs),
        "case_id_coverage": {
            "expected_count": EXPECTED_CASES,
            "all_present": {pair["case_id"] for pair in pairs} == expected_case_ids,
        },
        "unique_seed_count": len(output_source_key),
        "each_seed_has_full_and_flat": all(
            mapping["proposal_a"] != mapping["proposal_b"]
            for mapping in output_source_key.values()
        ),
        "completion_status_inherited_from_input": input_validation,
        "orientation_balance": {
            "full_as_a": full_as_a,
            "full_as_b": EXPECTED_CASES - full_as_a,
            "flat_as_a": flat_as_a,
            "flat_as_b": EXPECTED_CASES - flat_as_a,
        },
        "block_orientation_balance": block_distributions,
        "empty_required_fields": empty_fields,
        "proposal_field_structure_identical": all(
            set(pair["proposal_a"]) == REQUIRED_PROPOSAL_FIELDS
            and set(pair["proposal_b"]) == REQUIRED_PROPOSAL_FIELDS
            for pair in pairs
        ),
        "format_fingerprint_counts_after": fingerprint_counts,
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "source_label_leakage": leakage,
        "invalid_cases": invalid_cases,
        "scientific_content_modified": False,
        "normalization_scope": [
            "remove explicit template/schema labels",
            "replace explicit snake_case schema tokens with neutral wording",
            "deterministic whitespace cleanup",
            "fixed-seed case ordering and A/B assignment",
        ],
    }
    summary = {
        "total_cases": len(pairs),
        "full_as_a": full_as_a,
        "flat_as_a": flat_as_a,
        "block_orientation_balance": block_distributions,
        "missing_cases": sorted(expected_case_ids - {pair["case_id"] for pair in pairs}),
        "invalid_cases": invalid_cases,
        "full_word_count_before": _stats(full_before),
        "full_word_count_after": _stats(full_after),
        "flat_word_count_before": _stats(flat_before),
        "flat_word_count_after": _stats(flat_after),
        "average_word_count_delta": {
            "full": sum(full_after) / len(full_after) - sum(full_before) / len(full_before),
            "flat": sum(flat_after) / len(flat_after) - sum(flat_before) / len(flat_before),
        },
        "word_count_definition": "whitespace-separated tokens across the three proposal fields",
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "leakage_flags": leakage,
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "blind_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(args.output_dir / "source_key.json", output_source_key)
    _write_json(args.output_dir / "validation_report.json", validation)
    _write_json(args.output_dir / "dataset_summary.json", summary)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(pairs),
        "unique_seed_count": len(output_source_key),
        "full_as_a": full_as_a,
        "flat_as_a": flat_as_a,
        "block_balance": block_distributions,
        "validation_passed": validation["passed"],
        "remaining_format_fingerprint_counts": fingerprint_counts,
    }, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
