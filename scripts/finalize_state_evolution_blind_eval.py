#!/usr/bin/env python3
"""Finalize the anonymous Full-vs-Flat blind dataset.

The source text is deliberately rebuilt from the pre-normalization blind
dataset.  The current normalized source key supplies the already-established
case identity and A/B orientation; it is copied, never reshuffled.

No LLM, API, semantic rewriting, or proposal regeneration is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval"
DEFAULT_CURRENT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval_normalized"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_blind_eval_final"
EXPECTED_CASES = 80
BLOCK_SIZE = 20
REQUIRED_PROPOSAL_FIELDS = {
    "research_problem",
    "proposed_method",
    "experimental_plan",
}


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
    r"(?:C\d+|Component\s+\d+)\s*[:：]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_INLINE_COMPONENT_LABEL_RE = re.compile(r"\bC\d+\s*[:：]\s*", re.IGNORECASE)
_EXPERIMENT_LABEL_RE = re.compile(
    r"^(?P<prefix>\s*(?:[-*]\s*)?)"
    r"Experiment\s+\d+\s*(?::|：)?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# These are schema artifacts in the pre-normalization Full representation.
# They are removed or shortened, never expanded into a repeated source-like
# phrase.  ``core_solution`` becomes a neutral single word to preserve local
# sentence grammar without manufacturing "proposed method" occurrences.
_SCHEMA_TOKEN_REPLACEMENTS = {
    "core_solution": "approach",
    "proposed_solution": "proposal",
    "solution_rationale": "rationale",
    "method_components": "components",
    "experimental_plan": "experimental plan",
    "research_gap": "research problem",
    "what_it_validates": "what it validates",
    "validation_criterion": "validation criterion",
}
_SCHEMA_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _SCHEMA_TOKEN_REPLACEMENTS)) + r")\b",
    re.IGNORECASE,
)
_SEED_METHOD_RE = re.compile(r"\bSeed Method\b", re.IGNORECASE)

_FORBIDDEN_KEYS = {
    "source_method",
    "source_run",
    "source_label",
    "filename",
    "source_path",
    "output_path",
    "trajectory_path",
    "metadata",
}
_STRONG_SOURCE_MARKER_RE = re.compile(
    r"\b(?:RISE\s*[-_]\s*(?:Full|Flat)|Full\s+Runner|Flat\s+Runner)\b",
    re.IGNORECASE,
)
_PATH_MARKERS = ("/home/tmh/", "outputs/", "trajectory_path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize anonymous state-evolution blind evaluation data"
    )
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY_DIR)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
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


def _load_pairs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing pair file: {path}")
    pairs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            case_id = _nonempty_string(pair.get("case_id"), f"{path}:{line_number} case_id")
            if case_id in pairs:
                raise RuntimeError(f"duplicate case_id: {case_id}")
            pairs[case_id] = pair
    if len(pairs) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} pairs in {path}, found {len(pairs)}")
    return pairs


def _load_source_key(path: Path) -> dict[str, dict[str, str]]:
    value = _read_json(path)
    if not isinstance(value, dict) or len(value) != EXPECTED_CASES:
        raise RuntimeError(f"{path} must contain exactly {EXPECTED_CASES} mappings")
    result: dict[str, dict[str, str]] = {}
    for case_id, mapping in value.items():
        if not isinstance(mapping, dict):
            raise RuntimeError(f"{case_id} source mapping is not an object")
        seed_id = _nonempty_string(mapping.get("seed_id"), f"{case_id}.seed_id")
        proposal_a = _nonempty_string(mapping.get("proposal_a"), f"{case_id}.proposal_a")
        proposal_b = _nonempty_string(mapping.get("proposal_b"), f"{case_id}.proposal_b")
        if {proposal_a, proposal_b} != {"full", "flat"}:
            raise RuntimeError(f"{case_id} does not map one full and one flat proposal")
        result[case_id] = {
            "seed_id": seed_id,
            "proposal_a": proposal_a,
            "proposal_b": proposal_b,
        }
    if len({mapping["seed_id"] for mapping in result.values()}) != EXPECTED_CASES:
        raise RuntimeError("source key does not cover 80 unique seeds")
    return result


def _normalize_original_proposal(proposal: Any) -> tuple[dict[str, str], Counter[str]]:
    if not isinstance(proposal, dict) or set(proposal) != REQUIRED_PROPOSAL_FIELDS:
        raise RuntimeError("proposal does not have the required three fields")
    stats: Counter[str] = Counter()
    normalized: dict[str, str] = {}
    for field in REQUIRED_PROPOSAL_FIELDS:
        text = _nonempty_string(proposal[field], f"proposal.{field}")
        retained_lines: list[str] = []
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

            # A generated line can contain nested labels such as
            # ``C1: Component 1: description``.
            component_match = _COMPONENT_LABEL_RE.match(line)
            while component_match:
                stats["component_number_labels_removed"] += 1
                line = component_match.group("prefix") + component_match.group("rest").strip()
                component_match = _COMPONENT_LABEL_RE.match(line)

            line, inline_count = _INLINE_COMPONENT_LABEL_RE.subn("", line)
            if inline_count:
                stats["inline_component_number_labels_removed"] += inline_count

            line = _SCHEMA_TOKEN_RE.sub(
                lambda match: _SCHEMA_TOKEN_REPLACEMENTS[match.group(0).lower()],
                line,
            )
            line, seed_method_count = _SEED_METHOD_RE.subn("baseline", line)
            if seed_method_count:
                stats["seed_method_labels_neutralized"] += seed_method_count
            retained_lines.append(line)

        # All paragraph and line-break structure is intentionally normalized
        # only after labels are removed. Whitespace collapse cannot reorder or
        # alter any non-whitespace character.
        flattened = re.sub(r"\s+", " ", "\n".join(retained_lines)).strip()
        normalized[field] = _nonempty_string(flattened, f"normalized proposal.{field}")
    return normalized, stats


def _word_count(proposal: dict[str, str]) -> int:
    return sum(len(re.findall(r"\S+", proposal[field])) for field in REQUIRED_PROPOSAL_FIELDS)


def _line_count(proposal: dict[str, str]) -> int:
    return sum(proposal[field].count("\n") + 1 for field in REQUIRED_PROPOSAL_FIELDS)


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
    source_identity_hits: list[dict[str, str]] = []
    path_hits: list[dict[str, str]] = []
    generic_terms: Counter[str] = Counter()
    generic_re = re.compile(r"\b(?:full|flat)\b", re.IGNORECASE)
    for pair in pairs:
        for key in pair:
            if key in _FORBIDDEN_KEYS:
                forbidden_keys.append(key)
        for path, value in _walk_strings(pair):
            if path.endswith(".case_id"):
                continue
            for match in _STRONG_SOURCE_MARKER_RE.finditer(value):
                source_identity_hits.append({
                    "case_id": pair.get("case_id", ""),
                    "path": path,
                    "text": match.group(0),
                })
            lowered = value.lower()
            for token in _PATH_MARKERS:
                if token in lowered:
                    path_hits.append({
                        "case_id": pair.get("case_id", ""),
                        "path": path,
                        "token": token,
                    })
            for match in generic_re.finditer(value):
                generic_terms[match.group(0).lower()] += 1
    return {
        "leak_found": bool(forbidden_keys or source_identity_hits or path_hits),
        "forbidden_keys": sorted(set(forbidden_keys)),
        "source_identity_hits": source_identity_hits,
        "path_hits": path_hits,
        "generic_full_flat_terms_for_review": dict(sorted(generic_terms.items())),
    }


def _fingerprint_audit(pairs: list[dict[str, Any]]) -> dict[str, int]:
    patterns = {
        "Proposed Solution:": re.compile(r"^\s*Proposed Solution\s*:", re.I | re.M),
        "Technical Rationale:": re.compile(r"^\s*Technical Rationale\s*:", re.I | re.M),
        "Method Components:": re.compile(r"^\s*Method Components\s*:", re.I | re.M),
        "Objective:": re.compile(r"^\s*Objective\s*:", re.I | re.M),
        "Setup:": re.compile(r"^\s*Setup\s*:", re.I | re.M),
        "Baselines:": re.compile(r"^\s*Baselines\s*:", re.I | re.M),
        "Metrics:": re.compile(r"^\s*Metrics\s*:", re.I | re.M),
        "What It Validates:": re.compile(r"^\s*What It Validates\s*:", re.I | re.M),
        "Validation Criterion:": re.compile(r"^\s*Validation Criterion\s*:", re.I | re.M),
        "core_solution": re.compile(r"\bcore_solution\b", re.I),
        "component_number_label": re.compile(r"^\s*(?:[-*]\s*)?(?:C\d+|Component\s+\d+)\s*:", re.I | re.M),
        "experiment_number_label": re.compile(r"^\s*Experiment\s+\d+\s*:?", re.I | re.M),
        "Seed Method": re.compile(r"\bSeed Method\b", re.I),
    }
    counts = {name: 0 for name in patterns}
    for pair in pairs:
        for side in ("proposal_a", "proposal_b"):
            for field in REQUIRED_PROPOSAL_FIELDS:
                text = pair[side][field]
                for name, pattern in patterns.items():
                    counts[name] += len(pattern.findall(text))
    return counts


def _stats(values: list[int]) -> dict[str, float | int]:
    return {"average": sum(values) / len(values), "min": min(values), "max": max(values)}


def _phrase_counts(pairs_by_case: dict[str, dict[str, Any]], source_key: dict[str, dict[str, str]], phrase: str) -> dict[str, int]:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    counts = {"full": 0, "flat": 0}
    for case_id, mapping in source_key.items():
        pair = pairs_by_case[case_id]
        for method, side in ((mapping["proposal_a"], "proposal_a"), (mapping["proposal_b"], "proposal_b")):
            counts[method] += sum(
                len(pattern.findall(_nonempty_string(pair[side][field], f"{case_id}.{side}.{field}")))
                for field in REQUIRED_PROPOSAL_FIELDS
            )
    return counts


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")

    legacy_pairs = _load_pairs(args.legacy_dir / "blind_pairs.jsonl")
    legacy_key = _load_source_key(args.legacy_dir / "source_key.json")
    current_pairs = _load_pairs(args.current_dir / "blind_pairs.jsonl")
    current_key = _load_source_key(args.current_dir / "source_key.json")
    expected_case_ids = {f"B2_C{index:03d}" for index in range(1, EXPECTED_CASES + 1)}
    if set(current_pairs) != expected_case_ids or set(current_key) != expected_case_ids:
        raise RuntimeError("current dataset case IDs do not exactly cover B2_C001-B2_C080")

    legacy_by_seed: dict[str, dict[str, Any]] = {}
    for case_id, mapping in legacy_key.items():
        pair = legacy_pairs[case_id]
        seed_id = mapping["seed_id"]
        if seed_id in legacy_by_seed:
            raise RuntimeError(f"legacy source key has duplicate seed: {seed_id}")
        legacy_by_seed[seed_id] = {
            "title": _nonempty_string(pair.get("seed_title"), f"legacy {case_id}.seed_title"),
            "abstract": _nonempty_string(pair.get("seed_abstract"), f"legacy {case_id}.seed_abstract"),
            mapping["proposal_a"]: pair["proposal_a"],
            mapping["proposal_b"]: pair["proposal_b"],
        }

    current_seed_ids = {mapping["seed_id"] for mapping in current_key.values()}
    if current_seed_ids != set(legacy_by_seed):
        raise RuntimeError("current and legacy source keys cover different seeds")

    normalized_by_seed: dict[str, dict[str, Any]] = {}
    normalization_counts: Counter[str] = Counter()
    for seed_id, record in legacy_by_seed.items():
        normalized: dict[str, dict[str, str]] = {}
        for method in ("full", "flat"):
            normalized[method], stats = _normalize_original_proposal(record[method])
            normalization_counts.update({f"{method}.{key}": value for key, value in stats.items()})
        normalized_by_seed[seed_id] = {
            "title": record["title"],
            "abstract": record["abstract"],
            "full": normalized["full"],
            "flat": normalized["flat"],
        }

    final_pairs: list[dict[str, Any]] = []
    for case_id in sorted(current_key):
        mapping = current_key[case_id]
        current_pair = current_pairs[case_id]
        record = normalized_by_seed[mapping["seed_id"]]
        final_pairs.append({
            "case_id": case_id,
            "seed_title": _nonempty_string(current_pair.get("seed_title"), f"current {case_id}.seed_title"),
            "seed_abstract": _nonempty_string(current_pair.get("seed_abstract"), f"current {case_id}.seed_abstract"),
            "proposal_a": record[mapping["proposal_a"]],
            "proposal_b": record[mapping["proposal_b"]],
        })

    # The A/B source key is copied byte-for-byte as a mapping, so this step
    # cannot change the prior assignment or case identity.
    output_key = current_key
    leakage = _leakage_audit(final_pairs)
    fingerprints = _fingerprint_audit(final_pairs)
    empty_fields = [
        {"case_id": pair["case_id"], "side": side, "field": field}
        for pair in final_pairs
        for side in ("proposal_a", "proposal_b")
        for field in REQUIRED_PROPOSAL_FIELDS
        if not isinstance(pair[side].get(field), str) or not pair[side][field].strip()
    ]
    invalid_cases: list[str] = []
    if len(final_pairs) != EXPECTED_CASES:
        invalid_cases.append("case_count")
    if {pair["case_id"] for pair in final_pairs} != expected_case_ids:
        invalid_cases.append("case_id_coverage")
    if len({mapping["seed_id"] for mapping in output_key.values()}) != EXPECTED_CASES:
        invalid_cases.append("unique_seed_count")
    if empty_fields:
        invalid_cases.append("empty_fields")
    if leakage["leak_found"]:
        invalid_cases.append("source_leakage")
    if any(fingerprints.values()):
        invalid_cases.append("format_fingerprint")
    if any(
        set(pair[side]) != REQUIRED_PROPOSAL_FIELDS
        for pair in final_pairs
        for side in ("proposal_a", "proposal_b")
    ):
        invalid_cases.append("proposal_field_structure")
    if any("\n" in pair[side][field] for pair in final_pairs for side in ("proposal_a", "proposal_b") for field in REQUIRED_PROPOSAL_FIELDS):
        invalid_cases.append("internal_line_breaks")

    full_final = [
        final_pairs[index][side][field]
        for index, case_id in enumerate(sorted(current_key))
        for side in ("proposal_a", "proposal_b")
        for field in REQUIRED_PROPOSAL_FIELDS
        if current_key[case_id][side] == "full"
    ]
    flat_final = [
        final_pairs[index][side][field]
        for index, case_id in enumerate(sorted(current_key))
        for side in ("proposal_a", "proposal_b")
        for field in REQUIRED_PROPOSAL_FIELDS
        if current_key[case_id][side] == "flat"
    ]
    # The comprehensions above contain field strings; construct proposal-level
    # metrics directly to keep the definition explicit.
    full_word_counts: list[int] = []
    flat_word_counts: list[int] = []
    full_line_counts: list[int] = []
    flat_line_counts: list[int] = []
    for index, case_id in enumerate(sorted(current_key)):
        pair = final_pairs[index]
        mapping = current_key[case_id]
        for side, method in (("proposal_a", mapping["proposal_a"]), ("proposal_b", mapping["proposal_b"])):
            proposal = pair[side]
            if method == "full":
                full_word_counts.append(_word_count(proposal))
                full_line_counts.append(_line_count(proposal))
            else:
                flat_word_counts.append(_word_count(proposal))
                flat_line_counts.append(_line_count(proposal))

    current_full_words: list[int] = []
    current_flat_words: list[int] = []
    current_full_lines: list[int] = []
    current_flat_lines: list[int] = []
    for case_id, mapping in current_key.items():
        pair = current_pairs[case_id]
        for side, method in (("proposal_a", mapping["proposal_a"]), ("proposal_b", mapping["proposal_b"])):
            proposal = pair[side]
            if method == "full":
                current_full_words.append(_word_count(proposal))
                current_full_lines.append(_line_count(proposal))
            else:
                current_flat_words.append(_word_count(proposal))
                current_flat_lines.append(_line_count(proposal))

    phrase_names = ["method in the seed paper", "proposed method", "Seed Method"]
    phrase_before = {phrase: _phrase_counts(current_pairs, current_key, phrase) for phrase in phrase_names}
    phrase_after = {phrase: _phrase_counts({pair["case_id"]: pair for pair in final_pairs}, current_key, phrase) for phrase in phrase_names}
    block_balance = []
    for start in range(0, EXPECTED_CASES, BLOCK_SIZE):
        block = [output_key[f"B2_C{index:03d}"] for index in range(start + 1, start + BLOCK_SIZE + 1)]
        full_as_a = sum(mapping["proposal_a"] == "full" for mapping in block)
        block_balance.append({
            "case_ids": f"B2_C{start + 1:03d}-B2_C{start + BLOCK_SIZE:03d}",
            "full_as_a": full_as_a,
            "flat_as_a": BLOCK_SIZE - full_as_a,
        })
    full_as_a = sum(mapping["proposal_a"] == "full" for mapping in output_key.values())

    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legacy_input_dir": str(args.legacy_dir.resolve()),
        "current_input_dir": str(args.current_dir.resolve()),
        "case_count": len(final_pairs),
        "paper_count": EXPECTED_CASES,
        "case_id_coverage": {"all_present": {pair["case_id"] for pair in final_pairs} == expected_case_ids},
        "unique_seed_count": len({mapping["seed_id"] for mapping in output_key.values()}),
        "each_seed_has_full_and_flat": all(mapping["proposal_a"] != mapping["proposal_b"] for mapping in output_key.values()),
        "orientation_balance": {
            "full_as_a": full_as_a,
            "full_as_b": EXPECTED_CASES - full_as_a,
            "flat_as_a": EXPECTED_CASES - full_as_a,
            "flat_as_b": full_as_a,
        },
        "block_orientation_balance": block_balance,
        "empty_required_fields": empty_fields,
        "proposal_field_structure_identical": all(
            set(pair["proposal_a"]) == REQUIRED_PROPOSAL_FIELDS and set(pair["proposal_b"]) == REQUIRED_PROPOSAL_FIELDS
            for pair in final_pairs
        ),
        "internal_line_breaks_remaining": sum(
            "\n" in pair[side][field]
            for pair in final_pairs
            for side in ("proposal_a", "proposal_b")
            for field in REQUIRED_PROPOSAL_FIELDS
        ),
        "fixed_phrase_counts_before_current_input": phrase_before,
        "fixed_phrase_counts_after_final_output": phrase_after,
        "average_internal_line_count": {
            "definition": "sum of line counts across the three proposal fields; each field is expected to be one line",
            "full": sum(full_line_counts) / len(full_line_counts),
            "flat": sum(flat_line_counts) / len(flat_line_counts),
            "full_per_field": sum(full_line_counts) / (len(full_line_counts) * 3),
            "flat_per_field": sum(flat_line_counts) / (len(flat_line_counts) * 3),
        },
        "format_fingerprint_counts_after": fingerprints,
        "source_label_leakage": leakage,
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "invalid_cases": invalid_cases,
        "passed": not invalid_cases,
        "scientific_content_modified": False,
        "a_b_assignment_reused_without_reshuffle": True,
    }
    summary = {
        "total_cases": len(final_pairs),
        "full_as_a": full_as_a,
        "flat_as_a": EXPECTED_CASES - full_as_a,
        "block_orientation_balance": block_balance,
        "fixed_phrase_counts_before_current_input": phrase_before,
        "fixed_phrase_counts_after_final_output": phrase_after,
        "full_average_internal_line_count": sum(full_line_counts) / len(full_line_counts),
        "flat_average_internal_line_count": sum(flat_line_counts) / len(flat_line_counts),
        "full_word_count": {"before": _stats(current_full_words), "after": _stats(full_word_counts)},
        "flat_word_count": {"before": _stats(current_flat_words), "after": _stats(flat_word_counts)},
        "empty_fields": empty_fields,
        "leakage_flags": leakage,
        "normalization_counts": dict(sorted(normalization_counts.items())),
        "word_count_definition": "whitespace-separated tokens across the three proposal fields",
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "blind_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in final_pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(args.output_dir / "source_key.json", output_key)
    _write_json(args.output_dir / "validation_report.json", validation)
    _write_json(args.output_dir / "dataset_summary.json", summary)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(final_pairs),
        "unique_seed_count": validation["unique_seed_count"],
        "full_as_a": full_as_a,
        "flat_as_a": EXPECTED_CASES - full_as_a,
        "block_balance": block_balance,
        "phrase_counts_after": phrase_after,
        "full_average_internal_line_count": validation["average_internal_line_count"]["full"],
        "flat_average_internal_line_count": validation["average_internal_line_count"]["flat"],
        "validation_passed": validation["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
