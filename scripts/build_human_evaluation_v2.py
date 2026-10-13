#!/usr/bin/env python3
"""Create the second, compact and sessionized human-evaluation dataset.

The script reads the existing human-evaluation assignment and private keys so
that the experiment design is unchanged.  H1 content is rebuilt from the
original result fields with the compact field selection requested for v2; H2
content is taken from the existing v1 evaluation view and only cleaned for
format markers.  No LLM or external service is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_human_evaluation_dataset import (  # noqa: E402
    SEED_IDS,
    EXPERTS,
    Artifact,
    DatasetError,
    SeedContext,
    concat_existing,
    discover_direct,
    discover_flat,
    discover_markdown,
    discover_researchagent,
    discover_rise,
    json_dump,
    paper_id,
    rise_components,
    rise_experiments,
    stable_text,
)


ARTIFACT_PATTERNS = {
    "core_solution": re.compile(r"(?i)(?<![A-Za-z0-9_])\(?\s*core[ _-]+solution\s*\)?\s*(?::|-)?\s*"),
    "seed_method": re.compile(r"(?i)(?<![A-Za-z0-9_])seed[ _-]+method\s*(?::|-)?\s*"),
    "related_number": re.compile(r"(?i)(?<![A-Za-z0-9_])related\s*#\s*\d+\s*(?::|-)?\s*"),
}
EXPLICIT_SOURCE_PATTERNS = {
    # Keep this case-sensitive: lowercase ``rise`` can be an ordinary verb
    # inside a scientific proposal and is not a system-identity marker.
    "RISE": re.compile(r"\bRISE(?:-Full|-Flat)?\b"),
    "ResearchAgent": re.compile(r"(?i)\bResearchAgent\b"),
    "Direct-Full": re.compile(r"(?i)\bDirect-Full\b"),
    "execution_trace": re.compile(r"(?i)\bexecution trace\b"),
    "run_metadata": re.compile(r"(?i)\brun metadata\b"),
    "output_path": re.compile(r"(?i)/home/tmh/|(?:source|trajectory) path"),
}
COMMON_FIELDS = ("research_problem", "proposed_method", "experimental_plan")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            raise DatasetError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"JSONL record is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def clean_layout(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    output: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and output:
                output.append("")
            previous_blank = True
        else:
            output.append(line)
            previous_blank = False
    return "\n".join(output).strip()


def clean_artifacts(text: Any) -> str:
    result = stable_text(text)
    for pattern in ARTIFACT_PATTERNS.values():
        result = pattern.sub("", result)
    result = clean_layout(result)
    if not result:
        raise DatasetError("artifact cleanup emptied a substantive proposal field")
    return result


def cleaned_proposal(values: dict[str, Any]) -> dict[str, str]:
    result = {field: clean_artifacts(values.get(field, "")) for field in COMMON_FIELDS}
    if any(not result[field] for field in COMMON_FIELDS):
        raise DatasetError("empty field after proposal normalization")
    return result


def compact_from_artifact(artifact: Artifact, method: str) -> dict[str, str]:
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
        payload = payload["payload"]
    if method == "direct":
        values = {
            "research_problem": payload.get("research_problem"),
            "proposed_method": payload.get("proposed_method"),
            "experimental_plan": payload.get("experimental_plan"),
        }
    elif method == "rise":
        values = {
            "research_problem": payload.get("research_gap"),
            "proposed_method": concat_existing(
                payload.get("proposed_solution"),
                rise_components(payload.get("method_components")),
            ),
            "experimental_plan": rise_experiments(payload.get("experimental_plan")),
        }
    elif method == "researchagent":
        values = {
            "research_problem": payload.get("problem"),
            "proposed_method": payload.get("method"),
            "experimental_plan": payload.get("experiment"),
        }
    else:
        raise DatasetError(f"unknown method: {method}")
    return cleaned_proposal(values)


def validate_case(case: dict[str, Any], positions: tuple[str, ...]) -> None:
    if not re.fullmatch(r"P\d{3}", str(case.get("case_id", ""))):
        raise DatasetError(f"invalid case id: {case.get('case_id')}")
    for position in positions:
        proposal = case.get(f"proposal_{position.lower()}")
        if not isinstance(proposal, dict) or set(proposal) != set(COMMON_FIELDS):
            raise DatasetError(f"proposal fields are not common in {case['case_id']}")
        for field in COMMON_FIELDS:
            if not isinstance(proposal[field], str) or not proposal[field].strip():
                raise DatasetError(f"empty proposal field {case['case_id']}/{position}/{field}")


def source_texts_from_v1(
    v1_root: Path,
    source_key: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for expert in EXPERTS:
        rows = read_jsonl(v1_root / expert / "H1_end_to_end_cases.jsonl")
        for row in rows:
            pid = row["case_id"]
            mapping = source_key[expert].get(pid)
            if not mapping:
                raise DatasetError(f"missing H1 source mapping for {expert}/{pid}")
            for position in ("A", "B", "C"):
                method = mapping[position]
                key = (pid, method)
                value = row[f"proposal_{position.lower()}"]
                if key in result and result[key] != value:
                    raise DatasetError(f"v1 proposal differs across assigned experts for {key}")
                result[key] = value
    return result


def transform_h1(
    v1_root: Path,
    h1_key: dict[str, dict[str, dict[str, str]]],
    contexts: dict[str, SeedContext],
    artifacts: dict[str, dict[str, Artifact]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for expert in EXPERTS:
        rows = read_jsonl(v1_root / expert / "H1_end_to_end_cases.jsonl")
        expert_rows: list[dict[str, Any]] = []
        for row in rows:
            pid = row["case_id"]
            mapping = h1_key[expert].get(pid)
            if not mapping or set(mapping) != {"A", "B", "C"}:
                raise DatasetError(f"invalid H1 mapping for {expert}/{pid}")
            if pid not in contexts:
                raise DatasetError(f"missing seed context for {pid}")
            case = {
                "case_id": pid,
                "seed_title": contexts[pid].title,
                "seed_abstract": contexts[pid].abstract,
            }
            for position in ("A", "B", "C"):
                method = mapping[position]
                case[f"proposal_{position.lower()}"] = compact_from_artifact(artifacts[method][pid], method)
            validate_case(case, ("A", "B", "C"))
            expert_rows.append(case)
        if len(expert_rows) != 48:
            raise DatasetError(f"H1 {expert} has {len(expert_rows)} cases, expected 48")
        output[expert] = expert_rows
    return output


def transform_h2(v1_root: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for expert in EXPERTS:
        rows = read_jsonl(v1_root / expert / "H2_full_flat_cases.jsonl")
        expert_rows: list[dict[str, Any]] = []
        for row in rows:
            case = {
                "case_id": row["case_id"],
                "seed_title": row["seed_title"],
                "seed_abstract": row["seed_abstract"],
                "proposal_a": cleaned_proposal(row["proposal_a"]),
                "proposal_b": cleaned_proposal(row["proposal_b"]),
            }
            validate_case(case, ("A", "B"))
            expert_rows.append(case)
        if len(expert_rows) != 48:
            raise DatasetError(f"H2 {expert} has {len(expert_rows)} cases, expected 48")
        output[expert] = expert_rows
    return output


def flatten_proposal_texts(cases_by_expert: dict[str, list[dict[str, Any]]]) -> list[str]:
    values: list[str] = []
    for rows in cases_by_expert.values():
        for case in rows:
            for key, value in case.items():
                if key.startswith("proposal_"):
                    values.extend(value[field] for field in COMMON_FIELDS)
    return values


def marker_counts(texts: list[str]) -> dict[str, int]:
    return {
        "core_solution": len(ARTIFACT_PATTERNS["core_solution"].findall("\n".join(texts))),
        "Seed Method": len(ARTIFACT_PATTERNS["seed_method"].findall("\n".join(texts))),
        "Related #<number>": len(ARTIFACT_PATTERNS["related_number"].findall("\n".join(texts))),
        **{
            name: len(pattern.findall("\n".join(texts)))
            for name, pattern in EXPLICIT_SOURCE_PATTERNS.items()
        },
    }


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def length_stats(proposals: dict[tuple[str, str], dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in COMMON_FIELDS:
        values = [
            words(" ".join(proposal[field] for field in [proposal]))
            if field == "total"
            else words(proposal[field])
            for proposal in proposals.values()
        ]
        if field == "total":
            values = [words("\n".join(proposal.values())) for proposal in proposals.values()]
        result[field] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": percentile(values, 0.90),
            "max": max(values),
            "min": min(values),
        }
    total_values = [words("\n".join(proposal.values())) for proposal in proposals.values()]
    result["total"] = {
        "mean": statistics.mean(total_values),
        "median": statistics.median(total_values),
        "p90": percentile(total_values, 0.90),
        "max": max(total_values),
        "min": min(total_values),
    }
    return result


def unique_h1_proposals(
    cases_by_expert: dict[str, list[dict[str, Any]]],
    key_by_expert: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for expert, rows in cases_by_expert.items():
        for case in rows:
            mapping = key_by_expert[expert][case["case_id"]] if key_by_expert else None
            for position in ("A", "B", "C"):
                method = mapping[position] if mapping else f"position_{position}"
                key = (case["case_id"], method)
                proposal = case[f"proposal_{position.lower()}"]
                if key not in result:
                    result[key] = proposal
                elif result[key] != proposal:
                    raise DatasetError(f"inconsistent proposal for {key}")
    return result


def raw_h1_proposals_from_v1(
    v1_root: Path,
    h1_key: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    return unique_h1_proposals(
        {
            expert: read_jsonl(v1_root / expert / "H1_end_to_end_cases.jsonl")
            for expert in EXPERTS
        },
        h1_key,
    )


def compact_unique_h1_proposals(
    h1_cases: dict[str, list[dict[str, Any]]],
    h1_key: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    return unique_h1_proposals(h1_cases, h1_key)


def write_session(
    session_dir: Path,
    cases: list[dict[str, Any]],
    positions: tuple[str, ...],
    task: str,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=False)
    jsonl = "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n"
    (session_dir / "cases.jsonl").write_text(jsonl, encoding="utf-8")
    blocks = [
        f"# {task} evaluation session",
        "",
        "Use only the supplied seed context and anonymized proposals. Do not use ties.",
        "",
    ]
    for case in cases:
        blocks += [f"## Case {case['case_id']}", "", "### Seed Paper Title", case["seed_title"], "", "### Seed Paper Abstract", case["seed_abstract"], ""]
        for position in positions:
            proposal = case[f"proposal_{position.lower()}"]
            blocks += [f"### Proposal {position}", "", "#### Research Problem", proposal["research_problem"], "", "#### Proposed Method", proposal["proposed_method"], "", "#### Experimental Plan", proposal["experimental_plan"], ""]
    (session_dir / "cases.md").write_text("\n".join(blocks), encoding="utf-8")
    if task == "H1":
        header = "case_id,ranking,confidence,primary_reason,notes"
        rows = [f"{case['case_id']},,,," for case in cases]
    else:
        header = "case_id,winner,confidence,primary_reason,notes"
        rows = [f"{case['case_id']},,,," for case in cases]
    (session_dir / "ratings.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_sessions(root: Path, h1_cases: dict[str, list[dict[str, Any]]], h2_cases: dict[str, list[dict[str, Any]]]) -> None:
    for expert in EXPERTS:
        for task, cases, positions in (
            ("H1", h1_cases[expert], ("A", "B", "C")),
            ("H2", h2_cases[expert], ("A", "B")),
        ):
            task_root = root / expert / task
            task_root.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                chunk = cases[index * 16 : (index + 1) * 16]
                if len(chunk) != 16:
                    raise DatasetError(f"{expert}/{task}/session_{index + 1:02d} is not 16 cases")
                write_session(task_root / f"session_{index + 1:02d}", chunk, positions, task)


def assignment_validation(
    h1_cases: dict[str, list[dict[str, Any]]],
    h2_cases: dict[str, list[dict[str, Any]]],
    h1_key: dict[str, dict[str, dict[str, str]]],
    h2_key: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    issues: list[str] = []
    h1_counts = {expert: len(h1_cases[expert]) for expert in EXPERTS}
    h2_counts = {expert: len(h2_cases[expert]) for expert in EXPERTS}
    h1_case_counts = Counter(case["case_id"] for rows in h1_cases.values() for case in rows)
    h2_case_counts = Counter(case["case_id"] for rows in h2_cases.values() for case in rows)
    if set(h1_case_counts) != set(SEED_IDS) or set(h2_case_counts) != set(SEED_IDS):
        issues.append("H1/H2 do not cover exactly P001-P080")
    if set(h1_case_counts.values()) != {3}:
        issues.append(f"H1 case counts are not 3: {dict(h1_case_counts)}")
    if set(h2_case_counts.values()) != {3}:
        issues.append(f"H2 case counts are not 3: {dict(h2_case_counts)}")
    if set(h1_counts.values()) != {48}:
        issues.append(f"H1 expert counts: {h1_counts}")
    if set(h2_counts.values()) != {48}:
        issues.append(f"H2 expert counts: {h2_counts}")
    h1_positions = {expert: {method: {position: 0 for position in "ABC"} for method in ("rise", "direct", "researchagent")} for expert in EXPERTS}
    for expert, rows in h1_cases.items():
        for case in rows:
            mapping = h1_key[expert][case["case_id"]]
            for position, method in mapping.items():
                h1_positions[expert][method][position] += 1
    if any(value != 16 for expert in h1_positions.values() for method in expert.values() for value in method.values()):
        issues.append(f"H1 position balance is not 16 each: {h1_positions}")
    h2_full_as_a = {expert: sum(h2_key[expert][case["case_id"]]["A"] == "full" for case in h2_cases[expert]) for expert in EXPERTS}
    if set(h2_full_as_a.values()) != {24}:
        issues.append(f"H2 Full-as-A counts: {h2_full_as_a}")
    return {
        "passed": not issues,
        "issues": issues,
        "h1_unique_seed_cases": len(h1_case_counts),
        "h2_unique_seed_cases": len(h2_case_counts),
        "h1_ratings_per_seed": sorted(set(h1_case_counts.values()))[0] if h1_case_counts else 0,
        "h2_ratings_per_seed": sorted(set(h2_case_counts.values()))[0] if h2_case_counts else 0,
        "h1_cases_per_expert": h1_counts,
        "h2_cases_per_expert": h2_counts,
        "h1_position_counts": h1_positions,
        "h2_full_as_a": h2_full_as_a,
        "session_sizes": {expert: {"H1": [16, 16, 16], "H2": [16, 16, 16]} for expert in EXPERTS},
    }


def length_report(
    pre: dict[str, dict[str, dict[str, float]]],
    post: dict[str, dict[str, dict[str, float]]],
    post_proposals: dict[tuple[str, str], dict[str, str]],
) -> str:
    lines = [
        "# H1 length audit",
        "",
        "Counts are whitespace-delimited tokens over the 80 unique seed-level proposals for each method. The v2 change is field selection and deterministic marker cleanup only; no proposal was truncated.",
        "",
    ]
    for method in ("rise", "direct", "researchagent"):
        lines += [f"## {method}", "", "| Version | Field | Mean | Median | P90 | Max | Min |", "|---|---|---:|---:|---:|---:|---:|"]
        for version, values in (("v1", pre[method]), ("v2", post[method])):
            for field in (*COMMON_FIELDS, "total"):
                stat = values[field]
                lines.append(f"| {version} | {field} | {stat['mean']:.1f} | {stat['median']:.1f} | {stat['p90']:.1f} | {stat['max']:.0f} | {stat['min']:.0f} |")
        lines.append("")
    lines += ["## Length warnings", ""]
    warning_count = 0
    all_medians = {name: post[name]["total"]["median"] for name in ("rise", "direct", "researchagent")}
    shortest = min(all_medians.values())
    for method in ("rise", "direct", "researchagent"):
        if post[method]["total"]["median"] > shortest * 2:
            lines.append(f"- {method}: v2 median total length is more than twice the shortest method median.")
            warning_count += 1
    for method in ("rise", "direct", "researchagent"):
        if post[method]["total"]["max"] > 2500:
            lines.append(f"- {method}: at least one v2 proposal exceeds 2500 words; no automatic truncation was applied.")
            for (case_id, proposal_method), proposal in sorted(post_proposals.items()):
                if proposal_method == method:
                    total_words = words("\n".join(proposal.values()))
                    if total_words > 2500:
                        lines.append(f"  - {case_id}: {total_words} words total")
            warning_count += 1
    if not warning_count:
        lines.append("- None under the specified thresholds.")
    return "\n".join(lines) + "\n"


def source_discovery_report(issues: list[str]) -> str:
    lines = [
        "# Source and alignment validation",
        "",
        "All four supplied result roots were inspected deterministically.",
        "",
        "- Markdown: 80 files, P001-P080; title and abstract available for 80/80.",
        "- Direct: 80 final proposals under `papers/*/final_proposal.json`.",
        "- ResearchAgent: 80 `Pxxx.json` final exports plus manifest/results JSONL. The export has no per-record status field; all required final-stage fields are present and the manifest lists 80 IDs.",
        "- RISE-Full: 80 final proposals under `papers/*/final_research_proposal.json`.",
        "- RISE-Flat: 80 final proposals under `state_evolution_ablation/flat/Pxxx/final_proposal.json`; the E3 assembly manifest also reports 80/80.",
        "- Title alignment against the Markdown parser passed for all available result titles.",
        "",
        "## Unresolved source issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in issues or ["None."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-v1", type=Path, default=Path("/home/tmh/baseline/human_evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/tmh/baseline/human_evaluation_v2"))
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/nlp")
    parser.add_argument("--direct-dir", type=Path, default=Path("/home/tmh/baseline/output_direct"))
    parser.add_argument("--researchagent-dir", type=Path, default=Path("/home/tmh/baseline/output_ResearchAgent"))
    parser.add_argument("--rise-dir", type=Path, default=Path("/home/tmh/baseline/output_rise"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("/home/tmh/baseline/state_evolution_ablation"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise DatasetError(f"refusing to overwrite existing output directory: {args.output_dir}")
    if not args.input_v1.exists():
        raise DatasetError(f"v1 input does not exist: {args.input_v1}")
    contexts, issues = discover_markdown(args.input_dir)
    direct, direct_issues = discover_direct(args.direct_dir)
    rise, rise_issues = discover_rise(args.rise_dir)
    researchagent, ra_issues = discover_researchagent(args.researchagent_dir)
    flat, flat_issues = discover_flat(args.ablation_dir)
    issues.extend(direct_issues + rise_issues + ra_issues + flat_issues)
    if set(contexts) != set(SEED_IDS) or any(not context.title or not context.abstract for context in contexts.values()):
        issues.append("seed title/abstract coverage is not exactly 80 complete cases")
    for label, artifacts in (("Direct", direct), ("RISE", rise), ("ResearchAgent", researchagent), ("Flat", flat)):
        if set(artifacts) != set(SEED_IDS):
            issues.append(f"{label} final proposal coverage is not exactly P001-P080")
    if issues:
        raise DatasetError("source validation failed; no v2 output was created: " + "; ".join(issues))

    h1_key = json.loads((args.input_v1 / "private/h1_source_key.json").read_text(encoding="utf-8"))
    h2_key = json.loads((args.input_v1 / "private/h2_source_key.json").read_text(encoding="utf-8"))
    for expert in EXPERTS:
        if set(h1_key.get(expert, {})) != set(read_jsonl(args.input_v1 / expert / "H1_end_to_end_cases.jsonl")[i]["case_id"] for i in range(48)):
            raise DatasetError(f"H1 source key does not match existing order/content for {expert}")
        if len(h2_key.get(expert, {})) != 48:
            raise DatasetError(f"H2 source key has incorrect case count for {expert}")
    h1_cases = transform_h1(
        args.input_v1,
        h1_key,
        contexts,
        {"direct": direct, "rise": rise, "researchagent": researchagent},
    )
    h2_cases = transform_h2(args.input_v1)
    assignment = assignment_validation(h1_cases, h2_cases, h1_key, h2_key)
    if not assignment["passed"]:
        raise DatasetError("assignment validation failed: " + "; ".join(assignment["issues"]))

    pre_h1 = raw_h1_proposals_from_v1(args.input_v1, h1_key)
    post_h1 = compact_unique_h1_proposals(h1_cases, h1_key)
    pre_lengths = {
        "rise": length_stats({key: value for key, value in pre_h1.items() if key[1] == "rise"}),
        "direct": length_stats({key: value for key, value in pre_h1.items() if key[1] == "direct"}),
        "researchagent": length_stats({key: value for key, value in pre_h1.items() if key[1] == "researchagent"}),
    }
    post_lengths = {
        "rise": length_stats({key: value for key, value in post_h1.items() if key[1] == "rise"}),
        "direct": length_stats({key: value for key, value in post_h1.items() if key[1] == "direct"}),
        "researchagent": length_stats({key: value for key, value in post_h1.items() if key[1] == "researchagent"}),
    }
    before_cases: dict[str, list[dict[str, Any]]] = {}
    for expert in EXPERTS:
        before_cases[expert] = read_jsonl(args.input_v1 / expert / "H1_end_to_end_cases.jsonl") + read_jsonl(args.input_v1 / expert / "H2_full_flat_cases.jsonl")
    after_cases = {
        expert: h1_cases[expert] + h2_cases[expert]
        for expert in EXPERTS
    }
    before_counts = marker_counts(flatten_proposal_texts(before_cases))
    after_counts = marker_counts(flatten_proposal_texts(after_cases))

    temp_dir = Path(tempfile.mkdtemp(prefix="human_evaluation_v2.", dir=str(args.output_dir.parent)))
    try:
        (temp_dir / "private").mkdir(parents=True)
        for expert in EXPERTS:
            (temp_dir / expert).mkdir()
            (temp_dir / expert / "INSTRUCTIONS.md").write_text("""# Anonymous evaluation instructions\n\nThis is a strictly anonymous blind evaluation. Do not guess the source of any proposal, and do not use length, formatting, wording, or language style as a proxy for identity. Do not search external literature or use another case to influence the current case.\n\nJudge only the supplied Seed Paper Title, Seed Paper Abstract, and anonymized proposal content.\n\nFor H1, rank the three proposals from strongest to weakest with no ties. For H2, choose exactly one of the two proposals with no tie. Evaluate overall scientific quality as a follow-up research direction: research problem quality, technical mechanism, soundness/feasibility, experimental testability, scientific significance, and perceived differentiation from the seed paper. Perceived differentiation is relative to the supplied seed context and is not exhaustive novelty verification.\n\nConfidence: 1 = low, 2 = medium, 3 = high. Primary reason must use one of: problem, mechanism, feasibility, experiment, significance, differentiation, overall.\n""", encoding="utf-8")
        write_sessions(temp_dir, h1_cases, h2_cases)

        json_dump(temp_dir / "private/h1_source_key.json", h1_key)
        json_dump(temp_dir / "private/h2_source_key.json", h2_key)
        assignment_manifest = json.loads((args.input_v1 / "private/assignment_manifest.json").read_text(encoding="utf-8"))
        assignment_manifest.update({
            "dataset_version": "human_evaluation_v2",
            "session_size": 16,
            "h1_case_order_preserved_from_v1": True,
            "h2_case_order_preserved_from_v1": True,
        })
        json_dump(temp_dir / "private/assignment_manifest.json", assignment_manifest)
        (temp_dir / "private/normalization_mapping.md").write_text("""# v2 normalization mapping\n\nAll proposal content is copied from existing result fields. No LLM, paraphrase, summary, translation, polishing, truncation, or scientific correction is used. H1 is rebuilt with core fields only; H2 retains the v1 field selection and receives only deterministic marker/whitespace cleanup.\n\n## H1\n\n- Direct: `research_problem`; `proposed_method`; `experimental_plan`. The standalone `technical_rationale` is not included.\n- RISE-Full: `research_gap`; `proposed_solution` followed by `method_components[*].description` in source order; `experimental_plan`. The standalone `solution_rationale` is not included. Component IDs and schema labels are not displayed.\n- ResearchAgent: `problem`; `method`; `experiment`. Standalone rationale fields are not included.\n\n## H2\n\nThe existing v1 H2 proposal field selection and A/B assignment are retained. Only the deterministic artifact cleanup documented in `blindness_audit.md` and whitespace/layout cleanup are applied.\n\n## Shared formatting\n\nThe three common outer fields are `research_problem`, `proposed_method`, and `experimental_plan`. Existing text is only stripped, line-ending normalized, repeated whitespace normalized, and exact internal marker labels removed.\n""", encoding="utf-8")
        blindness_lines = [
            "# Blindness audit",
            "",
            "Counts are over proposal text in the v1 and v2 expert-facing JSONL views. Labels are removed only when they match the deterministic marker rules; no source identity is inserted or inferred.",
            "",
            "| Marker | Before | After |",
            "|---|---:|---:|",
        ]
        for marker in before_counts:
            blindness_lines.append(f"| {marker} | {before_counts[marker]} | {after_counts[marker]} |")
        blindness_lines += ["", "Explicit method-identity markers remaining in v2 proposal text:"]
        remaining_identity = {key: after_counts[key] for key in ("RISE", "ResearchAgent", "Direct-Full") if after_counts[key]}
        blindness_lines.extend(f"- {key}: {value}" for key, value in remaining_identity.items())
        if not remaining_identity:
            blindness_lines.append("- None.")
        contextual = {key: after_counts[key] for key in ("execution_trace", "run_metadata", "output_path") if after_counts[key]}
        blindness_lines += ["", "Contextual phrases retained because they occur as substantive scientific proposal text (not runtime metadata):"]
        blindness_lines.extend(f"- {key}: {value}" for key, value in contextual.items())
        if not contextual:
            blindness_lines.append("- None.")
        blindness_lines += ["", "The v2 expert directories contain no source key files, result paths, trajectory metadata, or method mapping fields."]
        (temp_dir / "private/blindness_audit.md").write_text("\n".join(blindness_lines) + "\n", encoding="utf-8")
        (temp_dir / "private/length_audit.md").write_text(length_report(pre_lengths, post_lengths, post_h1), encoding="utf-8")

        validation = assignment_validation(h1_cases, h2_cases, h1_key, h2_key)
        validation.update({
            "compact_h1_fields": list(COMMON_FIELDS),
            "h2_content_selection": "retained from v1; marker and whitespace cleanup only",
            "source_title_abstract_available": "80/80",
            "content_transform": "deterministic field selection, concatenation, exact marker removal, and layout cleanup; no LLM",
            "before_artifact_counts": before_counts,
            "after_artifact_counts": after_counts,
            "h1_length_audit": post_lengths,
            "source_validation_issues": [],
            "source_key_in_expert_dirs": False,
            "method_identity_in_instructions": False,
            "scientific_content_rewritten": False,
        })
        json_dump(temp_dir / "validation_report.json", validation)
        validation_text = [
            "# Human evaluation v2 validation",
            "",
            f"Overall: **{'PASS' if validation['passed'] else 'FAIL'}**",
            "",
            "## Coverage and allocation",
            "",
            "- H1 unique seeds: 80; three ratings per seed; 48 cases per expert.",
            "- H2 unique seeds: 80; three ratings per seed; 48 cases per expert.",
            "- H1 A/B/C position balance: each method occurs 16 times in each position for every expert.",
            "- H2 Full-as-A balance: 24/24 per expert.",
            "- Sessions: every expert has H1/session_01..03 and H2/session_01..03, each with 16 cases.",
            "",
            "## Content and blindness",
            "",
            "- H1 compact view uses core fields only and excludes standalone rationale fields.",
            "- H2 retains the v1 substantive field selection; only deterministic artifact/whitespace cleanup is applied.",
            "- No LLM rewriting, paraphrasing, summarization, or truncation was performed.",
            "- Source keys are private and absent from expert directories.",
            "",
            "## Issues",
            "",
        ]
        validation_text.extend(f"- {issue}" for issue in validation["issues"] or ["None."])
        (temp_dir / "validation_report.md").write_text("\n".join(validation_text) + "\n", encoding="utf-8")
        (temp_dir / "README.md").write_text("""# Human evaluation v2\n\nThis directory is a deterministic second preparation of the anonymous evaluation tasks. H1 uses compact core-content field selection; H2 retains the previous substantive selection and only receives marker/whitespace cleanup. Each task has 80 stable P001-P080 seed cases, three ratings per case, and three 16-case sessions per expert.\n\nGive experts only their own `expert_xx` directory. Keep `private/` separate; it contains source keys, assignment records, audits, and validation.\n""", encoding="utf-8")
        # Source keys must not accidentally be copied into expert-facing dirs.
        if any(path.name in {"h1_source_key.json", "h2_source_key.json", "source_key.json"} for path in temp_dir.glob("expert_*/**/*")):
            raise DatasetError("source key leaked into expert directory")
        if any(token in (temp_dir / expert / "INSTRUCTIONS.md").read_text(encoding="utf-8") for expert in EXPERTS for token in ("RISE", "Direct", "ResearchAgent", "Full", "Flat")):
            raise DatasetError("method identity leaked into instructions")
        temp_dir.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"created: {args.output_dir}")
    print("H1: 80 seeds, 48 cases per expert, compact core fields")
    print("H2: 80 seeds, 48 cases per expert, v1 content selection retained")
    print("sessions: H1/H2 session_01..03, 16 cases each")
    print("validation: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
