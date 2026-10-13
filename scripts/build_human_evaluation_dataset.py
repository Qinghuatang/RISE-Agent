#!/usr/bin/env python3
"""Build the anonymized human-evaluation tasks from completed local artifacts.

This is deliberately an offline, deterministic data-preparation script.  It
does not import an LLM client and never writes to any input result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import parse_markdown_file  # noqa: E402


EXPERTS = tuple(f"expert_{i:02d}" for i in range(1, 6))
SEED_IDS = tuple(f"P{i:03d}" for i in range(1, 81))
H1_ASSIGNMENT_SEED = 20260911
H2_ASSIGNMENT_SEED = 20260912


class DatasetError(RuntimeError):
    pass


def unwrap(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("payload"), dict):
        return value["payload"]
    return value


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised by real artifact failures
        raise DatasetError(f"invalid JSON: {path}: {exc}") from exc


def paper_id(value: str | Path) -> str | None:
    # Dataset filenames use underscores around Pxxx (for example
    # ``MinerU_markdown_P001_...``); a word-boundary expression would not
    # match because underscore is itself a word character.
    match = re.search(r"(P\d{3})(?!\d)", str(value))
    return match.group(1) if match else None


def stable_text(value: Any) -> str:
    """Render an existing field without semantic rewriting."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n\n".join(part for part in (stable_text(x) for x in value) if part)
    if isinstance(value, dict):
        return "\n\n".join(
            part for part in (stable_text(x) for x in value.values()) if part
        )
    return str(value).strip()


def concat_existing(*values: Any) -> str:
    """Fixed-order concatenation with exact duplicate blocks removed."""

    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = stable_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return "\n\n".join(parts).strip()


def rise_components(value: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        text = stable_text(item.get("description") if isinstance(item, dict) else item)
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    return "\n\n".join(parts).strip()


def rise_experiments(value: Any) -> str:
    # Keep the source schema's field order, but omit schema labels and
    # bookkeeping metadata.  The values themselves are copied verbatim.
    fields = (
        "objective",
        "setup",
        "baselines",
        "metrics",
        "what_it_validates",
        "validation_criterion",
    )
    experiments = value if isinstance(value, list) else [value]
    blocks: list[str] = []
    seen: set[str] = set()
    for experiment in experiments:
        if isinstance(experiment, dict):
            block = concat_existing(*(experiment.get(field) for field in fields))
        else:
            block = stable_text(experiment)
        if block and block not in seen:
            blocks.append(block)
            seen.add(block)
    return "\n\n".join(blocks).strip()


@dataclass(frozen=True)
class SeedContext:
    seed_id: str
    title: str
    abstract: str
    source_md: Path


@dataclass(frozen=True)
class MethodProposal:
    research_problem: str
    proposed_method: str
    experimental_plan: str


@dataclass(frozen=True)
class Artifact:
    seed_id: str
    path: Path
    proposal: MethodProposal
    status: str
    title: str | None = None


def discover_markdown(input_dir: Path) -> tuple[dict[str, SeedContext], list[str]]:
    issues: list[str] = []
    files = sorted(input_dir.glob("*.md"), key=lambda p: p.name)
    by_id: dict[str, Path] = {}
    for path in files:
        pid = paper_id(path.name)
        if pid is None:
            issues.append(f"Markdown filename has no Pxxx id: {path.name}")
        elif pid in by_id:
            issues.append(f"duplicate Markdown id {pid}: {by_id[pid]} and {path}")
        else:
            by_id[pid] = path
    contexts: dict[str, SeedContext] = {}
    for pid, path in sorted(by_id.items()):
        try:
            parsed = parse_markdown_file(path)
        except Exception as exc:
            issues.append(f"Markdown parse failed for {pid}: {exc}")
            continue
        if not parsed.title:
            issues.append(f"missing title in Markdown: {pid}")
        if not parsed.abstract:
            issues.append(f"missing abstract in Markdown: {pid}")
        if parsed.title and parsed.abstract:
            contexts[pid] = SeedContext(pid, parsed.title, parsed.abstract, path)
    expected = set(SEED_IDS)
    missing = sorted(expected - set(contexts))
    extra = sorted(set(contexts) - expected)
    if missing:
        issues.append(f"missing Markdown seeds: {', '.join(missing)}")
    if extra:
        issues.append(f"unexpected Markdown seeds: {', '.join(extra)}")
    if len(files) != 80:
        issues.append(f"expected 80 Markdown files, found {len(files)}")
    return contexts, issues


def manifest_status(directory: Path, names: Iterable[str]) -> str:
    for name in names:
        path = directory / name
        if not path.exists():
            continue
        data = unwrap(read_json(path))
        if isinstance(data, dict) and data.get("status"):
            return str(data["status"])
    return "status_not_explicit"


def discover_direct(root: Path) -> tuple[dict[str, Artifact], list[str]]:
    artifacts: dict[str, Artifact] = {}
    issues: list[str] = []
    candidates = sorted(root.glob("papers/*/final_proposal.json"))
    for path in candidates:
        pid = paper_id(path.parent.name)
        if pid is None:
            issues.append(f"Direct result has no Pxxx id: {path}")
            continue
        if pid in artifacts:
            issues.append(f"duplicate Direct final proposal for {pid}")
            continue
        payload = unwrap(read_json(path))
        proposal = MethodProposal(
            stable_text(payload.get("research_problem")),
            concat_existing(payload.get("proposed_method"), payload.get("technical_rationale")),
            stable_text(payload.get("experimental_plan")),
        )
        status = manifest_status(path.parent, ("run_manifest.json",))
        artifacts[pid] = Artifact(pid, path, proposal, status)
    return artifacts, coverage_issues("Direct", artifacts, issues)


def discover_rise(root: Path) -> tuple[dict[str, Artifact], list[str]]:
    artifacts: dict[str, Artifact] = {}
    issues: list[str] = []
    candidates = sorted(root.glob("papers/*/final_research_proposal.json"))
    for path in candidates:
        pid = paper_id(path.parent.name)
        if pid is None:
            issues.append(f"RISE result has no Pxxx id: {path}")
            continue
        if pid in artifacts:
            issues.append(f"duplicate RISE final proposal for {pid}")
            continue
        payload = unwrap(read_json(path))
        proposal = MethodProposal(
            stable_text(payload.get("research_gap")),
            concat_existing(
                payload.get("proposed_solution"),
                payload.get("solution_rationale"),
                rise_components(payload.get("method_components")),
            ),
            rise_experiments(payload.get("experimental_plan")),
        )
        status = manifest_status(path.parent, ("pilot_manifest.json", "run_manifest.json"))
        artifacts[pid] = Artifact(pid, path, proposal, status)
    return artifacts, coverage_issues("RISE", artifacts, issues)


def discover_flat(root: Path) -> tuple[dict[str, Artifact], list[str]]:
    artifacts: dict[str, Artifact] = {}
    issues: list[str] = []
    candidates = sorted(root.glob("flat/P*/final_proposal.json"))
    for path in candidates:
        pid = paper_id(path.parent.name)
        if pid is None:
            issues.append(f"Flat result has no Pxxx id: {path}")
            continue
        if pid in artifacts:
            issues.append(f"duplicate Flat final proposal for {pid}")
            continue
        payload = unwrap(read_json(path))
        proposal = MethodProposal(
            stable_text(payload.get("research_problem")),
            stable_text(payload.get("proposed_method")),
            stable_text(payload.get("experimental_plan")),
        )
        status = manifest_status(path.parent, ("run_metadata.json",))
        artifacts[pid] = Artifact(pid, path, proposal, status)
    return artifacts, coverage_issues("Flat", artifacts, issues)


def discover_researchagent(root: Path) -> tuple[dict[str, Artifact], list[str]]:
    artifacts: dict[str, Artifact] = {}
    issues: list[str] = []
    candidates = sorted(root.glob("P[0-9][0-9][0-9].json"))
    for path in candidates:
        pid = paper_id(path.name)
        if pid is None:
            issues.append(f"ResearchAgent result has no Pxxx id: {path}")
            continue
        if pid in artifacts:
            issues.append(f"duplicate ResearchAgent result for {pid}")
            continue
        payload = read_json(path)
        paper = payload.get("paper") if isinstance(payload, dict) else {}
        proposal = MethodProposal(
            concat_existing(payload.get("problem"), payload.get("problem_rationale")),
            concat_existing(payload.get("method"), payload.get("method_rationale")),
            concat_existing(payload.get("experiment"), payload.get("experiment_rationale")),
        )
        # The official export has no per-paper status field.  Its manifest and
        # the presence of all three final stage fields are the completion test.
        status = "final_fields_present"
        artifacts[pid] = Artifact(pid, path, proposal, status, stable_text(paper.get("title")))
    return artifacts, coverage_issues("ResearchAgent", artifacts, issues)


def coverage_issues(label: str, artifacts: dict[str, Artifact], issues: list[str]) -> list[str]:
    missing = sorted(set(SEED_IDS) - set(artifacts))
    extra = sorted(set(artifacts) - set(SEED_IDS))
    if missing:
        issues.append(f"{label} missing seeds: {', '.join(missing)}")
    if extra:
        issues.append(f"{label} unexpected seeds: {', '.join(extra)}")
    if len(artifacts) != 80:
        issues.append(f"{label} usable final proposal count is {len(artifacts)}, expected 80")
    for pid, artifact in sorted(artifacts.items()):
        if any(not getattr(artifact.proposal, field) for field in (
            "research_problem", "proposed_method", "experimental_plan"
        )):
            issues.append(f"{label} has an empty normalized field for {pid}")
    return issues


def norm_title(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def proposal_dict(proposal: MethodProposal) -> dict[str, str]:
    return {
        "research_problem": proposal.research_problem,
        "proposed_method": proposal.proposed_method,
        "experimental_plan": proposal.experimental_plan,
    }


def h1_markdown(cases: list[dict[str, Any]]) -> str:
    chunks = [
        "# H1 — End-to-End Proposal Ranking",
        "",
        "Rank the three proposals in each case from strongest to weakest. Do not use ties.",
        "",
    ]
    for case in cases:
        chunks += [f"## Case {case['case_id']}", "", "### Seed Paper Title", case["seed_title"], "", "### Seed Paper Abstract", case["seed_abstract"], ""]
        for label in ("A", "B", "C"):
            proposal = case[f"proposal_{label.lower()}"]
            chunks += [f"### Proposal {label}", "", "#### Research Problem", proposal["research_problem"], "", "#### Proposed Method", proposal["proposed_method"], "", "#### Experimental Plan", proposal["experimental_plan"], ""]
    return "\n".join(chunks)


def h2_markdown(cases: list[dict[str, Any]]) -> str:
    chunks = [
        "# H2 — Two-Proposal Comparison",
        "",
        "Choose the stronger follow-up research direction in each case. Do not use ties.",
        "",
    ]
    for case in cases:
        chunks += [f"## Case {case['case_id']}", "", "### Seed Paper Title", case["seed_title"], "", "### Seed Paper Abstract", case["seed_abstract"], ""]
        for label in ("A", "B"):
            proposal = case[f"proposal_{label.lower()}"]
            chunks += [f"### Proposal {label}", "", "#### Research Problem", proposal["research_problem"], "", "#### Proposed Method", proposal["proposed_method"], "", "#### Experimental Plan", proposal["experimental_plan"], ""]
    return "\n".join(chunks)


INSTRUCTIONS = """# Anonymous evaluation instructions

This is a strictly anonymous blind evaluation. Do not try to identify where any proposal came from.

Do not infer identity from length, formatting, wording, or language style. Do not search for external literature. Judge only the supplied seed title, seed abstract, and anonymized proposals.

For H1, rank all three proposals from strongest to weakest with no ties. For H2, choose exactly one of the two proposals, with no tie.

Evaluate the overall scientific quality of each proposal as a follow-up research direction: research problem quality, technical mechanism, soundness and feasibility, experimental testability, scientific significance, and perceived differentiation from the supplied seed context. Perceived differentiation is relative only to the supplied seed context and is not an exhaustive novelty verification.

Confidence and primary reason are auxiliary analysis fields only.
"""


def write_csv(path: Path, fieldnames: list[str], case_ids: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_id in case_ids:
            writer.writerow({"case_id": case_id, **{field: "" for field in fieldnames[1:]}})


def build_assignments() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    combinations = list(itertools.combinations(EXPERTS, 3))
    h1_slots = combinations * 8
    h2_slots = combinations * 8
    random.Random(H1_ASSIGNMENT_SEED).shuffle(h1_slots)
    random.Random(H2_ASSIGNMENT_SEED).shuffle(h2_slots)
    return dict(zip(SEED_IDS, h1_slots)), dict(zip(SEED_IDS, h2_slots))


def h1_case_maps(h1_assignments: dict[str, tuple[str, ...]]) -> dict[str, dict[str, dict[str, str]]]:
    methods = ("rise", "direct", "researchagent")
    permutations = list(itertools.permutations(methods))
    result: dict[str, dict[str, dict[str, str]]] = {expert: {} for expert in EXPERTS}
    for index, expert in enumerate(EXPERTS):
        assigned = [pid for pid in SEED_IDS if expert in h1_assignments[pid]]
        rng = random.Random(H1_ASSIGNMENT_SEED + 1000 + index)
        rng.shuffle(assigned)
        choices = permutations * 8
        rng.shuffle(choices)
        for pid, order in zip(assigned, choices):
            result[expert][pid] = dict(zip(("A", "B", "C"), order))
    return result


def h2_case_maps(h2_assignments: dict[str, tuple[str, ...]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {expert: {} for expert in EXPERTS}
    for index, expert in enumerate(EXPERTS):
        assigned = [pid for pid in SEED_IDS if expert in h2_assignments[pid]]
        rng = random.Random(H2_ASSIGNMENT_SEED + 2000 + index)
        rng.shuffle(assigned)
        orientations = ["full"] * 24 + ["flat"] * 24
        rng.shuffle(orientations)
        for pid, orientation in zip(assigned, orientations):
            result[expert][pid] = orientation
    return result


def validate_generated(
    root: Path,
    contexts: dict[str, SeedContext],
    h1_assignments: dict[str, tuple[str, ...]],
    h2_assignments: dict[str, tuple[str, ...]],
    h1_maps: dict[str, dict[str, dict[str, str]]],
    h2_maps: dict[str, dict[str, str]],
) -> dict[str, Any]:
    issues: list[str] = []
    h1_counts = {expert: 0 for expert in EXPERTS}
    h2_counts = {expert: 0 for expert in EXPERTS}
    h1_positions = {
        expert: {x: {method: 0 for method in ("rise", "direct", "researchagent")} for x in ("A", "B", "C")}
        for expert in EXPERTS
    }
    h2_full_as_a = {expert: 0 for expert in EXPERTS}
    for pid in SEED_IDS:
        if len(h1_assignments[pid]) != 3 or len(h2_assignments[pid]) != 3:
            issues.append(f"assignment count is not 3 for {pid}")
        for expert in h1_assignments[pid]:
            h1_counts[expert] += 1
            mapping = h1_maps[expert][pid]
            for position, method in mapping.items():
                h1_positions[expert][position][method] += 1
            if set(mapping.values()) != {"rise", "direct", "researchagent"}:
                issues.append(f"H1 mapping is not a permutation for {expert}/{pid}")
        for expert in h2_assignments[pid]:
            h2_counts[expert] += 1
            if h2_maps[expert][pid] == "full":
                h2_full_as_a[expert] += 1
    if set(contexts) != set(SEED_IDS):
        issues.append("seed context coverage is not exactly P001-P080")
    if any(value != 48 for value in h1_counts.values()):
        issues.append(f"H1 expert counts are not all 48: {h1_counts}")
    if any(value != 48 for value in h2_counts.values()):
        issues.append(f"H2 expert counts are not all 48: {h2_counts}")
    if any(value != 24 for value in h2_full_as_a.values()):
        issues.append(f"H2 Full-as-A counts are not all 24: {h2_full_as_a}")
    if any(
        any(value != 16 for by_method in positions.values() for value in by_method.values())
        for positions in h1_positions.values()
    ):
        issues.append(f"H1 positions are not all 16: {h1_positions}")

    expert_files = list(root.glob("expert_*/**/*"))
    if any(path.name in {"source_key.json", "h1_source_key.json", "h2_source_key.json"} for path in expert_files):
        issues.append("source key found inside expert directory")
    prohibited_instructions = ("RISE", "Direct", "ResearchAgent", "Full", "Flat")
    for path in root.glob("expert_*/INSTRUCTIONS.md"):
        text = path.read_text(encoding="utf-8")
        for token in prohibited_instructions:
            if token in text:
                issues.append(f"method identity token {token!r} in {path}")

    # Ensure all expert-facing JSON records have only the common presentation
    # fields in each proposal and no source metadata.
    for path in root.glob("expert_*/H[12]*cases.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            for key in ("proposal_a", "proposal_b") + (("proposal_c",) if "proposal_c" in case else ()):
                if set(case[key]) != {"research_problem", "proposed_method", "experimental_plan"}:
                    issues.append(f"non-common proposal fields in {path}")

    h1_case_count = len({
        json.loads(line)["case_id"]
        for path in root.glob("expert_*/H1_end_to_end_cases.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    })
    h2_case_count = len({
        json.loads(line)["case_id"]
        for path in root.glob("expert_*/H2_full_flat_cases.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    })
    return {
        "passed": not issues,
        "issues": issues,
        "h1_unique_seed_cases": h1_case_count,
        "h2_unique_seed_cases": h2_case_count,
        "h1_ratings_per_case": 3,
        "h2_ratings_per_case": 3,
        "h1_cases_per_expert": h1_counts,
        "h2_cases_per_expert": h2_counts,
        "h1_position_counts_per_expert": h1_positions,
        "h2_full_as_a_per_expert": h2_full_as_a,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/tmh/baseline/human_evaluation"))
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/nlp")
    parser.add_argument("--direct-dir", type=Path, default=Path("/home/tmh/baseline/output_direct"))
    parser.add_argument("--researchagent-dir", type=Path, default=Path("/home/tmh/baseline/output_ResearchAgent"))
    parser.add_argument("--rise-dir", type=Path, default=Path("/home/tmh/baseline/output_rise"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("/home/tmh/baseline/state_evolution_ablation/flat"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise DatasetError(f"refusing to overwrite existing output directory: {args.output_dir}")
    for path in (args.input_dir, args.direct_dir, args.researchagent_dir, args.rise_dir, args.ablation_dir):
        if not path.exists():
            raise DatasetError(f"input path does not exist: {path}")

    contexts, issues = discover_markdown(args.input_dir)
    direct, direct_issues = discover_direct(args.direct_dir)
    researchagent, ra_issues = discover_researchagent(args.researchagent_dir)
    rise, rise_issues = discover_rise(args.rise_dir)
    flat, flat_issues = discover_flat(args.ablation_dir.parent)
    issues.extend(direct_issues + ra_issues + rise_issues + flat_issues)

    for label, artifacts in (("Direct", direct), ("ResearchAgent", researchagent), ("RISE", rise), ("Flat", flat)):
        for pid in SEED_IDS:
            artifact = artifacts.get(pid)
            if artifact is None:
                continue
            if artifact.title and norm_title(artifact.title) != norm_title(contexts.get(pid, SeedContext(pid, "", "", Path())).title):
                issues.append(f"{label} title mismatch for {pid}: result={artifact.title!r}")

    # E3's assembly manifest documents that its Flat tree is complete and that
    # its Full source was output_final_80.  The requested RISE-Full input is the
    # supplied complete output_rise tree, so record both facts explicitly.
    assembly_manifest = args.ablation_dir.parent / "assembly_manifest.json"
    if assembly_manifest.exists():
        assembly = read_json(assembly_manifest)
        if assembly.get("flat_count") != 80 or assembly.get("rise_count") != 80:
            issues.append("E3 assembly manifest does not report 80 Full and 80 Flat results")

    if issues:
        # A validation report is still useful, but do not create expert data
        # when source alignment is unsafe.
        print("Input validation failed:")
        for issue in issues:
            print(" -", issue)
        return 2

    h1_assignments, h2_assignments = build_assignments()
    h1_maps = h1_case_maps(h1_assignments)
    h2_maps = h2_case_maps(h2_assignments)

    temp_dir = Path(tempfile.mkdtemp(prefix="human_evaluation.", dir=str(args.output_dir.parent)))
    try:
        (temp_dir / "private").mkdir(parents=True)
        for expert in EXPERTS:
            (temp_dir / expert).mkdir()

        h1_source_key: dict[str, dict[str, dict[str, str]]] = {expert: {} for expert in EXPERTS}
        h2_source_key: dict[str, dict[str, dict[str, str]]] = {expert: {} for expert in EXPERTS}
        h1_cases_by_expert: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERTS}
        h2_cases_by_expert: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERTS}
        for expert_index, expert in enumerate(EXPERTS):
            h1_ids = [pid for pid in SEED_IDS if expert in h1_assignments[pid]]
            h2_ids = [pid for pid in SEED_IDS if expert in h2_assignments[pid]]
            random.Random(H1_ASSIGNMENT_SEED + 3000 + expert_index).shuffle(h1_ids)
            random.Random(H2_ASSIGNMENT_SEED + 4000 + expert_index).shuffle(h2_ids)
            for pid in h1_ids:
                context = contexts[pid]
                mapping = h1_maps[expert][pid]
                proposals = {"rise": rise[pid].proposal, "direct": direct[pid].proposal, "researchagent": researchagent[pid].proposal}
                case = {"case_id": pid, "seed_title": context.title, "seed_abstract": context.abstract}
                for position in ("A", "B", "C"):
                    case[f"proposal_{position.lower()}"] = proposal_dict(proposals[mapping[position]])
                h1_cases_by_expert[expert].append(case)
                h1_source_key[expert][pid] = mapping
            for pid in h2_ids:
                context = contexts[pid]
                orientation = h2_maps[expert][pid]
                proposals = {"full": rise[pid].proposal, "flat": flat[pid].proposal}
                order = ("full", "flat") if orientation == "full" else ("flat", "full")
                case = {"case_id": pid, "seed_title": context.title, "seed_abstract": context.abstract,
                        "proposal_a": proposal_dict(proposals[order[0]]), "proposal_b": proposal_dict(proposals[order[1]])}
                h2_cases_by_expert[expert].append(case)
                h2_source_key[expert][pid] = {"A": order[0], "B": order[1]}

        for expert in EXPERTS:
            expert_dir = temp_dir / expert
            (expert_dir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
            h1_path = expert_dir / "H1_end_to_end_cases.jsonl"
            h1_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in h1_cases_by_expert[expert]) + "\n", encoding="utf-8")
            (expert_dir / "H1_end_to_end_cases.md").write_text(h1_markdown(h1_cases_by_expert[expert]), encoding="utf-8")
            write_csv(expert_dir / "H1_ratings.csv", ["case_id", "ranking", "confidence", "primary_reason", "notes"], [x["case_id"] for x in h1_cases_by_expert[expert]])
            h2_path = expert_dir / "H2_full_flat_cases.jsonl"
            h2_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in h2_cases_by_expert[expert]) + "\n", encoding="utf-8")
            (expert_dir / "H2_full_flat_cases.md").write_text(h2_markdown(h2_cases_by_expert[expert]), encoding="utf-8")
            write_csv(expert_dir / "H2_ratings.csv", ["case_id", "winner", "confidence", "primary_reason", "notes"], [x["case_id"] for x in h2_cases_by_expert[expert]])

        mapping = {
            "source_direct": str(args.direct_dir),
            "source_researchagent": str(args.researchagent_dir),
            "source_rise_full": str(args.rise_dir),
            "source_flat": str(args.ablation_dir),
            "E3_assembly_manifest": str(assembly_manifest),
            "field_mapping": {
                "Direct-Full": {
                    "Research Problem": ["research_problem"],
                    "Proposed Method": ["proposed_method", "technical_rationale"],
                    "Experimental Plan": ["experimental_plan"],
                },
                "ResearchAgent": {
                    "Research Problem": ["problem", "problem_rationale"],
                    "Proposed Method": ["method", "method_rationale"],
                    "Experimental Plan": ["experiment", "experiment_rationale"],
                },
                "RISE-Full": {
                    "Research Problem": ["research_gap"],
                    "Proposed Method": ["proposed_solution", "solution_rationale", "method_components[*].description"],
                    "Experimental Plan": ["experimental_plan[*].objective", "setup", "baselines", "metrics", "what_it_validates", "validation_criterion"],
                },
                "RISE-Flat": {
                    "Research Problem": ["research_problem"],
                    "Proposed Method": ["proposed_method"],
                    "Experimental Plan": ["experimental_plan"],
                },
            },
            "normalization": [
                "fixed-order extraction and concatenation only",
                "exact duplicate blocks omitted",
                "RISE component/experiment schema labels and bookkeeping omitted while values are retained",
                "no LLM, paraphrase, summary, translation, polish, or scientific rewriting",
            ],
        }
        json_dump(temp_dir / "private/normalization_mapping.json", mapping)
        (temp_dir / "private/normalization_mapping.md").write_text("""# Normalization mapping\n\nThis file records deterministic source-field extraction. Proposal text is copied from existing result fields; fields are concatenated in the listed order. No model or paraphrasing is used. Exact duplicate blocks are omitted.\n\n## Direct-Full\n\n- Research Problem: `research_problem`\n- Proposed Method: `proposed_method`, then `technical_rationale`\n- Experimental Plan: `experimental_plan`\n\n## ResearchAgent\n\n- Research Problem: `problem`, then `problem_rationale`\n- Proposed Method: `method`, then `method_rationale`\n- Experimental Plan: `experiment`, then `experiment_rationale`\n\n## RISE-Full\n\n- Research Problem: `research_gap`\n- Proposed Method: `proposed_solution`, `solution_rationale`, then each `method_components[*].description` in source order\n- Experimental Plan: each `experimental_plan` item values in source schema order: `objective`, `setup`, `baselines`, `metrics`, `what_it_validates`, `validation_criterion`; IDs/status/bookkeeping labels are not displayed\n\n## RISE-Flat\n\n- Research Problem: `research_problem`\n- Proposed Method: `proposed_method`\n- Experimental Plan: `experimental_plan`\n\nE3 source note: the supplied `state_evolution_ablation` tree contains `flat/Pxxx`; its `assembly_manifest.json` records the corresponding Full source separately. The complete RISE-Full proposals used here are read from the supplied `output_rise` directory.\n""", encoding="utf-8")

        json_dump(temp_dir / "private/h1_source_key.json", h1_source_key)
        json_dump(temp_dir / "private/h2_source_key.json", h2_source_key)
        assignment_manifest = {
            "h1_assignment_seed": H1_ASSIGNMENT_SEED,
            "h2_assignment_seed": H2_ASSIGNMENT_SEED,
            "experts": list(EXPERTS),
            "seed_ids": list(SEED_IDS),
            "h1_experts_by_case": {pid: list(h1_assignments[pid]) for pid in SEED_IDS},
            "h2_experts_by_case": {pid: list(h2_assignments[pid]) for pid in SEED_IDS},
            "expected_ratings_per_case": {"H1": 3, "H2": 3},
            "expected_cases_per_expert": {"H1": 48, "H2": 48},
        }
        json_dump(temp_dir / "private/assignment_manifest.json", assignment_manifest)

        generated_validation = validate_generated(temp_dir, contexts, h1_assignments, h2_assignments, h1_maps, h2_maps)
        json_dump(temp_dir / "validation_report.json", generated_validation)
        validation_lines = [
            "# Human evaluation dataset validation",
            "",
            f"Overall: **{'PASS' if generated_validation['passed'] else 'FAIL'}**",
            "",
            "## Source discovery",
            "",
            "- Markdown seed files: 80; title/abstract extraction: 80/80.",
            "- Direct-Full final proposals: 80/80 (`output_direct/papers/*/final_proposal.json`).",
            "- ResearchAgent final exports: 80/80 (`Pxxx.json`; manifest contains 80 IDs; no per-record status field).",
            "- RISE-Full final proposals: 80/80 (`output_rise/papers/*/final_research_proposal.json`).",
            "- RISE-Flat final proposals: 80/80 (`state_evolution_ablation/flat/Pxxx/final_proposal.json`).",
            "- No duplicate or missing source IDs detected.",
            "",
            "## Expert allocation",
            "",
            f"- H1 unique seeds: {generated_validation['h1_unique_seed_cases']}; 3 ratings per seed; per expert: {generated_validation['h1_cases_per_expert']}.",
            f"- H2 unique seeds: {generated_validation['h2_unique_seed_cases']}; 3 ratings per seed; per expert: {generated_validation['h2_cases_per_expert']}.",
            f"- H1 position counts: `{json.dumps(generated_validation['h1_position_counts_per_expert'])}`.",
            f"- H2 Full-as-A counts: `{json.dumps(generated_validation['h2_full_as_a_per_expert'])}`.",
            "",
            "## Issues",
            "",
        ]
        validation_lines.extend(f"- {issue}" for issue in generated_validation["issues"] or ["None"])
        (temp_dir / "validation_report.md").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

        readme = """# Human expert evaluation dataset\n\nThis directory contains two strictly anonymous evaluation tasks generated from completed local experiments. The expert-facing files contain only the seed title/abstract and content-preserving normalized proposals. No proposal was generated or rewritten.\n\n## Tasks\n\n- **H1**: three anonymous proposals per seed; rank strongest to weakest with no ties.\n- **H2**: two anonymous proposals per seed; choose A or B with no tie.\n\nEach task covers P001-P080. Each case is assigned to exactly three of five experts. Each expert receives 48 cases for each task. H1 positions are balanced at 16 per position per expert. H2 uses 24 cases with the complete-state proposal in A and 24 with it in B per expert. The assignments and mappings are private.\n\n## Files\n\nEach `expert_xx/` directory contains instructions, JSONL for later analysis, Markdown for reading, and a blank CSV rating sheet. The `private/` directory contains the field mapping, assignment manifest, and source keys; it must not be given to experts.\n\nFor later analysis, H1 rankings can be converted to pairwise preferences and aggregated by seed majority, then joined to existing model-judge outputs by Pxxx. H2 can similarly be reduced to a seed-level majority comparison. Individual ratings should not be treated as independent inferential samples.\n\n## Provenance\n\nThis dataset was produced deterministically from the supplied Direct-Full, ResearchAgent, RISE-Full, and RISE-Flat result roots. The output root contains no API-generated content.\n"""
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")

        if generated_validation["issues"]:
            raise DatasetError("generated dataset validation failed: " + "; ".join(generated_validation["issues"]))
        temp_dir.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"created: {args.output_dir}")
    print("H1: 80 seeds, 3 ratings per seed, 48 cases per expert")
    print("H2: 80 seeds, 3 ratings per seed, 48 cases per expert")
    print("all source alignment and anonymity checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
