#!/usr/bin/env python3
"""Build a fresh anonymous pairwise package from completed Ours papers.

The package is evaluation-only: it performs deterministic field mapping and
does not call an LLM or any external service.  The anonymous input contains
only seed metadata and Proposal A/B.  The source key is written separately
for optional post-review unblinding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import parse_markdown_file  # noqa: E402


DEFAULT_OURS_DIR = REPO_ROOT / "outputs" / "ours_paired64_segment_a"
DEFAULT_OURS_REPORT = DEFAULT_OURS_DIR / "evaluation_report.json"
DEFAULT_DIRECT_DIRS = [
    REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_2",
    REPO_ROOT / "outputs" / "direct_llm_neutral_nlp2_workers8",
]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "blind_review_successful_ours_v1"

COMMON_FIELDS = (
    "research_problem",
    "proposed_method",
    "technical_rationale",
    "experimental_plan",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build anonymous paired review data from successful Ours runs"
    )
    parser.add_argument("--ours-dir", type=Path, default=DEFAULT_OURS_DIR)
    parser.add_argument("--ours-report", type=Path, default=DEFAULT_OURS_REPORT)
    parser.add_argument(
        "--direct-dir",
        type=Path,
        action="append",
        dest="direct_dirs",
        help="Direct-Neutral root; may be supplied more than once",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _payload(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid artifact envelope: {path}")
    return payload


def _text(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_components(components: Any, *, path: Path) -> str:
    if not isinstance(components, list) or not components:
        raise ValueError(f"{path}: method_components must be a non-empty list")
    lines: list[str] = []
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            raise ValueError(f"{path}: method_components[{index - 1}] must be an object")
        name = component.get("name") or component.get("id") or f"Component {index}"
        description = _text(
            component.get("description"),
            field=f"method_components[{index - 1}].description",
            path=path,
        )
        lines.append(f"{name}: {description}")
    return "\n".join(lines)


def _format_experiments(experiments: Any, *, path: Path) -> str:
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{path}: experimental_plan must be a non-empty list")
    blocks: list[str] = []
    for index, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise ValueError(f"{path}: experimental_plan[{index - 1}] must be an object")
        labels = (
            ("Objective", "objective"),
            ("Setup", "setup"),
            ("Baselines", "baselines"),
            ("Metrics", "metrics"),
            ("What it validates", "what_it_validates"),
            ("Validation criterion", "validation_criterion"),
        )
        lines: list[str] = []
        for label, key in labels:
            value = experiment.get(key)
            if value is None:
                continue
            rendered = value if isinstance(value, str) else _json_text(value)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{path}: experimental_plan[{index - 1}].{key} is empty")
            lines.append(f"{label}: {rendered}")
        if not lines:
            raise ValueError(f"{path}: experimental_plan[{index - 1}] has no content")
        blocks.append(f"Experiment {index}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _normalize_ours(paper_dir: Path) -> dict[str, str]:
    path = paper_dir / "final_research_proposal.json"
    payload = _payload(path)
    components = _format_components(payload.get("method_components"), path=path)
    method = "\n\n".join(
        (
            _text(payload.get("proposed_solution"), field="proposed_solution", path=path),
            "Major components:\n" + components,
        )
    )
    proposal = {
        "research_problem": _text(payload.get("research_gap"), field="research_gap", path=path),
        "proposed_method": method,
        "technical_rationale": _text(
            payload.get("solution_rationale"), field="solution_rationale", path=path
        ),
        "experimental_plan": _format_experiments(payload.get("experimental_plan"), path=path),
    }
    return _validate_proposal(proposal, path)


def _normalize_direct(paper_dir: Path) -> dict[str, str]:
    path = paper_dir / "final_proposal.json"
    payload = _payload(path)
    proposal = {
        "research_problem": _text(payload.get("research_problem"), field="research_problem", path=path),
        "proposed_method": _text(payload.get("proposed_method"), field="proposed_method", path=path),
        "technical_rationale": _text(
            payload.get("technical_rationale"), field="technical_rationale", path=path
        ),
        "experimental_plan": _text(
            payload.get("experimental_plan"), field="experimental_plan", path=path
        ),
    }
    return _validate_proposal(proposal, path)


def _validate_proposal(proposal: dict[str, str], path: Path) -> dict[str, str]:
    if tuple(proposal) != COMMON_FIELDS:
        raise ValueError(f"{path}: proposal fields do not match common blind schema")
    for field in COMMON_FIELDS:
        _text(proposal[field], field=field, path=path)
    return proposal


def _successful_ours_ids(report_path: Path) -> list[str]:
    document = _read_json(report_path)
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise ValueError(f"invalid Ours evaluation report: {report_path}")
    ids = [
        row.get("paper_id")
        for row in payload["papers"]
        if isinstance(row, dict) and row.get("status") == "committed"
    ]
    if any(not isinstance(paper_id, str) or not paper_id for paper_id in ids):
        raise ValueError(f"invalid committed paper identity in: {report_path}")
    return sorted(set(ids))


def _proposal_dirs(root: Path, artifact_name: str) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"proposal directory does not exist: {root}")
    result: dict[str, Path] = {}
    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if (paper_dir / artifact_name).is_file():
            result[paper_dir.name] = paper_dir
    return result


def _direct_map(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for paper_id, paper_dir in _proposal_dirs(root, "final_proposal.json").items():
            if paper_id in result:
                raise ValueError(f"duplicate Direct-Neutral proposal for {paper_id}")
            result[paper_id] = paper_dir
    return result


def _paper_number(paper_id: str) -> int | None:
    match = re.search(r"(?:^|_)P(\d+)(?:_|$)", paper_id)
    return int(match.group(1)) if match else None


def _assignment(seed: int, paper_id: str) -> bool:
    digest = hashlib.sha256(f"{seed}:paired-blind:{paper_id}".encode()).digest()
    return digest[0] % 2 == 0


def _write_review_instructions(path: Path) -> None:
    path.write_text(
        """# Anonymous paired proposal review

Review each pair independently using only the seed title, seed abstract, and
Proposal A/B. Do not infer system identity or use prior scores. Judge scientific
content rather than writing style or length.

For each dimension use: +2 A clearly better, +1 A somewhat better, 0 no
meaningful difference, -1 B somewhat better, -2 B clearly better. Use the
dimensions: research problem quality, novelty, technical mechanism quality,
technical soundness and feasibility, experimental quality, scientific
significance, and overall proposal quality. Give a short reason for each
dimension. Do not open `source_key.json` during scoring.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    direct_roots = args.direct_dirs or DEFAULT_DIRECT_DIRS
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    successful_ids = _successful_ours_ids(args.ours_report)
    ours_dirs = _proposal_dirs(args.ours_dir, "final_research_proposal.json")
    direct_dirs = _direct_map(direct_roots)
    missing_ours = sorted(set(successful_ids) - set(ours_dirs))
    missing_direct = sorted(set(successful_ids) - set(direct_dirs))
    if missing_ours or missing_direct:
        raise SystemExit(
            "incomplete pairing; missing Ours="
            + repr(missing_ours)
            + "; missing Direct-Neutral="
            + repr(missing_direct)
        )

    paper_paths = {path.stem: path for path in args.input_dir.glob("*.md")}
    missing_markdown = sorted(set(successful_ids) - set(paper_paths))
    if missing_markdown:
        raise SystemExit(f"missing source Markdown for committed papers: {missing_markdown}")

    cases: list[dict[str, Any]] = []
    source_key: list[dict[str, Any]] = []
    for index, paper_id in enumerate(successful_ids, start=1):
        markdown_path = paper_paths[paper_id]
        parsed = parse_markdown_file(markdown_path)
        title = _text(parsed.title, field="title", path=markdown_path)
        abstract = _text(parsed.abstract, field="abstract", path=markdown_path)
        ours = _normalize_ours(ours_dirs[paper_id])
        direct = _normalize_direct(direct_dirs[paper_id])
        pair_id = f"pair_{index:04d}"
        if _assignment(args.seed, paper_id):
            proposal_a, proposal_b = ours, direct
            a_method, b_method = "ours", "direct_neutral"
        else:
            proposal_a, proposal_b = direct, ours
            a_method, b_method = "direct_neutral", "ours"
        cases.append(
            {
                "pair_id": pair_id,
                "seed_title": title,
                "seed_abstract": abstract,
                "proposal_a": proposal_a,
                "proposal_b": proposal_b,
            }
        )
        source_key.append(
            {
                "pair_id": pair_id,
                "paper_id": paper_id,
                "paper_number": _paper_number(paper_id),
                "a_source_method": a_method,
                "b_source_method": b_method,
                "source_markdown": str(markdown_path),
                "ours_source_dir": str(ours_dirs[paper_id]),
                "direct_source_dir": str(direct_dirs[paper_id]),
            }
        )

    anonymous_fields = {
        "pair_id",
        "seed_title",
        "seed_abstract",
        "proposal_a",
        "proposal_b",
    }
    proposal_complete = all(
        set(case["proposal_a"]) == set(COMMON_FIELDS)
        and set(case["proposal_b"]) == set(COMMON_FIELDS)
        for case in cases
    )
    # Do not scan proposal prose for ordinary words such as "direct" or
    # "ours": those can occur naturally and removing them would alter the
    # scientific content.  Leakage is checked against the anonymous schema's
    # structural keys only; source labels live exclusively in source_key.json.
    structural_keys = set()
    for case in cases:
        structural_keys.update(case.keys())
        structural_keys.update(case["proposal_a"].keys())
        structural_keys.update(case["proposal_b"].keys())
    leaked_labels = sorted(
        structural_keys.intersection({"ours", "direct", "candidate_a", "candidate_b"})
    )
    manifest = {
        "experiment": "anonymous_paired_proposal_review",
        "paper_count": len(cases),
        "pair_count": len(cases),
        "proposals_per_pair": 2,
        "anonymous_fields": sorted(anonymous_fields),
        "proposal_fields": list(COMMON_FIELDS),
        "title_abstract_complete": all(
            bool(case["seed_title"].strip()) and bool(case["seed_abstract"].strip())
            for case in cases
        ),
        "proposal_fields_complete": proposal_complete,
        "unique_pair_ids": len({case["pair_id"] for case in cases}) == len(cases),
        "source_key_separate": True,
        "anonymous_source_label_leakage": leaked_labels,
        "previous_scores_or_winners_used": False,
        "external_llm_or_api_used": False,
        "assignment_seed": args.seed,
    }

    args.output_dir.mkdir()
    (args.output_dir / "paired_anonymous_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "source_key.json").write_text(
        json.dumps(source_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pairing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_review_instructions(args.output_dir / "review_instructions.md")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
