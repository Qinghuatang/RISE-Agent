#!/usr/bin/env python3
"""Build anonymized pairwise review packages from completed proposal artifacts.

This is an evaluation-only utility. It does not call an LLM, modify workflow
artifacts, or assign scientific quality scores. Each package contains blinded
candidate A/B packets and a separate unblinding key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OURS = REPO_ROOT / "outputs" / "formal_nlp_40_successful_merged" / "papers"
DEFAULT_STRONG = REPO_ROOT / "outputs" / "direct_llm_formal_40"
DEFAULT_NEUTRAL = REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_2"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "blind_review_ours_vs_direct_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build blinded Ours-vs-Direct review packages")
    parser.add_argument("--ours-dir", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--direct-strong-dir", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--direct-neutral-dir", type=Path, default=DEFAULT_NEUTRAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", choices=("strong", "neutral", "both"), default="both")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force", action="store_true")
    return parser


def _payload(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid artifact envelope: {path}")
    return payload


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_components(components: Any) -> str:
    if not isinstance(components, list):
        return ""
    lines: list[str] = []
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            lines.append(f"Component {index}: {_json_text(component)}")
            continue
        name = component.get("name") or component.get("id") or f"Component {index}"
        description = component.get("description", "")
        lines.append(f"{name}: {description}")
    return "\n".join(lines)


def _format_experiments(experiments: Any) -> str:
    if isinstance(experiments, str):
        return experiments
    if not isinstance(experiments, list):
        return _json_text(experiments)
    blocks: list[str] = []
    for index, experiment in enumerate(experiments, start=1):
        if isinstance(experiment, dict):
            fields = [
                ("Objective", experiment.get("objective")),
                ("Setup", experiment.get("setup")),
                ("Baselines", experiment.get("baselines")),
                ("Metrics", experiment.get("metrics")),
                ("What it validates", experiment.get("what_it_validates")),
                ("Validation criterion", experiment.get("validation_criterion")),
            ]
            content = "\n".join(
                f"{label}: {value if isinstance(value, str) else _json_text(value)}"
                for label, value in fields
                if value is not None
            )
            blocks.append(f"Experiment {index}\n{content}")
        else:
            blocks.append(f"Experiment {index}\n{_json_text(experiment)}")
    return "\n\n".join(blocks)


def _normalize_ours(path: Path) -> dict[str, Any]:
    payload = _payload(path / "final_research_proposal.json")
    return {
        "research_problem": payload["research_gap"],
        "proposed_method": "\n\n".join(
            part
            for part in (
                payload.get("proposed_solution", ""),
                "Major components:\n" + _format_components(payload.get("method_components")),
            )
            if part
        ),
        "technical_rationale": payload["solution_rationale"],
        "experimental_plan": _format_experiments(payload.get("experimental_plan", [])),
        "expected_contributions": payload.get("expected_contributions", []),
    }


def _normalize_direct(
    path: Path,
    *,
    artifact_name: str,
    neutral_schema: bool,
) -> dict[str, Any]:
    payload = _payload(path / artifact_name)
    proposed_method = payload.get("proposed_method", {})
    if neutral_schema:
        method_text = str(proposed_method)
    elif isinstance(proposed_method, dict):
        method_text = "\n\n".join(
            part
            for part in (
                proposed_method.get("core_idea", ""),
                "Major components:\n" + _format_components(proposed_method.get("components")),
            )
            if part
        )
    else:
        method_text = str(proposed_method)
    return {
        "research_problem": payload["research_problem"],
        "proposed_method": method_text,
        "technical_rationale": payload["technical_rationale"],
        "experimental_plan": _format_experiments(payload.get("experimental_plan", [])),
        "expected_contributions": payload.get("expected_contributions", []),
    }


def _proposal_dirs(root: Path, filename: str) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"proposal directory does not exist: {root}")
    result: dict[str, Path] = {}
    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        proposal = paper_dir / filename
        if proposal.is_file():
            result[paper_dir.name] = paper_dir
    return result


def _assignment(seed: int, paper_id: str, baseline: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{baseline}:{paper_id}".encode()).digest()
    return digest[0] % 2 == 0


def _review_template(pair_id: str, paper_id: str) -> dict[str, Any]:
    empty = {"better": None, "confidence": None, "notes": ""}
    return {
        "pair_id": pair_id,
        "paper_id": paper_id,
        "dimensions": {
            "research_problem": dict(empty),
            "technical_mechanism": dict(empty),
            "technical_soundness": dict(empty),
            "experimental_plan": dict(empty),
            "scientific_value": dict(empty),
            "overall_proposal": dict(empty),
        },
        "overall_preference": None,
        "reviewer_notes": "",
    }


def _write_package(
    *,
    name: str,
    baseline_label: str,
    ours: dict[str, Path],
    direct: dict[str, Path],
    direct_artifact_name: str,
    neutral_schema: bool,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    package = output_root / name
    package.mkdir(parents=True, exist_ok=False)
    pairs_dir = package / "pairs"
    pairs_dir.mkdir()
    key: list[dict[str, Any]] = []
    forms: list[dict[str, Any]] = []
    common = sorted(set(ours) & set(direct))
    for index, paper_id in enumerate(common, start=1):
        pair_id = f"case_{index:03d}"
        ours_proposal = _normalize_ours(ours[paper_id])
        direct_proposal = _normalize_direct(
            direct[paper_id],
            artifact_name=direct_artifact_name,
            neutral_schema=neutral_schema,
        )
        ours_is_a = _assignment(seed, paper_id, baseline_label)
        candidates = (
            ("A", ours_proposal, "ours") if ours_is_a else ("A", direct_proposal, baseline_label),
            ("B", direct_proposal, baseline_label) if ours_is_a else ("B", ours_proposal, "ours"),
        )
        packet = {
            "pair_id": pair_id,
            "paper_id": paper_id,
            "evaluation_instructions": {
                "blindness": "Do not infer system identity from style or content; judge only the supplied candidates.",
                "allowed_better_values": ["A", "B", "tie", "not_judged"],
                "allowed_confidence_values": ["low", "medium", "high"],
            },
            "candidate_a": candidates[0][1],
            "candidate_b": candidates[1][1],
        }
        (pairs_dir / f"{pair_id}.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        forms.append(_review_template(pair_id, paper_id))
        key.append(
            {
                "pair_id": pair_id,
                "paper_id": paper_id,
                "candidate_a_source": candidates[0][2],
                "candidate_b_source": candidates[1][2],
                "ours_source_dir": str(ours[paper_id]),
                "direct_source_dir": str(direct[paper_id]),
            }
        )

    (package / "review_forms.json").write_text(
        json.dumps(forms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package / "source_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package / "README.md").write_text(
        "# Blind pairwise review package\n\n"
        f"Baseline: {baseline_label}\n\n"
        "Review `pairs/*.json` without opening `source_key.json`. Fill the corresponding "
        "entries in `review_forms.json`. The key is only for unblinding after review.\n\n"
        "Dimensions: research problem, technical mechanism, technical soundness, "
        "experimental plan, and overall proposal. Use A/B/tie and low/medium/high "
        "confidence. Do not infer a winner from execution metadata.\n",
        encoding="utf-8",
    )
    manifest = {
        "package": name,
        "baseline": baseline_label,
        "pair_count": len(common),
        "paper_ids": common,
        "assignment_seed": seed,
        "ours_proposal_count": len(ours),
        "direct_proposal_count": len(direct),
        "excluded_from_pairing": {
            "ours_only": sorted(set(ours) - set(direct)),
            "direct_only": sorted(set(direct) - set(ours)),
        },
        "quality_scores_generated": False,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    ours = _proposal_dirs(args.ours_dir, "final_research_proposal.json")
    strong = _proposal_dirs(args.direct_strong_dir, "direct_proposal.json")
    neutral = _proposal_dirs(args.direct_neutral_dir, "final_proposal.json")
    manifests: list[dict[str, Any]] = []
    if args.baseline in {"strong", "both"}:
        manifests.append(
            _write_package(
                name="direct_strong",
                baseline_label="direct_strong",
                ours=ours,
                direct=strong,
                direct_artifact_name="direct_proposal.json",
                neutral_schema=False,
                output_root=args.output_dir,
                seed=args.seed,
            )
        )
    if args.baseline in {"neutral", "both"}:
        manifests.append(
            _write_package(
                name="direct_neutral",
                baseline_label="direct_neutral",
                ours=ours,
                direct=neutral,
                direct_artifact_name="final_proposal.json",
                neutral_schema=True,
                output_root=args.output_dir,
                seed=args.seed,
            )
        )
    root_manifest = {
        "ours_dir": str(args.ours_dir),
        "direct_strong_dir": str(args.direct_strong_dir),
        "direct_neutral_dir": str(args.direct_neutral_dir),
        "ours_proposal_count": len(ours),
        "direct_strong_proposal_count": len(strong),
        "direct_neutral_proposal_count": len(neutral),
        "packages": manifests,
        "quality_scores_generated": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
