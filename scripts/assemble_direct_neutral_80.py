#!/usr/bin/env python3
"""Assemble all 80 complete Direct-Neutral results into one fresh directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from research_agent.ingestion.markdown import parse_markdown_file
from research_agent.models.direct_neutral import DirectNeutralProposal, DirectNeutralRunManifest


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "nlp"
OUTPUT_DIR = Path("/home/tmh/baseline/output_direct")
EXPECTED_INPUT_PAPERS = 80

SOURCE_ROOTS = (
    ("direct_llm_neutral_nlp40_2", REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_2"),
    ("direct_llm_neutral_nlp40_retry_6", REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_retry_6"),
    ("direct_llm_neutral_nlp40_retry_2", REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_retry_2"),
    ("direct_llm_neutral_nlp2_workers8", REPO_ROOT / "outputs" / "direct_llm_neutral_nlp2_workers8"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _input_papers() -> dict[str, Path]:
    if not INPUT_DIR.is_dir():
        raise ValueError(f"input directory does not exist: {INPUT_DIR}")
    paths = sorted(
        path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    if len(paths) != EXPECTED_INPUT_PAPERS:
        raise ValueError(
            f"expected exactly {EXPECTED_INPUT_PAPERS} Markdown files, found {len(paths)}"
        )
    result: dict[str, Path] = {}
    for path in paths:
        parsed = parse_markdown_file(path)
        if parsed.paper_id != path.stem:
            raise ValueError(f"paper_id does not match Markdown stem: {path.name}")
        if parsed.paper_id in result:
            raise ValueError(f"duplicate input paper_id: {parsed.paper_id}")
        result[parsed.paper_id] = path.resolve()
    return result


def _select_complete(input_by_id: dict[str, Path]) -> dict[str, tuple[str, Path]]:
    selected: dict[str, tuple[str, Path]] = {}
    for source_name, root in SOURCE_ROOTS:
        if not root.is_dir():
            raise ValueError(f"source result directory does not exist: {root}")
        for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            paper_id = paper_dir.name
            if paper_id not in input_by_id:
                continue
            proposal_path = paper_dir / "final_proposal.json"
            manifest_path = paper_dir / "run_manifest.json"
            if not proposal_path.is_file() or not manifest_path.is_file():
                continue
            try:
                proposal_doc = json.loads(proposal_path.read_text(encoding="utf-8"))
                proposal_payload = proposal_doc.get("payload", proposal_doc)
                DirectNeutralProposal.model_validate(proposal_payload)
                manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_payload = manifest_doc.get("payload", manifest_doc)
                run_manifest = DirectNeutralRunManifest.model_validate(manifest_payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid Direct-Neutral artifacts for {paper_id}") from exc
            if run_manifest.paper_id != paper_id or run_manifest.status != "succeeded":
                continue
            # Later retry roots replace an earlier complete result if both exist.
            selected[paper_id] = (source_name, paper_dir.resolve())

    missing = sorted(set(input_by_id) - set(selected))
    if missing:
        raise ValueError("complete Direct-Neutral result missing: " + ", ".join(missing))
    if len(selected) != EXPECTED_INPUT_PAPERS:
        raise ValueError(f"expected {EXPECTED_INPUT_PAPERS} complete results, found {len(selected)}")
    return selected


def _aggregate_report(
    input_by_id: dict[str, Path],
    selected: dict[str, tuple[str, Path]],
    destination_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for paper_id in sorted(selected):
        source_name, source_dir = selected[paper_id]
        manifest_doc = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest = DirectNeutralRunManifest.model_validate(manifest_doc.get("payload", manifest_doc))
        rows.append(
            {
                "paper_id": paper_id,
                "source_md": str(input_by_id[paper_id]),
                "output_dir": str((destination_root / "papers" / paper_id).resolve()),
                "source_run": source_name,
                "status": manifest.status,
                "logical_scientific_generation_calls": manifest.logical_scientific_generation_calls,
                "physical_api_attempts": manifest.physical_api_attempts,
                "prompt_tokens": manifest.prompt_tokens,
                "completion_tokens": manifest.completion_tokens,
                "total_tokens": manifest.total_tokens,
            }
        )

    def known_sum(field: str) -> int:
        return sum(int(row[field]) for row in rows if isinstance(row[field], int))

    complete_usage = sum(
        all(isinstance(row[field], int) for field in ("prompt_tokens", "completion_tokens", "total_tokens"))
        for row in rows
    )
    return {
        "baseline_name": "Direct-Neutral",
        "paper_count": len(rows),
        "succeeded": sum(row["status"] == "succeeded" for row in rows),
        "failed": sum(row["status"] != "succeeded" for row in rows),
        "logical_llm_calls": sum(row["logical_scientific_generation_calls"] for row in rows),
        "physical_api_attempts": sum(row["physical_api_attempts"] for row in rows),
        "prompt_tokens": known_sum("prompt_tokens"),
        "completion_tokens": known_sum("completion_tokens"),
        "total_known_tokens": known_sum("total_tokens"),
        "calls_with_usage": complete_usage,
        "calls_missing_usage": len(rows) - complete_usage,
        "token_metadata_complete": complete_usage == len(rows),
        "papers": rows,
    }


def main() -> int:
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT_DIR}")
    input_by_id = _input_papers()
    selected = _select_complete(input_by_id)
    papers_dir = OUTPUT_DIR / "papers"
    papers_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    for paper_id in sorted(selected):
        source_name, source_dir = selected[paper_id]
        destination = papers_dir / paper_id
        shutil.copytree(source_dir, destination)
        manifest_rows.append(
            {
                "paper_id": paper_id,
                "source_run": source_name,
                "source_dir": str(source_dir),
                "copied_dir": str(destination.resolve()),
                "has_final_proposal": (destination / "final_proposal.json").is_file(),
            }
        )

    report = _aggregate_report(input_by_id, selected, OUTPUT_DIR)
    _write_json(OUTPUT_DIR / "evaluation_report.json", report)
    _write_json(
        OUTPUT_DIR / "assembly_manifest.json",
        {
            "purpose": "aggregate complete Direct-Neutral per-paper artifacts",
            "paper_count": EXPECTED_INPUT_PAPERS,
            "final_proposal_count": sum(row["has_final_proposal"] for row in manifest_rows),
            "source_roots": [str(root) for _, root in SOURCE_ROOTS],
            "selection_rule": "complete retry result in the latest available source root wins",
            "papers": manifest_rows,
        },
    )
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "paper_count": report["paper_count"],
        "succeeded": report["succeeded"],
        "failed": report["failed"],
        "final_proposal_count": report["paper_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
