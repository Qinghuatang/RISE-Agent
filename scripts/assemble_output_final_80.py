#!/usr/bin/env python3
"""Assemble the 80 complete Ours paper results into one fresh directory.

Only paper directories containing ``final_research_proposal.json`` are
selected.  Retry outputs take precedence naturally because source roots are
checked in chronological order and a paper is copied only once.  Existing
source artifacts are never modified.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from research_agent.ingestion.markdown import parse_markdown_file
from research_agent.runner.batch_evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationError,
    build_evaluation_report,
    summarize_paper_output,
)
from research_agent.serialization import write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "nlp"
OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final_80"
EXPECTED_PAPERS = 80

# Ordered from the original aggregate to the successive retry outputs.  If a
# paper has complete artifacts in more than one root, the later retry root
# replaces the earlier entry.
SOURCE_ROOTS = (
    ("output_final", REPO_ROOT / "outputs" / "output_final" / "papers"),
    ("output_final_30", REPO_ROOT / "outputs" / "output_final_30"),
    ("output_final_6", REPO_ROOT / "outputs" / "output_final_6"),
    ("output_final_2", REPO_ROOT / "outputs" / "output_final_2"),
)


def _input_paths() -> dict[str, Path]:
    if not INPUT_DIR.is_dir():
        raise BatchEvaluationError(f"input directory does not exist: {INPUT_DIR}")
    paths = sorted(
        path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    if len(paths) != EXPECTED_PAPERS:
        raise BatchEvaluationError(
            f"expected exactly {EXPECTED_PAPERS} Markdown files, found {len(paths)}"
        )
    result: dict[str, Path] = {}
    for path in paths:
        parsed_id = parse_markdown_file(path).paper_id
        if parsed_id != path.stem:
            raise BatchEvaluationError(
                f"paper_id does not match Markdown stem: {path.name} -> {parsed_id}"
            )
        if parsed_id in result:
            raise BatchEvaluationError(f"duplicate input paper_id: {parsed_id}")
        result[parsed_id] = path.resolve()
    return result


def _select_sources(input_by_id: dict[str, Path]) -> dict[str, tuple[str, Path]]:
    selected: dict[str, tuple[str, Path]] = {}
    for source_name, root in SOURCE_ROOTS:
        if not root.is_dir():
            raise BatchEvaluationError(f"source result directory does not exist: {root}")
        for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            paper_id = paper_dir.name
            if paper_id not in input_by_id:
                continue
            if not (paper_dir / "final_research_proposal.json").is_file():
                continue
            selected[paper_id] = (source_name, paper_dir.resolve())

    missing = sorted(set(input_by_id) - set(selected))
    if missing:
        raise BatchEvaluationError(
            "complete result is missing for paper(s): " + ", ".join(missing)
        )
    extra = sorted(set(selected) - set(input_by_id))
    if extra:
        raise BatchEvaluationError(
            "selected result contains paper(s) absent from input: " + ", ".join(extra)
        )
    if len(selected) != EXPECTED_PAPERS:
        raise BatchEvaluationError(
            f"expected {EXPECTED_PAPERS} selected results, found {len(selected)}"
        )
    return selected


def main() -> int:
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT_DIR}")

    input_by_id = _input_paths()
    selected = _select_sources(input_by_id)
    papers_dir = OUTPUT_DIR / "papers"
    papers_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    evaluations = []
    for paper_id in sorted(selected):
        source_name, source_dir = selected[paper_id]
        destination = papers_dir / paper_id
        shutil.copytree(source_dir, destination)
        evaluation = summarize_paper_output(
            input_by_id[paper_id],
            destination,
        )
        evaluations.append(evaluation)
        manifest_rows.append(
            {
                "paper_id": paper_id,
                "source_run": source_name,
                "source_dir": str(source_dir),
                "copied_dir": str(destination.resolve()),
                "has_final_proposal": True,
            }
        )

    batch_config = BatchEvaluationConfig(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        batch_name="ours_output_final_80",
        max_papers=None,
        selection_seed=None,
        allow_overwrite=False,
    )
    report = build_evaluation_report(
        batch_config,
        evaluations,
        sorted(selected),
    )
    write_json(OUTPUT_DIR / "evaluation_report.json", "evaluation_report", report)

    manifest = {
        "purpose": "aggregate complete Ours per-paper artifacts",
        "paper_count": EXPECTED_PAPERS,
        "final_proposal_count": len(evaluations),
        "source_roots": [str(root) for _, root in SOURCE_ROOTS],
        "selection_rule": (
            "select only paper directories with final_research_proposal.json; "
            "first complete result in the ordered source roots is retained"
        ),
        "papers": manifest_rows,
    }
    write_json(OUTPUT_DIR / "assembly_manifest.json", "assembly_manifest", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "paper_count": report.paper_count,
                "completed_count": report.completed_count,
                "failed_count": report.failed_count,
                "stopped_count": report.stopped_count,
                "final_proposal_count": len(evaluations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
