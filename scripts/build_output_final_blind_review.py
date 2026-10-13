#!/usr/bin/env python3
"""Merge successful Ours runs and build a fresh anonymous review package.

This is an evaluation-data utility only.  It copies completed per-paper
artifacts, performs no scientific processing, and calls no LLM or external
service.  The existing deterministic blind-package builder performs the final
proposal field mapping and A/B anonymization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OURS_RUNS = [
    REPO_ROOT / "outputs" / "ours_paired64_segment_a",
    REPO_ROOT / "outputs" / "ours_paired64_segment_b",
]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final"
DEFAULT_BLIND_DIR = REPO_ROOT / "outputs" / "blind_review_output_final_v1"
DEFAULT_DIRECT_DIRS = [
    REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_2",
    REPO_ROOT / "outputs" / "direct_llm_neutral_nlp2_workers8",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge successful Ours artifacts and build anonymous A/B review data"
    )
    parser.add_argument(
        "--ours-run",
        type=Path,
        action="append",
        dest="ours_runs",
        help="Ours batch output; may be supplied more than once",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blind-output-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument(
        "--direct-dir",
        type=Path,
        action="append",
        dest="direct_dirs",
        help="Direct-Neutral output root; may be supplied more than once",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    return parser


def _read_payload(run: Path) -> dict[str, Any]:
    report_path = run / "evaluation_report.json"
    if not report_path.is_file():
        raise ValueError(f"missing evaluation report: {report_path}")
    document = json.loads(report_path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise ValueError(f"invalid evaluation report: {report_path}")
    return payload


def _collect_committed(runs: list[Path]) -> list[tuple[str, Path, dict[str, Any], str]]:
    records: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for run in runs:
        payload = _read_payload(run)
        for row in payload["papers"]:
            if not isinstance(row, dict) or not isinstance(row.get("paper_id"), str):
                raise ValueError(f"malformed paper row in {run / 'evaluation_report.json'}")
            if row.get("status") != "committed":
                continue
            paper_id = row["paper_id"]
            source_dir = run / paper_id
            if not source_dir.is_dir():
                raise ValueError(f"missing committed paper artifacts: {source_dir}")
            if not (source_dir / "final_research_proposal.json").is_file():
                raise ValueError(f"committed paper has no final proposal: {source_dir}")
            if paper_id in records:
                raise ValueError(f"duplicate committed paper across runs: {paper_id}")
            records[paper_id] = (source_dir, row, str(run))
    return [(paper_id, *records[paper_id]) for paper_id in sorted(records)]


def _write_merged_report(output_dir: Path, records: list[tuple[str, Path, dict[str, Any], str]]) -> None:
    rows = [dict(row, output_dir=str(output_dir / "papers" / paper_id)) for paper_id, _, row, _ in records]
    payload = {
        "batch_name": "output_final",
        "input_dir": str(DEFAULT_INPUT_DIR),
        "output_dir": str(output_dir),
        "paper_count": len(rows),
        "completed_count": len(rows),
        "failed_count": 0,
        "commit_status_counts": {"committed": len(rows)},
        "papers": rows,
    }
    document = {"payload": payload, "schema": "batch_evaluation_report", "version": 1}
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_merge_manifest(
    output_dir: Path,
    records: list[tuple[str, Path, dict[str, Any], str]],
    runs: list[Path],
) -> None:
    manifest = {
        "purpose": "read-only aggregation of successful Ours per-paper artifacts",
        "source_runs": [str(run) for run in runs],
        "selection": "status == committed",
        "paper_count": len(records),
        "papers": [
            {
                "paper_id": paper_id,
                "source_run": source_run,
                "source_dir": str(source_dir),
                "copied_dir": str(output_dir / "papers" / paper_id),
                "status": "committed",
                "has_final_proposal": True,
            }
            for paper_id, source_dir, _, source_run in records
        ],
        "external_llm_or_api_used": False,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parser().parse_args()
    ours_runs = args.ours_runs or DEFAULT_OURS_RUNS
    direct_dirs = args.direct_dirs or DEFAULT_DIRECT_DIRS
    for path in (args.output_dir, args.blind_output_dir):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"Refusing to overwrite existing output path: {path}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")

    records = _collect_committed(ours_runs)
    if not records:
        raise SystemExit("no committed Ours papers found")

    papers_dir = args.output_dir / "papers"
    papers_dir.mkdir(parents=True)
    for paper_id, source_dir, _, _ in records:
        shutil.copytree(source_dir, papers_dir / paper_id)
    _write_merged_report(args.output_dir, records)
    _write_merge_manifest(args.output_dir, records, ours_runs)

    builder = REPO_ROOT / "scripts" / "build_successful_ours_blind_package.py"
    command = [
        sys.executable,
        str(builder),
        "--ours-dir",
        str(papers_dir),
        "--ours-report",
        str(args.output_dir / "evaluation_report.json"),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.blind_output_dir),
        "--seed",
        str(args.seed),
    ]
    for direct_dir in direct_dirs:
        command.extend(("--direct-dir", str(direct_dir)))
    subprocess.run(command, check=True)
    print(
        json.dumps(
            {
                "successful_ours_papers": len(records),
                "merged_output_dir": str(args.output_dir),
                "blind_output_dir": str(args.blind_output_dir),
                "source_key": str(args.blind_output_dir / "source_key.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
