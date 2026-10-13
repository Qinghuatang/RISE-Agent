#!/usr/bin/env python3
"""Create a read-only 40-paper view from successive parallel Ours runs.

For duplicate paper IDs, the retry run wins. The script creates symlinks to
the original artifacts and never rewrites them. It reports committed, stopped,
and failed papers separately so downstream evaluation cannot mistake coverage
for successful final proposals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "outputs" / "ours_parallel_nlp40_workers8_2"
DEFAULT_RETRY = REPO_ROOT / "outputs" / "ours_parallel_nlp40_retry_non_budget"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ours_parallel_nlp40_merged_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge successive parallel Ours batch artifacts")
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--retry-run", type=Path, default=DEFAULT_RETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _report(run: Path) -> dict[str, Any]:
    report_path = run / "evaluation_report.json"
    if not report_path.is_file():
        raise SystemExit(f"missing evaluation report: {report_path}")
    document = json.loads(report_path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise SystemExit(f"invalid evaluation report: {report_path}")
    return payload


def _rows(run: Path, payload: dict[str, Any]) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in payload["papers"]:
        if not isinstance(row, dict) or not isinstance(row.get("paper_id"), str):
            raise SystemExit(f"malformed paper row in {run / 'evaluation_report.json'}")
        paper_id = row["paper_id"]
        source = run / paper_id
        if not source.is_dir():
            raise SystemExit(f"missing paper artifact directory: {source}")
        result[paper_id] = (source.resolve(), row)
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    base_payload = _report(args.base_run)
    retry_payload = _report(args.retry_run)
    base = _rows(args.base_run, base_payload)
    retry = _rows(args.retry_run, retry_payload)
    merged = dict(base)
    source_run: dict[str, str] = {paper_id: "base" for paper_id in base}
    for paper_id, item in retry.items():
        merged[paper_id] = item
        source_run[paper_id] = "retry"

    if len(merged) != 40:
        raise SystemExit(f"expected 40 merged papers, found {len(merged)}")

    papers_dir = args.output_dir / "papers"
    papers_dir.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    for paper_id in sorted(merged):
        source, row = merged[paper_id]
        link = papers_dir / paper_id
        os.symlink(source, link, target_is_directory=True)
        manifest_rows.append(
            {
                "paper_id": paper_id,
                "source_run": source_run[paper_id],
                "source_dir": str(source),
                "status": row.get("status"),
                "failure_type": row.get("failure_type"),
                "has_final_proposal": (source / "final_research_proposal.json").is_file(),
            }
        )

    counts = {
        status: sum(row["status"] == status for row in manifest_rows)
        for status in sorted({row["status"] for row in manifest_rows})
    }
    manifest = {
        "base_run": str(args.base_run),
        "retry_run": str(args.retry_run),
        "paper_count": len(manifest_rows),
        "status_counts": counts,
        "final_proposal_count": sum(row["has_final_proposal"] for row in manifest_rows),
        "papers": manifest_rows,
        "selection_rule": "retry artifact wins for duplicate paper_id; otherwise base artifact",
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("paper_count", "status_counts", "final_proposal_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
