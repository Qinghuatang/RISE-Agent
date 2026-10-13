#!/usr/bin/env python3
"""Assemble the complete 80-paper Flat and original RISE artifacts.

The script only copies existing local artifacts.  It never calls an LLM or
retrieval service and refuses to overwrite an existing destination.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import parse_markdown_file  # noqa: E402


EXPECTED_PAPERS = 80
PAPER_ID_RE = re.compile(r"(P\d{3})(?:_|$)")
INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_OUTPUT_DIR = Path("/home/tmh/baseline/state_evolution_ablation")
RISE_ROOT = REPO_ROOT / "outputs" / "output_final_80" / "papers"
FLAT_ROOTS = (
    ("pilot8_v2", REPO_ROOT / "outputs" / "state_evolution_ablation_pilot8_v2"),
    ("remaining72", REPO_ROOT / "outputs" / "state_evolution_ablation_flat_remaining72"),
    ("retry15", REPO_ROOT / "outputs" / "state_evolution_ablation_flat_retry15"),
    ("retry2", REPO_ROOT / "outputs" / "state_evolution_ablation_flat_retry2"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble 80-paper Flat and RISE artifacts")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--rise-root", type=Path, default=RISE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_papers(input_dir: Path) -> dict[str, dict[str, Any]]:
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    paths = tuple(sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"))
    if len(paths) != EXPECTED_PAPERS:
        raise RuntimeError(f"expected {EXPECTED_PAPERS} Markdown files, found {len(paths)}")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        parsed = parse_markdown_file(path)
        if parsed.paper_id != path.stem:
            raise RuntimeError(f"paper_id does not match Markdown stem: {path.name} -> {parsed.paper_id}")
        match = PAPER_ID_RE.search(path.stem)
        if match is None:
            raise RuntimeError(f"cannot determine Pxxx ID from filename: {path.name}")
        short_id = match.group(1)
        if short_id in result:
            raise RuntimeError(f"duplicate short paper ID: {short_id}")
        result[short_id] = {"paper_id": parsed.paper_id, "short_id": short_id, "path": path.resolve()}
    if len(result) != EXPECTED_PAPERS:
        raise RuntimeError(f"expected {EXPECTED_PAPERS} unique paper IDs, found {len(result)}")
    return result


def _flat_branch_complete(branch: Path) -> bool:
    if not branch.is_dir():
        return False
    metadata_path = branch / "run_metadata.json"
    proposal_path = branch / "final_proposal.json"
    if not metadata_path.is_file() or not proposal_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(metadata, dict) and metadata.get("status") in {"committed", "stopped"}


def _flat_status(branch: Path) -> str | None:
    try:
        value = json.loads((branch / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value.get("status") if isinstance(value, dict) else None


def _select_flat(input_by_short_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    # Later roots have precedence, so a successful retry replaces an older
    # successful/stopped result for the same paper.
    for source_name, root in FLAT_ROOTS:
        if not root.is_dir():
            raise RuntimeError(f"Flat source directory does not exist: {root}")
        for short_id, info in input_by_short_id.items():
            branch = root / short_id / "flat"
            if _flat_branch_complete(branch):
                selected[short_id] = {
                    "source_run": source_name,
                    "source_dir": branch.resolve(),
                    "status": _flat_status(branch),
                    "paper_id": info["paper_id"],
                }
    missing = sorted(set(input_by_short_id) - set(selected))
    if missing:
        raise RuntimeError("complete Flat result missing: " + ", ".join(missing))
    if len(selected) != EXPECTED_PAPERS:
        raise RuntimeError(f"expected {EXPECTED_PAPERS} complete Flat results, found {len(selected)}")
    return selected


def _select_rise(input_by_short_id: dict[str, dict[str, Any]], rise_root: Path) -> dict[str, dict[str, Any]]:
    if not rise_root.is_dir():
        raise RuntimeError(f"RISE source directory does not exist: {rise_root}")
    selected: dict[str, dict[str, Any]] = {}
    for short_id, info in input_by_short_id.items():
        source_dir = rise_root / info["paper_id"]
        if not source_dir.is_dir():
            raise RuntimeError(f"RISE result directory missing for {short_id}: {source_dir}")
        if not (source_dir / "final_research_proposal.json").is_file():
            raise RuntimeError(f"RISE final proposal missing for {short_id}: {source_dir}")
        selected[short_id] = {
            "source_run": "output_final_80",
            "source_dir": source_dir.resolve(),
            "paper_id": info["paper_id"],
        }
    if len(selected) != EXPECTED_PAPERS:
        raise RuntimeError(f"expected {EXPECTED_PAPERS} complete RISE results, found {len(selected)}")
    return selected


def _copy_selected(
    selected: dict[str, dict[str, Any]],
    destination_root: Path,
    *,
    system: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    destination_system = destination_root / system
    destination_system.mkdir(parents=True, exist_ok=False)
    for short_id in sorted(selected):
        item = selected[short_id]
        destination = destination_system / short_id
        shutil.copytree(item["source_dir"], destination)
        rows.append(
            {
                "short_id": short_id,
                "paper_id": item["paper_id"],
                "source_run": item["source_run"],
                "source_dir": str(item["source_dir"]),
                "copied_dir": str(destination.resolve()),
                "status": item.get("status"),
                "file_count": sum(1 for path in destination.rglob("*") if path.is_file()),
            }
        )
    return rows


def main() -> int:
    args = _parser().parse_args()
    if args.output_dir.is_symlink() or (
        args.output_dir.exists()
        and (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))
    ):
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")

    input_by_short_id = _input_papers(args.input_dir)
    flat = _select_flat(input_by_short_id)
    rise = _select_rise(input_by_short_id, args.rise_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = _copy_selected(flat, args.output_dir, system="flat")
    rise_rows = _copy_selected(rise, args.output_dir, system="rise")
    manifest = {
        "experiment": "state_evolution_ablation_80",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_count": EXPECTED_PAPERS,
        "flat_count": len(flat_rows),
        "rise_count": len(rise_rows),
        "input_dir": str(args.input_dir.resolve()),
        "rise_source_root": str(args.rise_root.resolve()),
        "flat_source_roots": [{"name": name, "path": str(root.resolve())} for name, root in FLAT_ROOTS],
        "selection_rule": "latest complete Flat result wins in source-root order: pilot8_v2, remaining72, retry15, retry2",
        "systems": {
            "flat": {"directory": str((args.output_dir / "flat").resolve()), "papers": flat_rows},
            "rise": {"directory": str((args.output_dir / "rise").resolve()), "papers": rise_rows},
        },
    }
    _write_json(args.output_dir / "assembly_manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "flat_count": len(flat_rows),
        "rise_count": len(rise_rows),
        "assembly_manifest": str((args.output_dir / "assembly_manifest.json").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
