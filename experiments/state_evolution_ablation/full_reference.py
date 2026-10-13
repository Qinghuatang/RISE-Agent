"""Read-only references to the already completed RISE Full runs.

The ablation does not execute the Full branch.  This module only verifies that
the selected seed has a complete, compatible historical RISE output and saves
a small reference record under the new pilot directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .pilot8_config import DEFAULT_MAX_SEARCHES, DEFAULT_MAX_STEPS, PilotSeed


REQUIRED_FULL_ARTIFACTS = (
    "run_manifest.json",
    "seed_context.json",
    "seed_extraction_audit.json",
    "research_state.json",
    "research_memory.json",
    "trajectory.jsonl",
    "trajectory_replay.jsonl",
    "llm_calls.jsonl",
    "final_research_proposal.json",
)


class FullReferenceError(ValueError):
    """The historical Full output is absent or incompatible."""


def _read_envelope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullReferenceError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
        raise FullReferenceError(f"invalid artifact envelope: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_full_reference(seed: PilotSeed, full_root: str | Path) -> dict[str, Any]:
    """Validate and summarize one historical Full output without mutation."""

    paper_dir = Path(full_root) / seed.paper_id
    if not paper_dir.is_dir():
        raise FullReferenceError(f"historical Full directory does not exist: {paper_dir}")
    missing = [name for name in REQUIRED_FULL_ARTIFACTS if not (paper_dir / name).is_file()]
    if missing:
        raise FullReferenceError(
            f"historical Full output is incomplete for {seed.short_id}; missing: {', '.join(missing)}"
        )
    manifest = _read_envelope(paper_dir / "run_manifest.json")["payload"]
    paper_id = manifest.get("paper_id")
    if paper_id != seed.paper_id:
        raise FullReferenceError(
            f"historical Full paper_id mismatch: expected {seed.paper_id}, got {paper_id}"
        )
    status = manifest.get("status")
    if status not in {"committed", "stopped"}:
        raise FullReferenceError(f"historical Full status is not complete: {status!r}")
    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        raise FullReferenceError("historical Full manifest has no budget summary")
    max_steps = budget.get("max_steps")
    max_searches = budget.get("max_searches")
    if max_steps != DEFAULT_MAX_STEPS or max_searches != DEFAULT_MAX_SEARCHES:
        raise FullReferenceError(
            f"historical Full budget is incompatible: max_steps={max_steps}, max_searches={max_searches}"
        )
    providers: set[str] = set()
    models: set[str] = set()
    call_count = 0
    for line in (paper_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FullReferenceError(f"invalid historical LLM audit: {paper_dir / 'llm_calls.jsonl'}") from exc
        payload = record.get("payload") if isinstance(record, dict) else None
        if not isinstance(payload, dict):
            raise FullReferenceError(f"invalid historical LLM audit record: {paper_dir / 'llm_calls.jsonl'}")
        call_count += 1
        if isinstance(payload.get("provider"), str):
            providers.add(payload["provider"])
        if isinstance(payload.get("model"), str):
            models.add(payload["model"])
    return {
        "seed_id": seed.short_id,
        "paper_id": seed.paper_id,
        "system": "full",
        "status": status,
        "source_output_dir": str(paper_dir.resolve()),
        "source_artifacts": {
            name: {"path": str((paper_dir / name).resolve()), "sha256": _sha256(paper_dir / name)}
            for name in REQUIRED_FULL_ARTIFACTS
        },
        "historical_config": {
            "max_steps": max_steps,
            "max_searches": max_searches,
            "steps_used": budget.get("steps_used"),
            "searches_used": budget.get("searches_used"),
        },
        "historical_llm": {
            "call_count": call_count,
            "providers": sorted(providers),
            "models": sorted(models),
        },
        "read_only": True,
        "api_calls_in_this_experiment": 0,
    }


def write_full_reference(seed: PilotSeed, full_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Inspect one Full run and write only a reference/summary JSON."""

    summary = inspect_full_reference(seed, full_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "full_reference.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "FullReferenceError",
    "REQUIRED_FULL_ARTIFACTS",
    "inspect_full_reference",
    "write_full_reference",
]
