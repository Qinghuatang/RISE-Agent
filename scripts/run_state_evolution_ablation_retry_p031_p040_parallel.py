#!/usr/bin/env python3
"""Retry the two failed Flat pilot seeds with two paper-level workers.

This is orchestration only.  The existing FlatRunner remains the single-paper
entry point, and the historical Full branch is copied as a read-only
reference.  No RISE scientific workflow is implemented here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from research_agent.llm.config import DeepSeekConfig
from research_agent.retrieval.arxiv import ArxivConfig
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig

from experiments.state_evolution_ablation.flat_runner import FlatRunConfig, FlatRunner
from experiments.state_evolution_ablation.full_reference import (
    FullReferenceError,
    inspect_full_reference,
    write_full_reference,
)
from experiments.state_evolution_ablation.pilot8_config import (
    DEFAULT_FULL_OUTPUT_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    PilotSeed,
    get_seed,
    resolve_seed_paths,
)


RETRY_SEED_IDS = ("P031", "P040")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_pilot8_retry_p031_p040"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry the P031/P040 RISE-Flat pilot branches in parallel"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--full-output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--semantic-scholar-config", type=Path, default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-searches", type=int, default=DEFAULT_MAX_SEARCHES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="rerun existing retry branches")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without calling services")
    parser.add_argument("--allow-real-services", action="store_true", help="required for a real Flat run")
    return parser


def _plain_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_seeds() -> tuple[PilotSeed, ...]:
    return tuple(get_seed(seed_id) for seed_id in RETRY_SEED_IDS)


def _static_check(args: argparse.Namespace, seeds: tuple[PilotSeed, ...]) -> dict[str, Any]:
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    if not args.semantic_scholar_config.is_file():
        raise SystemExit(f"Semantic Scholar config does not exist: {args.semantic_scholar_config}")
    paths = resolve_seed_paths(args.input_dir, seeds)
    missing = [short_id for short_id, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"configured Markdown files are missing: {', '.join(missing)}")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_steps <= 0 or args.max_searches < 0:
        raise SystemExit("--max-steps must be positive and --max-searches non-negative")
    return {
        "input_dir": str(args.input_dir.resolve()),
        "full_output_dir": str(args.full_output_dir.resolve()),
        "semantic_scholar_config": str(args.semantic_scholar_config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "workers": args.workers,
        "max_steps": args.max_steps,
        "max_searches": args.max_searches,
        "seeds": [
            {
                "seed_id": seed.short_id,
                "paper_id": seed.paper_id,
                "source_md": str(paths[seed.short_id].resolve()),
            }
            for seed in seeds
        ],
    }


def _check_real_config(args: argparse.Namespace) -> dict[str, Any]:
    deepseek = DeepSeekConfig.from_env()
    semantic = SemanticScholarConfig.from_file(args.semantic_scholar_config)
    arxiv = ArxivConfig.from_env()
    deepseek.validate_for_request()
    semantic.validate_for_request()
    return {
        "deepseek_model": deepseek.model,
        "deepseek_base_url": deepseek.base_url,
        "semantic_scholar_base_url": semantic.base_url,
        "semantic_scholar_cache_dir": semantic.cache_dir,
        "arxiv_cache_dir": arxiv.cache_dir,
        "api_keys_configured": {
            "deepseek": bool(deepseek.api_key),
            "semantic_scholar": bool(semantic.api_key),
        },
    }


def _run_one(seed: PilotSeed, args: argparse.Namespace) -> str:
    branch = args.output_dir / seed.short_id / "flat"
    config = FlatRunConfig(
        markdown_path=args.input_dir / seed.filename,
        output_dir=branch,
        semantic_scholar_config=args.semantic_scholar_config,
        max_steps=args.max_steps,
        max_searches=args.max_searches,
        allow_real_services=True,
        allow_overwrite=args.overwrite,
    )
    try:
        result = FlatRunner(config).run()
        return str(result.get("status", "completed"))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        branch.mkdir(parents=True, exist_ok=True)
        _plain_write(branch / "error.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        return "failed"


def main() -> int:
    args = _parser().parse_args()
    seeds = _selected_seeds()
    static = _static_check(args, seeds)

    full_reference_status: dict[str, str] = {}
    for seed in seeds:
        try:
            full_reference_status[seed.short_id] = inspect_full_reference(seed, args.full_output_dir)["status"]
        except FullReferenceError as exc:
            full_reference_status[seed.short_id] = f"invalid: {exc}"

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "selected_seeds": list(RETRY_SEED_IDS),
            "static": static,
            "full_reference_status": full_reference_status,
        }, ensure_ascii=False, indent=2))
        return 0 if all(not value.startswith("invalid:") for value in full_reference_status.values()) else 1

    invalid = {key: value for key, value in full_reference_status.items() if value.startswith("invalid:")}
    if invalid:
        raise SystemExit(
            "historical Full references are not compatible; refusing to run Flat: "
            + "; ".join(f"{key}: {value}" for key, value in invalid.items())
        )
    if not args.allow_real_services:
        raise SystemExit("real Flat retry requires --allow-real-services")
    try:
        runtime_config = _check_real_config(args)
    except Exception as exc:
        raise SystemExit(f"real provider configuration is invalid: {exc}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        write_full_reference(seed, args.full_output_dir, args.output_dir / seed.short_id / "full")
    _plain_write(args.output_dir / "manifest.json", {
        "experiment": "state_evolution_ablation_pilot8_retry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_seeds": list(RETRY_SEED_IDS),
        "static_config": static,
        "runtime_config": runtime_config,
        "full_is_read_only": True,
        "full_api_calls_in_this_experiment": 0,
    })

    outcomes: dict[str, str] = {}
    executor = ProcessPoolExecutor(max_workers=args.workers)
    try:
        futures = {executor.submit(_run_one, seed, args): seed for seed in seeds}
        finished = 0
        for future in as_completed(futures):
            seed = futures[future]
            outcomes[seed.short_id] = future.result()
            finished += 1
            print(f"[Flat {finished}/{len(seeds)}] {seed.short_id}: {outcomes[seed.short_id]}", flush=True)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted by user; completed artifacts were retained.", file=sys.stderr)
        raise
    else:
        executor.shutdown(wait=True)

    ordered_outcomes = {seed.short_id: outcomes.get(seed.short_id, "missing") for seed in seeds}
    _plain_write(args.output_dir / "pilot_run_summary.json", {
        "experiment": "state_evolution_ablation_pilot8_retry",
        "workers": args.workers,
        "runtime_config": runtime_config,
        "outcomes": ordered_outcomes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"output_dir": str(args.output_dir), "outcomes": ordered_outcomes}, ensure_ascii=False, indent=2))
    return 0 if all(value in {"committed", "stopped", "skipped_completed"} for value in ordered_outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
