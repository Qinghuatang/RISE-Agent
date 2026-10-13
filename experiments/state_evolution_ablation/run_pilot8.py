#!/usr/bin/env python3
"""Run the eight-paper RISE-Flat pilot and reference historical RISE-Full.

Full is deliberately read-only here: it is never instantiated as a runner and
therefore cannot call an LLM or an external retrieval service.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from research_agent.llm.config import DeepSeekConfig
from research_agent.retrieval.arxiv import ArxivConfig
from research_agent.retrieval.semantic_scholar import SemanticScholarConfig

from experiments.state_evolution_ablation.flat_runner import FlatRunConfig, FlatRunner
from experiments.state_evolution_ablation.full_reference import FullReferenceError, inspect_full_reference, write_full_reference
from experiments.state_evolution_ablation.pilot8_config import (
    DEFAULT_FULL_OUTPUT_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEMANTIC_SCHOLAR_CONFIG,
    PILOT_SEEDS,
    PilotSeed,
    get_seed,
    resolve_seed_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only RISE-Flat for the eight-paper state evolution pilot"
    )
    parser.add_argument("--system", choices=("flat", "full", "both"), default="both")
    parser.add_argument("--seed", help="short ID such as P004; defaults to all eight")
    parser.add_argument("--all", action="store_true", help="explicitly select all eight configured seeds")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--full-output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--semantic-scholar-config", type=Path, default=DEFAULT_SEMANTIC_SCHOLAR_CONFIG)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-searches", type=int, default=DEFAULT_MAX_SEARCHES)
    parser.add_argument("--overwrite", action="store_true", help="explicitly rerun existing incomplete/completed Flat branches")
    parser.add_argument("--resume", action="store_true", help="resume by skipping completed Flat branches (the default behavior)")
    parser.add_argument("--workers", type=int, default=8, help="number of paper-level Flat workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="validate paths and Full references without calling services")
    parser.add_argument("--allow-real-services", action="store_true", help="required for a real Flat run")
    return parser


def _selected(args: argparse.Namespace) -> tuple[PilotSeed, ...]:
    if args.seed and args.all:
        raise SystemExit("use either --seed or --all, not both")
    if args.seed:
        try:
            return (get_seed(args.seed),)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    return PILOT_SEEDS


def _plain_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_static(args: argparse.Namespace, seeds: tuple[PilotSeed, ...]) -> dict[str, Any]:
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    if not args.semantic_scholar_config.is_file():
        raise SystemExit(f"Semantic Scholar config does not exist: {args.semantic_scholar_config}")
    paths = resolve_seed_paths(args.input_dir, seeds)
    missing = [short_id for short_id, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"configured pilot Markdown files are missing: {', '.join(missing)}")
    if args.max_steps <= 0 or args.max_searches < 0:
        raise SystemExit("--max-steps must be positive and --max-searches non-negative")
    return {
        "input_dir": str(args.input_dir.resolve()),
        "semantic_scholar_config": str(args.semantic_scholar_config.resolve()),
        "seed_count": len(seeds),
        "seeds": [
            {
                "seed_id": seed.short_id,
                "paper_id": seed.paper_id,
                "category": seed.category,
                "source_md": str(paths[seed.short_id].resolve()),
            }
            for seed in seeds
        ],
        "max_steps": args.max_steps,
        "max_searches": args.max_searches,
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


def _status_from_metadata(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid"
    return value.get("status") if isinstance(value, dict) else "invalid"


def _branch_ready(branch: Path) -> bool:
    status = _status_from_metadata(branch / "run_metadata.json")
    return status in {"committed", "stopped"} and (branch / "final_proposal.md").is_file()


def _ensure_branch_available(branch: Path, *, overwrite: bool) -> bool:
    if not branch.exists():
        return False
    if _branch_ready(branch) and not overwrite:
        return True
    if not overwrite:
        raise SystemExit(
            f"incomplete Flat output exists: {branch}; use --overwrite to rerun it"
        )
    return False


def _write_top_manifest(args: argparse.Namespace, static: dict[str, Any], seeds: tuple[PilotSeed, ...]) -> None:
    path = args.output_dir / "manifest.json"
    if path.exists() and not args.overwrite:
        return
    payload = {
        "experiment": "state_evolution_ablation_pilot8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "systems": {"full": "read_only_historical_reference", "flat": "new_run"},
        "full_source_dir": str(args.full_output_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "selected_seeds": [seed.short_id for seed in seeds],
        "static_config": static,
        "flat_config": {
            "max_steps": args.max_steps,
            "max_searches": args.max_searches,
            "workers": args.workers,
            "semantic_scholar_config": str(args.semantic_scholar_config.resolve()),
        },
        "full_api_calls_in_this_experiment": 0,
    }
    _plain_write(path, payload)


def _run_full_reference(seed: PilotSeed, args: argparse.Namespace) -> str:
    branch = args.output_dir / seed.short_id / "full"
    if branch.is_dir() and (branch / "full_reference.json").is_file() and not args.overwrite:
        return "skipped_existing_reference"
    try:
        write_full_reference(seed, args.full_output_dir, branch)
        return "referenced"
    except FullReferenceError as exc:
        branch.mkdir(parents=True, exist_ok=True)
        _plain_write(branch / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        return "invalid_full_reference"


def _run_flat(seed: PilotSeed, args: argparse.Namespace) -> str:
    branch = args.output_dir / seed.short_id / "flat"
    if _ensure_branch_available(branch, overwrite=args.overwrite):
        return "skipped_completed"
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
        # FlatRunner writes its own error artifact after initialization.  This
        # also covers failures during preflight/initialization.
        branch.mkdir(parents=True, exist_ok=True)
        _plain_write(branch / "error.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        return "failed"


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    seeds = _selected(args)
    static = _check_static(args, seeds)
    ref_results = {}
    for seed in seeds:
        try:
            ref_results[seed.short_id] = inspect_full_reference(seed, args.full_output_dir)["status"]
        except FullReferenceError as exc:
            ref_results[seed.short_id] = f"invalid: {exc}"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "system": args.system, "workers": args.workers, "output_dir": str(args.output_dir), "full_output_dir": str(args.full_output_dir), "static": static, "full_reference_status": ref_results}, ensure_ascii=False, indent=2))
        return 0 if all(not str(value).startswith("invalid:") for value in ref_results.values()) else 1
    invalid_references = {
        short_id: value for short_id, value in ref_results.items()
        if str(value).startswith("invalid:")
    }
    if invalid_references:
        raise SystemExit(
            "historical Full references are not all compatible; refusing to run Flat: "
            + "; ".join(f"{key}: {value}" for key, value in invalid_references.items())
        )
    if args.system in {"flat", "both"} and not args.allow_real_services:
        raise SystemExit("real Flat pilot requires --allow-real-services")
    if args.system in {"flat", "both"}:
        try:
            runtime_config = _check_real_config(args)
        except Exception as exc:
            raise SystemExit(f"real provider configuration is invalid: {exc}") from exc
    else:
        runtime_config = {"real_services": False, "full_is_read_only": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_top_manifest(args, static, seeds)
    outcomes: dict[str, dict[str, str]] = {}
    try:
        for seed in seeds:
            outcomes[seed.short_id] = {}
            if args.system in {"full", "both"}:
                outcomes[seed.short_id]["full"] = _run_full_reference(seed, args)

        if args.system in {"flat", "both"}:
            # Validate every branch before starting workers so an output-path
            # error cannot leave a partially launched batch.
            for seed in seeds:
                _ensure_branch_available(
                    args.output_dir / seed.short_id / "flat",
                    overwrite=args.overwrite,
                )
            future_to_seed = {}
            executor = ProcessPoolExecutor(max_workers=args.workers)
            try:
                for seed in seeds:
                    future_to_seed[executor.submit(_run_flat, seed, args)] = seed
                finished = 0
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    outcomes[seed.short_id]["flat"] = future.result()
                    finished += 1
                    print(
                        f"[Flat {finished}/{len(seeds)}] {seed.short_id}: "
                        f"{outcomes[seed.short_id]['flat']}",
                        flush=True,
                    )
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    except KeyboardInterrupt:
        print("Interrupted by user; completed artifacts were retained.", file=sys.stderr)
        raise
    _plain_write(args.output_dir / "pilot_run_summary.json", {
        "system": args.system,
        "workers": args.workers,
        "runtime_config": runtime_config,
        "outcomes": outcomes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"output_dir": str(args.output_dir), "outcomes": outcomes}, ensure_ascii=False, indent=2))
    return 0 if all(value.get("flat") not in {"failed", "invalid"} for value in outcomes.values() if "flat" in value) else 1


if __name__ == "__main__":
    raise SystemExit(main())
