#!/usr/bin/env python3
"""Run the frozen Direct-Neutral baseline on the formal 40-paper cohort."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.runner.direct_llm_neutral import (
    CountingJsonTransport,
    DirectNeutralRunner,
    direct_neutral_prompt_hash,
    discover_neutral_inputs,
    run_direct_neutral_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel one-shot Direct-Neutral 40-paper baseline runner"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "nlp",
        help="directory containing exactly 40 Markdown papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40",
    )
    parser.add_argument("--batch-name", default="direct_llm_neutral_nlp40")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=None, help="optional DeepSeek model override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _safe_config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    return config


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive when provided")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output path: {args.output_dir}")

    try:
        paths = discover_neutral_inputs(args.input_dir)
        prompt_hash = direct_neutral_prompt_hash()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    config = _safe_config(args)
    config_ready = True
    try:
        config.validate_for_request()
    except Exception:
        config_ready = False
        if not args.dry_run:
            raise

    if args.dry_run:
        print(
            json.dumps(
                {
                    "baseline_name": "Direct-Neutral",
                    "dry_run": True,
                    "input_dir": str(args.input_dir),
                    "output_dir": str(args.output_dir),
                    "total_papers": len(paths),
                    "workers": args.workers,
                    "model": config.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "prompt_hash": prompt_hash,
                    "deepseek_config_ready": config_ready,
                    "papers": [path.name for path in paths],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    def factory() -> DirectNeutralRunner:
        transport = CountingJsonTransport()
        client = DeepSeekClient(config, transport=transport)
        return DirectNeutralRunner(
            client,
            raw_response_source=transport,
            physical_attempt_source=transport,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=config.model,
            provider="deepseek",
        )

    report = run_direct_neutral_batch(
        paths,
        args.output_dir,
        factory,
        batch_name=args.batch_name,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
