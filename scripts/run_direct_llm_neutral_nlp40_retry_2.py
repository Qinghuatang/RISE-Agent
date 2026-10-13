#!/usr/bin/env python3
"""Retry the two remaining failed papers from the Direct-Neutral retry run.

The script selects incomplete rows from the latest six-paper report and
delegates execution to the existing one-shot Direct-Neutral batch runner.
The previous retry directory is read only and is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.ingestion.markdown import build_full_paper_reading_context, parse_markdown_file  # noqa: E402
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.runner.direct_llm_neutral import (  # noqa: E402
    CountingJsonTransport,
    DirectNeutralRunner,
    direct_neutral_prompt_hash,
    run_direct_neutral_batch,
)


EXPECTED_INPUT_PAPERS = 80
EXPECTED_SOURCE_PAPERS = 6
EXPECTED_RETRY_PAPERS = 2
DEFAULT_WORKERS = 2

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_SOURCE_RUN = REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_retry_6"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "direct_llm_neutral_nlp40_retry_2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry the two incomplete papers from Direct-Neutral retry_6"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default="direct_llm_neutral_nlp40_retry_2")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_source_report(source_run: Path) -> dict[str, Any]:
    report_path = source_run / "evaluation_report.json"
    if not report_path.is_file():
        raise ValueError(f"missing Direct-Neutral source report: {report_path}")
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Direct-Neutral source report: {report_path}") from exc
    payload = document.get("payload", document) if isinstance(document, dict) else None
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise ValueError(f"invalid Direct-Neutral source report: {report_path}")
    if payload.get("total_papers") != EXPECTED_SOURCE_PAPERS:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_PAPERS} papers in source report, "
            f"found {payload.get('total_papers')}"
        )
    return payload


def _discover_inputs(input_dir: Path) -> dict[str, Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
    )
    if len(paths) != EXPECTED_INPUT_PAPERS:
        raise ValueError(
            f"expected exactly {EXPECTED_INPUT_PAPERS} Markdown papers, found {len(paths)}"
        )
    result: dict[str, Path] = {}
    for path in paths:
        parsed = parse_markdown_file(path)
        build_full_paper_reading_context(parsed)
        if parsed.paper_id != path.stem:
            raise ValueError(f"paper_id does not match Markdown stem: {path.name}")
        if parsed.paper_id in result:
            raise ValueError(f"duplicate paper ID: {parsed.paper_id}")
        result[parsed.paper_id] = path.resolve()
    return result


def _select_retry_paths(
    input_by_id: dict[str, Path], source_run: Path
) -> tuple[list[Path], list[str]]:
    payload = _read_source_report(source_run)
    retry: list[Path] = []
    seen: set[str] = set()
    for row in payload["papers"]:
        if not isinstance(row, dict) or not isinstance(row.get("paper_id"), str):
            raise ValueError("source report contains a malformed paper row")
        paper_id = row["paper_id"]
        if paper_id in seen:
            raise ValueError(f"duplicate paper ID in source report: {paper_id}")
        seen.add(paper_id)
        source = input_by_id.get(paper_id)
        if source is None:
            raise ValueError(f"source Markdown missing for paper: {paper_id}")
        final_proposal = source_run / paper_id / "final_proposal.json"
        if row.get("status") != "succeeded" or not final_proposal.is_file():
            retry.append(source)

    if len(seen) != EXPECTED_SOURCE_PAPERS:
        raise ValueError(
            f"source report contains {len(seen)} papers, expected {EXPECTED_SOURCE_PAPERS}"
        )
    retry.sort(key=lambda path: path.name)
    if len(retry) != EXPECTED_RETRY_PAPERS:
        raise ValueError(
            f"expected exactly {EXPECTED_RETRY_PAPERS} failed papers, found {len(retry)}"
        )
    return retry, [path.stem for path in retry]


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive when provided")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output path: {args.output_dir}")

    try:
        input_by_id = _discover_inputs(args.input_dir)
        retry_paths, retry_ids = _select_retry_paths(input_by_id, args.source_run)
        prompt_hash = direct_neutral_prompt_hash()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
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
                    "source_run": str(args.source_run),
                    "output_dir": str(args.output_dir),
                    "total_input_papers": len(input_by_id),
                    "retry_papers": retry_ids,
                    "retry_count": len(retry_paths),
                    "workers": args.workers,
                    "model": config.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "prompt_hash": prompt_hash,
                    "deepseek_config_ready": config_ready,
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
        retry_paths,
        args.output_dir,
        factory,
        batch_name=args.batch_name,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
