#!/usr/bin/env python3
"""Retry the failed Direct-LLM2 papers in an assembled output directory.

The target papers are discovered from ``evaluation_report.json`` instead of
being hard-coded.  The script requires exactly four failed papers for this
retry batch, runs them with bounded paper-level concurrency, and replaces only
those failed paper directories after successful retry.  Previous failed
artifacts are copied to ``retry_history`` before replacement.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from research_agent.ingestion.markdown import (  # noqa: E402
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.llm.config import DeepSeekConfig  # noqa: E402
from research_agent.llm.deepseek import DeepSeekClient  # noqa: E402
from research_agent.runner.direct_llm_neutral import CountingJsonTransport  # noqa: E402
from research_agent.serialization import write_json  # noqa: E402

from run_direct_llm2_p001_p010_parallel import (  # noqa: E402
    DirectLLM2SingleRunner,
    _error_payload,
    _prompt_hash,
)
from run_direct_llm2_p011_p110_parallel_assemble import _token_stats  # noqa: E402


DEFAULT_WORKERS = 4
DEFAULT_INPUT_DIRS = (REPO_ROOT / "data" / "nlp", REPO_ROOT / "data" / "nlp3")
DEFAULT_OUTPUT_DIR = Path("/home/tmh/final_baseline/output_direct2")
DEFAULT_BATCH_NAME = "direct_llm2_retry_failed_4"
PAPER_ID_PATTERN = re.compile(r"(?:^|_)P(?P<number>\d{3})(?:_|$)")
EXPECTED_FAILED_COUNT = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry the four failed Direct-LLM2 papers in output_direct2"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="input directory; repeat for multiple roots (defaults to data/nlp and data/nlp3)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument("--model", default=None, help="optional DeepSeek model override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload", value)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object payload in {path}")
    return payload


def _paper_number(value: str | Path) -> int | None:
    match = PAPER_ID_PATTERN.search(Path(value).stem)
    return int(match.group("number")) if match else None


def _config(args: argparse.Namespace) -> DeepSeekConfig:
    config = DeepSeekConfig.from_env()
    if args.model:
        config = replace(config, model=args.model)
    return config


def _discover_failed(output_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    report_path = output_dir / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"evaluation report does not exist: {report_path}")
    report = _read_payload(report_path)
    rows = report.get("papers")
    if not isinstance(rows, list):
        raise ValueError(f"evaluation report has no papers list: {report_path}")
    failed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "failed":
            continue
        number = _paper_number(str(row.get("paper_id", "")))
        if number is None:
            raise ValueError(f"failed report row has no parseable paper_id: {row}")
        if number in failed:
            raise ValueError(f"duplicate failed paper P{number:03d} in report")
        failed[number] = row
    if len(failed) != EXPECTED_FAILED_COUNT:
        names = ", ".join(f"P{n:03d}" for n in sorted(failed)) or "none"
        raise ValueError(
            f"expected exactly {EXPECTED_FAILED_COUNT} failed papers, found {len(failed)}: {names}"
        )
    return failed, report


def _find_inputs(input_dirs: list[Path], numbers: set[int]) -> dict[int, Path]:
    by_number: dict[int, Path] = {}
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        for path in sorted(input_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".md":
                continue
            number = _paper_number(path)
            if number not in numbers:
                continue
            if number in by_number:
                raise ValueError(f"duplicate P{number:03d} Markdown input: {path}")
            parsed = parse_markdown_file(path)
            build_full_paper_reading_context(parsed)
            by_number[number] = path.resolve()
    missing = sorted(numbers - set(by_number))
    if missing:
        raise ValueError("missing retry inputs: " + ", ".join(f"P{n:03d}" for n in missing))
    return by_number


def _validate_existing_dirs(output_dir: Path, failed: dict[int, dict[str, Any]]) -> None:
    papers_root = output_dir / "papers"
    if not papers_root.is_dir():
        raise FileNotFoundError(f"paper output directory does not exist: {papers_root}")
    current_prompt_hash = _prompt_hash()
    for number, row in failed.items():
        paper_id = str(row["paper_id"])
        paper_dir = papers_root / paper_id
        if not paper_dir.is_dir():
            raise FileNotFoundError(f"failed paper directory does not exist: {paper_dir}")
        manifest_path = paper_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"failed paper manifest does not exist: {manifest_path}")
        manifest = _read_payload(manifest_path)
        if manifest.get("status") != "failed":
            raise ValueError(f"P{number:03d} directory is not marked failed")
        if manifest.get("prompt_hash") not in (None, current_prompt_hash):
            raise ValueError(f"P{number:03d} uses a different Direct-LLM2 prompt")


def _validate(args: argparse.Namespace) -> tuple[dict[int, Path], dict[int, dict[str, Any]], dict[str, Any], DeepSeekConfig]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when provided")
    if not args.batch_name or "/" in args.batch_name or "\\" in args.batch_name:
        raise ValueError("--batch-name must be a simple filename component")
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {args.output_dir}")

    failed, report = _discover_failed(args.output_dir)
    _validate_existing_dirs(args.output_dir, failed)
    input_dirs = list(args.input_dir) if args.input_dir else list(DEFAULT_INPUT_DIRS)
    inputs = _find_inputs(input_dirs, set(failed))

    for path in (
        args.output_dir / "retry_history" / args.batch_name,
        args.output_dir / "retry_attempts" / args.batch_name,
        args.output_dir / ".retry_work" / args.batch_name,
    ):
        if path.exists():
            raise FileExistsError(f"retry batch path already exists: {path}")

    config = _config(args)
    config_ready = True
    try:
        config.validate_for_request()
    except Exception:
        config_ready = False
        if not args.dry_run:
            raise
    setattr(args, "config_ready", config_ready)
    setattr(args, "input_dirs", input_dirs)
    return inputs, failed, report, config


def _unexpected_failure(source: Path, destination: Path, error: Exception) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    detail = _error_payload(error)
    write_json(destination / "error.json", "direct_llm2_error", detail)
    return {
        "paper_id": source.stem,
        "source_md": str(source),
        "baseline_name": "Direct-LLM2",
        "status": "failed",
        "logical_scientific_generation_calls": 0,
        "physical_api_attempts": 0,
        "output_dir": str(destination),
        "error_type": detail["error_type"],
        "error": detail,
    }


def _run_retry(
    inputs: dict[int, Path],
    work_root: Path,
    args: argparse.Namespace,
    config: DeepSeekConfig,
) -> list[dict[str, Any]]:
    work_root.mkdir(parents=True, exist_ok=False)
    futures: dict[Future[dict[str, Any]], tuple[Path, Path]] = {}
    rows: list[dict[str, Any]] = []

    def factory() -> DirectLLM2SingleRunner:
        transport = CountingJsonTransport()
        client = DeepSeekClient(config, transport=transport)
        return DirectLLM2SingleRunner(
            client,
            transport,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=config.model,
        )

    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="direct-llm2-retry")
    interrupted = False
    try:
        for number in sorted(inputs):
            source = inputs[number]
            destination = work_root / source.stem
            futures[executor.submit(factory().run, source, destination)] = (source, destination)
        completed = 0
        for future in as_completed(futures):
            source, destination = futures[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                row = _unexpected_failure(source, destination, exc)
            rows.append(row)
            print(f"[retry {completed}/{len(inputs)}] {source.stem}: {row.get('status', 'failed')}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return rows


def _commit_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    batch_name: str,
) -> list[dict[str, Any]]:
    work_root = output_dir / ".retry_work" / batch_name
    history_root = output_dir / "retry_history" / batch_name
    attempts_root = output_dir / "retry_attempts" / batch_name
    history_root.mkdir(parents=True, exist_ok=False)
    attempts_root.mkdir(parents=True, exist_ok=False)
    papers_root = output_dir / "papers"
    committed: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda item: str(item.get("paper_id", ""))):
        paper_id = str(row["paper_id"])
        retry_dir = work_root / paper_id
        final_dir = papers_root / paper_id
        if row.get("status") == "succeeded":
            backup_dir = history_root / paper_id
            shutil.copytree(final_dir, backup_dir)
            shutil.rmtree(final_dir)
            shutil.move(str(retry_dir), str(final_dir))
            row["output_dir"] = str(final_dir)
            row["retry_history_dir"] = str(backup_dir)
            print(f"[replace] {paper_id}: succeeded -> {final_dir}", flush=True)
        else:
            attempt_dir = attempts_root / paper_id
            shutil.move(str(retry_dir), str(attempt_dir))
            row["retry_output_dir"] = str(attempt_dir)
            row["output_dir"] = str(final_dir)
            print(f"[keep-failed] {paper_id}: retry failed; original preserved", flush=True)
        committed.append(row)
    shutil.rmtree(work_root)
    return committed


def _update_report(
    report: dict[str, Any],
    retry_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: str,
    started_clock: float,
) -> dict[str, Any]:
    by_id = {str(row.get("paper_id")): dict(row) for row in report.get("papers", [])}
    for row in retry_rows:
        paper_id = str(row["paper_id"])
        old = by_id.get(paper_id, {})
        merged = dict(old)
        merged.update(row)
        merged["retry_count"] = int(old.get("retry_count", 0) or 0) + 1
        merged["last_retry_batch"] = args.batch_name
        by_id[paper_id] = merged
    rows = sorted(by_id.values(), key=lambda row: str(row.get("paper_id", "")))
    failed = sum(row.get("status") == "failed" for row in rows)
    report = dict(report)
    report.update({
        "succeeded": sum(row.get("status") == "succeeded" for row in rows),
        "failed": failed,
        "logical_llm_calls": sum(int(row.get("logical_scientific_generation_calls", 0) or 0) for row in rows),
        "physical_api_attempts": sum(int(row.get("physical_api_attempts", 0) or 0) for row in rows),
        "finished_at": _utc_now(),
        "retry_batch_name": args.batch_name,
        "retry_started_at": started_at,
        "retry_wall_clock_seconds": time.monotonic() - started_clock,
        "papers": rows,
    })
    report.update(_token_stats(rows))
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        inputs, failed, report, config = _validate(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    failed_ids = [f"P{number:03d}" for number in sorted(failed)]
    if args.dry_run:
        print(json.dumps({
            "baseline_name": "Direct-LLM2",
            "dry_run": True,
            "failed_papers": failed_ids,
            "paper_count": len(inputs),
            "workers": args.workers,
            "input_dirs": [str(path) for path in args.input_dirs],
            "output_dir": str(args.output_dir),
            "model": config.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_hash": _prompt_hash(),
            "deepseek_config_ready": args.config_ready,
        }, ensure_ascii=False, indent=2))
        return 0

    started_at = _utc_now()
    started_clock = time.monotonic()
    work_root = args.output_dir / ".retry_work" / args.batch_name
    retry_rows = _run_retry(inputs, work_root, args, config)
    committed_rows = _commit_results(retry_rows, args.output_dir, args.batch_name)
    updated = _update_report(report, committed_rows, args, started_at, started_clock)
    write_json(args.output_dir / "evaluation_report.json", "direct_llm2_evaluation_report", updated)
    write_json(
        args.output_dir / "assembly_manifest.json",
        "direct_llm2_assembly_manifest",
        {
            "baseline_name": "Direct-LLM2",
            "output_dir": str(args.output_dir),
            "last_retry_batch": args.batch_name,
            "retried_papers": failed_ids,
            "status": "succeeded" if updated["failed"] == 0 else "completed_with_failures",
            "prompt_hash": _prompt_hash(),
        },
    )
    print(json.dumps({
        "retried_papers": failed_ids,
        "succeeded_after_retry": sum(row.get("status") == "succeeded" for row in committed_rows),
        "failed_after_retry": sum(row.get("status") == "failed" for row in committed_rows),
        "batch_succeeded": updated["succeeded"],
        "batch_failed": updated["failed"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0 if updated["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
