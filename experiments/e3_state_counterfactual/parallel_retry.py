"""Paper-level parallel retry helpers for E3 unified verification.

This module deliberately reuses :func:`run_unified_case`.  The two branches
inside one case therefore retain their existing order and isolation; only
different paper cases are scheduled concurrently.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from research_agent.llm.config import DeepSeekConfig
from research_agent.modules.related_work_verification import VerificationConfig
from research_agent.serialization import write_json

from .unified import UnifiedVerificationCase, run_unified_case


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_has_both_branch_artifacts(output_dir: Path, paper_id: str) -> bool:
    """Return whether both unified branch results are already materialized."""

    case_dir = output_dir / "unified_verification" / paper_id
    return all(
        (case_dir / f"unified_verification_{branch}.json").is_file()
        for branch in ("targeted", "global")
    )


def select_incomplete_cases(
    cases: Iterable[UnifiedVerificationCase],
    output_dir: str | Path,
) -> list[UnifiedVerificationCase]:
    """Select only cases with at least one missing unified branch artifact.

    Selection is deterministic by paper id.  A successful branch in a
    partially completed case is not removed or rerun: ``run_unified_case``
    will skip it using its normal resume behavior.
    """

    root = Path(output_dir)
    return sorted(
        (
            case
            for case in cases
            if not case_has_both_branch_artifacts(root, case.paper_id)
        ),
        key=lambda case: case.paper_id,
    )


def _worker(
    case: UnifiedVerificationCase,
    output_dir: Path,
    *,
    deepseek: DeepSeekConfig,
    verification_config: VerificationConfig,
    case_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run one case and isolate unexpected infrastructure exceptions."""

    try:
        return case_runner(
            case,
            output_dir,
            deepseek=deepseek,
            verification_config=verification_config,
            overwrite=False,
        )
    except Exception as exc:
        # run_unified_case normally converts branch errors into per-case
        # artifacts.  This guard covers an unexpected worker-level exception
        # without cancelling other papers and leaves a diagnostic artifact.
        case_dir = output_dir / "unified_verification" / case.paper_id
        case_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "paper_id": case.paper_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
            "updated_at": _now(),
        }
        write_json(
            case_dir / "error_retry_worker.json",
            "e3_unified_retry_worker_error",
            error,
        )
        return error


def run_incomplete_cases_parallel(
    cases: Iterable[UnifiedVerificationCase],
    output_dir: str | Path,
    *,
    deepseek: DeepSeekConfig,
    verification_config: VerificationConfig,
    workers: int,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
    case_runner: Callable[..., dict[str, Any]] = run_unified_case,
) -> list[dict[str, Any]]:
    """Run incomplete E3 cases concurrently at paper granularity.

    ``case_runner`` is injectable for offline tests.  The default runner is
    the production E3 unified verifier and is called with ``overwrite=False``
    so completed Targeted/Global branches are never rerun accidentally.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    selected = sorted(cases, key=lambda case: case.paper_id)
    if not selected:
        return []

    root = Path(output_dir)
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="e3-unified-retry",
    )
    futures: dict[Future[dict[str, Any]], UnifiedVerificationCase] = {}
    try:
        for case in selected:
            future = executor.submit(
                _worker,
                case,
                root,
                deepseek=deepseek,
                verification_config=verification_config,
                case_runner=case_runner,
            )
            futures[future] = case
        results: list[dict[str, Any]] = []
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if progress is not None:
                progress(result, completed, len(selected))
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return sorted(results, key=lambda item: item.get("paper_id", ""))


__all__ = [
    "case_has_both_branch_artifacts",
    "run_incomplete_cases_parallel",
    "select_incomplete_cases",
]
