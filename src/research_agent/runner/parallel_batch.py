"""Paper-level parallel orchestration for the existing Ours workflow.

This module delegates every paper to the existing ``PaperExecutor``.  It does
not construct or modify a scientific module, Research State, Research Memory,
budget, or trajectory.  The only shared mutable object is the thread pool;
all paper artifacts and execution objects remain per-paper.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serialization import write_json
from .batch_evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationError,
    PaperExecutor,
    build_evaluation_report,
    summarize_paper_output,
)
from ..models.evaluation import EvaluationReport, PaperEvaluation
from ..ingestion.markdown import parse_markdown_file


@dataclass(frozen=True)
class ParallelBatchConfig:
    """Execution-only options layered over the existing batch config."""

    workers: int = 4
    expected_papers: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        if self.expected_papers is not None and self.expected_papers <= 0:
            raise ValueError("expected_papers must be positive when provided")


def _batch_error_payload(paper_id: str, error: Exception) -> dict[str, str]:
    return {
        "paper_id": paper_id,
        "error_type": type(error).__name__,
        "error": str(error)[:2_000],
    }


class ParallelBatchEvaluationRunner:
    """Run independent papers concurrently and aggregate only on the main thread."""

    def __init__(
        self,
        batch_config: BatchEvaluationConfig,
        executor: PaperExecutor,
        parallel_config: ParallelBatchConfig | None = None,
    ) -> None:
        self.batch_config = batch_config
        self.executor = executor
        self.parallel_config = parallel_config or ParallelBatchConfig()

    def discover_papers(self) -> list[Path]:
        """Discover the same sorted cohort as the existing serial runner."""

        input_dir = self.batch_config.input_dir
        if not input_dir.is_dir():
            raise BatchEvaluationError(f"batch input directory does not exist: {input_dir}")
        papers = sorted(
            path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
        )
        if not papers:
            raise BatchEvaluationError(
                f"batch input directory contains no Markdown files: {input_dir}"
            )
        if self.batch_config.max_papers is not None:
            # Preserve the existing serial runner's seeded selection behavior.
            from .batch_evaluation import BatchEvaluationRunner

            papers = BatchEvaluationRunner(self.batch_config, self.executor).discover_papers()
        expected = self.parallel_config.expected_papers
        if expected is not None and len(papers) != expected:
            raise BatchEvaluationError(
                f"expected exactly {expected} Markdown papers, found {len(papers)}: {input_dir}"
            )
        return papers

    @staticmethod
    def _paper_id(path: Path) -> str:
        paper_id = parse_markdown_file(path).paper_id
        if not paper_id or Path(paper_id).name != paper_id:
            raise BatchEvaluationError(f"unsafe paper id derived from Markdown filename: {paper_id!r}")
        return paper_id

    def _preflight(self, papers: list[Path]) -> list[tuple[Path, str, Path]]:
        paper_ids = [self._paper_id(path) for path in papers]
        if len(set(paper_ids)) != len(paper_ids):
            raise BatchEvaluationError("selected Markdown files contain duplicate paper IDs")
        output_dir = self.batch_config.output_dir
        if output_dir.exists() and not self.batch_config.allow_overwrite:
            raise BatchEvaluationError(
                f"refusing to overwrite existing batch output directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=self.batch_config.allow_overwrite)
        paper_outputs = [output_dir / paper_id for paper_id in paper_ids]
        if not self.batch_config.allow_overwrite:
            existing = [path for path in paper_outputs if path.exists()]
            if existing:
                raise BatchEvaluationError(
                    "paper output already exists: " + ", ".join(map(str, existing))
                )
        return list(zip(papers, paper_ids, paper_outputs))

    def _run_one(
        self,
        markdown_path: Path,
        paper_id: str,
        paper_output: Path,
    ) -> PaperEvaluation:
        error: Exception | None = None
        try:
            # The executor is the existing per-paper pilot adapter.  It creates
            # its own TrajectoryRunner, State, Memory, providers, and counters.
            self.executor(markdown_path, paper_output)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error = exc
            paper_output.mkdir(parents=True, exist_ok=True)
            write_json(
                paper_output / "batch_error.json",
                "batch_error",
                _batch_error_payload(paper_id, exc),
            )
        return summarize_paper_output(
            markdown_path,
            paper_output,
            executor_error=error,
        )

    def run(self) -> EvaluationReport:
        papers = self.discover_papers()
        assignments = self._preflight(papers)
        results: list[PaperEvaluation] = []
        executor = ThreadPoolExecutor(
            max_workers=self.parallel_config.workers,
            thread_name_prefix="ours-paper",
        )
        futures: dict[Future[PaperEvaluation], tuple[Path, str, Path]] = {}
        interrupted = False
        try:
            for markdown_path, paper_id, paper_output in assignments:
                future = executor.submit(
                    self._run_one,
                    markdown_path,
                    paper_id,
                    paper_output,
                )
                futures[future] = (markdown_path, paper_id, paper_output)
            for future in as_completed(futures):
                # KeyboardInterrupt is deliberately not caught here; it is
                # handled by the outer block and re-raised to the caller.
                results.append(future.result())
        except KeyboardInterrupt:
            interrupted = True
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

        by_id = {paper.paper_id: paper for paper in results}
        ordered_results = [by_id[paper_id] for _, paper_id, _ in assignments]
        ordered_ids = [paper_id for _, paper_id, _ in assignments]
        report = build_evaluation_report(
            self.batch_config,
            ordered_results,
            ordered_ids,
        )
        # Only the main thread reaches this write.  The existing report schema
        # and all per-paper artifact schemas remain unchanged.
        write_json(
            self.batch_config.output_dir / "evaluation_report.json",
            "evaluation_report",
            report,
        )
        return report


__all__ = ["ParallelBatchConfig", "ParallelBatchEvaluationRunner"]
