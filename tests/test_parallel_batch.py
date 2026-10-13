"""Offline regression tests for paper-level Ours batch parallelism."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research_agent.controller.budget import BudgetTracker
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.models import PaperEvaluation
from research_agent.retrieval.arxiv_cache import ArxivCache
from research_agent.retrieval.semantic_scholar_cache import SemanticScholarCache
from research_agent.runner import (
    BatchEvaluationConfig,
    BatchEvaluationRunner,
    ParallelBatchConfig,
    ParallelBatchEvaluationRunner,
    TrajectoryModules,
    TrajectoryRunner,
)


def _markdown(root: Path, paper_id: str) -> Path:
    path = root / f"{paper_id}.md"
    path.write_text(
        f"# Generic paper {paper_id}\n\n"
        "## Abstract\n\nA supplied abstract.\n\n"
        "## Method\n\nA supplied method.\n\n"
        "## Experiments\n\nA supplied experiment.\n",
        encoding="utf-8",
    )
    return path


def _offline_full_executor(markdown_path: Path, output_dir: Path):
    # This reuses the deterministic generic M11 fixture and creates all
    # per-paper state, memory, trajectory, replay, and final proposal locally.
    from test_m11 import modules, seed

    seed_context = seed().model_copy(update={"paper_id": markdown_path.stem})
    reading = build_full_paper_reading_context(parse_markdown_file(markdown_path))
    return TrajectoryRunner(
        modules(),
        budget=BudgetTracker(max_steps=5, max_searches=1),
        output_dir=output_dir,
    ).run(seed_context, full_paper_reading_context=reading)


def _semantic_payload() -> dict[str, object]:
    return {"papers": [{"paperId": "paper-1", "title": "A cached paper"}]}


class ParallelBatchTests(unittest.TestCase):
    def test_workers_one_and_four_have_same_workflow_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_a", "paper_b", "paper_c"):
                _markdown(input_dir, paper_id)

            serial_output = root / "serial"
            serial_config = BatchEvaluationConfig(
                input_dir=input_dir,
                output_dir=serial_output,
                batch_name="serial",
            )
            serial_report = BatchEvaluationRunner(
                serial_config,
                _offline_full_executor,
            ).run()

            parallel_output = root / "parallel"
            parallel_config = BatchEvaluationConfig(
                input_dir=input_dir,
                output_dir=parallel_output,
                batch_name="parallel",
            )
            parallel_report = ParallelBatchEvaluationRunner(
                parallel_config,
                _offline_full_executor,
                ParallelBatchConfig(workers=4),
            ).run()

            self.assertEqual(serial_report.selected_paper_ids, parallel_report.selected_paper_ids)
            self.assertEqual(serial_report.completed_count, parallel_report.completed_count)
            self.assertEqual(serial_report.failed_count, parallel_report.failed_count)
            self.assertEqual(serial_report.stopped_count, parallel_report.stopped_count)
            serial_by_id = {paper.paper_id: paper for paper in serial_report.papers}
            parallel_by_id = {paper.paper_id: paper for paper in parallel_report.papers}
            for paper_id in serial_report.selected_paper_ids:
                serial_paper = serial_by_id[paper_id].model_dump(mode="json")
                parallel_paper = parallel_by_id[paper_id].model_dump(mode="json")
                for payload in (serial_paper, parallel_paper):
                    payload.pop("markdown_path", None)
                    payload.pop("output_dir", None)
                self.assertEqual(serial_paper, parallel_paper)

                serial_dir = serial_output / paper_id
                parallel_dir = parallel_output / paper_id
                for artifact in (
                    "research_state.json",
                    "research_memory.json",
                    "final_research_proposal.json",
                ):
                    self.assertEqual(
                        json.loads((serial_dir / artifact).read_text(encoding="utf-8")),
                        json.loads((parallel_dir / artifact).read_text(encoding="utf-8")),
                    )
                serial_events = [
                    json.loads(line)["payload"]
                    for line in (serial_dir / "trajectory.jsonl").read_text().splitlines()
                ]
                parallel_events = [
                    json.loads(line)["payload"]
                    for line in (parallel_dir / "trajectory.jsonl").read_text().splitlines()
                ]
                for events in (serial_events, parallel_events):
                    for event in events:
                        event.pop("timestamp", None)
                self.assertEqual(serial_events, parallel_events)

    def test_parallel_completion_order_does_not_change_report_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_a", "paper_b", "paper_c"):
                _markdown(input_dir, paper_id)

            def delayed_executor(markdown_path: Path, output_dir: Path):
                time.sleep({"paper_a": 0.03, "paper_b": 0.01, "paper_c": 0.0}[markdown_path.stem])
                return _offline_full_executor(markdown_path, output_dir)

            report = ParallelBatchEvaluationRunner(
                BatchEvaluationConfig(input_dir=input_dir, output_dir=root / "batch"),
                delayed_executor,
                ParallelBatchConfig(workers=3),
            ).run()
            self.assertEqual(report.selected_paper_ids, ["paper_a", "paper_b", "paper_c"])
            self.assertEqual([paper.paper_id for paper in report.papers], report.selected_paper_ids)

    def test_worker_failure_isolated_and_report_written_by_main_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_ok", "paper_fail", "paper_later"):
                _markdown(input_dir, paper_id)

            def executor(markdown_path: Path, output_dir: Path):
                if markdown_path.stem == "paper_fail":
                    raise RuntimeError("synthetic worker failure")
                return _offline_full_executor(markdown_path, output_dir)

            output_dir = root / "batch"
            report = ParallelBatchEvaluationRunner(
                BatchEvaluationConfig(input_dir=input_dir, output_dir=output_dir),
                executor,
                ParallelBatchConfig(workers=2),
            ).run()
            by_id = {paper.paper_id: paper for paper in report.papers}
            self.assertEqual(report.completed_count, 2)
            self.assertEqual(report.failed_count, 1)
            self.assertEqual(by_id["paper_fail"].failure_type, "RuntimeError")
            self.assertTrue((output_dir / "paper_fail" / "batch_error.json").exists())
            self.assertTrue((output_dir / "evaluation_report.json").exists())
            self.assertTrue((output_dir / "paper_later" / "trajectory.jsonl").exists())

    def test_existing_output_is_rejected_and_workers_one_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            _markdown(input_dir, "paper_a")
            output_dir = root / "batch"
            output_dir.mkdir()
            with self.assertRaises(Exception):
                ParallelBatchEvaluationRunner(
                    BatchEvaluationConfig(input_dir=input_dir, output_dir=output_dir),
                    _offline_full_executor,
                    ParallelBatchConfig(workers=1),
                ).run()

    def test_keyboard_interrupt_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            _markdown(input_dir, "paper_a")

            def interrupting_executor(_markdown_path: Path, _output_dir: Path):
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                ParallelBatchEvaluationRunner(
                    BatchEvaluationConfig(input_dir=input_dir, output_dir=root / "batch"),
                    interrupting_executor,
                    ParallelBatchConfig(workers=1),
                ).run()

    def test_shared_semantic_scholar_cache_serializes_same_key_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SemanticScholarCache(Path(directory) / "semantic")
            request = {"url": "https://example.test/graph/v1/paper/search", "query": "same"}
            calls = 0
            calls_lock = threading.Lock()

            def fetcher():
                nonlocal calls
                with calls_lock:
                    calls += 1
                time.sleep(0.02)
                return _semantic_payload()

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(lambda _: cache.get_or_fetch("search", request, fetcher), range(4))
                )
            self.assertEqual(calls, 1)
            self.assertEqual([result[0] for result in results], [_semantic_payload()] * 4)
            self.assertEqual(sum(not result[1] for result in results), 1)

    def test_shared_arxiv_cache_serializes_same_key_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ArxivCache(Path(directory) / "arxiv")
            request = {"query": "same"}
            calls = 0
            calls_lock = threading.Lock()

            def fetcher():
                nonlocal calls
                with calls_lock:
                    calls += 1
                time.sleep(0.02)
                return "<feed><entry>cached</entry></feed>"

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(lambda _: cache.get_or_fetch("search", request, fetcher), range(4))
                )
            self.assertEqual(calls, 1)
            self.assertEqual([result[0] for result in results], [results[0][0]] * 4)
            self.assertEqual(sum(not result[1] for result in results), 1)


if __name__ == "__main__":
    unittest.main()
