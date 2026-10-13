"""M13 batch evaluation tests using generic offline trajectories only."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from research_agent.controller.budget import BudgetTracker
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_markdown_file,
)
from research_agent.models import EvaluationReport
from research_agent.runner import (
    BatchEvaluationConfig,
    BatchEvaluationRunner,
    TrajectoryRunner,
)
from research_agent.runner.batch_evaluation import (
    _inspiration_status,
    _selected_inspiration_ref,
)
from research_agent.serialization import write_json


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
    # Reuse the already tested generic M11 module bundle.  This is a fake
    # trajectory: it never initializes DeepSeek, Semantic Scholar, or arXiv.
    from test_m11 import modules, seed

    seed_context = seed().model_copy(update={"paper_id": markdown_path.stem})
    reading = build_full_paper_reading_context(parse_markdown_file(markdown_path))
    return TrajectoryRunner(
        modules(),
        budget=BudgetTracker(max_steps=5, max_searches=1),
        output_dir=output_dir,
    ).run(seed_context, full_paper_reading_context=reading)


class M13BatchEvaluationTests(unittest.TestCase):
    def test_inspiration_statistics_distinguish_available_triggered_found_and_used(self):
        found_event = {
            "event_type": "inspiration_retrieved",
            "result_payload": {
                "grounding_status": "arxiv",
                "result": {"status": "found", "inspirations": [{"id": "x"}]},
            },
        }
        no_useful_event = {
            "event_type": "inspiration_retrieved",
            "result_payload": {
                "grounding_status": "arxiv",
                "result": {"status": "no_useful_inspiration", "inspirations": []},
            },
        }
        revision_event = {
            "event_type": "solution_revised",
            "action": "TARGETED_REVISION",
            "result_payload": {
                "selected_inspiration_ref": "I1:1",
                "revision_result": {"status": "revised", "changed_units": ["C1"]},
            },
        }
        self.assertEqual(_inspiration_status(found_event), "found")
        self.assertEqual(_inspiration_status(no_useful_event), "no_useful_inspiration")
        self.assertIsNone(_selected_inspiration_ref(found_event))
        self.assertEqual(_selected_inspiration_ref(revision_event), "I1:1")

    def test_evaluation_contract_is_closed_and_review_template_is_human_only(self):
        with self.assertRaises(ValidationError):
            EvaluationReport.model_validate(
                {
                    "batch_name": "batch",
                    "input_dir": "in",
                    "output_dir": "out",
                    "paper_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "stopped_count": 0,
                    "token_metadata_complete": True,
                    "unexpected": "not allowed",
                }
            )
        template = json.loads(
            (Path(__file__).parents[1] / "human_review_template.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("gap_quality", template)
        self.assertIn("solution_quality", template)
        self.assertIn("final_proposal_quality", template)
        self.assertIsNone(template["gap_quality"]["method_level"])

    def test_selection_is_seeded_and_records_selected_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_a", "paper_b", "paper_c", "paper_d"):
                _markdown(input_dir, paper_id)
            config = BatchEvaluationConfig(
                input_dir=input_dir,
                output_dir=root / "batch",
                max_papers=2,
                selection_seed=17,
            )
            runner = BatchEvaluationRunner(config, lambda *_: None)
            first = [path.name for path in runner.discover_papers()]
            second = [path.name for path in runner.discover_papers()]
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)

    def test_fake_batch_isolates_trajectories_and_aggregates_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_a", "paper_b", "paper_c"):
                _markdown(input_dir, paper_id)
            output_dir = root / "batch"
            report = BatchEvaluationRunner(
                BatchEvaluationConfig(input_dir=input_dir, output_dir=output_dir),
                _offline_full_executor,
            ).run()

            self.assertEqual(report.paper_count, 3)
            self.assertEqual(report.completed_count, 3)
            self.assertEqual(report.failed_count, 0)
            self.assertEqual(report.selected_paper_ids, ["paper_a", "paper_b", "paper_c"])
            self.assertEqual(report.replay_verified_count, 3)
            self.assertEqual(report.state_consistent_count, 3)
            self.assertEqual(report.memory_consistent_count, 3)
            self.assertTrue((output_dir / "evaluation_report.json").exists())
            for paper_id in report.selected_paper_ids:
                paper_dir = output_dir / paper_id
                self.assertTrue((paper_dir / "trajectory.jsonl").exists())
                self.assertTrue((paper_dir / "research_state.json").exists())
                self.assertTrue((paper_dir / "research_memory.json").exists())
                self.assertTrue((paper_dir / "trajectory_replay.jsonl").exists())
                self.assertEqual(
                    report.papers[report.selected_paper_ids.index(paper_id)].paper_id,
                    paper_id,
                )

    def test_one_paper_failure_does_not_stop_remaining_papers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            for paper_id in ("paper_ok", "paper_fail", "paper_later"):
                _markdown(input_dir, paper_id)

            def executor(markdown_path: Path, output_dir: Path):
                if markdown_path.stem == "paper_fail":
                    raise RuntimeError("synthetic executor failure")
                return _offline_full_executor(markdown_path, output_dir)

            output_dir = root / "batch"
            report = BatchEvaluationRunner(
                BatchEvaluationConfig(input_dir=input_dir, output_dir=output_dir),
                executor,
            ).run()
            by_id = {paper.paper_id: paper for paper in report.papers}
            self.assertEqual(report.paper_count, 3)
            self.assertEqual(report.completed_count, 2)
            self.assertEqual(report.failed_count, 1)
            self.assertEqual(by_id["paper_fail"].failure_type, "RuntimeError")
            self.assertIn("synthetic executor failure", by_id["paper_fail"].error_message or "")
            self.assertTrue((output_dir / "paper_fail" / "batch_error.json").exists())
            self.assertTrue((output_dir / "paper_later" / "trajectory.jsonl").exists())

    def test_output_root_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "papers"
            input_dir.mkdir()
            _markdown(input_dir, "paper_a")
            output_dir = root / "batch"
            output_dir.mkdir()
            (output_dir / "paper_a").mkdir()
            with self.assertRaises(Exception):
                BatchEvaluationRunner(
                    BatchEvaluationConfig(input_dir=input_dir, output_dir=output_dir),
                    lambda *_: None,
                ).run()


if __name__ == "__main__":
    unittest.main()
