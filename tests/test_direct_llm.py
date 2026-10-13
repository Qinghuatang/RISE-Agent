"""Offline contracts for the one-shot Direct LLM baseline."""

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.llm.fake import FakeLLMClient
from research_agent.models.direct import DirectProposal
from research_agent.runner.direct_llm import DirectLLMRunner, direct_prompt_text, run_direct_batch
from research_agent.serialization import read_json


def _proposal() -> dict:
    return {
        "research_problem": "A method-level problem grounded in the supplied paper.",
        "proposed_method": {
            "core_idea": "A concrete mechanism for addressing the problem.",
            "components": [
                {"name": "Component A", "description": "A concrete operation."},
                {"name": "Component B", "description": "A complementary operation."},
            ],
        },
        "technical_rationale": "The components address the stated mechanism-level problem.",
        "experimental_plan": [
            {
                "objective": "Test the mechanism.",
                "setup": "Use the supplied task and controlled comparison.",
                "baselines": ["Seed Method", "Baseline X"],
                "metrics": ["accuracy"],
                "what_it_validates": "Whether the mechanism helps.",
                "validation_criterion": "Improvement over Seed Method.",
            },
            {
                "objective": "Test the component contribution.",
                "setup": "Run an ablation of Component B.",
                "baselines": ["Seed Method"],
                "metrics": ["accuracy"],
                "what_it_validates": "Whether Component B is necessary.",
                "validation_criterion": "The full method outperforms the ablation.",
            },
        ],
        "expected_contributions": ["A mechanism", "An evaluation protocol"],
    }


class DirectBaselineTests(unittest.TestCase):
    def test_schema_enforces_sizes_and_seed_baseline(self) -> None:
        proposal = DirectProposal.model_validate(_proposal())
        self.assertEqual(len(proposal.proposed_method.components), 2)
        self.assertEqual(len(proposal.experimental_plan), 2)
        invalid = json.loads(json.dumps(_proposal()))
        invalid["experimental_plan"][0]["baselines"] = ["Baseline X"]
        with self.assertRaises(Exception):
            DirectProposal.model_validate(invalid)

    def test_prompt_contains_frozen_task_and_generated_schema(self) -> None:
        prompt = direct_prompt_text()
        self.assertIn("propose ONE concrete follow-up research idea", prompt)
        self.assertIn('"research_problem"', prompt)
        self.assertIn("components", prompt)
        self.assertNotIn("{{DIRECT_PROPOSAL_JSON_SCHEMA}}", prompt)

    def test_single_run_calls_client_once_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "development_paper.md"
            markdown.write_text(
                "# Development Paper\n\n## Abstract\n\nA paper.\n\n## Method\n\nMethod text.\n\n## References\n\nExcluded.",
                encoding="utf-8",
            )
            client = FakeLLMClient(responses=[_proposal()])
            output = root / "output"
            result = DirectLLMRunner(client).run(markdown, output)
            self.assertEqual(result.manifest.status, "succeeded")
            self.assertEqual(result.manifest.llm_request_count, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertIn("Method text.", client.calls[0].messages[0].content)
            self.assertNotIn("Excluded.", client.calls[0].messages[0].content)
            self.assertTrue((output / "direct_proposal.json").exists())
            self.assertTrue((output / "llm_call.json").exists())
            self.assertTrue((output / "run_manifest.json").exists())

    def test_batch_is_serial_and_reports_one_call_per_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            clients = []
            for index in range(2):
                path = root / f"paper_{index}.md"
                path.write_text("# Paper\n\n## Method\n\nBody.", encoding="utf-8")
                paths.append(path)
                clients.append(FakeLLMClient(responses=[_proposal()]))
            iterator = iter(clients)
            report = run_direct_batch(
                paths,
                root / "batch",
                lambda: DirectLLMRunner(next(iterator)),
            )
            self.assertEqual(report["paper_count"], 2)
            self.assertEqual(report["succeeded_count"], 2)
            self.assertEqual(report["llm_request_count"], 2)
            self.assertTrue((root / "batch" / "direct_batch_report.json").exists())


if __name__ == "__main__":
    unittest.main()
