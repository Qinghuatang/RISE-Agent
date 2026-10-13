"""Offline contracts for the Direct-Neutral baseline."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from research_agent.errors import LLMTransportError
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.llm.fake import FakeLLMClient
from research_agent.models.direct_neutral import DirectNeutralProposal
from research_agent.runner.direct_llm_neutral import (
    CountingJsonTransport,
    DirectNeutralRunner,
    discover_neutral_inputs,
    direct_neutral_prompt_text,
    run_direct_neutral_batch,
)
from research_agent.serialization import read_json, read_jsonl


def _proposal() -> dict[str, str]:
    return {
        "research_problem": "A grounded research opportunity.",
        "proposed_method": "A concrete follow-up mechanism.",
        "technical_rationale": "The mechanism addresses the stated opportunity.",
        "experimental_plan": "Compare the proposal with the seed method and controlled variants.",
    }


class _FailOnceTransport:
    def __init__(self) -> None:
        self.calls = 0

    def post_json(self, url, headers, payload, timeout):
        self.calls += 1
        if self.calls == 1:
            raise LLMTransportError("temporary transport failure")
        return {
            "id": "request-2",
            "model": "offline-model",
            "choices": [{"message": {"content": json.dumps(_proposal())}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
        }


class DirectNeutralBaselineTests(unittest.TestCase):
    def test_prompt_is_frozen_minimal_task_prompt(self) -> None:
        prompt = direct_neutral_prompt_text()
        self.assertIn("You are an expert researcher in NLP and machine learning.", prompt)
        self.assertIn('"experimental_plan": "..."', prompt)
        self.assertIn("Do not invent experimental results.", prompt)
        self.assertNotIn("Seed Method", prompt)
        self.assertNotIn("components", prompt)
        self.assertNotIn("expected_contributions", prompt)

    def test_schema_has_exactly_four_non_empty_strings(self) -> None:
        proposal = DirectNeutralProposal.model_validate(_proposal())
        self.assertEqual(set(proposal.model_dump()), {
            "research_problem",
            "proposed_method",
            "technical_rationale",
            "experimental_plan",
        })
        invalid = dict(_proposal())
        invalid["experimental_plan"] = "   "
        with self.assertRaises(Exception):
            DirectNeutralProposal.model_validate(invalid)

    def test_single_run_uses_one_logical_call_and_shared_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "paper.md"
            markdown.write_text(
                "# Paper\n\n## Method\n\nKeep this body.\n\n## References\n\nDrop this tail.",
                encoding="utf-8",
            )
            client = FakeLLMClient(responses=[_proposal()])
            output = root / "paper-output"
            result = DirectNeutralRunner(client).run(markdown, output)

            self.assertEqual(result.manifest.status, "succeeded")
            self.assertEqual(result.manifest.logical_scientific_generation_calls, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertIn("Keep this body.", client.calls[0].messages[0].content)
            self.assertNotIn("Drop this tail.", client.calls[0].messages[0].content)
            self.assertTrue((output / "run_manifest.json").exists())
            self.assertTrue((output / "raw_response.txt").exists() is False)
            self.assertTrue((output / "final_proposal.json").exists())
            self.assertTrue((output / "llm_calls.jsonl").exists())
            self.assertFalse((output / "error.json").exists())
            self.assertEqual(len(read_jsonl(output / "llm_calls.jsonl")), 1)

    def test_transport_retry_is_physical_not_a_second_logical_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "paper.md"
            markdown.write_text("# Paper\n\n## Method\n\nBody.", encoding="utf-8")
            transport = CountingJsonTransport(_FailOnceTransport())
            client = DeepSeekClient(
                DeepSeekConfig(
                    api_key="offline",
                    base_url="https://offline.invalid",
                    model="offline-model",
                    max_retries=1,
                ),
                transport=transport,
            )
            output = root / "paper-output"
            result = DirectNeutralRunner(
                client,
                raw_response_source=transport,
                physical_attempt_source=transport,
                model="offline-model",
                provider="deepseek",
            ).run(markdown, output)

            self.assertEqual(result.manifest.status, "succeeded")
            self.assertEqual(result.manifest.logical_scientific_generation_calls, 1)
            self.assertEqual(result.manifest.physical_api_attempts, 2)
            call = read_jsonl(output / "llm_calls.jsonl")[0]["payload"]
            self.assertEqual(call["retry_count"], 1)
            self.assertEqual(call["total_tokens"], 24)

    def test_parallel_batch_isolates_papers_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(3):
                path = root / f"paper_{index}.md"
                path.write_text(f"# Paper {index}\n\n## Method\n\nBody {index}.", encoding="utf-8")
                paths.append(path)

            factory_lock = threading.Lock()
            factory_index = 0

            def factory() -> DirectNeutralRunner:
                nonlocal factory_index
                with factory_lock:
                    current = factory_index
                    factory_index += 1
                response: object = _proposal()
                if current == 0:
                    response = RuntimeError("synthetic worker failure")
                return DirectNeutralRunner(FakeLLMClient(responses=[response]))

            report = run_direct_neutral_batch(
                paths,
                root / "batch",
                factory,
                workers=2,
            )
            self.assertEqual(report["total_papers"], 3)
            self.assertEqual(report["succeeded"], 2)
            self.assertEqual(report["failed"], 1)
            self.assertEqual(report["logical_llm_calls"], 3)
            self.assertEqual(report["calls_with_usage"], 0)
            self.assertEqual(report["calls_missing_usage"], 3)
            self.assertTrue((root / "batch" / "evaluation_report.json").exists())
            self.assertEqual(
                sorted(path.name for path in (root / "batch").iterdir() if path.is_dir()),
                ["paper_0", "paper_1", "paper_2"],
            )
            for path in paths:
                manifest = read_json(root / "batch" / path.stem / "run_manifest.json")["payload"]
                self.assertEqual(manifest["paper_id"], path.stem)

    def test_workers_one_is_supported_and_existing_output_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "paper.md"
            markdown.write_text("# Paper\n\n## Method\n\nBody.", encoding="utf-8")
            output = root / "batch"
            report = run_direct_neutral_batch(
                [markdown],
                output,
                lambda: DirectNeutralRunner(FakeLLMClient(responses=[_proposal()])),
                workers=1,
            )
            self.assertEqual(report["succeeded"], 1)
            with self.assertRaises(FileExistsError):
                run_direct_neutral_batch(
                    [markdown],
                    output,
                    lambda: DirectNeutralRunner(FakeLLMClient(responses=[_proposal()])),
                    workers=1,
                )

    def test_discovery_requires_expected_count_and_parses_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                (root / f"paper_{index}.md").write_text(
                    f"# Paper {index}\n\n## Method\n\nBody.", encoding="utf-8"
                )
            self.assertEqual(len(discover_neutral_inputs(root, expected_count=2)), 2)
            with self.assertRaises(ValueError):
                discover_neutral_inputs(root)


if __name__ == "__main__":
    unittest.main()
