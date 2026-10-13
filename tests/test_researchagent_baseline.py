import tempfile
import unittest
from pathlib import Path

from research_agent.baselines.researchagent.input import load_seed_markdown
from research_agent.baselines.researchagent.llm import ResearchAgentDeepSeekClient
from research_agent.baselines.researchagent.retrieval import select_seed_candidate
from research_agent.baselines.researchagent.runner import _call_statistics
from research_agent.llm.config import DeepSeekConfig


class _FakeTransport:
    def __init__(self):
        self.calls = []

    def post_json(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        return {
            "id": "call-1",
            "model": "test-model",
            "choices": [{"message": {"content": "Problem: a\nRationale: b"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }


class ResearchAgentBaselineTests(unittest.TestCase):
    def test_seed_candidate_selection_is_deterministic_and_prefers_journal(self):
        candidates = [
            {"paperId": "z", "title": "A Method", "venue": "arXiv", "publicationTypes": [], "abstract": "x"},
            {"paperId": "b", "title": "A Method", "venue": "Conference", "publicationTypes": ["Conference"], "abstract": "x"},
            {"paperId": "a", "title": "A Method", "venue": "Journal", "publicationTypes": ["JournalArticle"], "abstract": "x"},
        ]
        selected, reason = select_seed_candidate(candidates)
        self.assertEqual(selected["paperId"], "a")
        self.assertEqual(reason, "journal_preferred")

    def test_seed_candidate_same_tier_prefers_abstract_then_paper_id(self):
        candidates = [
            {"paperId": "z", "title": "A Method", "venue": "Conference", "publicationTypes": ["Conference"], "abstract": None},
            {"paperId": "b", "title": "A Method", "venue": "Conference", "publicationTypes": ["Conference"], "abstract": "x"},
            {"paperId": "a", "title": "A Method", "venue": "Conference", "publicationTypes": ["Conference"], "abstract": "x"},
        ]
        selected, reason = select_seed_candidate(candidates)
        self.assertEqual(selected["paperId"], "a")
        self.assertEqual(reason, "stable_tiebreak")

    def test_markdown_input_is_deterministic_and_does_not_need_full_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text(
                "# A Generic Paper\n\n## Abstract\nA short abstract.\n\n"
                "## Method\nMain method.\n\n## References\nA reference.\n",
                encoding="utf-8",
            )
            seed = load_seed_markdown(path)
        self.assertEqual(seed.title, "A Generic Paper")
        self.assertEqual(seed.abstract, "A short abstract.")

    def test_deepseek_adapter_matches_official_text_generation_shape(self):
        transport = _FakeTransport()
        client = ResearchAgentDeepSeekClient(
            DeepSeekConfig(
                api_key="secret",
                base_url="https://example.invalid/v1",
                model="test-model",
            ),
            transport=transport,
        )
        result = client.call([
            {"role": "system", "content": "You are an assistant whose primary goal is to identify promising problems."},
            {"role": "user", "content": "Input"},
        ])
        self.assertEqual(result, "Problem: a\nRationale: b")
        self.assertEqual(len(transport.calls), 1)
        payload = transport.calls[0][2]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], 1000)
        self.assertEqual(payload["temperature"], 1.25)
        self.assertNotIn("response_format", payload)
        self.assertEqual(client.calls[0].stage, "problem_stage")

    def test_token_statistics_keep_known_values_when_metadata_is_missing(self):
        stats = _call_statistics([
            {"sequence": 1, "stage": "problem_stage", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            {"sequence": 2, "stage": "method_reviewer", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        ])
        self.assertFalse(stats["token_metadata_complete"])
        self.assertEqual(stats["known_prompt_tokens"], 10)
        self.assertEqual(stats["known_total_tokens"], 12)
        self.assertEqual(stats["calls_missing_token_metadata"], 1)
        self.assertIsNone(stats["total_tokens"])


if __name__ == "__main__":
    unittest.main()
