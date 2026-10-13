"""Offline contract tests for the four-field Direct-LLM2 output."""

from __future__ import annotations

import unittest

from research_agent.models.direct_llm2 import DirectLLM2Proposal


class DirectLLM2SchemaTests(unittest.TestCase):
    def test_accepts_the_four_required_non_empty_strings(self) -> None:
        proposal = DirectLLM2Proposal.model_validate(
            {
                "research_problem": "A concrete research problem.",
                "proposed_method": "A technically specific method.",
                "method_rationale": "A causal rationale.",
                "experimental_plan": "A plan with baselines, metrics, and ablations.",
            }
        )
        self.assertEqual(
            set(proposal.model_dump()),
            {
                "research_problem",
                "proposed_method",
                "method_rationale",
                "experimental_plan",
            },
        )

    def test_rejects_missing_extra_or_non_string_fields(self) -> None:
        with self.assertRaises(Exception):
            DirectLLM2Proposal.model_validate(
                {
                    "research_problem": "A problem.",
                    "proposed_method": {"core_idea": "Nested method."},
                    "method_rationale": "A rationale.",
                    "experimental_plan": "A plan.",
                }
            )
        with self.assertRaises(Exception):
            DirectLLM2Proposal.model_validate(
                {
                    "research_problem": "A problem.",
                    "proposed_method": "A method.",
                    "method_rationale": "A rationale.",
                    "experimental_plan": "A plan.",
                    "expected_contributions": ["Extra field."],
                }
            )


if __name__ == "__main__":
    unittest.main()
