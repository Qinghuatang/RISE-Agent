import json
import unittest

from research_agent.errors import ExtractionError
from research_agent.llm.fake import FakeLLMClient
from research_agent.models import (
    CandidateGap,
    ExperimentContext,
    GapVerificationResult,
    ResearchState,
    SeedPaperContext,
)
from research_agent.modules.gap_discovery import GapDiscoveryModule
from research_agent.modules.seed_context import SeedContextExtractor
from research_agent.modules.solution_ideation import SolutionIdeationModule
from research_agent.trajectory.context import TrajectoryContext
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_mineru_markdown,
)


GENERIC_MARKDOWN = """# Generic Method Paper

## ABSTRACT
This paper studies a generic method objective.

## 1 INTRODUCTION
The problem requires a method that remains stable under changing conditions.

## 2 RELATED WORK
Prior approaches use alternative mechanisms.

## 3 METHOD
The authors choose mechanism A rather than mechanism B because B violates
stability constraint C.  The choice is intended to preserve stable operation.

## 4 RESULTS
The reported results are positive.  Evaluation does not establish behavior
outside the tested setting.

## 5 CONCLUSION
The authors state that future work should extend the method while preserving
the relevant stability condition.

## REFERENCES
Reference-only material must not be supplied as evidence.

## APPENDIX
Supplementary details must not be supplied as evidence.
"""


def _section_id(heading: str) -> str:
    parsed = parse_mineru_markdown(GENERIC_MARKDOWN, paper_id="generic-paper")
    return next(section.section_id for section in parsed.sections if section.heading == heading)


def _reading_context(paper_id: str = "generic-paper"):
    return build_full_paper_reading_context(
        parse_mineru_markdown(GENERIC_MARKDOWN, paper_id=paper_id)
    )


def _seed_draft(method_id: str, results_id: str, *, invalid_constraint_id: str | None = None) -> dict:
    constraint_id = invalid_constraint_id or method_id
    return {
        "paper_id": "ignored-by-parser",
        "title": "ignored",
        "abstract": "ignored",
        "research_objective": {
            "value": "The method targets stable operation.",
            "source_section_ids": [_section_id("1 INTRODUCTION")],
        },
        "core_method": {
            "value": "The method chooses mechanism A for stability.",
            "source_section_ids": [method_id],
        },
        "main_components": [
            {"value": "Mechanism A performs the main transformation.", "source_section_ids": [method_id]}
        ],
        "main_findings": [
            {"value": "The method reports positive results.", "source_section_ids": [results_id]}
        ],
        "limitations": [
            {
                "value": "Behavior outside the tested setting remains unresolved.",
                "source_section_ids": [results_id],
            }
        ],
        "method_constraints": [
            {
                "value": "The authors choose mechanism A rather than B because B violates stability constraint C.",
                "source_section_ids": [constraint_id],
            }
        ],
        "experiment_context": {
            "tasks_or_datasets": [],
            "metrics": [],
            "major_baselines": [],
            "experimental_setting": {
                "value": "A generic controlled setting.",
                "source_section_ids": [results_id],
            },
        },
    }


def _seed_context() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-paper",
        title="Generic Method Paper",
        abstract="A generic method objective.",
        research_objective="The method targets stable operation.",
        core_method="The method chooses mechanism A for stability.",
        main_components=["Mechanism A performs the main transformation."],
        main_findings=["The method reports positive results."],
        limitations=["Behavior outside the tested setting remains unresolved."],
        method_constraints=[
            "The authors choose mechanism A rather than B because B violates stability constraint C."
        ],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Generic task"],
            metrics=["Generic metric"],
            major_baselines=["Generic baseline"],
            experimental_setting="A generic controlled setting.",
        ),
    )


def _gap() -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="The method leaves unresolved how to extend capability while preserving stability constraint C.",
        confidence="medium",
        confidence_reason="The seed states a method trade-off and future direction.",
        priority="high",
        priority_reason="The constraint affects the central method objective.",
        status="active",
        evidence_refs=["seed:method_constraints[0]"],
    )


class SeedDesignRationaleTests(unittest.TestCase):
    def test_author_rationale_is_extracted_with_main_body_anchor(self) -> None:
        method_id = _section_id("3 METHOD")
        results_id = _section_id("4 RESULTS")
        client = FakeLLMClient([_seed_draft(method_id, results_id)])
        result = SeedContextExtractor(client).extract_text(GENERIC_MARKDOWN, "generic-paper")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            result.context.method_constraints,
            ["The authors choose mechanism A rather than B because B violates stability constraint C."],
        )
        self.assertTrue(
            any(
                anchor.field_path == "method_constraints[0]"
                and anchor.section_id == method_id
                for anchor in result.audit.anchors
            )
        )
        self.assertNotIn("Reference-only material", client.calls[0].messages[1].content)
        self.assertNotIn("Supplementary details", client.calls[0].messages[1].content)
        self.assertNotIn("anchors", result.context.model_dump())

    def test_constraint_evidence_must_reference_a_main_body_section(self) -> None:
        method_id = _section_id("3 METHOD")
        results_id = _section_id("4 RESULTS")
        with self.assertRaises(ExtractionError):
            SeedContextExtractor(
                FakeLLMClient([_seed_draft(method_id, results_id, invalid_constraint_id="S999")])
            ).extract_text(GENERIC_MARKDOWN, "generic-paper")

    def test_seed_prompt_schema_mentions_constraints_and_evidence(self) -> None:
        method_id = _section_id("3 METHOD")
        results_id = _section_id("4 RESULTS")
        client = FakeLLMClient([_seed_draft(method_id, results_id)])
        SeedContextExtractor(client).extract_text(GENERIC_MARKDOWN, "generic-paper")
        prompt = client.calls[0].messages[0].content
        self.assertIn("method_constraints", prompt)
        self.assertIn("source_section_ids", prompt)
        self.assertIn("Do not infer a constraint", prompt)


class GapDesignRationaleTests(unittest.TestCase):
    def test_method_constraint_evidence_is_legal_and_exposed(self) -> None:
        response = {
            "gaps": [
                {
                    "statement": "The method leaves unresolved how to extend capability while preserving stability constraint C.",
                    "evidence_refs": ["S4"],
                }
            ]
        }
        client = FakeLLMClient([response])
        agenda = GapDiscoveryModule(client).generate_gaps(
            _seed_context(), full_paper_reading_context=_reading_context()
        )

        self.assertEqual(agenda.gaps[0].evidence_refs, ["S4"])
        user_prompt = client.calls[0].messages[1].content
        system_prompt = client.calls[0].messages[0].content
        self.assertIn('"S4"', user_prompt)
        self.assertIn("stability constraint C", user_prompt)
        self.assertIn("DESIGN-RATIONALE CONSISTENCY GATE", system_prompt)
        self.assertIn("rejected alternative", system_prompt)
        self.assertIn("compensates", system_prompt)

    def test_experiment_context_remains_illegal_gap_evidence(self) -> None:
        response = {
            "gaps": [
                {
                    "statement": "A generic evaluation issue.",
                    "evidence_refs": ["seed:experiment_context.metrics"],
                }
            ]
        }
        with self.assertRaises(Exception):
            GapDiscoveryModule(FakeLLMClient([response])).generate_gaps(
                _seed_context(), full_paper_reading_context=_reading_context()
            )

    def test_refine_receives_constraints_without_extra_critic_call(self) -> None:
        context = TrajectoryContext.initialize(_seed_context())
        agenda = GapDiscoveryModule(
            FakeLLMClient(
                [
                    {
                        "gaps": [
                            {
                                "statement": _gap().statement,
                                "evidence_refs": ["S4"],
                            }
                        ]
                    },
                    {
                        "evaluations": [
                            {
                                "candidate_id": "G1",
                                "confidence": "medium",
                                "confidence_reason": "The method rationale is explicit.",
                                "priority": "high",
                                "priority_reason": "It affects the objective.",
                            }
                        ]
                    },
                ]
            )
        ).discover(
            _seed_context(),
            full_paper_reading_context=_reading_context(),
            context=context,
        )
        verifier = GapVerificationResult(
            status="partially_resolved",
            summary="A broad case is covered.",
            remaining_gap="The constrained extension remains open.",
            evidence_level="abstract_only",
        )
        refine_client = FakeLLMClient(
            [
                {
                    "refined_statement": "The constrained extension remains open.",
                    "refinement_reason": "The refinement preserves the stated rationale.",
                }
            ]
        )
        result = GapDiscoveryModule(refine_client).refine_gap(
            agenda.active_gap, verifier, context=context
        )
        self.assertEqual(result.active_gap.id, "G1")
        self.assertEqual(len(refine_client.calls), 1)
        combined = "\n".join(message.content for message in refine_client.calls[0].messages)
        self.assertIn("stability constraint C", combined)
        self.assertIn("DESIGN-RATIONALE CONSISTENCY", refine_client.calls[0].messages[0].content)


class SolutionDesignRationaleTests(unittest.TestCase):
    def test_form_and_develop_keep_constraint_context_and_exact_call_count(self) -> None:
        form_response = {
            "candidates": [
                {"core_idea": "Extend mechanism A with a stability-preserving path.", "rationale": "It handles constraint C."},
                {"core_idea": "Add an adaptive mediator around mechanism A.", "rationale": "It preserves the rationale."},
                {"core_idea": "Use a redesigned alternative with an explicit stability safeguard.", "rationale": "It addresses why B was rejected."},
            ]
        }
        develop_response = {
            "selected_solution_id": "S3",
            "selection_reason": "It addresses the rejection reason while preserving stability.",
            "core_idea": "A redesigned alternative with a stability safeguard.",
            "rationale": "The new safeguard handles constraint C.",
            "components": [
                "Detect the condition that caused the original rejection.",
                "Apply a stability safeguard before using the redesigned path.",
            ],
        }
        client = FakeLLMClient([form_response, develop_response])
        module = SolutionIdeationModule(client)
        seed = _seed_context()
        verification = GapVerificationResult(
            status="valid",
            summary="The method opportunity remains valid.",
            remaining_gap=_gap().statement,
            evidence_level="abstract_only",
        )
        candidates = module.form_solutions(_gap(), seed, verification)
        selected, solution, reason = module.select_and_develop(
            _gap(), seed, candidates
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(selected.id, "S3")
        self.assertEqual(len(solution.components), 2)
        self.assertTrue(reason)
        self.assertEqual(len(client.calls), 2)
        combined = "\n".join(
            message.content for call in client.calls for message in call.messages
        )
        self.assertIn("method_constraints", combined)
        self.assertIn("stability constraint C", combined)
        self.assertIn("rejected alternative", combined)
        self.assertIn("constraint-handling", combined)

    def test_solution_context_is_structured_json(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "candidates": [
                        {"core_idea": "A", "rationale": "R"},
                        {"core_idea": "B", "rationale": "R"},
                        {"core_idea": "C", "rationale": "R"},
                    ]
                }
            ]
        )
        module = SolutionIdeationModule(client)
        module.form_solutions(
            _gap(),
            _seed_context(),
            GapVerificationResult(
                status="valid",
                summary="Valid.",
                remaining_gap="Open.",
                evidence_level="abstract_only",
            ),
        )
        user_content = client.calls[0].messages[1].content
        self.assertIn('"method_constraints"', user_content)
        payload_start = user_content.index("SEED PAPER CONTEXT")
        self.assertIn("stability constraint C", user_content[payload_start:])


if __name__ == "__main__":
    unittest.main()
