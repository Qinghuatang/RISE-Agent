import unittest
from pathlib import Path

from pydantic import ValidationError

from experiments.e3_state_counterfactual.global_revision import GlobalRevisionModule
from experiments.e3_state_counterfactual.models import (
    GlobalRevisionDraft,
    materialize_global_solution,
)
from experiments.e3_state_counterfactual.runner import E3RunConfig
from research_agent.llm.fake import FakeLLMClient
from research_agent.models import (
    CandidateGap,
    Component,
    CurrentSolution,
    ExperimentContext,
    RevisionNeed,
    SeedPaperContext,
)


def _seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="e3-generic-seed",
        title="A Generic Seed",
        abstract="A generic abstract.",
        research_objective="Study a mechanism-level question.",
        core_method="A method with cooperating components.",
        main_components=["Component one", "Component two"],
        main_findings=["A reported finding."],
        limitations=["A method boundary."],
        method_constraints=["Preserve the stated efficiency constraint."],
        experiment_context=ExperimentContext(),
    )


def _gap() -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="A generic mechanism-level gap.",
        confidence="medium",
        confidence_reason="The seed evidence supports the gap.",
        priority="high",
        priority_reason="It affects the central mechanism.",
        status="active",
        evidence_refs=["seed:core_method"],
    )


def _solution() -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="The current core mechanism.",
        rationale="The current rationale.",
        components=[
            Component(id="C1", description="First current unit.", status="ready"),
            Component(id="C2", description="Second current unit.", status="ready"),
        ],
        status="developed",
        novelty_status="distinct",
    )


class E3GlobalRevisionTests(unittest.TestCase):
    def test_e3_output_must_be_separate_from_read_only_source(self):
        with self.assertRaises(ValueError):
            E3RunConfig(
                source_dir=Path("/tmp/e3-source"),
                output_dir=Path("/tmp/e3-source/cases"),
            )

    def test_global_revision_materializes_complete_solution_with_reset_verification(self):
        response = {
            "status": "revised",
            "core_idea": "A complete revised mechanism.",
            "rationale": "A complete revised rationale.",
            "components": [
                {"id": "C1", "description": "Revised first unit."},
                {"id": "C3", "description": "A newly organized third unit."},
            ],
            "failure_reason": None,
        }
        fake = FakeLLMClient([response])
        module = GlobalRevisionModule(fake)
        execution = module.revise(
            seed_context=_seed(),
            active_gap=_gap(),
            current_solution=_solution(),
            revision_need=RevisionNeed(
                problem_to_fix="Repair the generic evidence-backed issue.",
                affected_units=["C1"],
                constraints=["Keep the active gap unchanged."],
            ),
            verification_evidence=None,
            selected_inspiration=None,
            relevant_failure_memory=[],
        )
        self.assertEqual(module.call_count, 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0].metadata["task"], "e3_global_revision")
        self.assertIsNotNone(execution.revised_solution)
        assert execution.revised_solution is not None
        self.assertEqual(execution.revised_solution.id, "S1")
        self.assertEqual(execution.revised_solution.novelty_status, "unchecked")
        self.assertTrue(
            all(component.status == "modified" for component in execution.revised_solution.components)
        )

    def test_global_draft_is_closed_and_requires_complete_rewrite(self):
        with self.assertRaises(ValidationError):
            GlobalRevisionDraft(
                status="revised",
                core_idea="A core.",
                rationale="A rationale.",
                components=[{"id": "C1", "description": "Only one."}],
                unexpected="not allowed",
            )

    def test_materialize_preserves_solution_identity_and_drops_old_novelty(self):
        draft = GlobalRevisionDraft(
            status="revised",
            core_idea="A new core.",
            rationale="A new rationale.",
            components=[
                {"id": "C1", "description": "New first."},
                {"id": "C2", "description": "New second."},
            ],
        )
        result = materialize_global_solution(_solution(), draft)
        self.assertEqual(result.id, "S1")
        self.assertEqual(result.novelty_status, "unchecked")
        self.assertEqual(result.evidence_refs, [])


if __name__ == "__main__":
    unittest.main()
