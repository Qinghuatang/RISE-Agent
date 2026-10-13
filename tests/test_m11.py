"""Generic offline end-to-end trajectory and replay tests."""

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.controller.budget import BudgetTracker
from research_agent.controller.scientific_controller import ScientificController
from research_agent.errors import SerializationError
from research_agent.errors import RetrievalError
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_mineru_markdown,
)
from research_agent.llm.fake import FakeLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    ExperimentContext,
    GapVerificationResult,
    PaperRecord,
    SeedPaperContext,
)
from research_agent.models.runner import TrajectoryReplayRecord
from research_agent.modules.commit import CommitModule
from research_agent.modules.experiment_design import ExperimentDesignModule
from research_agent.modules.finalizer import FinalizerModule
from research_agent.modules.gap_discovery import GapDiscoveryModule
from research_agent.modules.inspiration_retrieval import InspirationRetrievalModule
from research_agent.modules.related_work_verification import (
    RelatedWorkVerificationModule,
)
from research_agent.modules.solution_ideation import SolutionIdeationModule
from research_agent.modules.targeted_revision import TargetedRevisionModule
from research_agent.retrieval.fake import FakeInspirationSearchProvider, FakeRelatedPaperProvider
from research_agent.runner import TrajectoryModules, TrajectoryRunner
from research_agent.serialization.trajectory import (
    load_replay_journal,
    reconstruct_context,
)
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-m11-seed",
        title="Generic supplied seed",
        abstract="A supplied abstract.",
        research_objective="Study a method-level opportunity.",
        core_method="A method with interacting units.",
        main_components=["Unit A", "Unit B"],
        main_findings=["A supplied finding."],
        limitations=["A supplied limitation."],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task-A"],
            metrics=["metric-A"],
            major_baselines=["Baseline-A"],
            experimental_setting="A supplied setting.",
        ),
    )


def reading():
    markdown = """# Generic supplied seed

## ABSTRACT
A supplied abstract.

## 1 METHOD
A method with interacting units. Unit A and Unit B.

## 2 FINDINGS
A supplied finding and a supplied limitation.

## REFERENCES
Reference-only content.
"""
    return build_full_paper_reading_context(
        parse_mineru_markdown(markdown, paper_id=seed().paper_id)
    )


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id="related-1",
        title="A generic related paper",
        abstract="A supplied related abstract.",
        year=2025,
        venue="Fixture",
        source="fixture",
    )


def gap_draft() -> dict:
    return {
        "gaps": [
            {
                "statement": "The supplied method leaves a mechanism-level question unresolved.",
                "evidence_refs": ["S3"],
            }
        ]
    }


def gap_evaluation(*, count: int = 1) -> dict:
    return {
        "evaluations": [
            {
                "candidate_id": f"G{index}",
                "confidence": "medium",
                "confidence_reason": "The supplied method evidence supports the question.",
                "priority": "high",
                "priority_reason": "It affects the central mechanism.",
            }
            for index in range(1, count + 1)
        ]
    }


def two_gap_draft() -> dict:
    payload = gap_draft()
    second = dict(payload["gaps"][0])
    second["statement"] = "A second generic mechanism-level question remains open."
    payload["gaps"].append(second)
    return payload


def gap_verification() -> dict:
    return {
        "status": "valid",
        "summary": "The gap remains open in the supplied related-work pool.",
        "covered_aspects": [],
        "remaining_gap": "The mechanism-level question remains.",
        "related_papers": ["related-1"],
        "evidence_level": "abstract_only",
    }


def solution_verification() -> dict:
    return {
        "comparisons": [
            {
                "paper_ref": "related-1",
                "problem_relation": "related",
                "mechanism_relation": "different",
                "shared_elements": ["A standard comparison element."],
                "novelty_relevant_overlaps": [],
                "key_difference": "The supplied mechanism differs.",
            }
        ],
        "result": {
            "novelty_status": "distinct",
            "summary": "The supplied solution is distinct within the retrieved evidence.",
            "shared_elements": ["A standard comparison element."],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": ["A supplied distinctive mechanism."],
            "key_differences": ["A supplied key difference."],
            "related_papers": ["related-1"],
            "evidence_level": "abstract_only",
        },
    }


def solution_form() -> dict:
    return {
        "candidates": [
            {"core_idea": "Candidate one.", "rationale": "Rationale one."},
            {"core_idea": "Candidate two.", "rationale": "Rationale two."},
            {"core_idea": "Candidate three.", "rationale": "Rationale three."},
        ]
    }


def solution_development() -> dict:
    return {
        "selected_solution_id": "S1",
        "selection_reason": "The first candidate best addresses the active gap.",
        "core_idea": "A developed mechanism for the active gap.",
        "rationale": "The mechanism targets the supplied unresolved interaction.",
        "components": ["A core mechanism.", "A supporting mechanism."],
    }


def experiment_design() -> dict:
    return {
        "key_claims": [
            {"id": "K1", "statement": "The core solution is testable.", "target_unit": "core_solution"},
            {"id": "K2", "statement": "The first component is testable.", "target_unit": "C1"},
        ],
        "experiments": [
            {
                "existing_experiment_id": None,
                "objective": "Test the core mechanism.",
                "setup": "Task-A under the supplied setting.",
                "baselines": ["Seed Method"],
                "metrics": ["metric-A"],
                "what_it_validates": "core_solution behavior",
                "validation_criterion": "The observation tests the core claim.",
                "validates_claims": ["K1"],
            },
            {
                "existing_experiment_id": None,
                "objective": "Test the first component.",
                "setup": "Task-A with the first component isolated.",
                "baselines": ["Seed Method"],
                "metrics": ["metric-A"],
                "what_it_validates": "C1 behavior",
                "validation_criterion": "The observation tests the component claim.",
                "validates_claims": ["K2"],
            },
        ],
        "identifiability_checks": [
            {
                "claim_id": "K1",
                "required_observable": "Overall solution outcome.",
                "dataset_support": "Task-A exposes the outcome.",
                "ground_truth_availability": "The task target is available.",
                "control_or_baseline": "Seed Method is the control.",
                "falsification": "No improvement under the controlled test falsifies the claim.",
                "identifiable": True,
            },
            {
                "claim_id": "K2",
                "required_observable": "Component behavior.",
                "dataset_support": "Task-A supports the component comparison.",
                "ground_truth_availability": "The task target is available.",
                "control_or_baseline": "Seed Method is the baseline.",
                "falsification": "The isolated comparison does not support the claim.",
                "identifiable": True,
            },
        ],
        "design_summary": "The two experiments cover the committed claims.",
    }


def finalizer_draft() -> dict:
    return {
        "novelty_positioning": "The committed solution is distinct within the supplied evidence.",
        "expected_contributions": [
            "An expected mechanism-level contribution.",
            "A potential improvement evaluated by the planned experiments.",
        ],
    }


def inspiration_entities() -> dict:
    entities = [
        {
            "name": f"Generic bridge {index}",
            "type": "mechanism",
            "bridge_reason": "It supplies a transferable mechanism-level principle.",
        }
        for index in range(1, 6)
    ]
    return {
        "inspiration_need": {
            "current_problem": "A generic partial-overlap problem.",
            "required_function": "A transferable mechanism.",
            "current_mechanism": "The current supplied mechanism.",
            "rejected_mechanisms": [],
            "relevant_context": "The supplied gap and verification.",
        },
        "proposed_entities": entities,
        "selected_entities": [entities[0]],
    }


def inspiration_result() -> dict:
    return {
        "status": "found",
        "inspirations": [
            {
                "bridge_entity": {
                    "name": "Generic bridge 1",
                    "type": "mechanism",
                    "bridge_reason": "It supplies a transferable mechanism-level principle.",
                },
                "paper_ref": "inspiration-1",
                "source_problem": "A related generic problem.",
                "source_mechanism": "A related generic mechanism.",
                "transferable_principle": "A generic transferable principle.",
                "adaptation_hint": "Adapt the principle minimally.",
                "grounding": "arxiv",
            }
        ],
    }


def revision_draft() -> dict:
    return {
        "selected_inspiration_ref": "I1:1",
        "result": {
            "status": "revised",
            "revised_core_idea": None,
            "revised_rationale": None,
            "revised_components": [
                {"component_id": "C1", "description": "A revised first mechanism."}
            ],
            "changed_units": ["C1"],
            "revision_summary": "The first component is minimally revised.",
            "failure_reason": None,
        },
    }


def revision_without_inspiration() -> dict:
    payload = revision_draft()
    payload["selected_inspiration_ref"] = None
    return payload


def solution_partial_verification() -> dict:
    result = solution_verification()
    result["result"]["novelty_status"] = "partial_overlap"
    result["result"]["summary"] = "The solution partially overlaps with the supplied evidence."
    result["result"]["novelty_relevant_overlaps"] = [
        {
            "component_id": "C1",
            "overlap_summary": "The first component overlaps partially.",
            "matched_paper_ref": "related-1",
        }
    ]
    result["comparisons"][0]["novelty_relevant_overlaps"] = [
        "The first component overlaps partially."
    ]
    result["comparisons"][0]["mechanism_relation"] = "partial"
    return result


def modules(*, experiment_design_payload: dict | None = None) -> TrajectoryModules:
    verification_llm = FakeLLMClient([gap_verification(), solution_verification()])
    return TrajectoryModules(
        gap_discovery=GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation()])),
        solution_ideation=SolutionIdeationModule(
            FakeLLMClient([solution_form(), solution_development()])
        ),
        related_work_verification=RelatedWorkVerificationModule(
            verification_llm,
            FakeRelatedPaperProvider([paper()]),
        ),
        inspiration_retrieval=InspirationRetrievalModule(
            FakeLLMClient([]), FakeInspirationSearchProvider([])
        ),
        targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
        experiment_design=ExperimentDesignModule(
            FakeLLMClient([experiment_design_payload or experiment_design()])
        ),
        controller=ScientificController(FakeLLMClient([])),
        commit=CommitModule(),
        finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
    )


class M11TrajectoryTests(unittest.TestCase):
    def test_unable_experiment_design_is_scientific_stop_not_runner_failure(self):
        payload = experiment_design()
        payload["identifiability_checks"][1]["identifiable"] = False
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                modules(experiment_design_payload=payload),
                budget=BudgetTracker(max_steps=4, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
        self.assertEqual(result.status, "stopped")
        self.assertIsNone(result.error_type)
        self.assertEqual(result.final_state.experiments, [])

    def test_valid_commit_persists_and_reconstructs_exact_state_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrajectoryRunner(
                modules(),
                budget=BudgetTracker(max_steps=5, max_searches=1),
                output_dir=directory,
            )
            result = runner.run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "committed")
            self.assertIsNotNone(result.commit_snapshot)
            self.assertIsNotNone(result.final_proposal)
            self.assertEqual(result.budget.steps_used, 5)
            self.assertEqual(result.budget.searches_used, 1)
            self.assertTrue(Path(directory, "trajectory_replay.jsonl").exists())
            loaded = TrajectoryContext.load(seed(), directory)
            replayed = reconstruct_context(
                seed(),
                Path(directory, "trajectory.jsonl"),
                Path(directory, "trajectory_replay.jsonl"),
            )
            self.assertEqual(replayed.get_state(), loaded.get_state())
            self.assertEqual(replayed.get_memory(), loaded.get_memory())
            self.assertEqual(
                replayed.trajectory.get_events(), loaded.trajectory.get_events()
            )
            manifest = json.loads(Path(directory, "run_manifest.json").read_text())
            self.assertEqual(manifest["payload"]["status"], "committed")
            actions = [record.action for record in replayed.get_memory().decision_log]
            self.assertEqual(
                actions,
                [
                    "VERIFY_RELATED_WORK",
                    "SOLUTION_IDEATION",
                    "VERIFY_RELATED_WORK",
                    "DESIGN_EXPERIMENT",
                    "COMMIT",
                ],
            )

    def test_small_budget_stops_without_finalizer_or_fake_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            injected = modules()
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=1, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "stopped")
            self.assertIsNone(result.commit_snapshot)
            self.assertIsNone(result.final_proposal)
            self.assertEqual(result.budget.steps_used, 1)
            self.assertEqual(injected.finalizer.call_count, 0)
            replayed = reconstruct_context(
                seed(),
                Path(directory, "trajectory.jsonl"),
                Path(directory, "trajectory_replay.jsonl"),
            )
            self.assertEqual(replayed.get_state(), result.final_state)

    def test_bootstrap_and_internal_events_do_not_claim_scientific_actions(self):
        context = TrajectoryContext.initialize(seed())
        discovery = GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation(), {
            "refined_statement": "A narrower remaining mechanism question.",
            "refinement_reason": "The verification covered one aspect.",
        }]))
        discovery.discover(
            seed(), full_paper_reading_context=reading(), context=context, step=1
        )
        selected = context.trajectory.get_events()[-1]
        self.assertEqual(selected.event_type, "active_gap_selected")
        self.assertIsNone(selected.action)
        gap = context.get_state().active_gap
        assert gap is not None
        partial = GapVerificationResult(
            status="partially_resolved",
            summary="One aspect is covered.",
            remaining_gap="A residual question remains.",
            related_papers=[],
            evidence_level="abstract_only",
        )
        discovery.refine_gap(gap, partial, context=context, step=3)
        refined = context.trajectory.get_events()[-2]
        self.assertEqual(refined.event_type, "gap_refined")
        self.assertIsNone(refined.action)
        self.assertGreater(refined.step, 1)

    def test_replay_rejects_corrupt_sequence_and_memory_category(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrajectoryRunner(
                modules(), budget=BudgetTracker(max_steps=1, max_searches=1), output_dir=directory
            )
            runner.run(seed(), full_paper_reading_context=reading())
            replay_path = Path(directory, "trajectory_replay.jsonl")
            lines = replay_path.read_text().splitlines()
            payload = json.loads(lines[0])
            payload["payload"]["sequence"] = 9
            replay_path.write_text("\n".join([json.dumps(payload), *lines[1:]]) + "\n")
            with self.assertRaises(SerializationError):
                load_replay_journal(replay_path)
            payload["payload"]["sequence"] = 1
            payload["payload"]["memory_patch"] = {"unsupported": {}}
            replay_path.write_text("\n".join([json.dumps(payload), *lines[1:]]) + "\n")
            with self.assertRaises((SerializationError, ValueError)):
                load_replay_journal(replay_path)

    def test_inspiration_controller_choice_propagates_exact_record_to_revision(self):
        verification_llm = FakeLLMClient(
            [gap_verification(), solution_partial_verification(), solution_verification()]
        )
        controller_llm = FakeLLMClient(
            [{"action": "SEEK_INSPIRATION", "reason": "Use a transferable mechanism."}]
        )
        inspiration_paper = paper().model_copy(update={"paper_id": "inspiration-1"})
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation()])),
            solution_ideation=SolutionIdeationModule(
                FakeLLMClient([solution_form(), solution_development()])
            ),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([inspiration_entities(), inspiration_result()]),
                FakeInspirationSearchProvider([inspiration_paper]),
            ),
            targeted_revision=TargetedRevisionModule(FakeLLMClient([revision_draft()])),
            experiment_design=ExperimentDesignModule(FakeLLMClient([experiment_design()])),
            controller=ScientificController(controller_llm),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=8, max_searches=2),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "committed")
            self.assertEqual(injected.controller.call_count, 1)
            context = TrajectoryContext.load(seed(), directory)
            revisions = context.get_memory().revision_memory
            self.assertEqual(len(revisions), 1)
            self.assertEqual(revisions[0].selected_inspiration_ref, "I1:1")
            inspiration_calls = injected.inspiration_retrieval.llm_client.calls
            inspiration_call_one = json.loads(inspiration_calls[0].messages[1].content)
            self.assertEqual(
                inspiration_call_one["controller_exploration_rationale"],
                "Use a transferable mechanism.",
            )
            revision_calls = injected.targeted_revision.llm_client.calls
            revision_payload = json.loads(revision_calls[0].messages[1].content)
            self.assertEqual(
                revision_payload["controller_exploration_rationale"],
                "Use a transferable mechanism.",
            )
            self.assertEqual(
                revision_payload["inspiration_need"]["required_function"],
                "A transferable mechanism.",
            )
            self.assertEqual(result.budget.steps_used, 8)
            self.assertEqual(result.budget.searches_used, 2)

    def test_partial_overlap_uses_verification_only_revision_and_changed_units_focus(self):
        verification_llm = FakeLLMClient(
            [gap_verification(), solution_partial_verification(), solution_verification()]
        )
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation()])),
            solution_ideation=SolutionIdeationModule(
                FakeLLMClient([solution_form(), solution_development()])
            ),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(
                FakeLLMClient([revision_without_inspiration()])
            ),
            experiment_design=ExperimentDesignModule(FakeLLMClient([experiment_design()])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
        )
        result = TrajectoryRunner(
            injected, budget=BudgetTracker(max_steps=7, max_searches=1)
        ).run(seed(), full_paper_reading_context=reading())
        self.assertEqual(result.status, "committed")
        self.assertEqual(result.budget.steps_used, 7)
        self.assertEqual(
            injected.related_work_verification.call_count, 3
        )

    def test_second_revision_after_partial_reverification_is_allowed(self):
        second_revision = revision_without_inspiration()
        second_revision["result"]["revised_components"][0][
            "description"
        ] = "A second revised first mechanism."
        verification_llm = FakeLLMClient(
            [
                gap_verification(),
                solution_partial_verification(),
                solution_partial_verification(),
                solution_verification(),
            ]
        )
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation()])),
            solution_ideation=SolutionIdeationModule(
                FakeLLMClient([solution_form(), solution_development()])
            ),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(
                FakeLLMClient(
                    [revision_without_inspiration(), second_revision]
                )
            ),
            experiment_design=ExperimentDesignModule(FakeLLMClient([experiment_design()])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
        )
        result = TrajectoryRunner(
            injected, budget=BudgetTracker(max_steps=9, max_searches=1)
        ).run(seed(), full_paper_reading_context=reading())
        self.assertEqual(result.status, "committed")
        self.assertEqual(injected.targeted_revision.call_count, 2)
        self.assertEqual(injected.related_work_verification.call_count, 4)

    def test_resolved_gap_switches_without_inventing_an_action(self):
        verification_llm = FakeLLMClient(
            [
                {
                    **gap_verification(),
                    "status": "resolved",
                    "remaining_gap": None,
                },
                gap_verification(),
            ]
        )
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(FakeLLMClient([two_gap_draft(), gap_evaluation(count=2)])),
            solution_ideation=SolutionIdeationModule(FakeLLMClient([])),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
            experiment_design=ExperimentDesignModule(FakeLLMClient([])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([])),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=3, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.budget.steps_used, 3)
            context = TrajectoryContext.load(seed(), directory)
            self.assertEqual(
                [record.action for record in context.get_memory().decision_log],
                ["VERIFY_RELATED_WORK", "SWITCH_GAP", "VERIFY_RELATED_WORK"],
            )
            switch_events = [
                event for event in context.trajectory.get_events() if event.event_type == "gap_switched"
            ]
            self.assertEqual(len(switch_events), 1)
            self.assertEqual(switch_events[0].action, "SWITCH_GAP")
            self.assertEqual(context.get_state().active_gap.id, "G2")

    def test_partial_gap_refine_is_internal_and_does_not_consume_budget(self):
        verification_llm = FakeLLMClient(
            [
                {
                    **gap_verification(),
                    "status": "partially_resolved",
                    "remaining_gap": "A residual mechanism question remains.",
                },
                gap_verification(),
            ]
        )
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(
                FakeLLMClient(
                    [
                        gap_draft(),
                        gap_evaluation(),
                        {
                            "refined_statement": "A narrower residual mechanism question.",
                            "refinement_reason": "One aspect was covered.",
                        },
                    ]
                )
            ),
            solution_ideation=SolutionIdeationModule(FakeLLMClient([])),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
            experiment_design=ExperimentDesignModule(FakeLLMClient([])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([])),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=3, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.budget.steps_used, 2)
            context = TrajectoryContext.load(seed(), directory)
            events = context.trajectory.get_events()
            refined = [event for event in events if event.event_type == "gap_refined"]
            self.assertEqual(len(refined), 1)
            self.assertIsNone(refined[0].action)
            self.assertGreater(refined[0].step, 1)
            self.assertEqual(len(context.get_memory().decision_log), 2)

    def test_second_partial_after_one_refinement_rejects_gap_and_stops_cleanly(self):
        partial = {
            **gap_verification(),
            "status": "partially_resolved",
            "remaining_gap": "A residual mechanism question remains.",
        }
        verification_llm = FakeLLMClient([partial, partial])
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(
                FakeLLMClient(
                    [
                        gap_draft(),
                        gap_evaluation(),
                        {
                            "refined_statement": "The residual mechanism question remains.",
                            "refinement_reason": "The refinement keeps only the unresolved problem.",
                        },
                    ]
                )
            ),
            solution_ideation=SolutionIdeationModule(FakeLLMClient([])),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
            experiment_design=ExperimentDesignModule(FakeLLMClient([])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([])),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=3, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "stopped")
            self.assertIsNone(result.error_type)
            self.assertEqual(result.budget.steps_used, 3)
            self.assertIsNone(result.final_state.active_gap)
            self.assertEqual(injected.gap_discovery.call_count, 3)
            context = TrajectoryContext.load(seed(), directory)
            agenda = context.get_latest_gap_agenda()
            assert agenda is not None
            self.assertEqual(agenda.gaps[0].status, "rejected")
            self.assertEqual(
                [record.action for record in context.get_memory().decision_log],
                ["VERIFY_RELATED_WORK", "VERIFY_RELATED_WORK", "SWITCH_GAP"],
            )
            self.assertFalse(
                any(
                    event.event_type == "runner_action_failed"
                    for event in context.trajectory.get_events()
                )
            )

    def test_second_partial_switches_to_next_candidate_without_second_refine(self):
        partial = {
            **gap_verification(),
            "status": "partially_resolved",
            "remaining_gap": "A residual mechanism question remains.",
        }
        verification_llm = FakeLLMClient([partial, partial])
        injected = TrajectoryModules(
            gap_discovery=GapDiscoveryModule(
                FakeLLMClient(
                    [
                        two_gap_draft(),
                        gap_evaluation(count=2),
                        {
                            "refined_statement": "The residual mechanism question remains.",
                            "refinement_reason": "The refinement keeps only the unresolved problem.",
                        },
                    ]
                )
            ),
            solution_ideation=SolutionIdeationModule(FakeLLMClient([])),
            related_work_verification=RelatedWorkVerificationModule(
                verification_llm, FakeRelatedPaperProvider([paper()])
            ),
            inspiration_retrieval=InspirationRetrievalModule(
                FakeLLMClient([]), FakeInspirationSearchProvider([])
            ),
            targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
            experiment_design=ExperimentDesignModule(FakeLLMClient([])),
            controller=ScientificController(FakeLLMClient([])),
            commit=CommitModule(),
            finalizer=FinalizerModule(FakeLLMClient([])),
        )
        result = TrajectoryRunner(
            injected,
            budget=BudgetTracker(max_steps=3, max_searches=1),
        ).run(seed(), full_paper_reading_context=reading())
        self.assertEqual(result.status, "stopped")
        self.assertIsNone(result.error_type)
        self.assertIsNotNone(result.final_state.active_gap)
        self.assertEqual(result.final_state.active_gap.id, "G2")
        self.assertEqual(injected.gap_discovery.call_count, 3)

    def test_provider_failure_consumes_attempt_step_and_search_but_stops(self):
        class FailingProvider:
            provider_name = "failing_fixture"
            last_request_count = 1

            def __init__(self):
                self.calls = 0

            def get_related_paper_pool(self, seed_context, *, limit=20):
                self.calls += 1
                raise RetrievalError("generic provider failure")

        provider = FailingProvider()
        injected = modules()
        injected = TrajectoryModules(
            gap_discovery=injected.gap_discovery,
            solution_ideation=injected.solution_ideation,
            related_work_verification=RelatedWorkVerificationModule(
                FakeLLMClient([]), provider
            ),
            inspiration_retrieval=injected.inspiration_retrieval,
            targeted_revision=injected.targeted_revision,
            experiment_design=injected.experiment_design,
            controller=injected.controller,
            commit=injected.commit,
            finalizer=injected.finalizer,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_type, "RetrievalError")
            self.assertEqual(result.budget.steps_used, 1)
            self.assertEqual(result.budget.searches_used, 1)
            self.assertEqual(provider.calls, 1)
            context = TrajectoryContext.load(seed(), directory)
            self.assertTrue(
                any(event.event_type == "runner_action_failed" for event in context.trajectory.get_events())
            )
            self.assertEqual(len(context.get_memory().decision_log), 1)
            replayed = reconstruct_context(
                seed(), Path(directory, "trajectory.jsonl"), Path(directory, "trajectory_replay.jsonl")
            )
            self.assertEqual(replayed.get_memory(), context.get_memory())

    def test_finalizer_failure_keeps_commit_snapshot_and_terminal_outcome(self):
        injected = modules()
        injected.finalizer.llm_client = FakeLLMClient([])
        with tempfile.TemporaryDirectory() as directory:
            result = TrajectoryRunner(
                injected,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                output_dir=directory,
            ).run(seed(), full_paper_reading_context=reading())

            self.assertEqual(result.status, "finalization_failed")
            self.assertIsNotNone(result.commit_snapshot)
            self.assertIsNone(result.final_proposal)
            self.assertEqual(result.budget.steps_used, 5)
            self.assertEqual(result.budget.searches_used, 1)
            self.assertTrue(Path(directory, "commit_snapshot.json").exists())
            manifest = json.loads(Path(directory, "run_manifest.json").read_text())
            self.assertEqual(manifest["payload"]["status"], "finalization_failed")
            self.assertTrue(manifest["payload"]["commit_snapshot_written"])
            self.assertFalse(manifest["payload"]["final_proposal_written"])


if __name__ == "__main__":
    unittest.main()
