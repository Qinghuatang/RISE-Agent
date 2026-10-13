import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from research_agent.models import (
    BridgeEntity,
    BudgetMetadata,
    CandidateGap,
    CandidateSolution,
    CommitCheck,
    CommitSnapshot,
    ComparisonRecord,
    Component,
    ComponentRevisionPatch,
    ControllerDecision,
    CurrentSolution,
    DecisionRecord,
    Experiment,
    ExperimentContext,
    FailureRecord,
    FinalResearchProposal,
    GapAgenda,
    GapVerificationResult,
    InspirationCandidate,
    InspirationNeed,
    InspirationRecord,
    InspirationResult,
    InspirationSearchRecord,
    PaperMemory,
    PaperRecord,
    RelatedPaperPool,
    ResearchMemory,
    ResearchState,
    RevisionNeed,
    RevisionRecord,
    RevisionResult,
    SeedPaperContext,
    SolutionOverlap,
    SolutionVerificationResult,
    TrajectoryEvent,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.serialization import dumps_json, dumps_jsonl, loads_json, loads_jsonl


def make_gap(identifier: str = "G1") -> CandidateGap:
    return CandidateGap(
        id=identifier,
        statement="A meaningful unresolved research gap.",
        confidence="high",
        confidence_reason="The limitation is explicit.",
        priority="high",
        priority_reason="It affects the central objective.",
        status="active",
        evidence_refs=["seed:limitations"],
    )


def make_component(identifier: str) -> Component:
    return Component(
        id=identifier,
        description=f"Component {identifier}.",
        status="ready",
        evidence_refs=["seed:method"],
    )


def make_solution() -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="Use a mechanism that addresses the active gap.",
        rationale="The mechanism targets the stated failure mode.",
        components=[make_component("C1"), make_component("C2")],
        status="developed",
        novelty_status="unchecked",
        novelty_summary=None,
        evidence_refs=["gap:G1"],
    )


def make_experiment(identifier: str = "E1") -> Experiment:
    return Experiment(
        id=identifier,
        objective="Test whether the proposed mechanism helps.",
        setup="Use the seed task and a controlled split.",
        baselines=["baseline-A"],
        metrics=["accuracy"],
        what_it_validates="Whether the mechanism improves the target behavior.",
        validation_criterion="Improvement over the major baseline.",
        status="ready",
    )


class ModelContractTests(unittest.TestCase):
    def test_valid_examples_cover_all_core_contracts(self) -> None:
        context = ExperimentContext(
            tasks_or_datasets=["Dataset-A"],
            metrics=["accuracy"],
            major_baselines=["Baseline-A"],
            experimental_setting="Standard split.",
        )
        seed = SeedPaperContext(
            paper_id="seed-1",
            title="A seed paper",
            abstract="An abstract.",
            research_objective="Improve a target behavior.",
            core_method="A method.",
            main_components=["Encoder"],
            main_findings=["The method works on Dataset-A."],
            limitations=["The setting is narrow."],
            experiment_context=context,
        )
        gap = make_gap()
        agenda = GapAgenda(gaps=[gap])
        candidate = CandidateSolution(
            id="CS1", core_idea="An idea.", rationale="A reason.", status="candidate"
        )
        component = make_component("C1")
        solution = make_solution()
        experiment = make_experiment()
        state = ResearchState(
            active_gap=gap,
            candidate_solutions=[candidate],
            current_solution=solution,
            experiments=[experiment],
        )

        comparison = ComparisonRecord(
            paper_ref="paper-2",
            problem_relation="related",
            mechanism_relation="partial",
            shared_elements=["standard metric"],
            novelty_relevant_overlaps=["triggering mechanism"],
            key_difference="The intervention target differs.",
        )
        gap_verification = GapVerificationResult(
            status="partially_resolved",
            summary="Prior work addresses one aspect.",
            covered_aspects=["aspect A"],
            remaining_gap="Aspect B remains open.",
            related_papers=[comparison.paper_ref],
            evidence_level="abstract_only",
        )
        overlap = SolutionOverlap(
            component_id="C1",
            overlap_summary="The triggering mechanism is similar.",
            matched_paper_ref="paper-2",
        )
        solution_verification = SolutionVerificationResult(
            novelty_status="partial_overlap",
            summary="One novelty-bearing element overlaps.",
            shared_elements=["standard metric"],
            novelty_relevant_overlaps=[overlap],
            distinctive_elements=["different intervention target"],
            key_differences=["Different target."],
            related_papers=["paper-2"],
            evidence_level="full_text",
        )
        focus = VerificationFocus(mode="changed_units", changed_units=["C1"])

        bridge = BridgeEntity(
            name="Feedback inhibition",
            type="mechanism",
            bridge_reason="It performs the required corrective function elsewhere.",
        )
        need = InspirationNeed(
            current_problem="The current trigger overlaps prior work.",
            required_function="Activate correction when system dynamics require it.",
            current_mechanism="Uncertainty threshold.",
            rejected_mechanisms=["Static threshold"],
            relevant_context="Gap and verification summary.",
        )
        inspiration = InspirationCandidate(
            bridge_entity=bridge,
            paper_ref="arxiv:1234.5678",
            source_problem="Stabilize a dynamical system.",
            source_mechanism="Feedback inhibition.",
            transferable_principle="Use system response to trigger correction.",
            adaptation_hint="Map response dynamics to the target task.",
            grounding="arxiv",
        )
        inspiration_result = InspirationResult(status="found", inspirations=[inspiration])
        revision_need = RevisionNeed(
            problem_to_fix="Remove the overlap.",
            affected_units=["C1"],
            constraints=["Preserve the objective."],
        )
        revision_patch = ComponentRevisionPatch(
            component_id="C1", description="Use feedback inhibition instead."
        )
        revision_result = RevisionResult(
            status="revised",
            revised_core_idea=None,
            revised_rationale="The revised rationale preserves the objective.",
            revised_components=[revision_patch],
            changed_units=["C1", "rationale"],
            revision_summary="Replaced the overlapping trigger.",
            failure_reason=None,
        )

        paper = PaperRecord(
            paper_id="paper-2",
            title="A related paper",
            abstract="A related abstract.",
            authors=["Author A"],
            year=2026,
            venue="ICLR",
            source="seed-data",
            url=None,
        )
        pool = RelatedPaperPool(seed_paper_id=seed.paper_id, papers=[paper])
        search_record = InspirationSearchRecord(
            query="feedback correction mechanism",
            bridge_entities=[bridge],
            paper_refs=[paper.paper_id],
        )
        verification_record = VerificationRecord(
            id="V1",
            task="VERIFY_SOLUTION",
            target_ref=solution.id,
            verification_focus=focus,
            result=solution_verification,
            paper_refs=[paper.paper_id],
            created_at_step=3,
        )
        inspiration_record = InspirationRecord(
            id="I1",
            inspiration_need=need,
            proposed_entities=[bridge],
            selected_entities=[bridge],
            search_query="feedback correction mechanism",
            grounding_status="grounded",
            inspirations=[inspiration],
            paper_refs=[paper.paper_id],
            created_at_step=4,
        )
        revision_record = RevisionRecord(
            id="R1",
            revision_reason="The current trigger overlaps prior work.",
            affected_units=["C1"],
            selected_inspiration_ref="I1",
            changed_units=["C1"],
            revision_summary="Changed the trigger.",
            status="revised",
        )
        failure_record = FailureRecord(
            type="rejected_mechanism",
            object_ref="C1",
            summary="The trigger overlaps prior work.",
            reason="Novelty-relevant overlap.",
            evidence_refs=["V1"],
        )
        decision_record = DecisionRecord(
            step=5,
            action="VERIFY_RELATED_WORK",
            reason="Novelty needs checking.",
            result_summary="Verification completed.",
        )
        memory = ResearchMemory(
            paper_memory=PaperMemory(
                papers=[paper],
                related_paper_pool=pool,
                inspiration_search_records=[search_record],
            ),
            verification_memory=[verification_record],
            inspiration_memory=[inspiration_record],
            revision_memory=[revision_record],
            failure_memory=[failure_record],
            decision_log=[decision_record],
        )
        decision = ControllerDecision(action="COMMIT", reason="The state is ready.")
        commit_check = CommitCheck(commit_allowed=True)
        snapshot = CommitSnapshot(
            active_gap=gap,
            current_solution=solution,
            experiments=[experiment],
            latest_gap_verification=gap_verification,
            latest_solution_verification=solution_verification,
            relevant_evidence_refs=["V1"],
        )
        proposal = FinalResearchProposal(
            research_gap=gap.statement,
            proposed_solution=solution.core_idea,
            solution_rationale=solution.rationale,
            method_components=[component],
            novelty_positioning="The core mechanism is distinct.",
            experimental_plan=[experiment],
            expected_contributions=["A new mechanism."],
        )
        budget = BudgetMetadata(step_budget_remaining=5, search_budget_remaining=2)
        trajectory = TrajectoryEvent(
            step=5,
            event_type="controller_decision",
            action=decision.action,
            result_ref="decision:5",
            state_diff={"current_solution": "S1"},
            memory_writes=["decision_log"],
            budget=budget,
            timestamp=datetime.now(timezone.utc),
        )

        self.assertEqual(seed.paper_id, "seed-1")
        self.assertEqual(agenda.gaps, [gap])
        self.assertEqual(state.current_solution, solution)
        self.assertEqual(gap_verification.status, "partially_resolved")
        self.assertEqual(solution_verification.novelty_status, "partial_overlap")
        self.assertEqual(inspiration_result.status, "found")
        self.assertEqual(revision_need.affected_units, ["C1"])
        self.assertEqual(revision_result.revised_components, [revision_patch])
        self.assertEqual(memory.paper_memory.related_paper_pool, pool)
        self.assertTrue(commit_check.commit_allowed)
        self.assertEqual(snapshot.current_solution.id, "S1")
        self.assertEqual(proposal.experimental_plan, [experiment])
        self.assertEqual(trajectory.action, decision.action)

    def test_all_scientific_models_reject_unexpected_fields(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateGap(
                id="G1",
                statement="gap",
                confidence="high",
                confidence_reason="reason",
                priority="high",
                priority_reason="reason",
                status="candidate",
                unexpected="forbidden",
            )

    def test_invalid_statuses_and_m1_consistency_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            invalid_solution = make_solution().model_dump()
            invalid_solution["novelty_status"] = "unknown"
            CurrentSolution.model_validate(invalid_solution)
        with self.assertRaises(ValidationError):
            Component(id="C1", description="x", status="unchecked")
        with self.assertRaises(ValidationError):
            CurrentSolution(
                id="S1",
                core_idea="x",
                rationale="x",
                components=[make_component("C1"), make_component("C2")],
                status="candidate",
                novelty_status="unchecked",
                novelty_summary="x",
            )
        with self.assertRaises(ValidationError):
            Experiment(
                id="E1",
                objective="x",
                setup="x",
                design="x",
                what_it_validates="x",
                validation_criterion="x",
                status="ready",
            )
        with self.assertRaises(ValidationError):
            GapVerificationResult(
                status="unchecked",
                summary="x",
                evidence_level="abstract_only",
            )

        self.assertIsNone(make_solution().novelty_summary)

        gap_result = GapVerificationResult(
            status="valid", summary="The gap remains open.", evidence_level="abstract_only"
        )
        solution_result = SolutionVerificationResult(
            novelty_status="distinct",
            summary="No novelty-bearing overlap found.",
            evidence_level="abstract_only",
        )
        with self.assertRaises(ValidationError):
            VerificationRecord(
                id="V-gap-bad-result",
                task="VERIFY_GAP",
                target_ref="G1",
                result=solution_result,
                created_at_step=1,
            )
        with self.assertRaises(ValidationError):
            VerificationRecord(
                id="V-gap-bad-focus",
                task="VERIFY_GAP",
                target_ref="G1",
                verification_focus=VerificationFocus(mode="full_solution"),
                result=gap_result,
                created_at_step=1,
            )
        with self.assertRaises(ValidationError):
            VerificationRecord(
                id="V-solution-bad-result",
                task="VERIFY_SOLUTION",
                target_ref="S1",
                verification_focus=VerificationFocus(mode="full_solution"),
                result=gap_result,
                created_at_step=1,
            )
        with self.assertRaises(ValidationError):
            VerificationRecord(
                id="V-solution-bad-focus",
                task="VERIFY_SOLUTION",
                target_ref="S1",
                result=solution_result,
                created_at_step=1,
            )

        with self.assertRaises(ValidationError):
            InspirationResult(status="found")
        with self.assertRaises(ValidationError):
            InspirationResult(status="found", inspirations=[
                InspirationCandidate(
                    bridge_entity=BridgeEntity(
                        name="Entity", type="mechanism", bridge_reason="reason"
                    ),
                    source_problem="problem",
                    source_mechanism="mechanism",
                    transferable_principle="principle",
                    adaptation_hint="hint",
                    grounding="none",
                )
                for _ in range(4)
            ])
        with self.assertRaises(ValidationError):
            InspirationResult(
                status="no_useful_inspiration",
                inspirations=[InspirationCandidate(
                    bridge_entity=BridgeEntity(
                        name="Entity", type="mechanism", bridge_reason="reason"
                    ),
                    source_problem="problem",
                    source_mechanism="mechanism",
                    transferable_principle="principle",
                    adaptation_hint="hint",
                    grounding="none",
                )],
            )

        advisory_only = RevisionResult(
            status="revised",
            revised_core_idea="A changed core",
            revision_summary="The patch carries no authoritative unit metadata.",
        )
        self.assertEqual(advisory_only.changed_units, [])
        with self.assertRaises(ValidationError):
            RevisionResult(
                status="revised",
                revised_rationale="changed",
                revision_summary="bad failure",
                changed_units=["rationale"],
                failure_reason="not allowed",
            )
        with self.assertRaises(ValidationError):
            RevisionResult(status="unable_to_revise", revision_summary="missing reason")
        with self.assertRaises(ValidationError):
            RevisionResult(
                status="unable_to_revise",
                revised_rationale="not allowed",
                revision_summary="patch not allowed",
                failure_reason="failed",
            )

    def test_solution_and_agenda_size_constraints(self) -> None:
        with self.assertRaises(ValidationError):
            CurrentSolution(
                id="S1",
                core_idea="x",
                rationale="x",
                components=[make_component("C1")],
                status="developed",
                novelty_status="unchecked",
                novelty_summary="x",
            )
        with self.assertRaises(ValidationError):
            CurrentSolution(
                id="S1",
                core_idea="x",
                rationale="x",
                components=[make_component(f"C{i}") for i in range(6)],
                status="developed",
                novelty_status="unchecked",
                novelty_summary="x",
            )
        with self.assertRaises(ValidationError):
            GapAgenda(gaps=[make_gap(f"G{i}") for i in range(6)])

    def test_research_state_allows_intermediate_states(self) -> None:
        self.assertIsNone(ResearchState().active_gap)
        self.assertEqual(
            ResearchState(active_gap=make_gap(), experiments=[]).experiments, []
        )
        self.assertEqual(
            ResearchState(candidate_solutions=[CandidateSolution(
                id="CS1", core_idea="x", rationale="x", status="candidate"
            )]).current_solution,
            None,
        )
        self.assertEqual(make_solution().novelty_status, "unchecked")

    def test_verification_focus_requires_consistent_units(self) -> None:
        self.assertEqual(VerificationFocus(mode="full_solution").changed_units, [])
        with self.assertRaises(ValidationError):
            VerificationFocus(mode="full_solution", changed_units=["C1"])
        with self.assertRaises(ValidationError):
            VerificationFocus(mode="changed_units")

    def test_revision_patch_is_lightweight_and_closed(self) -> None:
        for field_name in ("status", "novelty_status", "evidence_refs"):
            with self.subTest(field_name=field_name):
                payload = {"component_id": "C1", "description": "updated"}
                payload[field_name] = "forbidden"
                with self.assertRaises(ValidationError):
                    ComponentRevisionPatch(**payload)

    def test_pydantic_json_and_jsonl_round_trip(self) -> None:
        solution = make_solution()
        text = dumps_json("research.current_solution", solution)
        decoded = loads_json(text, expected_schema="research.current_solution")
        restored = CurrentSolution.model_validate(decoded["payload"])
        self.assertEqual(restored, solution)

        jsonl = dumps_jsonl("research.current_solution", [solution, solution])
        records = loads_jsonl(jsonl, expected_schema="research.current_solution")
        self.assertEqual(len(records), 2)
        self.assertEqual(CurrentSolution.model_validate(records[0]["payload"]), solution)


if __name__ == "__main__":
    unittest.main()
