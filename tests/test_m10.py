import tempfile
import unittest
from pathlib import Path

from research_agent.errors import PreconditionError, StructuredOutputError
from research_agent.llm.fake import FakeLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    CandidateSolution,
    Component,
    CurrentSolution,
    Experiment,
    ExperimentContext,
    GapVerificationResult,
    ResearchState,
    SeedPaperContext,
    SolutionVerificationResult,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.models.commit import CommitSnapshot, FinalizerDraft
from research_agent.models.memory import PaperRecord, RelatedPaperPool
from research_agent.modules.commit import (
    CommitModule,
    check_commit_gate,
    create_commit_snapshot,
    load_commit_snapshot,
    save_commit_snapshot,
)
from research_agent.modules.finalizer import (
    FinalizerModule,
    load_final_research_proposal,
    save_final_research_proposal,
)
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-commit-seed",
        title="Generic Seed",
        abstract="Supplied abstract.",
        research_objective="Improve a supplied method.",
        core_method="A supplied method with interacting units.",
        main_components=["Unit one", "Unit two"],
        main_findings=["A supplied finding."],
        limitations=["A supplied limitation."],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task-A"],
            metrics=["metric-A"],
            major_baselines=["Seed Method"],
            experimental_setting="The supplied setting.",
        ),
    )


def gap(status: str = "active") -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="A supported method-level gap.",
        confidence="medium",
        confidence_reason="Supplied evidence supports the gap.",
        priority="high",
        priority_reason="It affects the method.",
        status=status,
        evidence_refs=["seed:core_method", "seed:core_method"],
    )


def solution(
    novelty: str = "distinct",
    component_status: tuple[str, str] = ("ready", "ready"),
) -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="A committed solution core.",
        rationale="A committed rationale.",
        components=[
            Component(
                id=f"C{index}",
                description=f"Committed unit {index}.",
                status=status,
                evidence_refs=["solution:evidence", "shared:evidence"],
            )
            for index, status in enumerate(component_status, start=1)
        ],
        status="developed",
        novelty_status=novelty,
        novelty_summary="The supplied novelty summary." if novelty != "unchecked" else None,
        evidence_refs=["solution:evidence", "shared:evidence"],
    )


def experiment(identifier: str, status: str = "ready") -> Experiment:
    return Experiment(
        id=identifier,
        objective=f"Evaluate {identifier}.",
        setup="Task-A under the supplied setting.",
        baselines=["Seed Method", "Baseline-A"],
        metrics=["metric-A"],
        what_it_validates="This validates core_solution and C1.",
        validation_criterion="The observation would support the committed claim.",
        status=status,
    )


def gap_verification(status: str = "valid", step: int = 1) -> VerificationRecord:
    return VerificationRecord(
        id=f"VG{step}",
        task="VERIFY_GAP",
        target_ref="G1",
        result=GapVerificationResult(
            status=status,
            summary="The supplied gap verification.",
            remaining_gap="The remaining gap." if status != "resolved" else None,
            related_papers=["paper:gap", "shared:evidence"],
            evidence_level="abstract_only",
        ),
        paper_refs=["paper:gap"],
        created_at_step=step,
    )


def solution_verification(novelty: str = "distinct", step: int = 2) -> VerificationRecord:
    return VerificationRecord(
        id=f"VS{step}",
        task="VERIFY_SOLUTION",
        target_ref="S1",
        verification_focus=VerificationFocus(mode="full_solution"),
        result=SolutionVerificationResult(
            novelty_status=novelty,
            summary="The supplied solution verification.",
            shared_elements=["A shared standard element."],
            novelty_relevant_overlaps=[],
            distinctive_elements=["A committed distinctive element."],
            key_differences=["A committed difference."],
            related_papers=["paper:solution", "shared:evidence"],
            evidence_level="abstract_only",
        ),
        paper_refs=["paper:solution"],
        created_at_step=step,
    )


_UNSET = object()


def committed_runtime(
    *,
    current_solution: CurrentSolution | None | object = _UNSET,
    experiments: list[Experiment] | None = None,
    gap_record: VerificationRecord | None = None,
    solution_record: VerificationRecord | None = None,
) -> TrajectoryContext:
    context = TrajectoryContext.initialize(seed())
    context.state_store.replace_state(
        ResearchState(
            active_gap=gap(),
            candidate_solutions=[
                CandidateSolution(
                    id="CS1",
                    core_idea="An old candidate.",
                    rationale="Candidate rationale.",
                    status="selected",
                )
            ],
            current_solution=solution() if current_solution is _UNSET else current_solution,
            experiments=experiments if experiments is not None else [experiment("E1"), experiment("E2")],
        )
    )
    if gap_record is not None:
        context.memory_store.save_verification(gap_record)
    if solution_record is not None:
        context.memory_store.save_verification(solution_record)
    return context


class CommitGateTests(unittest.TestCase):
    def test_gate_false_for_missing_or_invalid_conditions(self):
        cases = [
            ("no active gap", committed_runtime(), lambda c: c.state_store.replace_state(ResearchState())),
            ("no gap verification", committed_runtime(solution_record=solution_verification()), None),
            ("partial gap", committed_runtime(gap_record=gap_verification("partially_resolved"), solution_record=solution_verification()), None),
            ("resolved gap", committed_runtime(gap_record=gap_verification("resolved"), solution_record=solution_verification()), None),
            ("no solution", committed_runtime(gap_record=gap_verification()), None),
            ("unchecked novelty", committed_runtime(current_solution=solution("unchecked"), gap_record=gap_verification(), solution_record=solution_verification("distinct")), None),
            ("partial novelty", committed_runtime(current_solution=solution("partial_overlap"), gap_record=gap_verification(), solution_record=solution_verification("partial_overlap")), None),
            ("missing solution verification", committed_runtime(gap_record=gap_verification()), None),
            ("non-distinct verification", committed_runtime(current_solution=solution("partial_overlap"), gap_record=gap_verification(), solution_record=solution_verification("partial_overlap")), None),
            ("modified component", committed_runtime(current_solution=solution(component_status=("modified", "ready")), gap_record=gap_verification(), solution_record=solution_verification()), None),
            ("stale component", committed_runtime(current_solution=solution(component_status=("ready", "stale")), gap_record=gap_verification(), solution_record=solution_verification()), None),
            ("no experiments", committed_runtime(experiments=[], gap_record=gap_verification(), solution_record=solution_verification()), None),
            ("one experiment", committed_runtime(experiments=[experiment("E1")], gap_record=gap_verification(), solution_record=solution_verification()), None),
            ("stale experiment", committed_runtime(experiments=[experiment("E1"), experiment("E2", "stale")], gap_record=gap_verification(), solution_record=solution_verification()), None),
        ]
        for name, context, mutation in cases:
            with self.subTest(name=name):
                if mutation is not None:
                    mutation(context)
                check = check_commit_gate(context.get_state(), context.memory_store)
                self.assertFalse(check.commit_allowed)
                self.assertTrue(check.unmet_conditions)

    def test_no_solution_fixture_really_removes_current_solution(self):
        context = committed_runtime(
            current_solution=None,
            gap_record=gap_verification(),
            solution_record=None,
        )
        self.assertIsNone(context.get_state().current_solution)
        check = check_commit_gate(context.get_state(), context.memory_store)
        self.assertFalse(check.commit_allowed)
        self.assertIn("Current Solution is required.", check.unmet_conditions)

    def test_gate_true_for_complete_state(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        check = check_commit_gate(context.get_state(), context.memory_store)
        self.assertTrue(check.commit_allowed)
        self.assertEqual(check.unmet_conditions, [])

    def test_state_and_latest_solution_verification_conflict_is_explicit(self):
        context = committed_runtime(
            gap_record=gap_verification(),
            solution_record=solution_verification("partial_overlap", step=3),
        )
        with self.assertRaises(PreconditionError):
            check_commit_gate(context.get_state(), context.memory_store)

    def test_latest_verification_is_selected_by_step_and_matching_target(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        context.memory_store.save_verification(solution_verification("distinct", step=1))
        context.memory_store.save_verification(solution_verification("partial_overlap", step=3))
        with self.assertRaises(PreconditionError):
            check_commit_gate(context.get_state(), context.memory_store)

    def test_gate_has_no_side_effects(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        before_state = context.get_state()
        before_memory = context.get_memory()
        before_events = context.trajectory.get_events()
        check_commit_gate(before_state, context.memory_store)
        self.assertEqual(context.get_state(), before_state)
        self.assertEqual(context.get_memory(), before_memory)
        self.assertEqual(context.trajectory.get_events(), before_events)


class CommitSnapshotTests(unittest.TestCase):
    def _snapshot(self) -> tuple[TrajectoryContext, CommitSnapshot]:
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        return context, create_commit_snapshot(context.get_state(), context.memory_store)

    def test_snapshot_requires_gate(self):
        context = committed_runtime()
        with self.assertRaises(PreconditionError):
            create_commit_snapshot(context.get_state(), context.memory_store)

    def test_snapshot_contains_latest_results_and_stable_deduplicated_refs(self):
        context, snapshot = self._snapshot()
        self.assertEqual(snapshot.active_gap.id, "G1")
        self.assertEqual(snapshot.current_solution.id, "S1")
        self.assertEqual([e.id for e in snapshot.experiments], ["E1", "E2"])
        self.assertEqual(snapshot.latest_gap_verification.status, "valid")
        self.assertEqual(snapshot.latest_solution_verification.novelty_status, "distinct")
        self.assertEqual(
            snapshot.relevant_evidence_refs,
            [
                "seed:core_method",
                "solution:evidence",
                "shared:evidence",
                "paper:gap",
                "paper:solution",
            ],
        )
        self.assertFalse(hasattr(snapshot, "candidate_solutions"))
        self.assertEqual(context.get_memory().decision_log, [])

    def test_snapshot_is_structurally_isolated_from_state_memory_and_itself(self):
        context, snapshot = self._snapshot()
        context_state = context.state_store.get_state()
        context_state.active_gap.statement = "Changed after commit."
        context_state.current_solution.components[0].description = "Changed unit."
        context_state.experiments[0].objective = "Changed experiment."
        context.state_store.replace_state(context_state)
        memory = context.memory_store.get_memory()
        memory.verification_memory[0].result.summary = "Changed verification."
        context.memory_store.save_verification(memory.verification_memory[0])
        self.assertEqual(snapshot.active_gap.statement, "A supported method-level gap.")
        self.assertEqual(snapshot.current_solution.components[0].description, "Committed unit 1.")
        self.assertEqual(snapshot.experiments[0].objective, "Evaluate E1.")
        self.assertEqual(snapshot.latest_gap_verification.summary, "The supplied gap verification.")

        snapshot.active_gap.statement = "Detached snapshot edit."
        snapshot.current_solution.components[0].description = "Detached unit."
        snapshot.experiments[0].objective = "Detached experiment."
        self.assertNotEqual(context.get_state().active_gap.statement, snapshot.active_gap.statement)
        self.assertNotEqual(context.get_state().current_solution.components[0].description, snapshot.current_solution.components[0].description)

    def test_snapshot_artifact_round_trip(self):
        _, snapshot = self._snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = save_commit_snapshot(snapshot, Path(directory) / "commit_snapshot.json")
            loaded = load_commit_snapshot(path)
        self.assertEqual(loaded, snapshot)


class CommitActionTests(unittest.TestCase):
    def test_successful_commit_records_empty_state_diff_and_no_memory_writes(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        before_state = context.get_state()
        before_memory = context.get_memory()
        snapshot = CommitModule().commit(context=context, step=1)
        self.assertEqual(context.get_state(), before_state)
        self.assertEqual(context.get_memory(), before_memory)
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "research_committed")
        self.assertEqual(event.action, "COMMIT")
        self.assertEqual(event.state_diff, {})
        self.assertEqual(event.memory_writes, [])
        self.assertEqual(event.result_payload["active_gap_id"], snapshot.active_gap.id)

    def test_invalid_step_and_gate_failure_do_not_record_commit_event(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        with self.assertRaises(ValueError):
            CommitModule().commit(context=context, step=-1)
        self.assertEqual(len(context.trajectory.get_events()), 1)
        blocked = committed_runtime()
        with self.assertRaises(PreconditionError):
            CommitModule().commit(context=blocked, step=1)
        self.assertEqual(len(blocked.trajectory.get_events()), 1)


def finalizer_draft() -> dict:
    return {
        "novelty_positioning": "The solution shares standard elements while preserving the supplied distinctive mechanism within the retrieved related-work evidence.",
        "expected_contributions": [
            "An expected method-level improvement.",
            "A potential contribution evaluated by the committed experiments.",
        ],
    }


class FinalizerTests(unittest.TestCase):
    def _snapshot_and_seed(self):
        context = committed_runtime(
            gap_record=gap_verification(), solution_record=solution_verification()
        )
        snapshot = create_commit_snapshot(context.get_state(), context.memory_store)
        return context, snapshot, seed()

    def test_finalizer_uses_one_call_and_materializes_exact_state_units(self):
        context, snapshot, seed_context = self._snapshot_and_seed()
        llm = FakeLLMClient([finalizer_draft()])
        finalizer = FinalizerModule(llm)
        before_state = context.get_state()
        before_memory = context.get_memory()
        before_snapshot = snapshot.model_copy(deep=True)
        proposal = finalizer.finalize(snapshot, seed_context)
        self.assertEqual(finalizer.call_count, 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertIsInstance(llm.calls[0], object)
        self.assertEqual(proposal.method_components, snapshot.current_solution.components)
        self.assertEqual(proposal.experimental_plan, snapshot.experiments)
        self.assertEqual(proposal.expected_contributions, finalizer_draft()["expected_contributions"])
        self.assertEqual(proposal.research_gap, snapshot.active_gap.statement)
        self.assertEqual(proposal.proposed_solution, snapshot.current_solution.core_idea)
        self.assertEqual(proposal.solution_rationale, snapshot.current_solution.rationale)
        self.assertEqual(context.get_state(), before_state)
        self.assertEqual(context.get_memory(), before_memory)
        self.assertEqual(snapshot, before_snapshot)

    def test_finalizer_prompt_has_schema_and_boundary_rules_but_compact_input(self):
        _, snapshot, seed_context = self._snapshot_and_seed()
        llm = FakeLLMClient([finalizer_draft()])
        FinalizerModule(llm).finalize(snapshot, seed_context)
        system = llm.calls[0].messages[0].content
        user = llm.calls[0].messages[1].content
        self.assertIn("READ-ONLY FINALIZATION", system)
        self.assertIn("PRESERVE COMMITTED SCIENTIFIC CONTENT", system)
        self.assertIn("NOVELTY EVIDENCE BOUNDARY", system)
        self.assertIn("EXPERIMENTS ARE PLANS", system)
        self.assertIn("EXPECTED / POTENTIAL CONTRIBUTIONS", system)
        self.assertIn('"expected_contributions"', system)
        self.assertIn("shared_elements", system)
        self.assertIn("abstract_only", system)
        self.assertIn("research_objective", user)
        self.assertIn("core_method", user)
        self.assertIn("novelty_context", user)
        self.assertNotIn("related_paper_pool", user)
        self.assertNotIn("trajectory", user)
        self.assertNotIn("abstracts", user)

    def test_finalizer_draft_is_closed_and_contribution_count_is_two_to_three(self):
        _, snapshot, seed_context = self._snapshot_and_seed()
        extra = finalizer_draft()
        extra["method_components"] = []
        with self.assertRaises(StructuredOutputError):
            FinalizerModule(FakeLLMClient([extra])).finalize(snapshot, seed_context)
        for count in (1, 4):
            payload = finalizer_draft()
            payload["expected_contributions"] = ["Only this."] * count
            with self.subTest(count=count):
                with self.assertRaises(StructuredOutputError):
                    FinalizerModule(FakeLLMClient([payload])).finalize(snapshot, seed_context)
        three = finalizer_draft()
        three["expected_contributions"] = ["One.", "Two.", "Three."]
        proposal = FinalizerModule(FakeLLMClient([three])).finalize(snapshot, seed_context)
        self.assertEqual(len(proposal.expected_contributions), 3)

    def test_finalizer_artifact_round_trip(self):
        _, snapshot, seed_context = self._snapshot_and_seed()
        proposal = FinalizerModule(FakeLLMClient([finalizer_draft()])).finalize(snapshot, seed_context)
        with tempfile.TemporaryDirectory() as directory:
            path = save_final_research_proposal(
                proposal, Path(directory) / "final_research_proposal.json"
            )
            loaded = load_final_research_proposal(path)
        self.assertEqual(loaded, proposal)

    def test_finalizer_draft_schema_has_no_component_or_experiment_fields(self):
        schema = FinalizerDraft.model_json_schema()
        properties = set(schema["properties"])
        self.assertNotIn("method_components", properties)
        self.assertNotIn("experimental_plan", properties)
        self.assertNotIn("research_gap", properties)
        self.assertNotIn("proposed_solution", properties)
        self.assertNotIn("solution_rationale", properties)


if __name__ == "__main__":
    unittest.main()
