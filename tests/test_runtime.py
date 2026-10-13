import json
import tempfile
import unittest
from pathlib import Path

from research_agent.errors import SerializationError
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    CandidateSolution,
    Component,
    CurrentSolution,
    DecisionRecord,
    Experiment,
    ExperimentContext,
    FailureRecord,
    GapAgenda,
    GapVerificationResult,
    InspirationCandidate,
    InspirationNeed,
    InspirationRecord,
    InspirationSearchRecord,
    PaperRecord,
    RelatedPaperPool,
    ResearchMemory,
    ResearchState,
    RevisionRecord,
    SeedPaperContext,
    SolutionVerificationResult,
    TrajectoryEvent,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.models.inspiration import BridgeEntity
from research_agent.runtime.state_store import ResearchStateStore
from research_agent.serialization import dumps_json, dumps_jsonl
from research_agent.trajectory.context import TrajectoryContext
from research_agent.trajectory.diff import compute_state_diff
from research_agent.trajectory.recorder import TrajectoryRecorder


def make_seed(paper_id: str = "seed-1") -> SeedPaperContext:
    return SeedPaperContext(
        paper_id=paper_id,
        title="Seed paper",
        abstract="A seed abstract.",
        research_objective="Study an unresolved behavior.",
        core_method="A documented method.",
        main_components=["Encoder"],
        main_findings=["A finding."],
        limitations=[],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Dataset-A"],
            metrics=["accuracy"],
            major_baselines=["Baseline-A"],
            experimental_setting="Standard split.",
        ),
    )


def make_gap(identifier: str = "G1", status: str = "active") -> CandidateGap:
    return CandidateGap(
        id=identifier,
        statement="A meaningful unresolved gap.",
        confidence="high",
        confidence_reason="The limitation is explicit.",
        priority="high",
        priority_reason="It affects the objective.",
        status=status,
        evidence_refs=["seed:limitations"],
    )


def make_state() -> ResearchState:
    return ResearchState(
        active_gap=make_gap(),
        candidate_solutions=[
            CandidateSolution(
                id="CS1", core_idea="Candidate idea.", rationale="Candidate reason.", status="candidate"
            )
        ],
        current_solution=CurrentSolution(
            id="S1",
            core_idea="Core idea.",
            rationale="Rationale.",
            components=[
                Component(id="C1", description="First component.", status="ready"),
                Component(id="C2", description="Second component.", status="ready"),
            ],
            status="developed",
            novelty_status="unchecked",
        ),
        experiments=[
            Experiment(
                id="E1",
                objective="Test the mechanism.",
                setup="Use Dataset-A.",
                baselines=["Baseline-A"],
                metrics=["accuracy"],
                what_it_validates="The core mechanism.",
                validation_criterion="Improvement over baseline.",
                status="ready",
            )
        ],
    )


def make_paper(identifier: str = "paper-2") -> PaperRecord:
    return PaperRecord(
        paper_id=identifier,
        title="Related paper",
        abstract="Related abstract.",
        source="fixture",
    )


def make_gap_verification(identifier: str, step: int, target: str = "G1") -> VerificationRecord:
    return VerificationRecord(
        id=identifier,
        task="VERIFY_GAP",
        target_ref=target,
        result=GapVerificationResult(
            status="valid",
            summary=f"Verification {identifier}.",
            remaining_gap="The gap remains.",
            evidence_level="abstract_only",
        ),
        created_at_step=step,
    )


class RuntimeFoundationTests(unittest.TestCase):
    def test_new_trajectory_has_empty_state_and_memory_without_seed_copy(self) -> None:
        context = TrajectoryContext.initialize(make_seed())
        self.assertEqual(context.get_state(), ResearchState())
        self.assertEqual(context.get_memory(), ResearchMemory())
        self.assertNotIn("research_objective", context.get_state().model_dump())
        self.assertEqual(context.get_memory().paper_memory.papers, [])
        self.assertEqual(context.trajectory.get_events()[0].event_type, "trajectory_initialized")

    def test_state_store_validates_and_returns_defensive_snapshots(self) -> None:
        store = ResearchStateStore()
        snapshot = store.get_state()
        snapshot.candidate_solutions.append(
            CandidateSolution(id="CS1", core_idea="Idea.", rationale="Reason.", status="candidate")
        )
        self.assertEqual(store.get_state().candidate_solutions, [])
        store.replace_state(make_state())
        self.assertEqual(store.get_state().active_gap.id, "G1")
        with self.assertRaises(ValueError):
            store.replace_state({"unexpected": True})  # type: ignore[arg-type]

    def test_state_diff_reports_nested_changes_deterministically(self) -> None:
        before = make_state()
        after = before.model_copy(deep=True)
        after.current_solution.components[1].description = "Changed component."
        after.current_solution.novelty_status = "distinct"
        after.experiments[0].status = "stale"
        diff = compute_state_diff(before, after)
        self.assertEqual(
            diff["current_solution.components[1].description"],
            {"before": "Second component.", "after": "Changed component."},
        )
        self.assertEqual(
            diff["current_solution.novelty_status"],
            {"before": "unchecked", "after": "distinct"},
        )
        self.assertEqual(
            diff["experiments[0].status"],
            {"before": "ready", "after": "stale"},
        )
        json.dumps(diff)
        self.assertEqual(before.current_solution.components[1].description, "Second component.")

    def test_memory_save_paper_is_upsert_without_duplicates(self) -> None:
        store = ResearchMemoryStore("seed-1")
        paper = make_paper()
        store.save_paper(paper)
        store.save_paper(paper)
        self.assertEqual(len(store.get_memory().paper_memory.papers), 1)
        updated = paper.model_copy(update={"title": "Updated title"})
        store.save_paper(updated)
        self.assertEqual(len(store.get_memory().paper_memory.papers), 1)
        self.assertEqual(store.get_memory().paper_memory.papers[0].title, "Updated title")

    def test_related_paper_pool_must_match_seed_paper(self) -> None:
        store = ResearchMemoryStore("seed-1")
        with self.assertRaises(ValueError):
            store.set_related_paper_pool(RelatedPaperPool(seed_paper_id="other", papers=[]))
        pool = RelatedPaperPool(seed_paper_id="seed-1", papers=[make_paper()])
        store.set_related_paper_pool(pool)
        self.assertEqual([paper.paper_id for paper in store.get_related_papers()], ["paper-2"])

    def test_latest_gap_verification_uses_step_then_insertion_order(self) -> None:
        store = ResearchMemoryStore("seed-1")
        store.save_verification(make_gap_verification("V1", 2))
        store.save_verification(make_gap_verification("V2", 5))
        store.save_verification(make_gap_verification("V3", 5))
        self.assertEqual(store.get_latest_gap_verification("G1").id, "V3")
        self.assertIsNone(store.get_latest_gap_verification("G2"))
        self.assertEqual(store.get_latest_verification("G1", task="VERIFY_GAP").id, "V3")

    def test_memory_records_and_simple_relevant_queries(self) -> None:
        store = ResearchMemoryStore("seed-1")
        failure = FailureRecord(
            type="rejected_mechanism",
            object_ref="C1",
            summary="Rejected mechanism.",
            reason="Novelty overlap.",
        )
        store.save_failure(failure)
        self.assertEqual(len(store.get_relevant_failures(failure_type="rejected_mechanism")), 1)
        self.assertEqual(store.get_relevant_failures(object_ref="missing"), [])
        self.assertEqual(store.get_relevant_failures(recent_n=1)[0].object_ref, "C1")
        self.assertEqual(store.get_relevant_failures(recent_n=0), [])

        bridge = BridgeEntity(
            name="Feedback inhibition",
            type="mechanism",
            bridge_reason="It provides corrective feedback.",
        )
        need = InspirationNeed(
            current_problem="Problem.",
            required_function="Function.",
            current_mechanism="Mechanism.",
            relevant_context="Context.",
        )
        candidate = InspirationCandidate(
            bridge_entity=bridge,
            paper_ref="paper-2",
            source_problem="Source problem.",
            source_mechanism="Source mechanism.",
            transferable_principle="Transfer principle.",
            adaptation_hint="Adaptation.",
            grounding="arxiv",
        )
        inspiration = InspirationRecord(
            id="I1",
            inspiration_need=need,
            proposed_entities=[bridge],
            selected_entities=[bridge],
            search_query="feedback",
            grounding_status="grounded",
            inspirations=[candidate],
            created_at_step=4,
        )
        store.save_inspiration_search(
            InspirationSearchRecord(query="feedback", bridge_entities=[bridge], paper_refs=["paper-2"])
        )
        store.save_inspiration(inspiration)
        self.assertEqual(store.get_relevant_inspirations(record_id="I1")[0].id, "I1")
        self.assertEqual(store.get_relevant_inspirations(latest_n=0), [])

        revision = RevisionRecord(
            id="R1",
            revision_reason="Fix overlap.",
            affected_units=["C1"],
            changed_units=["C1"],
            revision_summary="Changed C1.",
            status="revised",
        )
        store.save_revision(revision)
        store.save_decision(
            DecisionRecord(
                step=5,
                action="VERIFY_RELATED_WORK",
                reason="Verify.",
                result_summary="Done.",
            )
        )
        self.assertEqual(store.get_revision_history()[0].id, "R1")
        self.assertEqual(store.get_memory().decision_log[0].step, 5)

    def test_failure_store_does_not_infer_shared_element_failures(self) -> None:
        store = ResearchMemoryStore("seed-1")
        self.assertEqual(store.get_memory().failure_memory, [])
        # A shared element is merely data supplied to a future module; the store
        # has no method that classifies it as a FailureRecord.
        store.save_paper(make_paper())
        self.assertEqual(store.get_memory().failure_memory, [])

    def test_memory_json_round_trip_and_trajectory_isolation(self) -> None:
        store = ResearchMemoryStore("seed-1")
        store.save_paper(make_paper())
        store.set_related_paper_pool(RelatedPaperPool(seed_paper_id="seed-1", papers=[]))
        other = ResearchMemoryStore("seed-2")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "research_memory.json"
            store.save(path)
            loaded = ResearchMemoryStore.load(path, "seed-1")
            self.assertEqual(loaded.get_memory(), store.get_memory())
            with self.assertRaises(SerializationError):
                ResearchMemoryStore.load(path, "seed-2")
        self.assertEqual(other.get_memory().paper_memory.papers, [])

    def test_trajectory_order_and_negative_steps_are_rejected(self) -> None:
        recorder = TrajectoryRecorder()
        recorder.append_event(1, "event_a")
        recorder.append_event(1, "event_b")
        with self.assertRaises(ValueError):
            recorder.append_event(0, "event_backwards")
        with self.assertRaises(ValueError):
            recorder.append_event(-1, "event_negative")
        self.assertEqual([event.event_type for event in recorder.get_events()], ["event_a", "event_b"])

    def test_gap_agenda_records_and_returns_latest(self) -> None:
        recorder = TrajectoryRecorder()
        recorder.record_gap_agenda(1, GapAgenda(gaps=[make_gap("G1")]))
        recorder.record_gap_agenda(2, GapAgenda(gaps=[make_gap("G2")]))
        latest = recorder.get_latest_gap_agenda()
        self.assertEqual(latest.gaps[0].id, "G2")
        with self.assertRaises(SerializationError):
            bad = TrajectoryRecorder()
            bad.append_event(1, "gap_agenda_updated", result_payload={"gaps": [{"bad": True}]})
            bad.get_latest_gap_agenda()

    def test_gap_refined_event_is_gap_specific(self) -> None:
        recorder = TrajectoryRecorder()
        recorder.record_gap_refined(1, "G1")
        self.assertTrue(recorder.has_refined_gap("G1"))
        self.assertFalse(recorder.has_refined_gap("G2"))

    def test_trajectory_jsonl_round_trip_and_corruption_is_rejected(self) -> None:
        recorder = TrajectoryRecorder()
        recorder.append_event(0, "trajectory_initialized", result_payload={"ok": True})
        recorder.append_event(2, "later", action="COMMIT")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.jsonl"
            recorder.save(path)
            loaded = TrajectoryRecorder.load(path)
            self.assertEqual(loaded.get_events(), recorder.get_events())

            invalid_order = Path(temp_dir) / "invalid-order.jsonl"
            events = recorder.get_events()
            invalid_order.write_text(
                dumps_jsonl(
                    "trajectory_event",
                    [events[1], events[0]],
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SerializationError):
                TrajectoryRecorder.load(invalid_order)

            invalid_envelope = Path(temp_dir) / "invalid-envelope.jsonl"
            invalid_envelope.write_text('{"schema":"wrong","version":1,"payload":{}}\n', encoding="utf-8")
            with self.assertRaises(SerializationError):
                TrajectoryRecorder.load(invalid_envelope)

    def test_trajectory_context_integration_persists_state_memory_and_events(self) -> None:
        context = TrajectoryContext.initialize(make_seed())
        context.record_gap_agenda(1, GapAgenda(gaps=[make_gap("G1")]))
        before = context.get_state()
        after = make_state()
        context.memory_store.save_verification(make_gap_verification("V1", 2))
        context.memory_store.save_decision(
            DecisionRecord(
                step=3,
                action="VERIFY_RELATED_WORK",
                reason="An active gap needs verification.",
                result_summary="Fixture verification recorded.",
            )
        )
        context.record_transition(
            3,
            "state_updated",
            before,
            after,
            action="VERIFY_RELATED_WORK",
            result_ref="V1",
            memory_writes=["verification:V1", "decision:3"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = context.persist(temp_dir)
            self.assertTrue(all(path.exists() for path in paths.values()))
            loaded = TrajectoryContext.load(make_seed(), temp_dir)
            self.assertEqual(loaded.get_state(), after)
            self.assertEqual(loaded.memory_store.get_latest_gap_verification("G1").id, "V1")
            self.assertEqual(loaded.get_latest_gap_agenda().gaps[0].id, "G1")
            self.assertTrue(any(event.state_diff for event in loaded.trajectory.get_events()))
            self.assertEqual(len(loaded.trajectory.get_events()), 3)


if __name__ == "__main__":
    unittest.main()
