import json
import unittest

from research_agent.errors import ModuleContractError, PreconditionError, StructuredOutputError
from research_agent.llm.fake import FakeLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    CandidateSolution,
    Component,
    CurrentSolution,
    Experiment,
    ExperimentContext,
    ExperimentDesignDraft,
    GapVerificationResult,
    ResearchState,
    SeedPaperContext,
    SolutionOverlap,
    SolutionVerificationResult,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.modules.experiment_design import ExperimentDesignModule
from research_agent.modules.commit import check_commit_gate
from research_agent.runtime.state_store import ResearchStateStore
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-experiment-seed",
        title="A Generic Experiment Seed",
        abstract="A generic supplied abstract.",
        research_objective="Test a method-level opportunity.",
        core_method="A mechanism with multiple components.",
        main_components=["Component one", "Component two"],
        main_findings=["A supplied finding."],
        limitations=["A supplied limitation."],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task-A", "Dataset-A"],
            metrics=["accuracy", "calibration"],
            major_baselines=["Baseline-A"],
            experimental_setting="Use the supplied standard split.",
        ),
    )


def gap() -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="A generic mechanism-level gap.",
        confidence="medium",
        confidence_reason="The supplied method evidence supports it.",
        priority="high",
        priority_reason="It affects the central method.",
        status="active",
        evidence_refs=["seed:core_method"],
    )


def solution(
    *,
    novelty: str = "distinct",
    component_statuses: tuple[str, str] = ("ready", "ready"),
) -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="A verified generic solution core.",
        rationale="A supplied rationale.",
        components=[
            Component(
                id=f"C{index}",
                description=f"Generic component {index}.",
                status=status,
            )
            for index, status in enumerate(component_statuses, start=1)
        ],
        status="developed",
        novelty_status=novelty,
        novelty_summary="The solution is distinct." if novelty == "distinct" else "Not distinct.",
        evidence_refs=["evidence:solution"],
    )


def gap_verification() -> VerificationRecord:
    return VerificationRecord(
        id="VG1",
        task="VERIFY_GAP",
        target_ref="G1",
        result=GapVerificationResult(
            status="valid",
            summary="The active gap remains valid.",
            remaining_gap="The mechanism-level gap remains.",
            related_papers=["paper:gap"],
            evidence_level="abstract_only",
        ),
        paper_refs=["paper:gap"],
        created_at_step=1,
    )


def solution_verification(
    *, focus: VerificationFocus | None = None, status: str = "distinct"
) -> VerificationRecord:
    focus = focus or VerificationFocus(mode="full_solution")
    result = SolutionVerificationResult(
        novelty_status=status,
        summary="The solution is distinct from the supplied related work.",
        shared_elements=["standard comparison"],
        novelty_relevant_overlaps=[],
        distinctive_elements=["A generic distinctive mechanism."],
        key_differences=["A generic key difference."],
        related_papers=["paper:solution"],
        evidence_level="abstract_only",
    )
    return VerificationRecord(
        id="VS1",
        task="VERIFY_SOLUTION",
        target_ref="S1",
        verification_focus=focus,
        result=result,
        paper_refs=["paper:solution"],
        created_at_step=2,
    )


def experiment(
    identifier: str,
    objective: str,
    what: str,
    *,
    status: str = "ready",
    setup: str = "Task-A with the supplied standard split.",
    baselines: list[str] | None = None,
) -> Experiment:
    return Experiment(
        id=identifier,
        objective=objective,
        setup=setup,
        baselines=baselines or ["Seed Method"],
        metrics=["accuracy"],
        what_it_validates=what,
        validation_criterion="The observation supports the claim without fabricated results.",
        status=status,
    )


def context_with_state(
    *,
    experiments: list[Experiment] | None = None,
    current_solution: CurrentSolution | None = None,
    active_gap: CandidateGap | None = None,
    add_gap_verification: bool = True,
    add_solution_verification: bool = True,
    focus: VerificationFocus | None = None,
) -> TrajectoryContext:
    context = TrajectoryContext.initialize(seed())
    context.state_store.replace_state(
        ResearchState(
            active_gap=active_gap if active_gap is not None else gap(),
            candidate_solutions=[
                CandidateSolution(
                    id="CS1",
                    core_idea="Candidate core.",
                    rationale="Candidate rationale.",
                    status="selected",
                )
            ],
            current_solution=current_solution or solution(),
            experiments=experiments or [],
        )
    )
    if add_gap_verification:
        context.memory_store.save_verification(gap_verification())
    if add_solution_verification:
        context.memory_store.save_verification(
            solution_verification(focus=focus)
        )
    return context


def draft_payload(
    *,
    proposals: list[dict],
    claims: list[dict] | None = None,
    summary: str = "A minimal sufficient generic experiment set.",
) -> dict:
    if claims is None:
        claims = [
            {
                "id": "K1",
                "statement": "The complete solution addresses the active gap.",
                "target_unit": "core_solution",
            },
            {
                "id": "K2",
                "statement": "The key mechanism contributes as intended.",
                "target_unit": "C1",
            },
        ]
    return {
        "key_claims": claims,
        "experiments": proposals,
        "identifiability_checks": [
            {
                "claim_id": claim["id"],
                "required_observable": "A directly observable claim outcome.",
                "dataset_support": "The supplied task exposes the observable.",
                "ground_truth_availability": "The supplied evaluation target is available.",
                "control_or_baseline": "Seed Method provides the control.",
                "falsification": "No difference under the controlled comparison falsifies the claim.",
                "identifiable": True,
            }
            for claim in claims
        ],
        "design_summary": summary,
    }


def initial_proposals(count: int = 2) -> list[dict]:
    proposals = [
        {
            "existing_experiment_id": None,
            "objective": "Test overall solution effectiveness.",
            "setup": "Task-A with the supplied standard split; compare methods.",
            "baselines": ["Seed Method", "Baseline-A"],
            "metrics": ["accuracy"],
            "what_it_validates": "core_solution; overall effectiveness",
            "validation_criterion": "The solution improves the gap-related behavior against Seed Method.",
            "validates_claims": ["K1"],
        },
        {
            "existing_experiment_id": None,
            "objective": "Test the key mechanism contribution.",
            "setup": "Task-A with a controlled mechanism comparison.",
            "baselines": ["Seed Method"],
            "metrics": ["calibration"],
            "what_it_validates": "C1; mechanism contribution",
            "validation_criterion": "The mechanism changes the targeted behavior in the expected direction.",
            "validates_claims": ["K2"],
        },
    ]
    if count <= 2:
        return proposals[:count]
    for index in range(3, count + 1):
        proposals.append(
            {
                "existing_experiment_id": None,
                "objective": f"Test boundary condition {index}.",
                "setup": f"Task-A under boundary condition {index}.",
                "baselines": ["Seed Method"],
                "metrics": ["accuracy"],
                "what_it_validates": "C1; boundary condition",
                "validation_criterion": "The mechanism remains interpretable under the condition.",
                "validates_claims": ["K2"],
            }
        )
    return proposals


class M8ExperimentDesignTests(unittest.TestCase):
    def test_unidentifiable_claim_returns_normal_unable_to_design_outcome(self):
        proposals = initial_proposals()
        payload = draft_payload(proposals=proposals)
        payload["identifiability_checks"][1]["identifiable"] = False
        context = context_with_state()
        result = ExperimentDesignModule(FakeLLMClient([payload])).design(context=context)
        self.assertEqual(result.status, "unable_to_design")
        self.assertIn("K2", result.failure_reason)
        self.assertEqual(context.get_state().experiments, [])
        self.assertFalse(check_commit_gate(context.get_state(), context.memory_store).commit_allowed)
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "experiments_design_unresolved")
        self.assertEqual(event.action, "DESIGN_EXPERIMENT")
        self.assertEqual(event.state_diff, {})

    def test_singleton_anchor_list_is_canonicalized_but_ambiguous_list_is_rejected(self):
        proposals = initial_proposals()
        proposals[0]["what_it_validates"] = ["core_solution"]
        proposals[1]["what_it_validates"] = ["C1"]
        result = ExperimentDesignModule(
            FakeLLMClient([draft_payload(proposals=proposals)])
        ).design(context=context_with_state())
        self.assertEqual(
            [experiment.what_it_validates for experiment in result.experiments],
            ["core_solution", "C1"],
        )

        ambiguous = initial_proposals()
        ambiguous[0]["what_it_validates"] = ["core_solution", "C1"]
        with self.assertRaises(StructuredOutputError):
            ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=ambiguous)])
            ).design(context=context_with_state())

    def test_prompt_schema_input_boundary_and_one_call(self):
        captured = {}

        def handler(request, output_schema):
            captured["system"] = request.messages[0].content
            captured["user"] = request.messages[1].content
            captured["schema"] = output_schema
            return draft_payload(proposals=initial_proposals())

        context = context_with_state()
        module = ExperimentDesignModule(FakeLLMClient(handler=handler))
        result = module.design(context=context)
        self.assertIs(captured["schema"], ExperimentDesignDraft)
        self.assertIn(
            json.dumps(ExperimentDesignDraft.model_json_schema(), ensure_ascii=False, separators=(",", ":")),
            captured["system"],
        )
        self.assertIn("Task-A", captured["user"])
        self.assertIn("accuracy", captured["user"])
        self.assertIn("Seed Method", captured["system"])
        self.assertIn("NO FABRICATED RESULTS", captured["system"])
        self.assertEqual(module.call_count, 1)
        self.assertEqual(result.status, "ready")

    def test_initial_design_materializes_ids_status_claims_and_audit(self):
        context = context_with_state()
        before_memory = context.get_memory().model_dump(mode="python")
        module = ExperimentDesignModule(
            FakeLLMClient([draft_payload(proposals=initial_proposals())])
        )
        result = module.design(context=context, step=3)
        state = context.get_state()
        self.assertEqual([experiment.id for experiment in state.experiments], ["E1", "E2"])
        self.assertEqual([experiment.id for experiment in result.experiments], ["E1", "E2"])
        self.assertTrue(all(experiment.status == "ready" for experiment in state.experiments))
        self.assertEqual(state.active_gap, context_with_state().get_state().active_gap)
        self.assertEqual(state.current_solution, context_with_state().get_state().current_solution)
        self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "experiments_designed")
        self.assertEqual(event.action, "DESIGN_EXPERIMENT")
        self.assertEqual(event.memory_writes, [])
        self.assertEqual(event.result_payload["experiment_ids"], ["E1", "E2"])
        self.assertIn("K1", event.result_payload["claim_coverage"])

    def test_initial_design_materializes_seed_method_baseline_for_core_claim(self):
        proposals = initial_proposals()
        proposals[0]["baselines"] = ["Original supplied method"]
        proposals[1]["baselines"] = ["Component control"]
        context = context_with_state()
        result = ExperimentDesignModule(
            FakeLLMClient([draft_payload(proposals=proposals)])
        ).design(context=context)
        core_experiment = next(
            experiment
            for experiment in result.experiments
            if "core_solution" in experiment.what_it_validates
        )
        self.assertEqual(core_experiment.baselines, ["Original supplied method", "Seed Method"])

    def test_initial_design_accepts_four_but_rejects_wrong_count(self):
        for count in (2, 4):
            context = context_with_state()
            module = ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=initial_proposals(count))])
            )
            result = module.design(context=context)
            self.assertEqual(len(result.experiments), count)
        for count in (1, 5):
            context = context_with_state()
            module = ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=initial_proposals(count))])
            )
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                module.design(context=context)

    def test_preconditions_fail_before_llm(self):
        cases = []
        missing_gap = context_with_state()
        missing_gap.state_store.replace_state(
            missing_gap.get_state().model_copy(update={"active_gap": None})
        )
        cases.append(missing_gap)
        missing_solution = context_with_state()
        missing_solution.state_store.replace_state(
            missing_solution.get_state().model_copy(update={"current_solution": None})
        )
        cases.append(missing_solution)
        cases.append(context_with_state(current_solution=solution(novelty="partial_overlap")))
        cases.append(context_with_state(current_solution=solution(novelty="core_overlap")))
        cases.append(context_with_state(current_solution=solution(component_statuses=("modified", "ready"))))
        cases.append(context_with_state(add_gap_verification=False))
        cases.append(context_with_state(add_solution_verification=False))
        for context in cases:
            llm = FakeLLMClient([draft_payload(proposals=initial_proposals())])
            module = ExperimentDesignModule(llm)
            before_state = context.get_state().model_dump(mode="python")
            before_memory = context.get_memory().model_dump(mode="python")
            before_events = context.trajectory.get_events()
            with self.assertRaises(PreconditionError):
                module.design(context=context)
            self.assertEqual(module.call_count, 0)
            self.assertEqual(llm.calls, [])
            self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
            self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
            self.assertEqual(context.trajectory.get_events(), before_events)

    def test_verification_focus_changed_units_is_supplied_deterministically(self):
        context = context_with_state(
            focus=VerificationFocus(mode="changed_units", changed_units=["C1"])
        )
        captured = {}

        def handler(request, output_schema):
            captured["payload"] = json.loads(request.messages[1].content)
            return draft_payload(proposals=initial_proposals())

        ExperimentDesignModule(FakeLLMClient(handler=handler)).design(context=context)
        self.assertEqual(captured["payload"]["changed_units"], ["C1"])

    def test_unchecked_stale_and_explicit_state_mismatch_fail_pre_llm(self):
        for current in (
            solution(novelty="unchecked"),
            solution(component_statuses=("stale", "ready")),
        ):
            context = context_with_state(current_solution=current)
            llm = FakeLLMClient([draft_payload(proposals=initial_proposals())])
            module = ExperimentDesignModule(llm)
            with self.assertRaises(PreconditionError):
                module.design(context=context)
            self.assertEqual(module.call_count, 0)
            self.assertEqual(llm.calls, [])

        context = context_with_state()
        module = ExperimentDesignModule(FakeLLMClient([]))
        with self.assertRaises(PreconditionError):
            module.design(
                active_gap=gap(),
                current_solution=solution().model_copy(update={"id": "S2"}),
                seed_context=seed(),
                state_store=context.state_store,
                memory_store=context.memory_store,
            )
        self.assertEqual(module.call_count, 0)

    def test_existing_experiment_count_above_four_is_not_truncated(self):
        experiments = [
            experiment(f"E{index}", f"Existing {index}", f"core_solution; E{index}")
            for index in range(1, 6)
        ]
        context = context_with_state(experiments=experiments)
        module = ExperimentDesignModule(FakeLLMClient([]))
        with self.assertRaises(PreconditionError):
            module.design(context=context)
        self.assertEqual(module.call_count, 0)

    def test_noop_with_all_ready_experiments_does_not_call_llm_or_write_event(self):
        experiments = [
            experiment("E1", "Existing overall", "core_solution; K1"),
            experiment("E2", "Existing mechanism", "C1; K2", setup="A different setup"),
        ]
        context = context_with_state(experiments=experiments)
        before_state = context.get_state().model_dump(mode="python")
        before_events = context.trajectory.get_events()
        module = ExperimentDesignModule(FakeLLMClient([]))
        result = module.design(context=context)
        self.assertEqual(module.call_count, 0)
        self.assertEqual(result.redesigned_experiment_ids, [])
        self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(context.trajectory.get_events(), before_events)

    def test_stale_redesign_preserves_ready_and_reuses_stale_id(self):
        experiments = [
            experiment("E1", "Existing overall", "core_solution; K1"),
            experiment("E2", "Old mechanism", "C1; old", status="stale", setup="Old setup"),
            experiment("E3", "Existing second mechanism", "C1; K2", setup="A third setup"),
        ]
        context = context_with_state(experiments=experiments)
        payload = draft_payload(
            proposals=[
                {
                    "existing_experiment_id": "E2",
                    "objective": "Redesign mechanism validation.",
                    "setup": "Task-A with a focused mechanism comparison.",
                    "baselines": ["Seed Method"],
                    "metrics": ["calibration"],
                    "what_it_validates": "C1; redesigned mechanism",
                    "validation_criterion": "The mechanism claim is supported by the comparison.",
                    "validates_claims": ["K1"],
                }
            ],
            claims=[
                {
                    "id": "K1",
                    "statement": "The redesigned mechanism now tests a new claim.",
                    "target_unit": "C1",
                },
            ],
        )
        result = ExperimentDesignModule(FakeLLMClient([payload])).design(context=context)
        final = context.get_state().experiments
        self.assertEqual([item.id for item in final], ["E1", "E2", "E3"])
        self.assertEqual(final[0], experiments[0])
        self.assertEqual(final[2], experiments[2])
        self.assertEqual(final[1].status, "ready")
        self.assertEqual(final[1].objective, "Redesign mechanism validation.")
        self.assertEqual(result.redesigned_experiment_ids, ["E2"])

    def test_ready_anchor_does_not_cover_a_new_redesign_claim(self):
        experiments = [
            experiment(
                "E1",
                "Existing overall and old component check",
                "core_solution; C1",
            ),
            experiment("E2", "Old stale mechanism", "C2", status="stale", setup="Old setup"),
        ]
        payload = draft_payload(
            proposals=[
                {
                    "existing_experiment_id": "E2",
                    "objective": "Redesign the second mechanism.",
                    "setup": "Task-A with a new second-mechanism comparison.",
                    "baselines": ["Seed Method"],
                    "metrics": ["calibration"],
                    "what_it_validates": "C2; new mechanism claim",
                    "validation_criterion": "The new mechanism claim is supported.",
                    "validates_claims": ["K2"],
                }
            ],
            claims=[
                {
                    "id": "K1",
                    "statement": "A new scientific meaning for the C1 claim.",
                    "target_unit": "C1",
                },
                {
                    "id": "K2",
                    "statement": "The stale mechanism claim.",
                    "target_unit": "C2",
                },
            ],
        )
        module = ExperimentDesignModule(FakeLLMClient([payload]))
        with self.assertRaises(ModuleContractError):
            module.design(context=context_with_state(experiments=experiments))

    def test_stale_redesign_rejects_ready_new_missing_duplicate_and_count_changes(self):
        existing = [
            experiment("E1", "Existing overall", "core_solution; K1"),
            experiment("E2", "Old mechanism", "C1; old", status="stale", setup="Old setup"),
            experiment("E3", "Existing second mechanism", "C1; K2", setup="A third setup"),
        ]
        payload_base = {
            "objective": "Redesign mechanism validation.",
            "setup": "Task-A with a focused mechanism comparison.",
            "baselines": ["Seed Method"],
            "metrics": ["calibration"],
            "what_it_validates": "C1; redesigned mechanism",
            "validation_criterion": "The mechanism claim is supported.",
            "validates_claims": ["K2"],
        }
        cases = [
            [{**payload_base, "existing_experiment_id": "E1"}],
            [{**payload_base, "existing_experiment_id": None}],
            [
                {**payload_base, "existing_experiment_id": "E2"},
                {**payload_base, "existing_experiment_id": "E2", "objective": "Duplicate"},
            ],
        ]
        for proposals in cases:
            context = context_with_state(experiments=existing)
            module = ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=proposals)])
            )
            with self.assertRaises(ModuleContractError):
                module.design(context=context)
            self.assertEqual(context.get_state().experiments, existing)

    def test_claim_and_anchor_validation_is_strict(self):
        invalid_payloads = [
            draft_payload(
                proposals=initial_proposals(),
                claims=[
                    {"id": "K1", "statement": "Only component.", "target_unit": "C1"},
                    {"id": "K2", "statement": "Unknown.", "target_unit": "C9"},
                ],
            ),
            draft_payload(
                proposals=[
                    {**initial_proposals()[0], "validates_claims": ["K9"]},
                    initial_proposals()[1],
                ]
            ),
            draft_payload(
                proposals=[
                    {**initial_proposals()[0], "what_it_validates": "overall effectiveness"},
                    initial_proposals()[1],
                ]
            ),
        ]
        for payload in invalid_payloads:
            context = context_with_state()
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                ExperimentDesignModule(FakeLLMClient([payload])).design(context=context)

    def test_empty_quality_fields_and_unexpected_state_fields_reject(self):
        empty_baseline = initial_proposals()
        empty_baseline[0]["baselines"] = []
        empty_metric = initial_proposals()
        empty_metric[0]["metrics"] = []
        for proposals in (empty_baseline, empty_metric):
            context = context_with_state()
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                ExperimentDesignModule(
                    FakeLLMClient([draft_payload(proposals=proposals)])
                ).design(context=context)
        no_seed_baseline = initial_proposals()
        no_seed_baseline[0]["baselines"] = ["Baseline-A"]
        no_seed_result = ExperimentDesignModule(
            FakeLLMClient([draft_payload(proposals=no_seed_baseline)])
        ).design(context=context_with_state())
        no_seed_core = next(
            experiment
            for experiment in no_seed_result.experiments
            if "core_solution" in experiment.what_it_validates
        )
        self.assertIn("Seed Method", no_seed_core.baselines)
        empty_criterion = initial_proposals()
        empty_criterion[0]["validation_criterion"] = ""
        with self.assertRaises(StructuredOutputError):
            ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=empty_criterion)])
            ).design(context=context_with_state())
        extra = draft_payload(proposals=initial_proposals())
        extra["experiments"][0]["status"] = "ready"
        with self.assertRaises(StructuredOutputError):
            ExperimentDesignModule(FakeLLMClient([extra])).design(
                context=context_with_state()
            )

    def test_duplicate_objective_and_setup_claim_coverage_reject(self):
        proposals = initial_proposals()
        proposals[1]["objective"] = proposals[0]["objective"]
        with self.assertRaises(ModuleContractError):
            ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=proposals)])
            ).design(context=context_with_state())
        proposals = initial_proposals()
        proposals[1]["setup"] = proposals[0]["setup"]
        proposals[1]["validates_claims"] = proposals[0]["validates_claims"]
        proposals[1]["what_it_validates"] = proposals[0]["what_it_validates"]
        with self.assertRaises(ModuleContractError):
            ExperimentDesignModule(
                FakeLLMClient([draft_payload(proposals=proposals)])
            ).design(context=context_with_state())

    def test_state_store_only_updates_only_experiments_without_trajectory(self):
        context = context_with_state()
        state_store = ResearchStateStore(context.get_state())
        memory_store = ResearchMemoryStore(seed().paper_id, context.get_memory())
        before = state_store.get_state()
        module = ExperimentDesignModule(
            FakeLLMClient([draft_payload(proposals=initial_proposals())])
        )
        result = module.design(
            active_gap=gap(),
            current_solution=solution(),
            seed_context=seed(),
            state_store=state_store,
            memory_store=memory_store,
        )
        after = state_store.get_state()
        self.assertEqual(result.status, "ready")
        self.assertEqual(after.active_gap, before.active_gap)
        self.assertEqual(after.current_solution, before.current_solution)
        self.assertEqual(len(after.experiments), 2)

    def test_invalid_trajectory_step_preflight_does_not_call_llm(self):
        context = context_with_state()
        llm = FakeLLMClient([draft_payload(proposals=initial_proposals())])
        module = ExperimentDesignModule(llm)
        before = context.get_state().model_dump(mode="python")
        events = context.trajectory.get_events()
        with self.assertRaises(PreconditionError):
            module.design(context=context, step=-1)
        self.assertEqual(module.call_count, 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(context.get_state().model_dump(mode="python"), before)
        self.assertEqual(context.trajectory.get_events(), events)


if __name__ == "__main__":
    unittest.main()
