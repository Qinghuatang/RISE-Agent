import json
import unittest

from research_agent.controller import (
    BudgetTracker,
    ControllerRoutingResult,
    ScientificController,
    record_decision_result,
)
from research_agent.errors import ControllerRoutingError, ModuleContractError, StructuredOutputError
from research_agent.llm.fake import FakeLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    CandidateSolution,
    Component,
    CurrentSolution,
    Experiment,
    ExperimentContext,
    FailureRecord,
    GapAgenda,
    GapVerificationResult,
    PaperRecord,
    RelatedPaperPool,
    ResearchState,
    SeedPaperContext,
    SolutionVerificationResult,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.trajectory.context import TrajectoryContext
from research_agent.modules.gap_discovery import GapDiscoveryModule
from research_agent.runner import TrajectoryRunner


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-controller-seed",
        title="Generic Seed",
        abstract="Supplied abstract.",
        research_objective="Improve a supplied mechanism.",
        core_method="A method with interacting components.",
        main_components=["Component one", "Component two"],
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
        statement="A method-level research gap.",
        confidence="medium",
        confidence_reason="Supplied mechanism evidence.",
        priority="high",
        priority_reason="It affects the central method.",
        status=status,
        evidence_refs=["seed:core_method"],
    )


def solution(novelty: str = "unchecked", component_status: str = "ready") -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="A generic solution core.",
        rationale="A supplied rationale.",
        components=[
            Component(id="C1", description="First unit.", status=component_status),
            Component(id="C2", description="Second unit.", status=component_status),
        ],
        status="developed",
        novelty_status=novelty,
        novelty_summary="The supplied novelty result." if novelty != "unchecked" else None,
        evidence_refs=["solution:evidence"],
    )


def experiment(identifier: str, status: str = "ready", anchor: str = "core_solution") -> Experiment:
    return Experiment(
        id=identifier,
        objective=f"Evaluate {identifier}.",
        setup="Task-A under the supplied setting.",
        baselines=["Seed Method"],
        metrics=["metric-A"],
        what_it_validates=f"This validates {anchor}.",
        validation_criterion="The observation would support the claim.",
        status=status,
    )


def gap_record(status: str = "valid", step: int = 1) -> VerificationRecord:
    return VerificationRecord(
        id=f"VG{step}",
        task="VERIFY_GAP",
        target_ref="G1",
        result=GapVerificationResult(
            status=status,
            summary="The supplied gap verification result.",
            remaining_gap="The remaining gap." if status != "resolved" else None,
            evidence_level="abstract_only",
        ),
        created_at_step=step,
    )


def solution_record(novelty: str, step: int = 2) -> VerificationRecord:
    return VerificationRecord(
        id=f"VS{step}",
        task="VERIFY_SOLUTION",
        target_ref="S1",
        verification_focus=VerificationFocus(mode="full_solution"),
        result=SolutionVerificationResult(
            novelty_status=novelty,
            summary="The supplied solution verification result.",
            distinctive_elements=["A supplied difference."],
            key_differences=["A supplied difference."],
            evidence_level="abstract_only",
        ),
        created_at_step=step,
    )


def runtime(
    *,
    current_solution: CurrentSolution | None = None,
    experiments: list[Experiment] | None = None,
    gap_verification: VerificationRecord | None = None,
    solution_verification: VerificationRecord | None = None,
    candidate_status: str = "rejected",
    related_pool: bool = False,
) -> TrajectoryContext:
    context = TrajectoryContext.initialize(seed())
    context.state_store.replace_state(
        ResearchState(
            active_gap=gap(),
            candidate_solutions=[
                CandidateSolution(
                    id="CS1",
                    core_idea="Candidate core.",
                    rationale="Candidate rationale.",
                    status=candidate_status,
                )
            ],
            current_solution=current_solution,
            experiments=experiments or [],
        )
    )
    if gap_verification is not None:
        context.memory_store.save_verification(gap_verification)
    if solution_verification is not None:
        context.memory_store.save_verification(solution_verification)
    if related_pool:
        context.memory_store.set_related_paper_pool(
            RelatedPaperPool(
                seed_paper_id=seed().paper_id,
                papers=[PaperRecord(paper_id="P1", title="Generic paper", source="fake")],
            )
        )
    return context


class M9DeterministicRoutingTests(unittest.TestCase):
    def test_exhausted_solution_space_switches_to_next_gap(self):
        context = runtime(
            current_solution=None,
            gap_verification=gap_record(),
        )
        context.state_store.replace_state(
            ResearchState(
                active_gap=gap(),
                candidate_solutions=[
                    CandidateSolution(
                        id="S1",
                        core_idea="An exhausted candidate.",
                        rationale="It failed repeatedly.",
                        status="rejected",
                    )
                ],
                current_solution=None,
                experiments=[],
            )
        )
        for index in range(2):
            context.memory_store.save_failure(
                FailureRecord(
                    type="rejected_solution",
                    object_ref="S1",
                    summary=f"Failure {index}.",
                    reason="The candidate did not resolve the gap.",
                )
            )
        context.record_gap_agenda(
            1,
            GapAgenda(
                gaps=[
                    gap(),
                    gap().model_copy(update={"id": "G2", "status": "candidate"}),
                ]
            ),
        )
        result = ScientificController(
            FakeLLMClient([{"action": "SWITCH_GAP", "reason": "Try the next gap."}]),
            budget_tracker=BudgetTracker(2, 0),
        ).decide(context)
        self.assertEqual(result.allowed_actions, ["SOLUTION_IDEATION", "SWITCH_GAP"])
        self.assertEqual(result.decision.action, "SWITCH_GAP")

    def test_valid_abandoned_gap_is_rejected_not_resolved(self):
        context = runtime(current_solution=None, gap_verification=gap_record("valid"))
        context.state_store.replace_state(
            ResearchState(
                active_gap=gap(),
                candidate_solutions=[
                    CandidateSolution(
                        id="S1",
                        core_idea="An exhausted candidate.",
                        rationale="It failed repeatedly.",
                        status="rejected",
                    )
                ],
                current_solution=None,
                experiments=[],
            )
        )
        for index in range(2):
            context.memory_store.save_failure(
                FailureRecord(
                    type="rejected_solution",
                    object_ref="S1",
                    summary=f"Failure {index}.",
                    reason="The candidate did not resolve the gap.",
                )
            )
        context.record_gap_agenda(
            1,
            GapAgenda(
                gaps=[gap(), gap().model_copy(update={"id": "G2", "status": "candidate"})]
            ),
        )
        terminal_status = TrajectoryRunner._switch_gap_terminal_status(context)
        self.assertEqual(terminal_status, "rejected")
        result = GapDiscoveryModule(FakeLLMClient([])).switch_gap(
            context=context, terminal_status=terminal_status, step=2
        )
        self.assertEqual(result.gap_agenda.gaps[0].status, "rejected")

    def test_failed_revision_routes_to_untried_solution_ideation(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
            candidate_status="candidate",
        )
        context.trajectory.append_event(
            3,
            "revision_failed",
            action="TARGETED_REVISION",
            result_payload={"revision_result": {"status": "unable_to_revise"}},
        )
        result = ScientificController(
            FakeLLMClient([]), budget_tracker=BudgetTracker(3, 1)
        ).decide(context)
        self.assertEqual(result.allowed_actions, ["SOLUTION_IDEATION"])
        self.assertEqual(result.decision.action, "SOLUTION_IDEATION")

    def test_controller_context_contains_compact_candidate_solution_summaries(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        state = context.get_state()
        state.candidate_solutions = [
            CandidateSolution(
                id="CS1",
                core_idea="A local mechanism variant.",
                rationale="It changes one existing unit.",
                status="selected",
            ),
            CandidateSolution(
                id="CS2",
                core_idea="An alternative interaction pattern.",
                rationale="It changes how units cooperate.",
                status="candidate",
            ),
            CandidateSolution(
                id="CS3",
                core_idea="A rejected direction.",
                rationale="It was not retained.",
                status="rejected",
            ),
        ]
        context.state_store.replace_state(state)
        llm = FakeLLMClient(
            [{"action": "SEEK_INSPIRATION", "reason": "Explore an unresolved mechanism."}]
        )
        controller = ScientificController(llm, budget_tracker=BudgetTracker(3, 1))
        controller.decide(context)
        payload = json.loads(llm.calls[0].messages[1].content)
        self.assertEqual(
            payload["controller_context"]["candidate_solutions"],
            [
                {
                    "id": "CS1",
                    "core_idea": "A local mechanism variant.",
                    "rationale": "It changes one existing unit.",
                    "status": "selected",
                },
                {
                    "id": "CS2",
                    "core_idea": "An alternative interaction pattern.",
                    "rationale": "It changes how units cooperate.",
                    "status": "candidate",
                },
                {
                    "id": "CS3",
                    "core_idea": "A rejected direction.",
                    "rationale": "It was not retained.",
                    "status": "rejected",
                },
            ],
        )

    def test_missing_gap_verification_routes_without_controller_llm(self):
        context = runtime(related_pool=True)
        llm = FakeLLMClient([])
        result = ScientificController(
            llm, budget_tracker=BudgetTracker(3, 0)
        ).decide(context)
        self.assertEqual(result.status, "decision")
        self.assertEqual(result.decision.action, "VERIFY_RELATED_WORK")
        self.assertFalse(result.used_llm)
        self.assertEqual(len(llm.calls), 0)

    def test_resolved_gap_routes_to_switch_with_reserve(self):
        context = runtime(gap_verification=gap_record("resolved"))
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 0)).decide(context)
        self.assertEqual(result.decision.action, "SWITCH_GAP")

    def test_valid_gap_without_solution_routes_to_ideation_even_when_candidates_rejected(self):
        context = runtime(gap_verification=gap_record(), candidate_status="rejected")
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 0)).decide(context)
        self.assertEqual(result.decision.action, "SOLUTION_IDEATION")
        self.assertEqual(result.allowed_actions, ["SOLUTION_IDEATION"])

    def test_unchecked_solution_routes_to_verification(self):
        context = runtime(
            current_solution=solution("unchecked"),
            gap_verification=gap_record(),
            related_pool=True,
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(1, 0)).decide(context)
        self.assertEqual(result.decision.action, "VERIFY_RELATED_WORK")

    def test_inspiration_found_routes_to_revision_deterministically(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        context.trajectory.append_event(
            3,
            "inspiration_retrieved",
            action="SEEK_INSPIRATION",
            result_ref="I1",
            result_payload={
                "grounding_status": "arxiv",
                "result": {"status": "found", "summary": "Found."},
            },
        )
        llm = FakeLLMClient([])
        result = ScientificController(llm, budget_tracker=BudgetTracker(2, 0)).decide(context)
        self.assertEqual(result.decision.action, "TARGETED_REVISION")
        self.assertEqual(len(llm.calls), 0)

    def test_non_found_inspiration_does_not_force_revision_from_commit_ready_state(self):
        context = runtime(
            current_solution=solution("distinct"),
            experiments=[experiment("E1"), experiment("E2", anchor="C1")],
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        context.trajectory.append_event(
            3,
            "inspiration_retrieved",
            action="SEEK_INSPIRATION",
            result_ref="I1",
            result_payload={
                "grounding_status": "arxiv",
                "result": {"status": "no_useful_inspiration", "inspirations": []},
            },
        )
        result = ScientificController(
            FakeLLMClient([]), budget_tracker=BudgetTracker(3, 1)
        ).decide(context, commit_allowed=True)
        self.assertEqual(result.allowed_actions, ["COMMIT"])
        self.assertEqual(result.decision.action, "COMMIT")

    def test_partial_overlap_uses_one_controller_call_for_two_legal_actions(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        llm = FakeLLMClient([{"action": "TARGETED_REVISION", "reason": "Evidence supports a minimal change."}])
        result = ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["TARGETED_REVISION", "SEEK_INSPIRATION"])
        # action_space order is canonical; targeted revision is the first
        # branch's scientific preference even though the mask is inspected as a set.
        self.assertEqual(result.decision.action, "TARGETED_REVISION")
        self.assertTrue(result.used_llm)
        self.assertEqual(len(llm.calls), 1)

    def test_partial_overlap_without_search_is_deterministic_revision(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        llm = FakeLLMClient([])
        result = ScientificController(llm, budget_tracker=BudgetTracker(2, 0)).decide(context)
        self.assertEqual(result.decision.action, "TARGETED_REVISION")
        self.assertEqual(len(llm.calls), 0)

    def test_partial_overlap_two_steps_masks_inspiration_but_keeps_revision(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        llm = FakeLLMClient([])
        result = ScientificController(llm, budget_tracker=BudgetTracker(2, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["TARGETED_REVISION"])
        self.assertEqual(result.decision.action, "TARGETED_REVISION")
        self.assertEqual(len(llm.calls), 0)

    def test_repeated_partial_overlap_routes_to_untried_solution(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
            candidate_status="candidate",
        )
        for revision_step, verification_step in ((3, 4), (5, 6)):
            context.trajectory.append_event(
                revision_step,
                "solution_revised",
                action="TARGETED_REVISION",
                result_ref="S1",
                result_payload={"revision_result": {"status": "revised"}},
            )
            context.trajectory.append_event(
                verification_step,
                "solution_verified",
                action="VERIFY_RELATED_WORK",
                result_ref="S1",
                result_payload={"result": {"novelty_status": "partial_overlap"}},
            )
        result = ScientificController(
            FakeLLMClient([]), budget_tracker=BudgetTracker(3, 1)
        ).decide(context)
        self.assertEqual(result.allowed_actions, ["SOLUTION_IDEATION"])
        self.assertEqual(result.decision.action, "SOLUTION_IDEATION")

    def test_repeated_partial_overlap_without_candidates_uses_inspiration(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
            candidate_status="rejected",
        )
        for revision_step, verification_step in ((3, 4), (5, 6)):
            context.trajectory.append_event(
                revision_step,
                "solution_revised",
                action="TARGETED_REVISION",
                result_ref="S1",
                result_payload={"revision_result": {"status": "revised"}},
            )
            context.trajectory.append_event(
                verification_step,
                "solution_verified",
                action="VERIFY_RELATED_WORK",
                result_ref="S1",
                result_payload={"result": {"novelty_status": "partial_overlap"}},
            )
        result = ScientificController(
            FakeLLMClient([]), budget_tracker=BudgetTracker(3, 1)
        ).decide(context)
        self.assertEqual(result.allowed_actions, ["SEEK_INSPIRATION"])
        self.assertEqual(result.decision.action, "SEEK_INSPIRATION")

    def test_illegal_controller_action_is_rejected_without_replacement(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        llm = FakeLLMClient([{"action": "COMMIT", "reason": "illegal"}])
        with self.assertRaises(ModuleContractError):
            ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)
        self.assertEqual(len(llm.calls), 1)

    def test_malformed_or_extra_controller_output_is_not_repaired(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        llm = FakeLLMClient([{"action": "TARGETED_REVISION", "reason": "ok", "extra": 1}])
        with self.assertRaises(StructuredOutputError):
            ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)

    def test_distinct_empty_experiments_with_search_reserve_offer_both_actions(self):
        context = runtime(
            current_solution=solution("distinct"),
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        llm = FakeLLMClient([{"action": "DESIGN_EXPERIMENT", "reason": "Design first."}])
        result = ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["DESIGN_EXPERIMENT", "SEEK_INSPIRATION"])
        self.assertEqual(result.decision.action, "DESIGN_EXPERIMENT")
        self.assertTrue(result.used_llm)
        self.assertEqual(len(llm.calls), 1)

    def test_distinct_empty_experiments_two_steps_masks_inspiration(self):
        context = runtime(
            current_solution=solution("distinct"),
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["DESIGN_EXPERIMENT"])

    def test_distinct_stale_experiments_mask_search_when_exhausted(self):
        context = runtime(
            current_solution=solution("distinct"),
            experiments=[experiment("E1", "stale")],
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 1)).decide(context)
        self.assertEqual(result.decision.action, "DESIGN_EXPERIMENT")

    def test_all_ready_experiments_do_not_offer_design_and_commit_without_search_is_single(self):
        context = runtime(
            current_solution=solution("distinct"),
            experiments=[experiment("E1"), experiment("E2", anchor="C1")],
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 1)).decide(
            context, commit_allowed=True
        )
        self.assertEqual(result.allowed_actions, ["COMMIT"])
        self.assertEqual(result.decision.action, "COMMIT")

    def test_commit_not_allowed_stops_when_search_is_exhausted(self):
        context = runtime(
            current_solution=solution("distinct"),
            experiments=[experiment("E1"), experiment("E2", anchor="C1")],
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 1)).decide(
            context, commit_allowed=False, unmet_commit_conditions=["commit gate unmet"]
        )
        self.assertEqual(result.status, "stopped")
        self.assertIn("commit gate unmet", result.unresolved_issues)

    def test_commit_ready_state_does_not_reopen_inspiration_without_new_evidence(self):
        current_solution = solution("distinct").model_copy(
            update={
                "core_idea": "Tune an existing objective through a local adjustment.",
                "rationale": "Refine an existing component incrementally.",
            }
        )
        context = runtime(
            current_solution=current_solution,
            experiments=[experiment("E1"), experiment("E2", anchor="C1")],
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        llm = FakeLLMClient([])
        result = ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(
            context, commit_allowed=True
        )
        self.assertEqual(result.allowed_actions, ["COMMIT"])
        self.assertEqual(result.decision.action, "COMMIT")
        self.assertEqual(len(llm.calls), 0)


class M12_2InspirationBoundaryTests(unittest.TestCase):
    def test_local_modification_opens_optional_inspiration(self):
        current_solution = solution("distinct").model_copy(
            update={
                "core_idea": "Make a local modification to an existing component.",
                "rationale": "Tune a parameter with a local adjustment.",
            }
        )
        context = runtime(
            current_solution=current_solution,
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        llm = FakeLLMClient(
            [{"action": "SEEK_INSPIRATION", "reason": "Explore a broader mechanism."}]
        )
        controller = ScientificController(llm, budget_tracker=BudgetTracker(3, 1))
        result = controller.decide(context)
        self.assertEqual(
            result.allowed_actions, ["DESIGN_EXPERIMENT", "SEEK_INSPIRATION"]
        )
        self.assertEqual(result.decision.action, "SEEK_INSPIRATION")
        self.assertTrue(result.used_llm)
        self.assertEqual(len(llm.calls), 1)

    def test_clear_new_mechanism_does_not_require_inspiration(self):
        current_solution = solution("distinct").model_copy(
            update={
                "core_idea": "Introduce a new mechanism with a new interaction and representation.",
                "rationale": "A substantive redesign directly addresses the gap.",
            }
        )
        context = runtime(
            current_solution=current_solution,
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        llm = FakeLLMClient(
            [{"action": "DESIGN_EXPERIMENT", "reason": "Direct experiments have higher current information value."}]
        )
        result = ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["DESIGN_EXPERIMENT", "SEEK_INSPIRATION"])
        self.assertEqual(result.decision.action, "DESIGN_EXPERIMENT")
        self.assertTrue(result.used_llm)
        self.assertEqual(len(llm.calls), 1)

    def test_high_priority_gap_with_incremental_solution_is_an_inspiration_candidate(self):
        current_solution = solution("distinct").model_copy(
            update={
                "core_idea": "Refine an existing objective and tune its loss locally.",
                "rationale": "An incremental adjustment may be insufficient for the high-impact gap.",
            }
        )
        context = runtime(
            current_solution=current_solution,
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        result = ScientificController(
            FakeLLMClient([{"action": "DESIGN_EXPERIMENT", "reason": "Test the refinement."}]),
            budget_tracker=BudgetTracker(3, 1),
        ).decide(context)
        self.assertIn("SEEK_INSPIRATION", result.allowed_actions)

    def test_distinct_solution_does_not_automatically_require_inspiration(self):
        current_solution = solution("distinct").model_copy(
            update={
                "core_idea": "A complete solution follows the supplied design rationale.",
                "rationale": "The solution is ready for direct experimental validation.",
            }
        )
        context = runtime(
            current_solution=current_solution,
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        llm = FakeLLMClient(
            [{"action": "DESIGN_EXPERIMENT", "reason": "The current proposal is ready for direct validation."}]
        )
        result = ScientificController(llm, budget_tracker=BudgetTracker(3, 1)).decide(context)
        self.assertEqual(result.allowed_actions, ["DESIGN_EXPERIMENT", "SEEK_INSPIRATION"])
        self.assertTrue(result.used_llm)
        self.assertEqual(len(llm.calls), 1)

    def test_inspiration_mask_is_independent_of_solution_wording(self):
        texts = [
            {
                "core_idea": "Modify the objective with a local adjustment.",
                "rationale": "Refine an existing component.",
            },
            {
                "core_idea": "Introduce a new mechanism and representation.",
                "rationale": "A substantive redesign is ready for validation.",
            },
        ]
        allowed = []
        for text in texts:
            current_solution = solution("distinct").model_copy(update=text)
            context = runtime(
                current_solution=current_solution,
                gap_verification=gap_record(),
                solution_verification=solution_record("distinct"),
            )
            controller = ScientificController(
                FakeLLMClient(
                    [{"action": "DESIGN_EXPERIMENT", "reason": "Direct validation is informative."}]
                ),
                budget_tracker=BudgetTracker(3, 1),
            )
            result = controller.decide(context)
            allowed.append(result.allowed_actions)
        self.assertEqual(allowed[0], allowed[1])
        self.assertEqual(allowed[0], ["DESIGN_EXPERIMENT", "SEEK_INSPIRATION"])

    def test_controller_prompt_describes_optional_generic_inspiration_gate(self):
        prompt = ScientificController(FakeLLMClient([])).prompt
        self.assertIn("Distinct verification alone does not require inspiration", prompt)
        self.assertIn("Do not inspect solution wording for keywords", prompt)
        self.assertIn("partial overlap", prompt)


class M9BoundaryAndBudgetTests(unittest.TestCase):
    def test_raw_partial_resolution_is_explicitly_rejected(self):
        context = runtime(
            gap_verification=gap_record("partially_resolved"), related_pool=True
        )
        with self.assertRaises(ControllerRoutingError):
            ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 0)).decide(context)

    def test_refined_partial_resolution_routes_to_verification(self):
        context = runtime(
            gap_verification=gap_record("partially_resolved"), related_pool=True
        )
        context.trajectory.append_event(
            2,
            "gap_refined",
            result_ref="G1",
            result_payload={"gap_id": "G1"},
        )
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(1, 0)).decide(context)
        self.assertEqual(result.decision.action, "VERIFY_RELATED_WORK")

    def test_action_space_has_no_refine_or_component_verification(self):
        self.assertNotIn("REFINE_GAP", ScientificController.action_space)
        self.assertNotIn("VERIFY_COMPONENT", ScientificController.action_space)

    def test_budget_is_updated_by_execution_not_decision(self):
        budget = BudgetTracker(3, 2)
        context = runtime(gap_verification=gap_record())
        controller = ScientificController(FakeLLMClient([]), budget_tracker=budget)
        before_state = context.get_state()
        result = controller.decide(context)
        self.assertEqual(budget.steps_used, 0)
        self.assertEqual(result.decision.action, "SOLUTION_IDEATION")
        budget.record_execution(scientific_searches=1)
        self.assertEqual((budget.steps_used, budget.searches_used), (1, 1))
        self.assertEqual(context.get_state(), before_state)

    def test_cache_hit_search_count_zero_and_http_count_is_not_budget(self):
        budget = BudgetTracker(2, 1)
        budget.record_execution(scientific_searches=0, provider_http_requests=7)
        self.assertEqual(budget.searches_used, 0)
        self.assertTrue(budget.can_execute("SEEK_INSPIRATION", scientific_searches=0))
        budget.record_execution(scientific_searches=1)
        self.assertEqual(budget.searches_used, 1)
        with self.assertRaises(ValueError):
            budget.record_execution(scientific_searches=1)

    def test_mandatory_verification_reserve_masks_ideation(self):
        context = runtime(gap_verification=gap_record())
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(1, 0)).decide(context)
        self.assertEqual(result.status, "stopped")
        self.assertIn("current solution missing", result.unresolved_issues[0])

    def test_verification_with_pool_is_allowed_at_zero_search_budget(self):
        context = runtime(related_pool=True)
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(1, 0)).decide(context)
        self.assertEqual(result.decision.action, "VERIFY_RELATED_WORK")

    def test_verification_without_pool_is_blocked_at_zero_search_budget(self):
        context = runtime()
        result = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(1, 0)).decide(context)
        self.assertEqual(result.status, "stopped")
        self.assertIn("search budget exhausted", result.unresolved_issues[0])

    def test_context_is_compact_and_prompt_has_dynamic_schema(self):
        context = runtime(
            current_solution=solution("partial_overlap"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
            related_pool=True,
        )
        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_solution",
                object_ref="CS1",
                summary="A compact failure.",
                reason="The supplied mechanism overlaps.",
            )
        )
        llm = FakeLLMClient([{"action": "SEEK_INSPIRATION", "reason": "Search for a new mechanism."}])
        controller = ScientificController(llm, budget_tracker=BudgetTracker(3, 1))
        result = controller.decide(context)
        self.assertEqual(result.decision.action, "SEEK_INSPIRATION")
        prompt = llm.calls[0].messages[0].content
        user = llm.calls[0].messages[1].content
        self.assertIn('"action"', prompt)
        self.assertIn("allowed_actions", user)
        self.assertIn("rejected_solution", user)
        self.assertNotIn("Generic paper", user)
        self.assertNotIn("papers", user)
        self.assertEqual(controller.last_context.budget["remaining_searches"], 1)

    def test_decision_logging_happens_after_execution_only(self):
        context = runtime(gap_verification=gap_record())
        controller = ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 0))
        result = controller.decide(context)
        self.assertEqual(context.get_memory().decision_log, [])
        record = record_decision_result(
            1, result.decision, "Fake Solution Ideation completed.", context.memory_store
        )
        self.assertEqual(record.step, 1)
        self.assertEqual(record.action, "SOLUTION_IDEATION")
        self.assertEqual(record.result_summary, "Fake Solution Ideation completed.")
        self.assertEqual(len(context.get_memory().decision_log), 1)

    def test_budget_never_becomes_negative(self):
        with self.assertRaises(ValueError):
            BudgetTracker(-1, 0)
        budget = BudgetTracker(0, 0)
        with self.assertRaises(ValueError):
            budget.record_execution()

    def test_distinct_modified_component_is_not_repaired_by_controller(self):
        context = runtime(
            current_solution=solution("distinct", component_status="modified"),
            gap_verification=gap_record(),
            solution_verification=solution_record("distinct"),
        )
        with self.assertRaises(ControllerRoutingError):
            ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(2, 0)).decide(context)

    def test_partial_overlap_modified_component_is_not_routed(self):
        context = runtime(
            current_solution=solution("partial_overlap", component_status="modified"),
            gap_verification=gap_record(),
            solution_verification=solution_record("partial_overlap"),
        )
        with self.assertRaises(ControllerRoutingError):
            ScientificController(FakeLLMClient([]), budget_tracker=BudgetTracker(3, 1)).decide(context)


if __name__ == "__main__":
    unittest.main()
