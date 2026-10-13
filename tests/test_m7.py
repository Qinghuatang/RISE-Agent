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
    FailureRecord,
    InspirationCandidate,
    InspirationNeed,
    InspirationRecord,
    ResearchState,
    RevisionNeed,
    RevisionResult,
    SeedPaperContext,
    SolutionOverlap,
    SolutionVerificationResult,
    TargetedRevisionDraft,
    VerificationFocus,
    VerificationRecord,
)
from research_agent.models.memory import RevisionRecord
from research_agent.modules.targeted_revision import (
    TargetedRevisionModule,
    apply_revision,
    canonicalize_revision_result,
)
from research_agent.runtime.state_store import ResearchStateStore
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-revision-seed",
        title="A Generic Revision Method",
        abstract="A generic abstract.",
        research_objective="Study a mechanism-level problem.",
        core_method="A current method with several units.",
        main_components=["Unit one", "Unit two", "Unit three"],
        main_findings=["A reported finding."],
        limitations=["A method boundary."],
        method_constraints=[
            "The original design preserves stability; the rejected alternative violates this constraint."
        ],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task"],
            metrics=["metric"],
            major_baselines=["baseline"],
            experimental_setting="Setting.",
        ),
    )


def gap(identifier: str = "G1") -> CandidateGap:
    return CandidateGap(
        id=identifier,
        statement="The active gap concerns a mechanism-level issue.",
        confidence="medium",
        confidence_reason="The method evidence supports the issue.",
        priority="high",
        priority_reason="It affects the central method.",
        status="active",
        evidence_refs=["seed:core_method"],
    )


def solution(
    *,
    component_statuses: tuple[str, str, str] = ("ready", "ready", "ready"),
) -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="The original core mechanism.",
        rationale="The original rationale.",
        components=[
            Component(id=f"C{i}", description=f"Original unit {i}.", status=status)
            for i, status in enumerate(component_statuses, start=1)
        ],
        status="developed",
        novelty_status="partial_overlap",
        novelty_summary="The previous verification summary.",
        evidence_refs=["evidence:solution"],
    )


def partial_result(
    *,
    component_ids: list[str] | None = None,
    papers: list[str] | None = None,
) -> SolutionVerificationResult:
    component_ids = component_ids if component_ids is not None else ["C2"]
    papers = papers if papers is not None else ["paper:related"]
    return SolutionVerificationResult(
        novelty_status="partial_overlap",
        summary="A local mechanism overlap needs a targeted change.",
        shared_elements=["A shared generic element."],
        novelty_relevant_overlaps=[
            SolutionOverlap(
                component_id=component_id,
                overlap_summary=f"The mechanism of {component_id} overlaps.",
                matched_paper_ref=papers[index % len(papers)],
            )
            for index, component_id in enumerate(component_ids)
        ],
        distinctive_elements=["A remaining distinction."],
        key_differences=["The core remains different."],
        related_papers=papers,
        evidence_level="abstract_only",
    )


def add_partial_verification(
    context: TrajectoryContext,
    *,
    result: SolutionVerificationResult | None = None,
) -> None:
    result = result or partial_result()
    context.memory_store.save_verification(
        VerificationRecord(
            id="V1",
            task="VERIFY_SOLUTION",
            target_ref="S1",
            verification_focus=VerificationFocus(mode="full_solution"),
            result=result,
            paper_refs=list(result.related_papers),
            created_at_step=1,
        )
    )


def inspiration_record(
    *, count: int = 1, grounding: str = "none", record_id: str = "I1"
) -> InspirationRecord:
    candidates = [
        InspirationCandidate(
            bridge_entity={
                "name": f"Bridge {index}",
                "type": "mechanism",
                "bridge_reason": "A generic bridge.",
            },
            paper_ref=(f"arxiv:{index}" if grounding == "arxiv" else None),
            source_problem="A source conceptual problem.",
            source_mechanism="A source mechanism supported by the available evidence.",
            transferable_principle="A transferable principle.",
            adaptation_hint="An adaptation hint.",
            grounding=grounding,
        )
        for index in range(1, count + 1)
    ]
    return InspirationRecord(
        id=record_id,
        inspiration_need=InspirationNeed(
            current_problem="The current solution needs a mechanism extension.",
            required_function="Provide the required function.",
            current_mechanism="The current mechanism.",
            rejected_mechanisms=["Rejected mechanism."],
            relevant_context="Relevant generic context.",
        ),
        proposed_entities=[],
        selected_entities=[],
        search_query="Bridge",
        grounding_status=grounding,
        inspirations=candidates,
        paper_refs=[candidate.paper_ref for candidate in candidates if candidate.paper_ref],
        created_at_step=2,
    )


def context_with_solution(
    *,
    statuses: tuple[str, str, str] = ("ready", "ready", "ready"),
    experiments: list[Experiment] | None = None,
) -> TrajectoryContext:
    context = TrajectoryContext.initialize(seed())
    context.state_store.replace_state(
        ResearchState(
            active_gap=gap(),
            candidate_solutions=[
                CandidateSolution(
                    id="S1",
                    core_idea="The original core mechanism.",
                    rationale="The original rationale.",
                    status="selected",
                ),
                CandidateSolution(
                    id="S2",
                    core_idea="An unused candidate.",
                    rationale="An unused rationale.",
                    status="candidate",
                ),
            ],
            current_solution=solution(component_statuses=statuses),
            experiments=experiments or [],
        )
    )
    add_partial_verification(context)
    return context


def revised_payload(
    *,
    selected_ref: str | None = None,
    changed_units: list[str] | None = None,
    core: str | None = None,
    rationale: str | None = "A consistent revised rationale.",
    components: list[dict[str, str]] | None = None,
    status: str = "revised",
    failure_reason: str | None = None,
) -> dict:
    changed_units = changed_units if changed_units is not None else ["C2", "rationale"]
    components = components if components is not None else [
        {"component_id": "C2", "description": "A revised second unit."}
    ]
    result = {
        "status": status,
        "revised_core_idea": core,
        "revised_rationale": rationale,
        "revised_components": components,
        "changed_units": changed_units,
        "revision_summary": "A minimum necessary revision was applied.",
        "failure_reason": failure_reason,
    }
    return {"selected_inspiration_ref": selected_ref, "result": result}


class M7TargetedRevisionTests(unittest.TestCase):
    def test_structural_revision_allows_coordinated_add_remove_replace(self):
        current = solution()
        result = RevisionResult(
            status="revised",
            revised_components=[
                {
                    "component_id": "C1",
                    "description": "A replacement for the first unit.",
                    "operation": "replace",
                },
                {"component_id": "C2", "operation": "remove"},
                {
                    "component_id": "C4",
                    "description": "A new coordinated unit.",
                    "operation": "add",
                },
            ],
            changed_units=["C3"],
            revision_summary="A coordinated repair changes the required units.",
        )
        canonical = canonicalize_revision_result(current, result)
        self.assertEqual(canonical.changed_units, ["C1", "C2", "C4"])
        merged = apply_revision(current, canonical)
        self.assertEqual([component.id for component in merged.components], ["C1", "C3", "C4"])
        self.assertEqual(merged.components[0].status, "modified")
        self.assertEqual(merged.components[-1].status, "modified")

    def test_structural_revision_stales_core_solution_experiments_without_core_text_change(self):
        experiments = [
            Experiment(
                id="E1",
                objective="Validate the complete solution.",
                setup="Setup",
                what_it_validates="core_solution",
                validation_criterion="Criterion",
                status="ready",
            ),
            Experiment(
                id="E2",
                objective="Validate the first component.",
                setup="Setup",
                what_it_validates="C1",
                validation_criterion="Criterion",
                status="ready",
            ),
        ]
        context = context_with_solution(experiments=experiments)
        payload = revised_payload(
            selected_ref=None,
            core=None,
            rationale=None,
            components=[
                {
                    "component_id": "C1",
                    "description": "A replacement with a changed topology.",
                    "operation": "replace",
                }
            ],
        )
        outcome = TargetedRevisionModule(FakeLLMClient([payload])).revise(context=context)
        self.assertEqual(outcome.revision_result.changed_units, ["C1"])
        self.assertEqual(
            [experiment.status for experiment in context.get_state().experiments],
            ["stale", "stale"],
        )
        self.assertEqual(outcome.verification_focus, VerificationFocus(mode="full_solution"))

    def test_changed_units_are_computed_from_actual_patch(self):
        current = solution()
        result = RevisionResult(
            status="revised",
            revised_core_idea="A genuinely changed core mechanism.",
            revised_components=[
                {"component_id": "C1", "description": "A coordinated changed unit one."},
                {"component_id": "C2", "description": "A genuinely changed unit."},
                {"component_id": "C3", "description": "A coordinated changed unit three."},
            ],
            changed_units=["C3"],
            revision_summary="A minimum necessary revision was applied.",
        )
        canonical = canonicalize_revision_result(current, result)
        self.assertEqual(canonical.changed_units, ["core_idea", "C1", "C2", "C3"])

    def test_core_revision_rejects_unpatched_existing_component(self):
        current = solution()
        result = RevisionResult(
            status="revised",
            revised_core_idea="A changed central mechanism.",
            revised_components=[
                {"component_id": "C2", "description": "A coordinated unit."}
            ],
            revision_summary="An incomplete central revision.",
        )
        with self.assertRaises(ModuleContractError) as raised:
            canonicalize_revision_result(current, result)
        self.assertIn("C1", str(raised.exception))

    def test_noop_revision_still_fails_after_deterministic_diff(self):
        current = solution()
        result = RevisionResult(
            status="revised",
            revised_core_idea=current.core_idea,
            revision_summary="No actual change.",
        )
        with self.assertRaises(ModuleContractError):
            canonicalize_revision_result(current, result)

    def test_prompt_injects_canonical_draft_schema_and_module_uses_one_call(self):
        captured = {}

        def handler(request, output_schema):
            captured["schema"] = output_schema
            captured["prompt"] = request.messages[0].content
            return revised_payload()

        context = context_with_solution()
        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        module.revise(context=context)
        self.assertIs(captured["schema"], TargetedRevisionDraft)
        self.assertIn(
            json.dumps(
                TargetedRevisionDraft.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            captured["prompt"],
        )
        self.assertIn("MINIMUM SUFFICIENT CHANGE", captured["prompt"])
        self.assertEqual(module.call_count, 1)

    def test_preconditions_fail_before_llm_and_mutation(self):
        cases = []
        empty = TrajectoryContext.initialize(seed())
        cases.append(empty)
        no_solution = TrajectoryContext.initialize(seed())
        no_solution.state_store.replace_state(ResearchState(active_gap=gap()))
        cases.append(no_solution)
        unfinished = context_with_solution(statuses=("ready", "modified", "ready"))
        cases.append(unfinished)
        no_evidence = TrajectoryContext.initialize(seed())
        no_evidence.state_store.replace_state(
            ResearchState(active_gap=gap(), current_solution=solution())
        )
        cases.append(no_evidence)
        for context in cases:
            llm = FakeLLMClient([])
            module = TargetedRevisionModule(llm)
            before_state = context.get_state().model_dump(mode="python")
            before_memory = context.get_memory().model_dump(mode="python")
            before_events = len(context.trajectory.get_events())
            with self.assertRaises(PreconditionError):
                module.revise(context=context)
            self.assertEqual(module.call_count, 0)
            self.assertEqual(llm.calls, [])
            self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
            self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
            self.assertEqual(len(context.trajectory.get_events()), before_events)

    def test_explicit_gap_and_solution_mismatches_fail_before_llm(self):
        context = context_with_solution()
        llm = FakeLLMClient([])
        module = TargetedRevisionModule(llm)
        with self.assertRaises(PreconditionError):
            module.revise(
                active_gap=gap("G2"),
                current_solution=solution(),
                seed_context=seed(),
                state_store=context.state_store,
                memory_store=context.memory_store,
            )
        self.assertEqual(module.call_count, 0)
        with self.assertRaises(PreconditionError):
            module.revise(
                active_gap=gap(),
                current_solution=solution().model_copy(update={"id": "S2"}),
                seed_context=seed(),
                state_store=context.state_store,
                memory_store=context.memory_store,
            )
        self.assertEqual(module.call_count, 0)

    def test_revision_need_is_deterministic_and_filters_relevant_failures(self):
        context = context_with_solution()
        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_mechanism",
                object_ref="M1",
                summary="Avoid the rejected mechanism.",
                reason="It was not suitable.",
            )
        )
        context.memory_store.save_failure(
            FailureRecord(
                type="failed_revision",
                object_ref="S1",
                summary="A previous revision failed.",
                reason="It was too broad.",
            )
        )
        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_gap",
                object_ref="G0",
                summary="This gap must not be a mechanism constraint.",
                reason="It was resolved.",
            )
        )
        context.memory_store.save_inspiration(inspiration_record(count=1))
        captured = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return revised_payload(selected_ref="I1:1")

        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        outcome = module.revise(context=context, inspiration_record_id="I1")
        self.assertEqual(outcome.revision_need.affected_units, ["C2"])
        self.assertIn("Preserve the Active Gap", outcome.revision_need.constraints[0])
        self.assertIn("Avoid the supplied rejected mechanisms", " ".join(outcome.revision_need.constraints))
        self.assertIn("I1:1", captured["content"])
        self.assertIn("Avoid the rejected mechanism.", captured["content"])
        self.assertIn("A previous revision failed.", captured["content"])
        self.assertNotIn("This gap must not be a mechanism constraint.", captured["content"])

    def test_revision_input_contains_seed_method_constraints(self):
        captured = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return revised_payload()

        context = context_with_solution()
        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        module.revise(context=context)
        self.assertIn('"seed_method_constraints"', captured["content"])
        self.assertIn("The original design preserves stability", captured["content"])
        self.assertIn("rejected alternative violates this constraint", captured["content"])
        self.assertEqual(module.call_count, 1)

    def test_inspiration_revision_input_contains_need_rationale_and_selected_principles(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=1))
        captured = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return revised_payload(
                selected_ref="I1:1",
                changed_units=["C2"],
                rationale=None,
                components=[
                    {
                        "component_id": "C2",
                        "description": "A minimum mechanism-level adaptation.",
                    }
                ],
            )

        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        module.revise(
            context=context,
            inspiration_record_id="I1",
            controller_exploration_rationale=(
                "The current mechanism family remains underexplored."
            ),
        )
        payload = json.loads(captured["content"])
        self.assertEqual(
            payload["controller_exploration_rationale"],
            "The current mechanism family remains underexplored.",
        )
        self.assertEqual(
            payload["inspiration_need"]["required_function"],
            "Provide the required function.",
        )
        self.assertEqual(payload["available_inspirations"][0]["inspiration_ref"], "I1:1")
        self.assertIn("mechanism-level change", module._prompt)
        self.assertIn("revision_summary must state", module._prompt)
        self.assertIn("GENERIC JSON FORMAT EXAMPLE", module._prompt)
        self.assertIn('"selected_inspiration_ref": "I1:1"', module._prompt)

    def test_revision_prompt_preserves_author_stated_constraints(self):
        prompt = TargetedRevisionModule(FakeLLMClient([]))._prompt
        self.assertIn("AUTHOR-STATED METHOD CONSTRAINTS", prompt)
        self.assertIn(
            "Do not simply reintroduce an alternative",
            " ".join(prompt.split()),
        )

    def test_inspiration_selection_requires_one_valid_runtime_ref(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=3))
        unknown = TargetedRevisionModule(
            FakeLLMClient([revised_payload(selected_ref="I1:9")])
        )
        before = context.get_memory().model_dump(mode="python")
        with self.assertRaises(ModuleContractError):
            unknown.revise(context=context, inspiration_record_id="I1")
        self.assertEqual(unknown.call_count, 1)
        self.assertEqual(context.get_memory().model_dump(mode="python"), before)

        missing = TargetedRevisionModule(
            FakeLLMClient([revised_payload(selected_ref=None)])
        )
        with self.assertRaises(ModuleContractError):
            missing.revise(context=context, inspiration_record_id="I1")
        self.assertEqual(missing.call_count, 1)

    def test_successful_component_revision_merges_state_focus_and_memory(self):
        experiments = [
            Experiment(
                id="E1",
                objective="Validate C2",
                setup="Setup",
                what_it_validates="C2 mechanism",
                validation_criterion="Criterion",
                status="ready",
            ),
            Experiment(
                id="E2",
                objective="Validate C1",
                setup="Setup",
                what_it_validates="C1 mechanism",
                validation_criterion="Criterion",
                status="ready",
            ),
        ]
        context = context_with_solution(experiments=experiments)
        before_candidates = context.get_state().candidate_solutions
        module = TargetedRevisionModule(FakeLLMClient([revised_payload()]))
        outcome = module.revise(context=context, step=4)
        state = context.get_state()
        self.assertEqual(state.current_solution.core_idea, "The original core mechanism.")
        self.assertEqual(state.current_solution.rationale, "A consistent revised rationale.")
        statuses = {component.id: component.status for component in state.current_solution.components}
        self.assertEqual(statuses, {"C1": "ready", "C2": "modified", "C3": "ready"})
        self.assertEqual(state.current_solution.novelty_status, "unchecked")
        self.assertIsNone(state.current_solution.novelty_summary)
        self.assertEqual([experiment.status for experiment in state.experiments], ["stale", "ready"])
        self.assertEqual(state.candidate_solutions, before_candidates)
        self.assertEqual(outcome.verification_focus, VerificationFocus(mode="changed_units", changed_units=["C2"]))
        self.assertEqual(outcome.next_required_action, "VERIFY_RELATED_WORK")
        self.assertEqual(len(context.get_memory().revision_memory), 1)
        self.assertEqual(context.get_memory().failure_memory, [])
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "solution_revised")
        self.assertEqual(event.action, "TARGETED_REVISION")
        self.assertIn("revision_memory", event.memory_writes)
        self.assertTrue(any(path.startswith("current_solution.") for path in event.state_diff))

    def test_core_revision_stales_all_components_and_experiments_and_uses_full_focus(self):
        experiments = [
            Experiment(
                id="E1",
                objective="Validate",
                setup="Setup",
                what_it_validates="C1",
                validation_criterion="Criterion",
                status="ready",
            ),
            Experiment(
                id="E2",
                objective="Validate",
                setup="Setup",
                what_it_validates="C2",
                validation_criterion="Criterion",
                status="stale",
            ),
        ]
        context = context_with_solution(experiments=experiments)
        payload = revised_payload(
            selected_ref=None,
            changed_units=["core_idea", "C2"],
            core="A revised core mechanism.",
            rationale=None,
            components=[
                {"component_id": "C1", "description": "A revised first unit."},
                {"component_id": "C2", "description": "A revised second unit."},
                {"component_id": "C3", "description": "A revised third unit."},
            ],
        )
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        outcome = module.revise(context=context)
        state = context.get_state()
        statuses = {component.id: component.status for component in state.current_solution.components}
        self.assertEqual(statuses, {"C1": "modified", "C2": "modified", "C3": "modified"})
        self.assertEqual([experiment.status for experiment in state.experiments], ["stale", "stale"])
        self.assertEqual(outcome.verification_focus, VerificationFocus(mode="full_solution"))

    def test_two_component_revision_uses_full_focus_and_deduplicates_affected_units(self):
        context = context_with_solution()
        add_partial_verification(
            context,
            result=partial_result(component_ids=["C2", "C2", "C3"]),
        )
        payload = revised_payload(
            changed_units=["C2", "C3"],
            rationale=None,
            components=[
                {"component_id": "C2", "description": "Revised unit two."},
                {"component_id": "C3", "description": "Revised unit three."},
            ],
        )
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        outcome = module.revise(context=context)
        self.assertEqual(outcome.revision_need.affected_units, ["C2", "C3"])
        self.assertEqual(outcome.verification_focus, VerificationFocus(mode="full_solution"))

    def test_apply_revision_preserves_evidence_and_rejects_invalid_patch_shapes(self):
        current = solution()
        valid = RevisionResult(
            status="revised",
            revised_components=[
                {"component_id": "C2", "description": "Changed unit."}
            ],
            changed_units=["C2"],
            revision_summary="Changed.",
        )
        merged = apply_revision(current, valid)
        self.assertEqual(merged.evidence_refs, current.evidence_refs)
        self.assertEqual(merged.components[1].evidence_refs, current.components[1].evidence_refs)
        self.assertEqual(merged.components[1].status, "modified")

        invalids = [
            revised_payload(changed_units=["C2"], components=[]),
            revised_payload(changed_units=["C9"], components=[{"component_id": "C9", "description": "New."}]),
            revised_payload(
                changed_units=["C2"],
                components=[
                    {"component_id": "C2", "description": "One."},
                    {"component_id": "C2", "description": "Two."},
                ],
            ),
            revised_payload(changed_units=["C2"], components=[{"component_id": "C2", "description": "Original unit 2."}]),
            revised_payload(changed_units=["rationale"], components=[], rationale="Only rationale."),
            revised_payload(changed_units=["unknown"], components=[]),
        ]
        for payload in invalids:
            context = context_with_solution()
            module = TargetedRevisionModule(FakeLLMClient([payload]))
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                module.revise(context=context)
            self.assertEqual(context.get_memory().revision_memory, [])
            self.assertEqual(context.get_state().current_solution, solution())

    def test_extra_state_fields_from_llm_are_rejected(self):
        payload = revised_payload()
        payload["result"]["novelty_status"] = "distinct"
        context = context_with_solution()
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        with self.assertRaises(StructuredOutputError):
            module.revise(context=context)
        self.assertEqual(module.call_count, 0)

    def test_unable_to_revise_writes_failure_memory_without_state_change(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=1))
        payload = revised_payload(
            selected_ref="I1:1",
            status="unable_to_revise",
            changed_units=[],
            rationale=None,
            components=[],
            failure_reason="The minimum necessary change is not defensible.",
        )
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        before_state = context.get_state().model_dump(mode="python")
        before_inspirations = context.get_memory().inspiration_memory
        outcome = module.revise(context=context, inspiration_record_id="I1", step=5)
        self.assertEqual(outcome.revision_result.status, "unable_to_revise")
        self.assertIsNone(outcome.next_required_action)
        self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(context.get_memory().inspiration_memory, before_inspirations)
        self.assertEqual(len(context.get_memory().revision_memory), 1)
        self.assertEqual(context.get_memory().revision_memory[0].status, "unable_to_revise")
        self.assertEqual(len(context.get_memory().failure_memory), 1)
        self.assertEqual(context.get_memory().failure_memory[0].type, "failed_revision")
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "revision_failed")
        self.assertEqual(event.state_diff, {})
        self.assertEqual(set(event.memory_writes), {"revision_memory", "failure_memory"})

    def test_inspiration_guided_revision_uses_one_candidate_and_does_not_call_verification(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=3))
        payload = revised_payload(
            selected_ref="I1:2",
            changed_units=["C2"],
            rationale=None,
            components=[{"component_id": "C2", "description": "Inspired unit."}],
        )
        llm = FakeLLMClient([payload])
        module = TargetedRevisionModule(llm)
        before_inspiration = context.get_memory().inspiration_memory
        outcome = module.revise(context=context, inspiration_record_id="I1")
        self.assertEqual(outcome.selected_inspiration_ref, "I1:2")
        self.assertEqual(context.get_memory().inspiration_memory, before_inspiration)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(outcome.next_required_action, "VERIFY_RELATED_WORK")

    def test_historical_inspiration_is_not_available_without_explicit_trigger(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=2))
        captured = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return revised_payload(selected_ref=None)

        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        outcome = module.revise(context=context)
        payload = json.loads(captured["content"])
        self.assertEqual(payload["available_inspirations"], [])
        self.assertIsNone(outcome.selected_inspiration_ref)

    def test_explicit_inspiration_trigger_supplies_only_one_record(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(inspiration_record(count=1, record_id="I1"))
        context.memory_store.save_inspiration(inspiration_record(count=2, record_id="I2"))
        captured = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return revised_payload(
                selected_ref="I2:1",
                changed_units=["C2"],
                rationale=None,
                components=[{"component_id": "C2", "description": "Selected bridge unit."}],
            )

        module = TargetedRevisionModule(FakeLLMClient(handler=handler))
        module.revise(context=context, inspiration_record_id="I2")
        payload = json.loads(captured["content"])
        refs = [item["inspiration_ref"] for item in payload["available_inspirations"]]
        self.assertEqual(refs, ["I2:1", "I2:2"])

    def test_unknown_or_empty_explicit_inspiration_fails_before_llm(self):
        for record in (None, inspiration_record(count=0)):
            context = context_with_solution()
            if record is not None:
                context.memory_store.save_inspiration(record)
                record_id = record.id
            else:
                record_id = "missing"
            llm = FakeLLMClient([])
            module = TargetedRevisionModule(llm)
            before_memory = context.get_memory().model_dump(mode="python")
            before_events = context.trajectory.get_events()
            with self.assertRaises(PreconditionError):
                module.revise(context=context, inspiration_record_id=record_id)
            self.assertEqual(module.call_count, 0)
            self.assertEqual(llm.calls, [])
            self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
            self.assertEqual(context.trajectory.get_events(), before_events)

    def test_selected_inspiration_is_the_only_inspiration_evidence_ref(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(
            inspiration_record(count=2, grounding="arxiv", record_id="I1")
        )
        payload = revised_payload(
            selected_ref="I1:1",
            status="unable_to_revise",
            changed_units=[],
            rationale=None,
            components=[],
            failure_reason="No defensible minimum change.",
        )
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        module.revise(context=context, inspiration_record_id="I1")
        evidence_refs = context.get_memory().failure_memory[-1].evidence_refs
        self.assertIn("arxiv:1", evidence_refs)
        self.assertNotIn("arxiv:2", evidence_refs)

    def test_unable_without_selected_inspiration_has_no_historical_inspiration_refs(self):
        context = context_with_solution()
        context.memory_store.save_inspiration(
            inspiration_record(count=2, grounding="arxiv", record_id="I1")
        )
        payload = revised_payload(
            selected_ref=None,
            status="unable_to_revise",
            changed_units=[],
            rationale=None,
            components=[],
            failure_reason="No defensible minimum change.",
        )
        module = TargetedRevisionModule(FakeLLMClient([payload]))
        module.revise(context=context)
        evidence_refs = context.get_memory().failure_memory[-1].evidence_refs
        self.assertNotIn("arxiv:1", evidence_refs)
        self.assertNotIn("arxiv:2", evidence_refs)

    def test_invalid_trajectory_step_is_preflighted_before_llm_and_memory(self):
        context = context_with_solution()
        llm = FakeLLMClient([revised_payload()])
        module = TargetedRevisionModule(llm)
        before_state = context.get_state().model_dump(mode="python")
        before_memory = context.get_memory().model_dump(mode="python")
        before_events = context.trajectory.get_events()
        with self.assertRaises(PreconditionError):
            module.revise(context=context, step=-1)
        self.assertEqual(module.call_count, 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
        self.assertEqual(context.trajectory.get_events(), before_events)


class M7StateStoreTests(unittest.TestCase):
    def test_state_store_only_success_updates_state_and_not_trajectory(self):
        context = context_with_solution()
        state_store = ResearchStateStore(context.get_state())
        memory_store = ResearchMemoryStore(seed().paper_id, context.get_memory())
        module = TargetedRevisionModule(
            FakeLLMClient([revised_payload()])
        )
        before = state_store.get_state()
        outcome = module.revise(
            active_gap=gap(),
            current_solution=solution(),
            seed_context=seed(),
            state_store=state_store,
            memory_store=memory_store,
        )
        self.assertEqual(outcome.next_required_action, "VERIFY_RELATED_WORK")
        self.assertEqual(state_store.get_state().current_solution.components[1].status, "modified")
        self.assertEqual(memory_store.get_revision_history()[0].id, "R1")
        self.assertEqual(state_store.get_state().active_gap, before.active_gap)


if __name__ == "__main__":
    unittest.main()
