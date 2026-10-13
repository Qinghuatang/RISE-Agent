import unittest

from pydantic import ValidationError

from research_agent.errors import ModuleContractError, PreconditionError, StructuredOutputError
from research_agent.ingestion.markdown import (
    build_full_paper_reading_context,
    parse_mineru_markdown,
)
from research_agent.llm.fake import FakeLLMClient
from research_agent.llm.recording import RecordingLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    CandidateGap,
    CandidateSolution,
    ExperimentContext,
    FailureRecord,
    GapAgenda,
    GapGenerationDraft,
    GapVerificationResult,
    ResearchState,
    SeedPaperContext,
    VerificationRecord,
)
from research_agent.modules.gap_discovery import GapDiscoveryModule
from research_agent.modules.solution_ideation import SolutionIdeationModule
from research_agent.trajectory.context import TrajectoryContext


def make_seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="seed-m4",
        title="Seed",
        abstract="Abstract",
        research_objective="Improve robustness under drift.",
        core_method="A state-aware method.",
        main_components=["Encoder", "Correction module"],
        main_findings=["The method helps on Dataset-A."],
        limitations=["It assumes stable intermediate states."],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Dataset-A"],
            metrics=["accuracy"],
            major_baselines=["Seed baseline"],
            experimental_setting="Standard split.",
        ),
    )


def reading_for(seed: SeedPaperContext):
    markdown = f"""# {seed.title or 'Generic Paper'}

## ABSTRACT
{seed.abstract or 'A supplied abstract.'}

## 1 INTRODUCTION
{seed.research_objective or 'A supplied research objective.'}

## 2 METHOD
{seed.core_method or 'A supplied core method.'}
{' '.join(seed.main_components)}
{' '.join(seed.method_constraints)}

## 3 FINDINGS
{' '.join(seed.main_findings)}

## 4 LIMITATIONS
{' '.join(seed.limitations)}

## REFERENCES
Reference-only content.
"""
    return build_full_paper_reading_context(
        parse_mineru_markdown(markdown, paper_id=seed.paper_id)
    )


def gap_draft(*refs: str) -> dict:
    mapping = {
        "seed:research_objective": "S3",
        "seed:core_method": "S4",
        "seed:main_components[0]": "S4",
        "seed:main_findings[0]": "S5",
        "seed:limitations[0]": "S6",
    }
    return {
        "gaps": [
            {
                "statement": "Intermediate states can drift without a reliable correction target.",
                "evidence_refs": [mapping.get(ref, ref) for ref in refs],
            }
        ]
    }


def gap_evaluation(
    *,
    candidate_id: str = "G1",
    confidence: str = "high",
    priority: str = "high",
) -> dict:
    return {
        "evaluations": [
            {
                "candidate_id": candidate_id,
                "confidence": confidence,
                "confidence_reason": "The supplied Seed Paper evidence supports this assessment.",
                "priority": priority,
                "priority_reason": "It affects a central method-level opportunity.",
            }
        ]
    }


def make_gap(identifier="G1", status="active", confidence="high") -> CandidateGap:
    return CandidateGap(
        id=identifier,
        statement="An unresolved research problem.",
        confidence=confidence,
        confidence_reason="The seed limitation is explicit.",
        priority="high",
        priority_reason="It affects the objective.",
        status=status,
        evidence_refs=["seed:limitations[0]"],
    )


def valid_verification(target="G1", step=1) -> VerificationRecord:
    return VerificationRecord(
        id="VG1",
        task="VERIFY_GAP",
        target_ref=target,
        result=GapVerificationResult(
            status="valid",
            summary="The gap remains open.",
            remaining_gap="The gap remains open.",
            evidence_level="abstract_only",
        ),
        created_at_step=step,
    )


class GapDiscoveryM4Tests(unittest.TestCase):
    def test_gap_evaluation_split_objects_get_one_format_retry(self):
        candidates = GapAgenda(
            gaps=[
                make_gap(identifier=f"G{index}", status="candidate")
                for index in range(1, 6)
            ]
        )
        legal_evidence_ref = reading_for(make_seed()).body_section_ids[0]
        for candidate in candidates.gaps:
            candidate.evidence_refs = [legal_evidence_ref]
        split_evaluations = []
        for index in range(1, 6):
            split_evaluations.extend(
                [
                    {
                        "candidate_id": f"G{index}",
                        "confidence": "medium",
                        "confidence_reason": "The supplied evidence supports the gap.",
                    },
                    {
                        "priority": "high",
                        "priority_reason": "The opportunity is method-level and testable.",
                    },
                ]
            )
        valid_evaluations = {
            "evaluations": [
                {
                    "candidate_id": f"G{index}",
                    "confidence": "medium",
                    "confidence_reason": "The supplied evidence supports the gap.",
                    "priority": "high",
                    "priority_reason": "The opportunity is method-level and testable.",
                }
                for index in range(1, 6)
            ]
        }
        provider = FakeLLMClient(
            [{"evaluations": split_evaluations}, valid_evaluations]
        )
        recorder = RecordingLLMClient(provider)
        module = GapDiscoveryModule(recorder)

        result = module.evaluate_gaps(
            candidates,
            make_seed(),
            full_paper_reading_context=reading_for(make_seed()),
        )

        self.assertEqual(len(result.gaps), 5)
        self.assertEqual([gap.id for gap in result.gaps], [f"G{i}" for i in range(1, 6)])
        self.assertEqual(module.call_count, 1)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(recorder.records), 2)
        self.assertFalse(recorder.records[0].success)
        self.assertEqual(recorder.records[0].validation_errors[0]["error_type"], "too_long")
        self.assertEqual(recorder.records[1].format_retry_count, 1)
        retry_prompt = recorder.records[1].retry_reason or ""
        self.assertIn("evaluations", retry_prompt)

    def test_discovery_uses_full_main_body_for_both_generation_and_evaluation(self):
        markdown = """# Generic Scientific Paper

## ABSTRACT
A generic abstract.

## 1 RELATED WORK
Related work is part of the supplied scientific context.

## 2 METHOD
The main method defines a mechanism.

## 3 EXPERIMENTS
The experiments describe observed behavior.

## REFERENCES
Reference-only material must not be supplied.

## APPENDIX
Appendix-only material must not be supplied.
"""
        reading = build_full_paper_reading_context(
            parse_mineru_markdown(markdown, paper_id="seed-m4")
        )
        llm = FakeLLMClient(
            [
                gap_draft("seed:core_method"),
                gap_evaluation(confidence="medium", priority="high"),
            ]
        )
        module = GapDiscoveryModule(llm)
        result = module.discover(make_seed(), full_paper_reading_context=reading)

        self.assertEqual(result.active_gap.id, "G1")
        self.assertEqual(module.call_count, 2)
        for call in llm.calls:
            content = call.messages[1].content
            self.assertIn("Related work is part", content)
            self.assertIn("The main method defines", content)
            self.assertIn("The experiments describe", content)
            self.assertNotIn("Reference-only material", content)
            self.assertNotIn("Appendix-only material", content)

    def test_one_call_materializes_ids_statuses_and_real_evidence(self):
        llm = FakeLLMClient(
            [
                    gap_draft(
                    "seed:limitations[0]",
                    "seed:research_objective",
                    "seed:core_method",
                    "seed:main_components[0]",
                    "seed:main_findings[0]",
                )
            ]
        )
        module = GapDiscoveryModule(llm)
        agenda = module.generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        self.assertEqual(module.call_count, 1)
        self.assertEqual(agenda.gaps[0].id, "G1")
        self.assertEqual(agenda.gaps[0].status, "candidate")
        self.assertIn('"evidence_refs"', llm.calls[0].messages[0].content)
        evidence_content = llm.calls[0].messages[1].content
        self.assertIn("LEGAL GAP EVIDENCE SECTION IDS", evidence_content)
        self.assertIn("S3", evidence_content)
        self.assertIn("S4", evidence_content)
        self.assertIn("S5", evidence_content)
        self.assertIn("S6", evidence_content)
        self.assertNotIn("seed:research_objective", evidence_content)
        self.assertNotIn("seed:experiment_context", evidence_content)
        self.assertIn("FULL SEED PAPER MAIN-BODY CONTENT", evidence_content)

    def test_experiment_context_refs_are_rejected(self):
        invalid_evaluation_gap = gap_draft("seed:experiment_context.metrics[0]")
        invalid_evaluation_gap["gaps"][0]["statement"] = (
            "The model has only been evaluated at 1.3B."
        )
        with self.assertRaises(ModuleContractError):
            GapDiscoveryModule(
                FakeLLMClient([invalid_evaluation_gap])
            ).generate_gaps(
                make_seed(), full_paper_reading_context=reading_for(make_seed())
            )

    def test_gap_prompt_declares_method_boundary_and_pure_evaluation_example(self):
        llm = FakeLLMClient([gap_draft("seed:core_method")])
        GapDiscoveryModule(llm).generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        prompt = llm.calls[0].messages[0].content
        self.assertIn("METHOD-LEVEL ACTIONABILITY", prompt)
        self.assertIn("substantive change", prompt)
        self.assertIn("more datasets", prompt)
        self.assertIn("more baselines", prompt)
        self.assertIn("a missing metric", prompt)
        self.assertIn("a missing ablation", prompt)
        self.assertIn("larger-scale evaluation", prompt)
        self.assertIn("do not return it as a gap", prompt)
        self.assertNotIn("should be preferred", prompt)
        self.assertNotIn("SOURCE PRIORITY AND GENERATION ORDER", prompt)

    def test_gap_prompt_declares_generic_reasoning_gates(self):
        llm = FakeLLMClient([gap_draft("seed:core_method")])
        GapDiscoveryModule(llm).generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        prompt = llm.calls[0].messages[0].content
        for phrase in (
            "EVIDENCE SUFFICIENCY GATE",
            "CONSERVATIVE METHOD OPPORTUNITY FALLBACK",
            "WHOLE-METHOD CONSISTENCY GATE",
            "A design fact is not automatically a failure",
            "complete method",
            "evidence that will be listed in `evidence_refs`",
            "compensates",
            "mitigates",
            "bypasses",
            "contradicts",
            "weakens the claimed severity or scope",
            "Paper-observed problems",
            "Mechanism-derived problems",
            "Boundary/system-level problems",
            "zero to five",
        ):
            self.assertIn(phrase, prompt)

    def test_synthetic_compensating_path_is_available_to_whole_method_call(self):
        seed = make_seed().model_copy(
            update={
                "paper_id": "synthetic-compensation",
                "core_method": "The method transforms an intermediate representation.",
                "main_components": [
                    "Component A transforms part of the intermediate representation.",
                    "Component B provides an alternative path carrying relevant original information.",
                ],
                "main_findings": [],
                "limitations": [],
            }
        )
        response = gap_draft("seed:core_method", "seed:main_components[0]")
        llm = FakeLLMClient([response])
        agenda = GapDiscoveryModule(llm).generate_gaps(
            seed, full_paper_reading_context=reading_for(seed)
        )
        self.assertEqual(len(agenda.gaps), 1)
        self.assertEqual(agenda.gaps[0].confidence, "low")
        self.assertEqual(len(llm.calls), 1)
        request_text = llm.calls[0].messages[1].content
        self.assertIn("Component A transforms part", request_text)
        self.assertIn("Component B provides an alternative path", request_text)

    def test_synthetic_constraint_without_failure_supports_low_opportunity_gap(self):
        seed = make_seed().model_copy(
            update={
                "paper_id": "synthetic-architecture-only",
                "research_objective": None,
                "main_findings": [],
                "limitations": [],
                "core_method": "The method uses a fixed architectural arrangement.",
                "main_components": ["A local processing component."],
            }
        )
        response = gap_draft("seed:core_method", "seed:main_components[0]")
        response["gaps"][0]["statement"] = (
            "The fixed architectural arrangement leaves flexibility for broader operating conditions unresolved."
        )
        llm = FakeLLMClient([response])
        agenda = GapDiscoveryModule(llm).generate_gaps(
            seed, full_paper_reading_context=reading_for(seed)
        )
        self.assertEqual(len(agenda.gaps), 1)
        self.assertEqual(agenda.gaps[0].confidence, "low")

    def test_explicit_future_direction_is_a_valid_gap_source(self):
        seed = make_seed().model_copy(
            update={
                "limitations": [
                    "The authors identify future work to extend the method beyond stable intermediate states."
                ]
            }
        )
        response = gap_draft("seed:limitations[0]")
        response["gaps"][0]["statement"] = (
            "The method remains limited to stable intermediate states, leaving extension to changing states as an open method problem."
        )
        agenda_module = GapDiscoveryModule(
            FakeLLMClient([response, gap_evaluation(confidence="high")])
        )
        agenda = agenda_module.generate_gaps(
            seed, full_paper_reading_context=reading_for(seed)
        )
        agenda = agenda_module.evaluate_gaps(
            agenda, seed, full_paper_reading_context=reading_for(seed)
        )
        self.assertEqual(agenda.gaps[0].confidence, "high")
        self.assertEqual(agenda.gaps[0].evidence_refs, ["S6"])

    def test_priority_is_independent_from_low_confidence(self):
        response = gap_draft("seed:core_method")
        agenda_module = GapDiscoveryModule(
            FakeLLMClient([response, gap_evaluation(confidence="low", priority="high")])
        )
        agenda = agenda_module.generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        agenda = agenda_module.evaluate_gaps(
            agenda,
            make_seed(),
            full_paper_reading_context=reading_for(make_seed()),
        )
        self.assertEqual(agenda.gaps[0].confidence, "low")
        self.assertEqual(agenda.gaps[0].priority, "high")

    def test_method_level_evidence_fields_remain_legal(self):
        llm = FakeLLMClient(
            [
                gap_draft(
                    "seed:research_objective",
                    "seed:core_method",
                    "seed:main_components[0]",
                    "seed:main_findings[0]",
                    "seed:limitations[0]",
                )
            ]
        )
        agenda = GapDiscoveryModule(llm).generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        self.assertEqual(len(agenda.gaps), 1)

    def test_zero_gaps_are_a_valid_conservative_result(self):
        module = GapDiscoveryModule(FakeLLMClient([{"gaps": []}]))
        agenda = module.generate_gaps(
            make_seed(), full_paper_reading_context=reading_for(make_seed())
        )
        self.assertEqual(agenda.gaps, [])
        self.assertEqual(module.call_count, 1)

    def test_gap_discovery_draft_allows_zero_to_five_candidates(self):
        self.assertEqual(
            GapGenerationDraft.model_validate({"gaps": []}).gaps,
            [],
        )
        valid = GapGenerationDraft.model_validate(gap_draft("seed:core_method"))
        self.assertEqual(len(valid.gaps), 1)

    def test_more_than_five_and_invalid_evidence_are_rejected(self):
        too_many = {
            "gaps": [
                {
                    "statement": f"Gap {i}",
                    "evidence_refs": ["seed:limitations[0]"],
                }
                for i in range(6)
            ]
        }
        with self.assertRaises(StructuredOutputError):
            GapDiscoveryModule(FakeLLMClient([too_many])).generate_gaps(
                make_seed(), full_paper_reading_context=reading_for(make_seed())
            )
        with self.assertRaises(ModuleContractError):
            GapDiscoveryModule(
                FakeLLMClient([gap_draft("seed:not-present")])
            ).generate_gaps(
                make_seed(), full_paper_reading_context=reading_for(make_seed())
            )

    def test_empty_evidence_and_extra_fields_are_rejected(self):
        empty = {
            "gaps": [
                {
                    "statement": "A gap",
                    "evidence_refs": [],
                }
            ]
        }
        with self.assertRaises(StructuredOutputError):
            GapDiscoveryModule(FakeLLMClient([empty])).generate_gaps(
                make_seed(), full_paper_reading_context=reading_for(make_seed())
            )
        extra = gap_draft("seed:limitations[0]")
        extra["gaps"][0]["rank"] = 1
        with self.assertRaises(StructuredOutputError):
            GapDiscoveryModule(FakeLLMClient([extra])).generate_gaps(
                make_seed(), full_paper_reading_context=reading_for(make_seed())
            )

    def test_selection_uses_first_ordered_candidate_even_when_low_confidence(self):
        agenda = GapAgenda(
            gaps=[
                make_gap("G1", status="candidate", confidence="low"),
                make_gap("G2", status="candidate", confidence="medium"),
                make_gap("G3", status="candidate", confidence="high"),
            ]
        )
        result = GapDiscoveryModule.select_active_gap(agenda)
        self.assertEqual(result.active_gap.id, "G1")
        self.assertEqual([gap.id for gap in result.gap_agenda.gaps], ["G1", "G2", "G3"])
        self.assertEqual(result.gap_agenda.gaps[0].status, "active")
        self.assertEqual(result.gap_agenda.gaps[2].status, "candidate")

    def test_all_low_still_selects_first_candidate(self):
        agenda = GapAgenda(
            gaps=[
                make_gap("G1", status="candidate", confidence="low"),
                make_gap("G2", status="candidate", confidence="low"),
            ]
        )
        result = GapDiscoveryModule.select_active_gap(agenda)
        self.assertEqual(result.active_gap.id, "G1")

    def test_discovery_updates_only_active_gap_and_records_agenda(self):
        llm = FakeLLMClient(
            [
                {
                    "gaps": [
                        gap_draft("seed:limitations[0]")["gaps"][0],
                        {
                            "statement": "A second supported problem.",
                            "evidence_refs": ["S5"],
                        },
                    ]
                },
                {
                    "evaluations": [
                        {
                            "candidate_id": "G1",
                            "confidence": "high",
                            "confidence_reason": "The limitation is explicit.",
                            "priority": "high",
                            "priority_reason": "It affects the objective.",
                        },
                        {
                            "candidate_id": "G2",
                            "confidence": "medium",
                            "confidence_reason": "The finding supports it.",
                            "priority": "medium",
                            "priority_reason": "It affects robustness.",
                        },
                    ]
                },
            ]
        )
        context = TrajectoryContext.initialize(make_seed())
        result = GapDiscoveryModule(llm).discover(
            make_seed(),
            full_paper_reading_context=reading_for(make_seed()),
            context=context,
            step=1,
        )
        self.assertEqual(context.get_state().active_gap.id, result.active_gap.id)
        self.assertEqual(context.get_state().candidate_solutions, [])
        self.assertEqual(context.get_state().current_solution, None)
        agenda = context.get_latest_gap_agenda()
        self.assertEqual(agenda.gaps[0].status, "active")
        self.assertTrue(
            {event.event_type for event in context.trajectory.get_events()}
            >= {"gap_agenda_updated", "gap_discovered", "active_gap_selected"}
        )

    def test_refine_is_one_call_and_preserves_gap_contract(self):
        context = TrajectoryContext.initialize(make_seed())
        initial = GapDiscoveryModule(
            FakeLLMClient(
                [gap_draft("seed:limitations[0]"), gap_evaluation()]
            )
        ).discover(
            make_seed(),
            full_paper_reading_context=reading_for(make_seed()),
            context=context,
        )
        verifier = GapVerificationResult(
            status="partially_resolved",
            summary="Prior work covers the broad case.",
            covered_aspects=["Broad case"],
            remaining_gap="The drift case remains.",
            evidence_level="abstract_only",
        )
        refine_llm = FakeLLMClient(
            [{"refined_statement": "The drift case remains unresolved.", "refinement_reason": "It is not covered."}]
        )
        module = GapDiscoveryModule(refine_llm)
        result = module.refine_gap(initial.active_gap, verifier, context=context, step=2)
        self.assertEqual(refine_llm.calls.__len__(), 1)
        refine_prompt = refine_llm.calls[0].messages[0].content
        self.assertIn("METHOD-LEVEL GAP REQUIREMENT", refine_prompt)
        self.assertIn("larger-scale", refine_prompt)
        self.assertIn("EVIDENCE SUFFICIENCY AND WHOLE-METHOD CONSISTENCY", refine_prompt)
        self.assertIn("compensate for, mitigate, bypass", refine_prompt)
        self.assertIn("new unsupported causal claim", refine_prompt)
        self.assertEqual(result.active_gap.id, "G1")
        self.assertEqual(result.active_gap.confidence, initial.active_gap.confidence)
        self.assertEqual(result.active_gap.priority, initial.active_gap.priority)
        self.assertEqual(result.active_gap.evidence_refs, initial.active_gap.evidence_refs)
        self.assertTrue(context.has_refined_gap("G1"))
        self.assertEqual(
            [event.event_type for event in context.trajectory.get_events()].count("gap_refined"),
            1,
        )
        with self.assertRaises(PreconditionError):
            module.refine_gap(initial.active_gap, verifier, context=context, step=3)
        self.assertEqual(len(refine_llm.calls), 1)

    def test_refine_valid_or_resolved_is_rejected_without_llm(self):
        llm = FakeLLMClient([])
        module = GapDiscoveryModule(llm)
        for status in ("valid", "resolved"):
            verification = GapVerificationResult(
                status=status,
                summary="Summary",
                evidence_level="abstract_only",
            )
            with self.assertRaises(PreconditionError):
                module.refine_gap(make_gap(), verification)
        self.assertEqual(len(llm.calls), 0)

    def test_switch_is_deterministic_and_resets_solution_state(self):
        context = TrajectoryContext.initialize(make_seed())
        context.record_gap_agenda(
            1,
            GapAgenda(
                gaps=[
                    make_gap("G1", status="active"),
                    make_gap("G2", status="candidate", confidence="low"),
                    make_gap("G3", status="candidate", confidence="medium"),
                ]
            ),
        )
        state = ResearchState(active_gap=make_gap("G1"))
        context.state_store.replace_state(state)
        context.memory_store.save_verification(
            valid_verification().model_copy(
                update={
                    "result": GapVerificationResult(
                        status="resolved",
                        summary="The active gap was resolved.",
                        evidence_level="abstract_only",
                    )
                }
            )
        )
        result = GapDiscoveryModule(FakeLLMClient([])).switch_gap(
            context=context, terminal_status="resolved", step=2
        )
        self.assertEqual(result.active_gap.id, "G2")
        self.assertEqual(result.gap_agenda.gaps[0].status, "resolved")
        self.assertEqual(context.get_state().active_gap.id, "G2")
        self.assertEqual(context.get_state().candidate_solutions, [])
        self.assertIsNone(context.get_state().current_solution)
        self.assertEqual(context.get_state().experiments, [])
        self.assertEqual(len(context.trajectory.get_events()), 4)
        self.assertEqual(len(GapDiscoveryModule(FakeLLMClient([])).llm_client.calls), 0)


def candidate_form_response(start="S1"):
    return {
        "candidates": [
            {"core_idea": f"Mechanism {start}-a", "rationale": "It targets the gap."},
            {"core_idea": f"Mechanism {start}-b", "rationale": "It changes another part."},
            {"core_idea": f"Mechanism {start}-c", "rationale": "It uses a distinct mechanism."},
        ]
    }


def development_response(selected="S1"):
    return {
        "selected_solution_id": selected,
        "selection_reason": "It best aligns with the verified gap.",
        "core_idea": "A developed mechanism for the active gap.",
        "rationale": "The mechanism directly addresses the remaining limitation.",
        "components": ["Detect the condition.", "Apply corrective intervention."],
    }


class SolutionIdeationM4Tests(unittest.TestCase):
    def make_context(self, candidates=None):
        context = TrajectoryContext.initialize(make_seed())
        context.state_store.replace_state(
            ResearchState(active_gap=make_gap(), candidate_solutions=candidates or [])
        )
        context.memory_store.save_verification(valid_verification())
        return context

    def test_missing_or_nonvalid_gap_verification_blocks_without_llm(self):
        llm = FakeLLMClient([])
        context = TrajectoryContext.initialize(make_seed())
        context.state_store.replace_state(ResearchState(active_gap=make_gap()))
        with self.assertRaises(PreconditionError):
            SolutionIdeationModule(llm).ideate(context)
        self.assertEqual(len(llm.calls), 0)
        context.memory_store.save_verification(
            VerificationRecord(
                id="VG2",
                task="VERIFY_GAP",
                target_ref="G1",
                result=GapVerificationResult(
                    status="partially_resolved",
                    summary="Not valid.",
                    remaining_gap="Still open.",
                    evidence_level="abstract_only",
                ),
                created_at_step=2,
            )
        )
        with self.assertRaises(PreconditionError):
            SolutionIdeationModule(llm).ideate(context)
        self.assertEqual(len(llm.calls), 0)

    def test_first_run_is_form_three_then_select_develop(self):
        llm = FakeLLMClient([candidate_form_response(), development_response()])
        context = self.make_context()
        result = SolutionIdeationModule(llm).ideate(context, step=2)
        self.assertTrue(result.generated_new_candidates)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.selected_solution_id, "S1")
        state = context.get_state()
        self.assertEqual([candidate.id for candidate in state.candidate_solutions], ["S1", "S2", "S3"])
        self.assertEqual([candidate.status for candidate in state.candidate_solutions], ["selected", "candidate", "candidate"])
        self.assertEqual(state.current_solution.status, "developed")
        self.assertEqual(state.current_solution.novelty_status, "unchecked")
        self.assertIsNone(state.current_solution.novelty_summary)
        self.assertEqual([component.id for component in state.current_solution.components], ["C1", "C2"])
        self.assertEqual([component.status for component in state.current_solution.components], ["ready", "ready"])
        self.assertEqual(state.experiments, [])
        self.assertEqual(result.next_required_action, "VERIFY_RELATED_WORK")
        prompt_text = "\n".join(message.content for call in llm.calls for message in call.messages)
        self.assertIn('"components"', prompt_text)
        self.assertIn('"selected_solution_id"', prompt_text)

    def test_candidate_pool_reuse_does_not_form_again(self):
        candidates = [
            CandidateSolution(id="S1", core_idea="Old 1", rationale="Old", status="rejected"),
            CandidateSolution(id="S2", core_idea="Old 2", rationale="Old", status="candidate"),
            CandidateSolution(id="S3", core_idea="Old 3", rationale="Old", status="candidate"),
        ]
        llm = FakeLLMClient([development_response("S2")])
        context = self.make_context(candidates)
        result = SolutionIdeationModule(llm).ideate(context)
        self.assertFalse(result.generated_new_candidates)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(context.get_state().candidate_solutions[1].status, "selected")
        self.assertEqual(context.get_state().candidate_solutions[0].status, "rejected")

    def test_all_rejected_gets_next_ids_and_replaces_pool(self):
        candidates = [
            CandidateSolution(id="S1", core_idea="Old 1", rationale="Old", status="rejected"),
            CandidateSolution(id="S2", core_idea="Old 2", rationale="Old", status="rejected"),
            CandidateSolution(id="S3", core_idea="Old 3", rationale="Old", status="rejected"),
        ]
        llm = FakeLLMClient([candidate_form_response("S4"), development_response("S4")])
        context = self.make_context(candidates)
        SolutionIdeationModule(llm).ideate(context)
        self.assertEqual([candidate.id for candidate in context.get_state().candidate_solutions], ["S4", "S5", "S6"])
        self.assertEqual(len(llm.calls), 2)

    def test_invalid_selected_id_and_rejected_id_are_rejected(self):
        candidate = CandidateSolution(id="S1", core_idea="Idea", rationale="Reason", status="candidate")
        module = SolutionIdeationModule(FakeLLMClient([development_response("S9")]))
        with self.assertRaises(ModuleContractError):
            module.select_and_develop(make_gap(), make_seed(), [candidate], [])
        rejected = candidate.model_copy(update={"status": "rejected"})
        module = SolutionIdeationModule(FakeLLMClient([development_response("S1")]))
        with self.assertRaises(PreconditionError):
            module.select_and_develop(make_gap(), make_seed(), [rejected], [])

    def test_only_allowed_failure_types_are_sent_to_prompts(self):
        failures = [
            FailureRecord(type="rejected_solution", object_ref="S1", summary="No", reason="Overlap"),
            FailureRecord(type="rejected_mechanism", object_ref="M", summary="No", reason="Overlap"),
            FailureRecord(type="rejected_gap", object_ref="G1", summary="No", reason="Resolved"),
            FailureRecord(type="failed_revision", object_ref="R1", summary="No", reason="Failed"),
        ]
        llm = FakeLLMClient([candidate_form_response(), development_response()])
        module = SolutionIdeationModule(llm)
        gap_verification = valid_verification().result
        module.form_solutions(make_gap(), make_seed(), gap_verification, failures)
        module.select_and_develop(make_gap(), make_seed(), [CandidateSolution(id="S1", core_idea="I", rationale="R", status="candidate")], failures)
        text = "\n".join(message.content for call in llm.calls for message in call.messages)
        self.assertIn("rejected_solution", text)
        self.assertIn("rejected_mechanism", text)
        self.assertNotIn("rejected_gap", text)
        self.assertNotIn("failed_revision", text)

    def test_solution_trajectory_has_candidate_and_developed_events(self):
        llm = FakeLLMClient([candidate_form_response(), development_response()])
        context = self.make_context()
        SolutionIdeationModule(llm).ideate(context)
        types = [event.event_type for event in context.trajectory.get_events()]
        self.assertIn("solution_candidates_generated", types)
        self.assertIn("solution_developed", types)


if __name__ == "__main__":
    unittest.main()
