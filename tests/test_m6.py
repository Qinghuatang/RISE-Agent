import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from research_agent.errors import (
    ModuleContractError,
    PreconditionError,
    RetrievalTransportError,
    StructuredOutputError,
)
from research_agent.llm.fake import FakeLLMClient
from research_agent.llm.recording import RecordingLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models import (
    BridgeEntity,
    BridgeGenerationDraft,
    CandidateGap,
    Component,
    CurrentSolution,
    ExperimentContext,
    InspirationCandidate,
    InspirationNeed,
    InspirationRecord,
    InspirationResult,
    InspirationSearchRecord,
    PaperRecord,
    ResearchState,
    SeedPaperContext,
)
from research_agent.modules.inspiration_retrieval import (
    InspirationRetrievalModule,
)
from research_agent.retrieval.arxiv import (
    ArxivConfig,
    ArxivInspirationSearchProvider,
    canonical_arxiv_id,
)
from research_agent.retrieval.fake import FakeInspirationSearchProvider
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-seed",
        title="A Generic Method",
        abstract="A generic abstract.",
        research_objective="Study a mechanism-level question.",
        core_method="A current mechanism with cooperating parts.",
        main_components=["Part A", "Part B"],
        main_findings=["The method has a reported finding."],
        limitations=["A boundary remains open."],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task"],
            metrics=["metric"],
            major_baselines=["baseline"],
            experimental_setting="Generic setting.",
        ),
    )


def gap() -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="The current method leaves a mechanism-level opportunity open.",
        confidence="medium",
        confidence_reason="The supplied method evidence supports the opportunity.",
        priority="high",
        priority_reason="It affects the central mechanism.",
        status="active",
        evidence_refs=["seed:core_method"],
    )


def solution() -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="A current mechanism for the active problem.",
        rationale="It addresses the active gap.",
        components=[
            Component(id="C1", description="The first mechanism.", status="ready"),
            Component(id="C2", description="The second mechanism.", status="ready"),
        ],
        status="developed",
        novelty_status="distinct",
    )


def entity(index: int) -> dict[str, Any]:
    return {
        "name": f"Transfer Mechanism {index}",
        "type": "mechanism",
        "bridge_reason": f"It may provide function {index}.",
    }


def draft(selected: int = 3) -> dict[str, Any]:
    entities = [entity(index) for index in range(1, 6)]
    return {
        "inspiration_need": {
            "current_problem": "The current mechanism leaves a function unresolved.",
            "required_function": "Provide a transferable function.",
            "current_mechanism": "The current mechanism.",
            "rejected_mechanisms": ["An explicitly rejected mechanism."],
            "relevant_context": "Compact generic context.",
        },
        "proposed_entities": entities,
        "selected_entities": entities[:selected],
    }


def paper(identifier: str = "arxiv:1000.0001", title: str = "A Transfer Paper") -> PaperRecord:
    return PaperRecord(
        paper_id=identifier,
        title=title,
        abstract="An abstract describing a transferable mechanism.",
        authors=["Author"],
        year=2025,
        venue="Generic Venue",
        source="arxiv",
        url="https://arxiv.org/abs/1000.0001",
    )


def grounded_result(paper_ref: str = "arxiv:1000.0001") -> dict[str, Any]:
    return {
        "status": "found",
        "inspirations": [
            {
                "bridge_entity": entity(1),
                "paper_ref": paper_ref,
                "source_problem": "The source paper addresses a related function.",
                "source_mechanism": "Its abstract supports a transferable mechanism.",
                "transferable_principle": "The mechanism suggests a general principle.",
                "adaptation_hint": "The principle may inform the current need.",
                "grounding": "arxiv",
            }
        ],
    }


def fallback_result() -> dict[str, Any]:
    return {
        "status": "found",
        "inspirations": [
            {
                "bridge_entity": entity(1),
                "paper_ref": None,
                "source_problem": "A conceptual problem in another mechanism family.",
                "source_mechanism": "A conceptual mechanism is relevant.",
                "transferable_principle": "The principle may transfer.",
                "adaptation_hint": "Consider whether it serves the current function.",
                "grounding": "none",
            }
        ],
    }


def context_with_solution() -> TrajectoryContext:
    context = TrajectoryContext.initialize(seed())
    context.state_store.replace_state(
        ResearchState(
            active_gap=gap(),
            current_solution=solution(),
            experiments=[],
        )
    )
    return context


class M6ModuleTests(unittest.TestCase):
    def test_grounded_provenance_retry_reuses_retrieval_and_can_drop_invalid_candidates(self):
        context = context_with_solution()
        bad = grounded_result("arxiv:unknown")
        llm = RecordingLLMClient(
            FakeLLMClient([draft(selected=1), bad, fallback_result()])
        )
        provider = FakeInspirationSearchProvider([paper()])
        module = InspirationRetrievalModule(llm, provider)
        result = module.retrieve(context=context)
        self.assertEqual(result.status, "no_useful_inspiration")
        self.assertEqual(provider.queries, ["Transfer Mechanism 1"])
        self.assertEqual(module.scientific_search_count, 1)
        self.assertEqual(len(llm.records), 3)
        self.assertEqual(llm.records[-1].format_retry_count, 1)
        self.assertEqual(len(context.get_memory().paper_memory.inspiration_search_records), 1)

    def test_successful_retrieval_with_null_paper_ref_is_not_downgraded(self):
        context = context_with_solution()
        invalid = grounded_result()
        invalid["inspirations"][0]["paper_ref"] = None
        module = InspirationRetrievalModule(
            FakeLLMClient([draft(selected=1), invalid]),
            FakeInspirationSearchProvider([paper()]),
        )
        with self.assertRaises(StructuredOutputError):
            module.retrieve(context=context)
        self.assertEqual(context.get_memory().inspiration_memory, [])

    def test_inspiration_prompts_distinguish_hypotheses_from_external_evidence(self):
        module = InspirationRetrievalModule(
            FakeLLMClient([]), FakeInspirationSearchProvider()
        )
        entities_prompt = module._entity_prompt
        extract_prompt = module._extract_prompt
        self.assertIn("general scientific knowledge", entities_prompt)
        self.assertIn("hypotheses/search directions", entities_prompt)
        self.assertIn("not retrieved evidence", entities_prompt)
        self.assertIn("transferable_principle", extract_prompt)
        self.assertIn("title and abstract", extract_prompt)
        self.assertIn("omit", extract_prompt)
        self.assertIn("source_problem and source_mechanism", extract_prompt)
        self.assertIn("supplied title and abstract", extract_prompt)
        self.assertIn("transferable_principle and adaptation_hint", extract_prompt)
        self.assertIn("general scientific knowledge may be used", extract_prompt)
        self.assertIn('grounding \"none\"', extract_prompt)
        self.assertIn("TRANSFER VALUE CHECK", extract_prompt)
        self.assertIn("added computation, memory, or complexity", extract_prompt)

    def test_inspiration_prompt_keeps_controller_need_and_constraints_in_call_one(self):
        context = context_with_solution()
        context.seed_context = context.seed_context.model_copy(
            update={
                "method_constraints": [
                    "The supplied design preserves a stated constraint."
                ]
            }
        )
        captured: dict[str, Any] = {}

        def handler(request, output_schema):
            captured["content"] = request.messages[1].content
            return draft(selected=0)

        module = InspirationRetrievalModule(
            FakeLLMClient(handler=handler), FakeInspirationSearchProvider()
        )
        output = module.retrieve(
            context=context,
            controller_exploration_rationale=(
                "The current candidates remain within one mechanism family."
            ),
        )
        self.assertEqual(output.status, "no_useful_inspiration")
        payload = json.loads(captured["content"])
        self.assertEqual(
            payload["controller_exploration_rationale"],
            "The current candidates remain within one mechanism family.",
        )
        self.assertEqual(
            payload["seed_method_constraints"],
            ["The supplied design preserves a stated constraint."],
        )

    def test_zero_selection_prompt_defers_search_decision_to_controller(self):
        prompt = InspirationRetrievalModule(
            FakeLLMClient([]), FakeInspirationSearchProvider()
        )._entity_prompt
        self.assertIn("has already selected SEEK_INSPIRATION", prompt)
        self.assertIn("normally select one to three", prompt)
        self.assertIn("Return zero only when no", prompt)

    def test_rejected_mechanisms_are_materialized_from_authoritative_memory(self):
        context = context_with_solution()
        from research_agent.models import FailureRecord

        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_mechanism",
                object_ref="M1",
                summary="Authoritative rejected mechanism.",
                reason="It was already rejected.",
            )
        )
        payload = draft(selected=0)
        payload["inspiration_need"]["rejected_mechanisms"] = [
            "Invented by the model.",
            "A rewritten failure.",
        ]
        module = InspirationRetrievalModule(
            FakeLLMClient([payload, fallback_result()]), FakeInspirationSearchProvider()
        )
        module.retrieve(context=context)
        assert module.last_record is not None
        self.assertEqual(
            module.last_record.inspiration_need.rejected_mechanisms,
            ["Authoritative rejected mechanism."],
        )

    def test_selected_same_name_uses_proposed_canonical_definition(self):
        context = context_with_solution()
        payload = draft(selected=1)
        payload["selected_entities"] = [
            {
                "name": payload["proposed_entities"][0]["name"],
                "type": "method",
                "bridge_reason": "A rewritten selected definition.",
            }
        ]
        module = InspirationRetrievalModule(
            FakeLLMClient([payload, fallback_result()]), FakeInspirationSearchProvider()
        )
        module.retrieve(context=context)
        assert module.last_record is not None
        proposed = module.last_record.proposed_entities[0]
        selected = module.last_record.selected_entities[0]
        self.assertEqual(selected, proposed)
        self.assertEqual(selected.type, "mechanism")
        self.assertEqual(selected.bridge_reason, proposed.bridge_reason)

    def test_candidate_same_name_uses_selected_canonical_definition(self):
        context = context_with_solution()
        result = grounded_result()
        result["inspirations"][0]["bridge_entity"] = {
            "name": "Transfer Mechanism 1",
            "type": "method",
            "bridge_reason": "A rewritten candidate definition.",
        }
        module = InspirationRetrievalModule(
            FakeLLMClient([draft(selected=1), result]),
            FakeInspirationSearchProvider([paper()]),
        )
        output = module.retrieve(context=context)
        self.assertEqual(output.status, "found")
        assert module.last_record is not None
        self.assertEqual(
            output.inspirations[0].bridge_entity,
            module.last_record.selected_entities[0],
        )

    def test_current_solution_is_required_before_llm_or_provider(self):
        context = TrajectoryContext.initialize(seed())
        context.state_store.replace_state(ResearchState(active_gap=gap()))
        llm = FakeLLMClient([])
        provider = FakeInspirationSearchProvider([paper()])
        module = InspirationRetrievalModule(llm, provider)
        before = context.get_state().model_dump(mode="python")
        with self.assertRaises(PreconditionError):
            module.retrieve(context=context)
        self.assertEqual(module.call_count, 0)
        self.assertEqual(provider.queries, [])
        self.assertEqual(context.get_state().model_dump(mode="python"), before)
        self.assertEqual(context.get_memory().inspiration_memory, [])

    def test_call_one_prompt_contains_previous_and_only_rejected_mechanisms(self):
        context = context_with_solution()
        old = BridgeEntity(
            name="Previously Explored Principle",
            type="system principle",
            bridge_reason="Previously explored.",
        )
        context.memory_store.save_inspiration(
            InspirationRecord(
                id="I1",
                inspiration_need=InspirationNeed(
                    current_problem="Old problem",
                    required_function="Old function",
                    current_mechanism="Old mechanism",
                    relevant_context="Old context",
                ),
                proposed_entities=[old],
                selected_entities=[old],
                search_query="Previously Explored Principle",
                grounding_status="none",
                created_at_step=1,
            )
        )
        from research_agent.models import FailureRecord

        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_mechanism",
                object_ref="M1",
                summary="Avoid this rejected mechanism.",
                reason="It did not address the function.",
            )
        )
        context.memory_store.save_failure(
            FailureRecord(
                type="rejected_gap",
                object_ref="G0",
                summary="Do not pass this rejected gap.",
                reason="It was resolved.",
            )
        )
        context.memory_store.save_failure(
            FailureRecord(
                type="failed_revision",
                object_ref="R0",
                summary="Do not pass this failed revision.",
                reason="It failed.",
            )
        )
        captured: dict[str, Any] = {}

        def handler(request, output_schema):
            captured["prompt"] = request.messages[1].content
            return draft(selected=0)

        module = InspirationRetrievalModule(
            FakeLLMClient(handler=handler), FakeInspirationSearchProvider()
        )
        result = module.retrieve(context=context)
        prompt = captured["prompt"]
        self.assertEqual(result.status, "no_useful_inspiration")
        self.assertIn("Previously Explored Principle", prompt)
        self.assertIn("Avoid this rejected mechanism.", prompt)
        self.assertNotIn("Do not pass this rejected gap.", prompt)
        self.assertNotIn("Do not pass this failed revision.", prompt)

    def test_exactly_five_and_selected_subset_contracts_are_strict(self):
        context = context_with_solution()
        provider = FakeInspirationSearchProvider()
        invalids = []
        invalids.append({**draft(), "proposed_entities": [entity(i) for i in range(1, 5)]})
        invalids.append({**draft(), "proposed_entities": [entity(i) for i in range(1, 7)]})
        duplicate = draft()
        duplicate["proposed_entities"][1] = duplicate["proposed_entities"][0].copy()
        invalids.append(duplicate)
        bad_type = draft()
        bad_type["proposed_entities"][0]["type"] = "unsupported"
        invalids.append(bad_type)
        not_member = draft()
        not_member["selected_entities"] = [entity(99)]
        invalids.append(not_member)
        too_many = draft()
        too_many["selected_entities"] = [*too_many["proposed_entities"], entity(99)]
        invalids.append(too_many)
        for payload in invalids:
            module = InspirationRetrievalModule(FakeLLMClient([payload]), provider)
            with self.assertRaises((StructuredOutputError, ModuleContractError)):
                module.retrieve(context=context)
            self.assertEqual(module.call_count, 0)

    def test_zero_selection_skips_search_and_call_two(self):
        context = context_with_solution()
        llm = FakeLLMClient([draft(selected=0)])
        provider = FakeInspirationSearchProvider([paper()])
        module = InspirationRetrievalModule(llm, provider)
        result = module.retrieve(context=context, step=3)
        self.assertEqual(result.status, "no_useful_inspiration")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(provider.queries, [])
        self.assertEqual(module.scientific_search_count, 0)
        self.assertEqual(module.last_next_required_action, None)
        self.assertEqual(len(context.get_memory().inspiration_memory), 1)
        self.assertEqual(context.get_memory().inspiration_memory[0].search_query, "")

    def test_combined_query_grounding_memory_and_state_read_only(self):
        context = context_with_solution()
        papers = [paper("arxiv:1000.0001"), paper("arxiv:1000.0002", "Another Transfer")]
        llm = FakeLLMClient([draft(selected=3), grounded_result()])
        provider = FakeInspirationSearchProvider(papers)
        module = InspirationRetrievalModule(llm, provider)
        before = context.get_state().model_dump(mode="python")
        result = module.retrieve(context=context, step=2)
        self.assertEqual(result.status, "found")
        self.assertEqual(provider.queries, [
            "Transfer Mechanism 1 OR Transfer Mechanism 2 OR Transfer Mechanism 3"
        ])
        self.assertEqual(provider.limits, [20])
        self.assertEqual(module.scientific_search_count, 1)
        self.assertEqual(module.call_count, 2)
        self.assertEqual(context.get_state().model_dump(mode="python"), before)
        memory = context.get_memory()
        self.assertEqual([p.paper_id for p in memory.paper_memory.papers], [
            "arxiv:1000.0001", "arxiv:1000.0002"
        ])
        self.assertEqual(len(memory.paper_memory.inspiration_search_records), 1)
        self.assertEqual(len(memory.inspiration_memory), 1)
        self.assertEqual(module.last_next_required_action, "TARGETED_REVISION")
        event = context.trajectory.get_events()[-1]
        self.assertEqual(event.event_type, "inspiration_retrieved")
        self.assertEqual(event.action, "SEEK_INSPIRATION")
        self.assertEqual(event.state_diff, {})
        self.assertIn("inspiration_search_records", event.memory_writes)

    def test_exact_query_cache_reuses_papers_without_provider_or_search_budget(self):
        context = context_with_solution()
        provider = FakeInspirationSearchProvider([paper()])
        first = InspirationRetrievalModule(
            FakeLLMClient([draft(selected=1), grounded_result()]), provider
        )
        first.retrieve(context=context, step=1)
        second = InspirationRetrievalModule(
            FakeLLMClient([draft(selected=1), grounded_result()]), provider
        )
        second.retrieve(context=context, step=2)
        self.assertEqual(provider.queries, ["Transfer Mechanism 1"])
        self.assertEqual(first.scientific_search_count, 1)
        self.assertEqual(second.scientific_search_count, 0)
        self.assertTrue(second.last_cache_hit)
        self.assertEqual(second.last_cache_scope, "trajectory")
        self.assertEqual(len(context.get_memory().paper_memory.inspiration_search_records), 1)

    def test_corrupt_query_cache_fails_explicitly_before_provider(self):
        context = context_with_solution()
        context.memory_store.save_inspiration_search(
            InspirationSearchRecord(
                query="Transfer Mechanism 1",
                bridge_entities=[BridgeEntity(**entity(1))],
                paper_refs=["arxiv:missing"],
            )
        )
        provider = FakeInspirationSearchProvider([paper()])
        module = InspirationRetrievalModule(FakeLLMClient([draft(selected=1)]), provider)
        with self.assertRaises(ModuleContractError):
            module.retrieve(context=context)
        self.assertEqual(provider.queries, [])
        self.assertEqual(module.call_count, 1)

    def test_disabled_provider_failure_and_empty_search_use_fallback_without_fake_success(self):
        for provider, allow_search in [
            (FakeInspirationSearchProvider([paper()]), False),
            (
                FakeInspirationSearchProvider(
                    error=RetrievalTransportError("arXiv unavailable")
                ),
                True,
            ),
            (FakeInspirationSearchProvider([]), True),
        ]:
            context = context_with_solution()
            module = InspirationRetrievalModule(
                FakeLLMClient([draft(selected=1), fallback_result()]), provider
            )
            result = module.retrieve(
                context=context, allow_external_search=allow_search
            )
            self.assertEqual(result.status, "found")
            self.assertEqual(result.inspirations[0].grounding, "none")
            self.assertIsNone(result.inspirations[0].paper_ref)
            self.assertEqual(module.last_grounding_status, "none")
            self.assertEqual(module.call_count, 2)
            if allow_search and provider.papers == [] and provider.error is None:
                self.assertEqual(
                    len(context.get_memory().paper_memory.inspiration_search_records), 1
                )
            else:
                self.assertEqual(
                    len(context.get_memory().paper_memory.inspiration_search_records), 0
                )

    def test_grounded_result_rejects_unknown_paper_unselected_entity_and_duplicates(self):
        bad_results = []
        bad_results.append(grounded_result("arxiv:unknown"))
        unselected = grounded_result()
        unselected["inspirations"][0]["bridge_entity"] = entity(5)
        bad_results.append(unselected)
        duplicate = grounded_result()
        duplicate["inspirations"].append(grounded_result()["inspirations"][0])
        bad_results.append(duplicate)
        for bad in bad_results:
            context = context_with_solution()
            module = InspirationRetrievalModule(
                FakeLLMClient([draft(selected=1), bad]),
                FakeInspirationSearchProvider([paper()]),
            )
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                module.retrieve(context=context)
            self.assertEqual(context.get_memory().paper_memory.papers, [])
            self.assertEqual(context.get_memory().inspiration_memory, [])


class FakeTextTransport:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get_text(self, url: str, headers: Mapping[str, str], timeout: float) -> str:
        self.calls.append((url, headers, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v2</id>
    <title> First paper </title>
    <summary> A useful abstract. </summary>
    <published>2025-01-02T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title> Duplicate version </title>
    <summary> Duplicate. </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2345.6789</id>
    <title> Second paper </title>
    <published>2024-02-03T00:00:00Z</published>
  </entry>
</feed>
"""


class M6ArxivProviderTests(unittest.TestCase):
    def test_arxiv_normalizes_versions_deduplicates_and_preserves_missing_abstract(self):
        transport = FakeTextTransport([ATOM])
        provider = ArxivInspirationSearchProvider(
            ArxivConfig(
                base_url="https://export.example/api/query",
                max_retries=0,
                min_request_interval_seconds=0,
            ),
            transport,
        )
        papers = provider.search("Transfer Mechanism 1 OR Transfer Mechanism 2")
        self.assertEqual([paper.paper_id for paper in papers], [
            "arxiv:1234.5678", "arxiv:2345.6789"
        ])
        self.assertEqual(papers[0].abstract, "A useful abstract.")
        self.assertIsNone(papers[1].abstract)
        self.assertEqual(provider.last_request_count, 1)
        self.assertIn("max_results=20", transport.calls[0][0])
        self.assertIn("all%3A%22Transfer+Mechanism+1%22", transport.calls[0][0])
        self.assertIn("OR", transport.calls[0][0])

    def test_arxiv_retry_counts_one_search_attempt_and_no_real_sleep_required(self):
        transport = FakeTextTransport(
            [RetrievalTransportError("temporary", retryable=True), ATOM]
        )
        sleeps: list[float] = []
        provider = ArxivInspirationSearchProvider(
            ArxivConfig(max_retries=1, min_request_interval_seconds=0),
            transport,
            sleep=sleeps.append,
        )
        provider.search("Transfer Mechanism")
        self.assertEqual(provider.last_request_count, 2)
        self.assertEqual(provider.last_retry_count, 1)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(provider.total_request_count, 2)

    def test_arxiv_global_cache_reuses_response_without_provider_request(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ArxivConfig(
                base_url="https://export.example/api/query",
                max_retries=0,
                min_request_interval_seconds=0,
                cache_dir=str(Path(directory) / "arxiv"),
            )
            first_transport = FakeTextTransport([ATOM])
            first = ArxivInspirationSearchProvider(config, first_transport)
            first_papers = first.search("Transfer Mechanism")
            self.assertEqual(first.last_request_count, 1)
            self.assertFalse(first.last_cache_hit)

            second_transport = FakeTextTransport(
                [AssertionError("arXiv API was called despite cache")]
            )
            second = ArxivInspirationSearchProvider(config, second_transport)
            second_papers = second.search("Transfer Mechanism")
            self.assertEqual(second_papers, first_papers)
            self.assertEqual(second.last_request_count, 0)
            self.assertTrue(second.last_cache_hit)
            self.assertEqual(second.last_cache_scope, "global")
            self.assertEqual(second_transport.calls, [])

    def test_arxiv_canonical_id_strips_version_only(self):
        self.assertEqual(canonical_arxiv_id("http://arxiv.org/abs/cs/010101v3"), "cs/010101")
        self.assertEqual(canonical_arxiv_id("arxiv:1234.5678v1"), "1234.5678")


if __name__ == "__main__":
    unittest.main()
