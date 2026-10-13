import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from research_agent.errors import (
    ModuleContractError,
    PreconditionError,
    RetrievalConfigurationError,
    RetrievalHTTPError,
    RetrievalTransportError,
    SeedPaperResolutionError,
    StructuredOutputError,
)
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
    PaperRecord,
    ResearchState,
    SeedPaperContext,
    SolutionVerificationResult,
    VerificationFocus,
)
from research_agent.models.verification import SolutionVerificationDraft
from research_agent.modules.related_work_verification import (
    RelatedWorkVerificationModule,
)
from research_agent.retrieval.fake import FakeRelatedPaperProvider
from research_agent.retrieval.semantic_scholar import (
    SemanticScholarConfig,
    SemanticScholarProvider,
)
from research_agent.retrieval.semantic_scholar_cache import SemanticScholarCache
from research_agent.runtime.state_store import ResearchStateStore
from research_agent.trajectory.context import TrajectoryContext


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="local-seed",
        title="Adaptive Representation for Structured Signals",
        abstract="A paper abstract.",
        research_objective="Study an unresolved method question.",
        core_method="A structured mechanism with two cooperating components.",
        main_components=["Component A", "Component B"],
        main_findings=["The method improves a reported task."],
        limitations=[],
        experiment_context=ExperimentContext(
            tasks_or_datasets=["Task-A"],
            metrics=["accuracy"],
            major_baselines=["Baseline-A"],
            experimental_setting="Standard split.",
        ),
    )


def gap() -> CandidateGap:
    return CandidateGap(
        id="G1",
        statement="The method leaves a mechanism-level question unresolved.",
        confidence="medium",
        confidence_reason="The supplied method evidence supports the question.",
        priority="high",
        priority_reason="It affects the central mechanism.",
        status="active",
        evidence_refs=["seed:core_method"],
    )


def paper(identifier: str = "p1", title: str = "A related method") -> PaperRecord:
    return PaperRecord(
        paper_id=identifier,
        title=title,
        abstract="An abstract describing a related mechanism.",
        year=2025,
        venue="Venue",
        source="fixture",
    )


def solution(status_one: str = "ready") -> CurrentSolution:
    return CurrentSolution(
        id="S1",
        core_idea="A new core mechanism.",
        rationale="It addresses the active gap.",
        components=[
            Component(id="C1", description="First mechanism.", status=status_one),
            Component(id="C2", description="Second mechanism.", status="ready"),
        ],
        status="developed",
        novelty_status="unchecked",
    )


def gap_result(status: str = "valid", refs: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "summary": "Within the retrieved related-paper evidence, the gap remains open.",
        "covered_aspects": [],
        "remaining_gap": None,
        "related_papers": refs or [],
        "evidence_level": "abstract_only",
    }
    if status == "partially_resolved":
        payload["covered_aspects"] = ["A narrower aspect is covered."]
        payload["remaining_gap"] = "The mechanism-level residual question remains."
    return payload


def comparison(
    paper_ref: str = "p1",
    *,
    novelty: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "paper_ref": paper_ref,
        "problem_relation": "related",
        "mechanism_relation": "different" if not novelty else "partial",
        "shared_elements": ["A standard element."],
        "novelty_relevant_overlaps": novelty or [],
        "key_difference": "The main mechanism differs.",
    }


def solution_draft(result: dict[str, Any], comparisons: list[dict[str, Any]] | None = None):
    return {"comparisons": comparisons or [], "result": result}


class FakeGetTransport:
    def __init__(self, responses: list[Any] | None = None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get_json(self, url: str, headers: Mapping[str, str], timeout: float):
        self.calls.append((url, headers, timeout))
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise RetrievalTransportError("no fixture response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class M5ProviderTests(unittest.TestCase):
    def test_empty_api_key_uses_environment_fallback(self):
        with patch.dict(
            os.environ,
            {"SEMANTIC_SCHOLAR_API_KEY": "environment-secret"},
            clear=False,
        ):
            config = SemanticScholarConfig.from_mapping(
                {
                    "api_key": "",
                    "base_url": "https://example.test",
                }
            )
        self.assertEqual(config.api_key, "environment-secret")
        self.assertTrue(config.public_dict()["api_key_configured"])
        self.assertNotIn("environment-secret", str(config.public_dict()))

    def test_nonempty_file_api_key_takes_precedence_over_environment(self):
        config = SemanticScholarConfig.from_mapping(
            {
                "api_key": "file-secret",
                "base_url": "https://example.test",
            },
            environment={"SEMANTIC_SCHOLAR_API_KEY": "environment-secret"},
        )
        self.assertEqual(config.api_key, "file-secret")

    def test_semantic_scholar_config_is_loaded_from_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_scholar.json"
            path.write_text(
                json.dumps(
                    {
                        "api_key": "file-secret",
                        "base_url": "https://example.test/",
                        "timeout_seconds": 12,
                        "max_retries": 2,
                        "min_request_interval_seconds": 1.5,
                        "cache_dir": "cache/test",
                        "auth_mode": "bearer",
                    }
                ),
                encoding="utf-8",
            )
            config = SemanticScholarConfig.from_file(path)
            self.assertEqual(config.api_key, "file-secret")
            self.assertEqual(config.base_url, "https://example.test")
            self.assertEqual(config.timeout_seconds, 12.0)
            self.assertEqual(config.max_retries, 2)
            self.assertEqual(config.min_request_interval_seconds, 1.5)
            self.assertEqual(config.cache_dir, "cache/test")
            self.assertEqual(config.auth_mode, "bearer")
            self.assertNotIn("file-secret", str(config.public_dict()))

    def test_global_cache_reuses_both_successful_api_responses_across_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "semantic_scholar"
            config = SemanticScholarConfig(
                base_url="https://example.test",
                cache_dir=str(cache_dir),
                max_retries=0,
                min_request_interval_seconds=0.0,
            )
            first_transport = FakeGetTransport(
                [
                    {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                    {"recommendedPapers": [{"paperId": "p1", "title": "Related"}]},
                ]
            )
            first = SemanticScholarProvider(config, first_transport)
            first_pool = first.get_related_paper_pool(seed())
            self.assertEqual(len(first_transport.calls), 2)
            self.assertEqual(sorted(first.last_global_cache_operations), [])
            self.assertTrue(list(cache_dir.rglob("*.json")))

            second_transport = FakeGetTransport(error=AssertionError("API was called"))
            second = SemanticScholarProvider(config, second_transport)
            second_pool = second.get_related_paper_pool(seed())
            self.assertEqual(second_pool, first_pool)
            self.assertEqual(second.last_request_count, 0)
            self.assertTrue(second.last_global_cache_hit)
            self.assertEqual(
                set(second.last_global_cache_operations),
                {"paper_search", "recommendations"},
            )
            self.assertEqual(second_transport.calls, [])

    def test_global_cache_does_not_persist_failed_provider_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SemanticScholarCache(Path(directory) / "semantic_scholar")
            provider = SemanticScholarProvider(
                SemanticScholarConfig(
                    base_url="https://example.test",
                    cache_dir=str(cache.root),
                    max_retries=0,
                    min_request_interval_seconds=0.0,
                ),
                FakeGetTransport(error=RetrievalTransportError("timeout")),
            )
            with self.assertRaises(RetrievalTransportError):
                provider.get_related_paper_pool(seed())
            self.assertEqual(list(cache.root.rglob("*.json")), [])

    def test_semantic_scholar_config_rejects_unknown_fields_and_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            unknown_path = Path(directory) / "unknown.json"
            unknown_path.write_text(
                json.dumps({"base_url": "https://example.test", "unexpected": True}),
                encoding="utf-8",
            )
            with self.assertRaises(RetrievalConfigurationError):
                SemanticScholarConfig.from_file(unknown_path)

            malformed_path = Path(directory) / "malformed.json"
            malformed_path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(RetrievalConfigurationError):
                SemanticScholarConfig.from_file(malformed_path)

    def test_semantic_scholar_normalizes_exact_title_top20_and_deduplicates(self):
        search = {
            "data": [
                {
                    "paperId": "seed-s2",
                    "title": "Adaptive Representation for Structured Signals",
                    "abstract": "Seed.",
                    "year": 2025,
                    "venue": "Venue",
                    "authors": [{"name": "Author"}],
                }
            ]
        }
        recommended = {
            "recommendedPapers": [
                {
                    "paperId": "p1",
                    "title": "First",
                    "abstract": "A.",
                    "authors": [],
                },
                {"paperId": "p1", "title": "First", "abstract": "A."},
                {"paperId": "p2", "title": "Second", "abstract": None},
            ]
        }
        transport = FakeGetTransport([search, recommended])
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            transport,
        )
        pool = provider.get_related_paper_pool(seed(), limit=20)
        self.assertEqual(provider.last_resolved_provider_id, "seed-s2")
        self.assertEqual(provider.last_seed_resolution_audit["candidate_count"], 1)
        self.assertEqual(provider.last_seed_resolution_audit["selection_reason"], "exact_unique")
        self.assertEqual([item.paper_id for item in pool.papers], ["p1", "p2"])
        self.assertIsNone(pool.papers[1].abstract)
        self.assertEqual(provider.last_request_count, 2)

    def test_api_key_header_and_request_pacing_use_injected_clock(self):
        responses = [
            {"data": [{"paperId": "seed-s2", "title": seed().title}]},
            {"recommendedPapers": []},
        ]
        transport = FakeGetTransport(responses)
        clock = FakeClock()
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                api_key="secret-value",
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=2.0,
            ),
            transport,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        provider.get_related_paper_pool(seed())
        self.assertEqual(transport.calls[0][1]["x-api-key"], "secret-value")
        self.assertEqual(clock.sleeps, [2.0])
        self.assertEqual(provider.last_pacing_waits, [2.0])
        self.assertNotIn("secret-value", str(provider.config.public_dict()))

    def test_bearer_auth_uses_authorization_header(self):
        transport = FakeGetTransport(
            [
                {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                {"recommendedPapers": []},
            ]
        )
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                api_key="proxy-token",
                base_url="https://s2api.ominiai.cn/s2",
                auth_mode="bearer",
                max_retries=0,
                min_request_interval_seconds=0.0,
                cache_dir=None,
            ),
            transport,
        )
        provider.get_related_paper_pool(seed())
        self.assertEqual(
            transport.calls[0][1]["Authorization"], "Bearer proxy-token"
        )
        self.assertNotIn("x-api-key", transport.calls[0][1])
        self.assertIn(
            "/s2/graph/v1/paper/search", transport.calls[0][0]
        )

    def test_429_retry_after_is_respected_without_real_sleep(self):
        transport = FakeGetTransport(
            [
                RetrievalHTTPError(429, "rate limited", retry_after_seconds=3.0),
                {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                {"recommendedPapers": []},
            ]
        )
        clock = FakeClock()
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=1,
                min_request_interval_seconds=0.0,
            ),
            transport,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        provider.get_related_paper_pool(seed())
        self.assertEqual(provider.last_request_count, 3)
        self.assertEqual(provider.last_retry_count, 1)
        self.assertEqual(provider.last_rate_limit_count, 1)
        self.assertEqual(provider.last_retry_waits, [3.0])
        self.assertEqual(clock.sleeps, [3.0])

    def test_429_without_retry_after_uses_bounded_backoff(self):
        transport = FakeGetTransport(
            [
                RetrievalHTTPError(429, "rate limited"),
                {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                {"recommendedPapers": []},
            ]
        )
        clock = FakeClock()
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=1,
                min_request_interval_seconds=0.0,
            ),
            transport,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        provider.get_related_paper_pool(seed())
        self.assertEqual(clock.sleeps, [0.5])
        self.assertEqual(provider.last_retry_waits, [0.5])

    def test_5xx_and_timeout_retry_but_401_403_404_do_not(self):
        for transient in (
            RetrievalHTTPError(500, "server error"),
            RetrievalTransportError("timeout", retryable=True),
        ):
            transport = FakeGetTransport(
                [
                    transient,
                    {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                    {"recommendedPapers": []},
                ]
            )
            provider = SemanticScholarProvider(
                SemanticScholarConfig(
                    base_url="https://example.test",
                    max_retries=1,
                    min_request_interval_seconds=0.0,
                ),
                transport,
                clock=lambda: 0.0,
                sleep=lambda _seconds: None,
            )
            provider.get_related_paper_pool(seed())
            self.assertEqual(provider.last_retry_count, 1)
        for status in (400, 401, 403, 404):
            transport = FakeGetTransport([RetrievalHTTPError(status, "permanent")])
            provider = SemanticScholarProvider(
                SemanticScholarConfig(
                    base_url="https://example.test",
                    max_retries=3,
                    min_request_interval_seconds=0.0,
                ),
                transport,
                clock=lambda: 0.0,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(RetrievalHTTPError):
                provider.get_related_paper_pool(seed())
            self.assertEqual(provider.last_request_count, 1)
            self.assertEqual(provider.last_retry_count, 0)

    def test_seed_resolution_prefers_formal_version_and_records_selection(self):
        ambiguous = FakeGetTransport(
            [
                {
                    "data": [
                        {
                            "paperId": "arxiv-record",
                            "title": seed().title,
                            "source": "arxiv",
                            "abstract": "preprint",
                        },
                        {
                            "paperId": "conference-record",
                            "title": seed().title,
                            "venue": "Conference",
                            "publicationTypes": ["Conference"],
                            "abstract": "published",
                        },
                    ]
                }
            ]
        )
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            ambiguous,
        )
        self.assertEqual(provider.resolve_seed_paper_id(seed()), "conference-record")
        self.assertEqual(
            provider.last_seed_resolution_audit["selection_reason"],
            "conference_preferred",
        )
        self.assertEqual(provider.last_seed_resolution_audit["candidate_count"], 2)

        journal = FakeGetTransport(
            [
                {
                    "data": [
                        {
                            "paperId": "conference-record",
                            "title": seed().title,
                            "publicationTypes": ["Conference"],
                        },
                        {
                            "paperId": "journal-record",
                            "title": seed().title,
                            "publicationTypes": ["JournalArticle"],
                        },
                    ]
                }
            ]
        )
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            journal,
        )
        self.assertEqual(provider.resolve_seed_paper_id(seed()), "journal-record")

        same_level = FakeGetTransport(
            [
                {
                    "data": [
                        {"paperId": "z", "title": seed().title, "venue": "Conference"},
                        {"paperId": "a", "title": seed().title, "venue": "Conference"},
                    ]
                }
            ]
        )
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            same_level,
        )
        self.assertEqual(provider.resolve_seed_paper_id(seed()), "a")
        self.assertEqual(provider.last_seed_resolution_audit["selection_reason"], "stable_tiebreak")

        abstract_preference = FakeGetTransport(
            [
                {
                    "data": [
                        {"paperId": "without-abstract", "title": seed().title, "venue": "Conference"},
                        {
                            "paperId": "with-abstract",
                            "title": seed().title,
                            "venue": "Conference",
                            "abstract": "available",
                        },
                    ]
                }
            ]
        )
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            abstract_preference,
        )
        self.assertEqual(provider.resolve_seed_paper_id(seed()), "with-abstract")
        self.assertEqual(provider.last_seed_resolution_audit["selection_reason"], "has_abstract")

        no_match = FakeGetTransport([{"data": [{"paperId": "a", "title": "Unrelated"}]}])
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            no_match,
        )
        with self.assertRaises(SeedPaperResolutionError):
            provider.resolve_seed_paper_id(seed())

    def test_provider_transport_failure_is_not_silent(self):
        provider = SemanticScholarProvider(
            SemanticScholarConfig(
                base_url="https://example.test",
                max_retries=0,
                min_request_interval_seconds=0.0,
            ),
            FakeGetTransport(error=RetrievalTransportError("timeout")),
        )
        with self.assertRaises(RetrievalTransportError):
            provider.get_related_paper_pool(seed())


class M5VerificationTests(unittest.TestCase):
    def make_context(self, current_solution: CurrentSolution | None = None) -> TrajectoryContext:
        context = TrajectoryContext.initialize(seed())
        context.state_store.replace_state(
            ResearchState(
                active_gap=gap(),
                candidate_solutions=(
                    [CandidateSolution(id="S1", core_idea="Idea", rationale="Reason", status="selected")]
                    if current_solution is not None
                    else []
                ),
                current_solution=current_solution,
                experiments=(
                    [
                        Experiment(
                            id="E1",
                            objective="Test",
                            setup="Setup",
                            what_it_validates="Claim",
                            validation_criterion="Criterion",
                            status="ready",
                        )
                    ]
                    if current_solution is not None
                    else []
                ),
            )
        )
        return context

    def test_gap_verification_saves_result_and_reuses_trajectory_local_cache(self):
        provider = FakeRelatedPaperProvider([paper()])
        llm = FakeLLMClient([gap_result("valid", ["p1"]), gap_result("resolved", ["p1"])])
        context = self.make_context()
        module = RelatedWorkVerificationModule(llm, provider)
        first = module.verify_gap(gap(), context=context, step=1)
        second = module.verify_gap(gap(), context=context, step=2)
        self.assertEqual(first.status, "valid")
        self.assertEqual(second.status, "resolved")
        self.assertEqual(provider.calls, ["local-seed"])
        self.assertEqual(module.scientific_search_count, 1)
        self.assertEqual(len(context.get_memory().verification_memory), 2)
        self.assertIsNotNone(context.memory_store.get_latest_gap_verification("G1"))
        event_types = [event.event_type for event in context.trajectory.get_events()]
        self.assertIn("related_pool_acquired", event_types)
        self.assertIn("related_pool_cache_hit", event_types)
        self.assertIn("gap_verified", event_types)
        self.assertEqual(context.get_state().active_gap.statement, gap().statement)

    def test_gap_verification_canonicalizes_unique_exact_title_to_paper_id(self):
        context = self.make_context()
        related = paper("P123", "A Generic Scientific Method")
        module = RelatedWorkVerificationModule(
            FakeLLMClient([gap_result("valid", [related.title])]),
            FakeRelatedPaperProvider([related]),
        )

        result = module.verify_gap(gap(), context=context, step=1)

        self.assertEqual(result.related_papers, ["P123"])
        self.assertEqual(context.get_memory().verification_memory[0].paper_refs, ["P123"])

    def test_canonical_paper_id_is_preserved(self):
        context = self.make_context()
        related = paper("P123", "A Generic Scientific Method")
        module = RelatedWorkVerificationModule(
            FakeLLMClient([gap_result("valid", ["P123"])]),
            FakeRelatedPaperProvider([related]),
        )

        result = module.verify_gap(gap(), context=context, step=1)

        self.assertEqual(result.related_papers, ["P123"])

    def test_unknown_title_is_not_silently_accepted(self):
        context = self.make_context()
        related = paper("P123", "A Generic Scientific Method")
        module = RelatedWorkVerificationModule(
            FakeLLMClient([gap_result("valid", ["Unknown Scientific Paper"])]),
            FakeRelatedPaperProvider([related]),
        )

        with self.assertRaises(ModuleContractError):
            module.verify_gap(gap(), context=context, step=1)

    def test_ambiguous_exact_title_is_not_guessed(self):
        context = self.make_context()
        first = paper("P123", "A Generic Scientific Method")
        second = paper("P456", "A Generic Scientific Method")
        module = RelatedWorkVerificationModule(
            FakeLLMClient([gap_result("valid", [first.title])]),
            FakeRelatedPaperProvider([first, second]),
        )

        with self.assertRaises(ModuleContractError):
            module.verify_gap(gap(), context=context, step=1)

    def test_solution_verification_canonicalizes_all_paper_reference_fields(self):
        context = self.make_context(solution())
        related = paper("P123", "A Generic Scientific Method")
        title = related.title
        result = {
            "novelty_status": "partial_overlap",
            "summary": "A local mechanism overlaps.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C1",
                    "overlap_summary": "A local overlap.",
                    "matched_paper_ref": title,
                }
            ],
            "distinctive_elements": ["A remaining distinction."],
            "key_differences": ["The mechanism differs."],
            "related_papers": [title],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([solution_draft(result, [comparison(title, novelty=["overlap"])])]),
            FakeRelatedPaperProvider([related]),
        )

        verified = module.verify_solution(
            gap(), context=context, verification_focus=VerificationFocus(mode="full_solution"), step=1
        )

        self.assertEqual(verified.related_papers, ["P123"])
        self.assertEqual(verified.novelty_relevant_overlaps[0].matched_paper_ref, "P123")
        stored_event = [
            event for event in context.trajectory.get_events()
            if event.event_type == "solution_verified"
        ][-1]
        self.assertEqual(stored_event.result_payload["comparisons"][0]["paper_ref"], "P123")

    def test_global_cache_avoids_provider_calls_in_a_new_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SemanticScholarConfig(
                base_url="https://example.test",
                cache_dir=str(Path(directory) / "semantic_scholar"),
                max_retries=0,
                min_request_interval_seconds=0.0,
            )
            first_provider = SemanticScholarProvider(
                config,
                FakeGetTransport(
                    [
                        {"data": [{"paperId": "seed-s2", "title": seed().title}]},
                        {"recommendedPapers": [{"paperId": "p1", "title": "Related"}]},
                    ]
                ),
            )
            first_context = self.make_context()
            first_module = RelatedWorkVerificationModule(
                FakeLLMClient([]), first_provider
            )
            first_module.get_related_paper_pool(
                seed(), context=first_context, step=1
            )
            self.assertEqual(first_module.scientific_search_count, 1)
            self.assertEqual(first_provider.last_request_count, 2)

            second_provider = SemanticScholarProvider(
                config,
                FakeGetTransport(error=AssertionError("global cache missed")),
            )
            second_context = self.make_context()
            second_module = RelatedWorkVerificationModule(
                FakeLLMClient([]), second_provider
            )
            second_module.get_related_paper_pool(
                seed(), context=second_context, step=1
            )
            self.assertEqual(second_module.scientific_search_count, 0)
            self.assertTrue(second_module.last_cache_hit)
            self.assertEqual(second_module.last_cache_scope, "global")
            self.assertEqual(second_provider.last_request_count, 0)

    def test_failed_pool_acquisition_consumes_one_scientific_search(self):
        class FailingProvider:
            provider_name = "fake-failing"
            last_request_count = 2
            last_retry_count = 1
            last_rate_limit_count = 1
            last_pacing_waits = [0.5]
            last_resolved_provider_id = None

            def get_related_paper_pool(self, _seed_context, *, limit=20):
                raise RetrievalHTTPError(429, "rate limited")

        context = self.make_context()
        module = RelatedWorkVerificationModule(FakeLLMClient([]), FailingProvider())
        with self.assertRaises(RetrievalHTTPError):
            module.verify_gap(gap(), context=context, step=1)
        self.assertEqual(module.scientific_search_count, 1)
        self.assertEqual(module.last_provider_request_count, 2)
        self.assertEqual(module.last_provider_retry_count, 1)
        self.assertIsNone(context.get_memory().paper_memory.related_paper_pool)
        self.assertEqual(len(context.get_memory().verification_memory), 0)

    def test_solution_verification_distinct_partial_and_core_transitions(self):
        distinct_result = {
            "novelty_status": "distinct",
            "summary": "The core mechanism remains distinct.",
            "shared_elements": ["A standard technique."],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": ["A different mechanism."],
            "key_differences": ["The mechanism differs."],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        distinct = solution_draft(distinct_result, [comparison()])
        partial_result = {
            **distinct_result,
            "novelty_status": "partial_overlap",
            "summary": "A local element overlaps.",
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C1",
                    "overlap_summary": "A local novelty-bearing element is covered.",
                    "matched_paper_ref": "p1",
                }
            ],
        }
        partial = solution_draft(partial_result, [comparison(novelty=["A local overlap."])])
        core_result = {
            **distinct_result,
            "novelty_status": "core_overlap",
            "summary": "The core mechanism is covered.",
            "novelty_relevant_overlaps": [
                {
                    "component_id": None,
                    "overlap_summary": "The supplied paper covers the core mechanism.",
                    "matched_paper_ref": "p1",
                }
            ],
        }
        core = solution_draft(core_result, [comparison(novelty=["The core mechanism."])])

        context = self.make_context(solution())
        module = RelatedWorkVerificationModule(
            FakeLLMClient([distinct, partial, core]),
            FakeRelatedPaperProvider([paper()]),
        )
        result = module.verify_solution(
            gap(), context=context, verification_focus=VerificationFocus(mode="full_solution"), step=1
        )
        self.assertEqual(result.novelty_status, "distinct")
        self.assertEqual(context.get_state().current_solution.novelty_status, "distinct")
        self.assertEqual(context.get_memory().failure_memory, [])
        module.verify_solution(
            gap(), context=context, verification_focus=VerificationFocus(mode="full_solution"), step=2
        )
        self.assertEqual(context.get_state().current_solution.novelty_status, "partial_overlap")
        self.assertEqual(context.get_memory().failure_memory, [])
        module.verify_solution(
            gap(), context=context, verification_focus=VerificationFocus(mode="full_solution"), step=3
        )
        self.assertIsNone(context.get_state().current_solution)
        self.assertEqual(context.get_state().candidate_solutions[0].status, "rejected")
        self.assertEqual(context.get_state().experiments, [])
        self.assertEqual(module.call_count, 3)
        self.assertEqual(len(context.get_memory().verification_memory), 3)
        self.assertEqual(len(context.get_memory().failure_memory), 1)
        failure = context.get_memory().failure_memory[0]
        self.assertEqual(failure.type, "rejected_solution")
        self.assertEqual(failure.object_ref, "S1")
        self.assertEqual(failure.reason, "The core mechanism is covered.")
        self.assertEqual(failure.evidence_refs, ["p1"])
        solution_event = [
            event
            for event in context.trajectory.get_events()
            if event.event_type == "solution_verified"
        ][-1]
        self.assertEqual(len(solution_event.result_payload["comparisons"]), 1)
        self.assertNotIn("comparisons", context.get_state().model_dump())

    def test_verify_solution_injects_canonical_draft_schema_and_output_type(self):
        result = {
            "novelty_status": "distinct",
            "summary": "The core mechanism remains distinct.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": ["A different mechanism."],
            "key_differences": ["The mechanism differs."],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        captured: dict[str, object] = {}

        def handler(request, output_schema):
            captured["output_schema"] = output_schema
            captured["system_prompt"] = request.messages[0].content
            return solution_draft(result, [comparison()])

        llm = FakeLLMClient(handler=handler)
        context = self.make_context(solution())
        module = RelatedWorkVerificationModule(
            llm,
            FakeRelatedPaperProvider([paper()]),
        )
        module.verify_solution(gap(), context=context, step=1)

        canonical_schema = json.dumps(
            SolutionVerificationDraft.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = captured["system_prompt"]
        self.assertIs(captured["output_schema"], SolutionVerificationDraft)
        self.assertIn(canonical_schema, prompt)
        self.assertIn('"comparisons"', prompt)
        self.assertIn('"result"', prompt)
        self.assertIn('"ComparisonRecord"', prompt)
        self.assertIn('"SolutionVerificationResult"', prompt)
        self.assertIn("copy the exact `paper_ref` identifier", prompt)
        self.assertNotIn("SOLUTION_VERIFICATION_RESULT_JSON_SCHEMA", prompt)

    def test_verify_solution_rejects_mismatched_explicit_solution_before_mutation(self):
        context = self.make_context(solution())
        mismatched_solution = solution().model_copy(update={"id": "S2"})
        llm = FakeLLMClient(
            [
                solution_draft(
                    {
                        "novelty_status": "distinct",
                        "summary": "Should never be used.",
                        "shared_elements": [],
                        "novelty_relevant_overlaps": [],
                        "distinctive_elements": [],
                        "key_differences": [],
                        "related_papers": [],
                        "evidence_level": "abstract_only",
                    }
                )
            ]
        )
        provider = FakeRelatedPaperProvider([paper()])
        module = RelatedWorkVerificationModule(llm, provider)
        before_state = context.get_state().model_dump(mode="python")
        before_memory = context.get_memory().model_dump(mode="python")
        before_event_count = len(context.trajectory.get_events())

        with self.assertRaises(PreconditionError):
            module.verify_solution(
                gap(),
                current_solution=mismatched_solution,
                context=context,
                step=1,
            )

        self.assertEqual(module.call_count, 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(provider.calls, [])
        self.assertEqual(context.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(context.get_memory().model_dump(mode="python"), before_memory)
        self.assertEqual(len(context.trajectory.get_events()), before_event_count)
        self.assertEqual(context.get_memory().verification_memory, [])
        self.assertEqual(context.get_memory().failure_memory, [])
        self.assertNotIn(
            "solution_verified",
            [event.event_type for event in context.trajectory.get_events()],
        )

    def test_state_store_only_verify_solution_rejects_mismatched_solution_before_io(self):
        state_store = ResearchStateStore(
            ResearchState(
                active_gap=gap(),
                candidate_solutions=[
                    CandidateSolution(
                        id="S1", core_idea="Idea", rationale="Reason", status="selected"
                    )
                ],
                current_solution=solution(),
                experiments=[],
            )
        )
        memory_store = ResearchMemoryStore(seed().paper_id)
        mismatched = solution().model_copy(update={"id": "S2"})
        llm = FakeLLMClient([])
        provider = FakeRelatedPaperProvider([paper()])
        module = RelatedWorkVerificationModule(llm, provider)
        before_state = state_store.get_state().model_dump(mode="python")
        before_memory = memory_store.get_memory().model_dump(mode="python")

        with self.assertRaises(PreconditionError):
            module.verify_solution(
                gap(),
                current_solution=mismatched,
                seed_context=seed(),
                state_store=state_store,
                memory_store=memory_store,
            )

        self.assertEqual(module.call_count, 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(provider.calls, [])
        self.assertEqual(state_store.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(memory_store.get_memory().model_dump(mode="python"), before_memory)

    def test_state_store_only_verify_gap_rejects_mismatched_gap_before_io(self):
        state_store = ResearchStateStore(ResearchState(active_gap=gap()))
        memory_store = ResearchMemoryStore(seed().paper_id)
        mismatched = gap().model_copy(update={"id": "G2"})
        llm = FakeLLMClient([])
        provider = FakeRelatedPaperProvider([paper()])
        module = RelatedWorkVerificationModule(llm, provider)
        before_state = state_store.get_state().model_dump(mode="python")
        before_memory = memory_store.get_memory().model_dump(mode="python")

        with self.assertRaises(PreconditionError):
            module.verify_gap(
                mismatched,
                seed_context=seed(),
                state_store=state_store,
                memory_store=memory_store,
            )

        self.assertEqual(module.call_count, 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(provider.calls, [])
        self.assertEqual(state_store.get_state().model_dump(mode="python"), before_state)
        self.assertEqual(memory_store.get_memory().model_dump(mode="python"), before_memory)

    def test_focused_distinct_marks_only_changed_modified_component_ready(self):
        current = solution(status_one="modified").model_copy(deep=True)
        current.components[1].status = "ready"
        context = self.make_context(current)
        result = {
            "novelty_status": "distinct",
            "summary": "The changed unit remains distinct.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": ["Distinct mechanism."],
            "key_differences": ["Different."],
            "related_papers": [],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([solution_draft(result)]), FakeRelatedPaperProvider([paper()])
        )
        module.verify_solution(
            gap(),
            context=context,
            verification_focus=VerificationFocus(mode="changed_units", changed_units=["C1"]),
            step=1,
        )
        statuses = {component.id: component.status for component in context.get_state().current_solution.components}
        self.assertEqual(statuses, {"C1": "ready", "C2": "ready"})

    def test_full_distinct_marks_stale_components_ready_after_core_revision(self):
        current = solution(status_one="modified").model_copy(deep=True)
        current.components[1].status = "stale"
        context = self.make_context(current)
        result = {
            "novelty_status": "distinct",
            "summary": "The revised full solution remains distinct.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": ["A revised distinctive mechanism."],
            "key_differences": ["The revised mechanism differs."],
            "related_papers": [],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([solution_draft(result)]), FakeRelatedPaperProvider([paper()])
        )
        module.verify_solution(
            gap(),
            context=context,
            verification_focus=VerificationFocus(mode="full_solution"),
            step=1,
        )
        statuses = {
            component.id: component.status
            for component in context.get_state().current_solution.components
        }
        self.assertEqual(statuses, {"C1": "ready", "C2": "ready"})

    def test_full_partial_overlap_marks_all_pending_components_ready(self):
        current = solution(status_one="modified").model_copy(deep=True)
        current.components[1].status = "stale"
        context = self.make_context(current)
        result = {
            "novelty_status": "partial_overlap",
            "summary": "A local overlap remains after full re-verification.",
            "shared_elements": ["A shared generic element."],
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C1",
                    "overlap_summary": "The local mechanism overlaps.",
                    "matched_paper_ref": "p1",
                }
            ],
            "distinctive_elements": ["The remaining mechanism differs."],
            "key_differences": ["The revised mechanism differs."],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([solution_draft(result, [comparison(novelty=["overlap"])])]),
            FakeRelatedPaperProvider([paper()]),
        )
        outcome = module.verify_solution(
            gap(),
            context=context,
            verification_focus=VerificationFocus(mode="full_solution"),
            step=1,
        )
        self.assertEqual(outcome.novelty_status, "partial_overlap")
        statuses = {
            component.id: component.status
            for component in context.get_state().current_solution.components
        }
        self.assertEqual(statuses, {"C1": "ready", "C2": "ready"})

    def test_changed_units_partial_overlap_settles_exact_pending_components(self):
        current = solution(status_one="ready").model_copy(deep=True)
        current.components[1].status = "modified"
        context = self.make_context(current)
        result = {
            "novelty_status": "partial_overlap",
            "summary": "The focused component remains partially overlapped.",
            "shared_elements": ["A shared generic element."],
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C2",
                    "overlap_summary": "C2 has a local overlap.",
                    "matched_paper_ref": "p1",
                }
            ],
            "distinctive_elements": ["The other mechanism differs."],
            "key_differences": ["The focused mechanism differs."],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([solution_draft(result, [comparison(novelty=["overlap"])])]),
            FakeRelatedPaperProvider([paper()]),
        )
        module.verify_solution(
            gap(),
            context=context,
            verification_focus=VerificationFocus(mode="changed_units", changed_units=["C2"]),
            step=1,
        )
        statuses = {
            component.id: component.status
            for component in context.get_state().current_solution.components
        }
        self.assertEqual(statuses, {"C1": "ready", "C2": "ready"})

    def test_changed_units_cannot_leave_pending_or_stale_components_unsettled(self):
        cases = []
        pending = solution(status_one="modified").model_copy(deep=True)
        pending.components[1].status = "modified"
        cases.append((pending, ["C2"]))
        stale = solution(status_one="ready").model_copy(deep=True)
        stale.components[1].status = "stale"
        cases.append((stale, ["C2"]))
        result = {
            "novelty_status": "partial_overlap",
            "summary": "A focused overlap remains.",
            "shared_elements": ["A shared element."],
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C2",
                    "overlap_summary": "C2 overlaps.",
                    "matched_paper_ref": "p1",
                }
            ],
            "distinctive_elements": ["A remaining distinction."],
            "key_differences": ["A focused difference."],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        for current, focus_units in cases:
            context = self.make_context(current)
            module = RelatedWorkVerificationModule(
                FakeLLMClient([solution_draft(result, [comparison(novelty=["overlap"])])]),
                FakeRelatedPaperProvider([paper()]),
            )
            with self.assertRaises(ModuleContractError):
                module.verify_solution(
                    gap(),
                    context=context,
                    verification_focus=VerificationFocus(
                        mode="changed_units", changed_units=focus_units
                    ),
                    step=1,
                )

    def test_semantic_validation_rejects_unknown_refs_and_components(self):
        context = self.make_context(solution())
        unknown_ref = {
            "status": "valid",
            "summary": "Summary.",
            "covered_aspects": [],
            "remaining_gap": None,
            "related_papers": ["not-in-memory"],
            "evidence_level": "abstract_only",
        }
        module = RelatedWorkVerificationModule(
            FakeLLMClient([unknown_ref]), FakeRelatedPaperProvider([paper()])
        )
        with self.assertRaises(ModuleContractError):
            module.verify_gap(gap(), context=context, step=1)
        bad_component_result = {
            "novelty_status": "partial_overlap",
            "summary": "Summary.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C99",
                    "overlap_summary": "Overlap.",
                    "matched_paper_ref": "p1",
                }
            ],
            "distinctive_elements": [],
            "key_differences": [],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        bad_component = solution_draft(
            bad_component_result, [comparison()]
        )
        module = RelatedWorkVerificationModule(
            FakeLLMClient([bad_component]), FakeRelatedPaperProvider([paper()])
        )
        with self.assertRaises(ModuleContractError):
            module.verify_solution(gap(), context=context, step=1)

    def test_comparison_contract_rejects_duplicates_limits_mismatch_and_missing_match(self):
        def run(result, comparisons, papers=None):
            context = self.make_context(solution())
            provider = FakeRelatedPaperProvider(papers or [paper(), paper("p2", "Another")])
            module = RelatedWorkVerificationModule(
                FakeLLMClient([solution_draft(result, comparisons)]), provider
            )
            with self.assertRaises((ModuleContractError, StructuredOutputError)):
                module.verify_solution(gap(), context=context, step=1)

        distinct_one = {
            "novelty_status": "distinct",
            "summary": "Distinct.",
            "shared_elements": [],
            "novelty_relevant_overlaps": [],
            "distinctive_elements": [],
            "key_differences": [],
            "related_papers": ["p1"],
            "evidence_level": "abstract_only",
        }
        run(distinct_one, [comparison(), comparison()])
        run(
            {**distinct_one, "related_papers": ["p9"]},
            [comparison("p9")],
        )
        run(
            distinct_one,
            [comparison(f"p{i}") for i in range(6)],
            papers=[paper(f"p{i}", f"Paper {i}") for i in range(6)],
        )
        partial = {
            **distinct_one,
            "novelty_status": "partial_overlap",
            "novelty_relevant_overlaps": [
                {
                    "component_id": "C1",
                    "overlap_summary": "Local overlap.",
                    "matched_paper_ref": "p2",
                }
            ],
        }
        run(partial, [comparison()])
        run({**distinct_one, "related_papers": []}, [comparison()])


if __name__ == "__main__":
    unittest.main()
