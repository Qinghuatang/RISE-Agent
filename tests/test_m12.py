"""Generic offline contracts for the M12 pilot and M11 trigger boundary."""

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.errors import StructuredOutputError
from research_agent.llm.client import LLMCallMetadata, LLMRequest, StructuredLLMResponse
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.fake import FakeLLMClient
from research_agent.llm.recording import RecordingLLMClient, aggregate_token_usage
from research_agent.models import (
    BridgeEntity,
    ExperimentContext,
    InspirationCandidate,
    InspirationNeed,
    InspirationRecord,
    InspirationResult,
    PaperRecord,
    SeedPaperContext,
    TargetedRevisionDraft,
)
from research_agent.models.pilot import PilotManifest
from research_agent.models.trajectory import TrajectoryEvent
from research_agent.runner.pilot import (
    PilotConfigurationError,
    RealPilotConfig,
    RealSinglePaperPilot,
    _AuditedArxivProvider,
    _AuditedSemanticScholarProvider,
)
from research_agent.runner.trajectory_runner import TrajectoryRunner
from research_agent.trajectory.context import TrajectoryContext


def _seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="generic-m12-seed",
        title="Generic supplied seed",
        abstract="A supplied abstract.",
        research_objective="A supplied objective.",
        core_method="A supplied method.",
        main_components=["Component A", "Component B"],
        main_findings=["A supplied finding."],
        limitations=["A supplied limitation."],
        experiment_context=ExperimentContext(),
    )


def _inspiration_record(record_id: str = "I1") -> InspirationRecord:
    entity = BridgeEntity(
        name="Generic bridge",
        type="mechanism",
        bridge_reason="A generic bridge.",
    )
    return InspirationRecord(
        id=record_id,
        inspiration_need=InspirationNeed(
            current_problem="A problem",
            required_function="A function",
            current_mechanism="A mechanism",
            relevant_context="A context",
        ),
        proposed_entities=[entity],
        selected_entities=[entity],
        search_query="generic query",
        grounding_status="arxiv",
        inspirations=[
            InspirationCandidate(
                bridge_entity=entity,
                paper_ref="arxiv:generic",
                source_problem="A source problem",
                source_mechanism="A source mechanism",
                transferable_principle="A principle",
                adaptation_hint="An adaptation",
                grounding="arxiv",
            )
        ],
        paper_refs=["arxiv:generic"],
        created_at_step=1,
    )


def _append_inspiration(context: TrajectoryContext, result_id: str = "I1") -> None:
    context.memory_store.save_inspiration(_inspiration_record(result_id))
    context.trajectory.append_event(
        1,
        "inspiration_retrieved",
        action="SEEK_INSPIRATION",
        result_ref=result_id,
        result_payload={"result": {"status": "found", "inspirations": [{}]}},
    )


class _TokenLLM:
    def __init__(self, *, fail: bool = False, secret: str = ""):
        self.config = DeepSeekConfig(
            api_key=secret or "configured",
            base_url="https://example.invalid",
            model="test-model",
        )
        self.fail = fail

    def complete_structured(self, request, output_schema):
        if self.fail:
            raise RuntimeError(f"request failed {self.config.api_key}")
        output = (
            InspirationResult(status="no_useful_inspiration", inspirations=[])
            if output_schema is InspirationResult
            else output_schema.model_validate({})
        )
        return StructuredLLMResponse(
            output=output,
            metadata=LLMCallMetadata(
                provider="fake",
                model="test-model",
                request_id="r1",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )


class M12RecordingTests(unittest.TestCase):
    def test_format_retry_recovers_invalid_json_once(self):
        from research_agent.llm.client import LLMMessage

        recorder = RecordingLLMClient(
            FakeLLMClient([
                "not a JSON object",
                {"status": "no_useful_inspiration", "inspirations": []},
            ])
        )
        response = recorder.complete_structured(
            LLMRequest(messages=(LLMMessage(role="user", content="x"),)),
            InspirationResult,
        )
        self.assertEqual(response.output.status, "no_useful_inspiration")
        self.assertEqual(len(recorder.records), 2)
        self.assertFalse(recorder.records[0].success)
        self.assertTrue(recorder.records[0].raw_response_excerpt)
        self.assertEqual(recorder.records[1].format_retry_count, 1)
        self.assertEqual(len(recorder.inner.calls), 2)

    def test_format_retry_does_not_repeat_after_second_failure(self):
        from research_agent.llm.client import LLMMessage

        recorder = RecordingLLMClient(FakeLLMClient(["sk-secret-token", "still bad"]))
        with self.assertRaises(StructuredOutputError):
            recorder.complete_structured(
                LLMRequest(messages=(LLMMessage(role="user", content="x"),)),
                InspirationResult,
            )
        self.assertEqual(len(recorder.records), 2)
        self.assertEqual(recorder.records[1].format_retry_count, 1)
        self.assertNotIn("sk-secret-token", recorder.records[0].raw_response_excerpt or "")

    def test_scientific_model_contract_failure_is_not_format_retried(self):
        from research_agent.llm.client import LLMMessage

        payload = {
            "selected_inspiration_ref": None,
            "result": {
                "status": "revised",
                "revised_core_idea": None,
                "revised_rationale": None,
                "revised_components": [],
                "changed_units": [],
                "revision_summary": "No patch.",
                "failure_reason": None,
            },
        }
        recorder = RecordingLLMClient(FakeLLMClient([payload, payload]))
        with self.assertRaises(StructuredOutputError):
            recorder.complete_structured(
                LLMRequest(messages=(LLMMessage(role="user", content="x"),)),
                TargetedRevisionDraft,
            )
        self.assertEqual(len(recorder.records), 1)
        self.assertFalse(recorder.records[0].format_retry_count)
    def test_recording_success_and_complete_usage(self):
        recorder = RecordingLLMClient(_TokenLLM())
        # LLMRequest validates its messages; construct a valid request without
        # relying on any scientific prompt.
        from research_agent.llm.client import LLMMessage

        request = LLMRequest(
            messages=(LLMMessage(role="user", content="generic input"),),
            metadata={"task": "generic_task", "paper_id": "generic-paper"},
            max_tokens=10,
        )
        recorder.complete_structured(request, InspirationResult)
        self.assertEqual(len(recorder.records), 1)
        self.assertEqual(recorder.records[0].task, "generic_task")
        self.assertTrue(recorder.records[0].success)
        self.assertEqual(aggregate_token_usage(recorder.records)["total_tokens"], 5)

    def test_recording_failure_redacts_secret_and_reraises(self):
        recorder = RecordingLLMClient(_TokenLLM(fail=True, secret="secret-token"))
        from research_agent.llm.client import LLMMessage

        request = LLMRequest(messages=(LLMMessage(role="user", content="x"),))
        with self.assertRaises(RuntimeError):
            recorder.complete_structured(request, InspirationResult)
        self.assertFalse(recorder.records[0].success)
        self.assertNotIn("secret-token", recorder.records[0].error_message or "")

    def test_structured_validation_failure_is_saved_with_diagnostics_and_payload(self):
        malformed = {
            "selected_inspiration_ref": None,
            "result": {
                "status": "revised",
                "revised_components": [
                    {"component_id": "C1", "description": "Changed."}
                ],
                "changed_units": "C1",
                "revision_summary": "Changed one unit.",
                "failure_reason": None,
            },
        }
        recorder = RecordingLLMClient(FakeLLMClient([malformed, malformed]))
        from research_agent.llm.client import LLMMessage

        with self.assertRaises(StructuredOutputError) as raised:
            recorder.complete_structured(
                LLMRequest(messages=(LLMMessage(role="user", content="x"),)),
                TargetedRevisionDraft,
            )

        error = raised.exception
        self.assertEqual(error.schema_name, "TargetedRevisionDraft")
        self.assertIn("result.changed_units", str(error))
        self.assertTrue(any(
            item["field_path"] == "result.changed_units"
            for item in error.validation_errors
        ))
        self.assertEqual(len(recorder.records), 2)
        self.assertEqual(recorder.records[0].parsed_payload, malformed)
        self.assertEqual(recorder.records[0].validation_errors, error.validation_errors)
        self.assertEqual(recorder.records[1].format_retry_count, 1)
        self.assertTrue(recorder.records[1].retry_reason)

    def test_missing_token_metadata_is_unavailable_not_zero(self):
        recorder = RecordingLLMClient(FakeLLMClient([{"status": "no_useful_inspiration", "inspirations": []}]))
        from research_agent.llm.client import LLMMessage

        recorder.complete_structured(
            LLMRequest(messages=(LLMMessage(role="user", content="x"),)),
            InspirationResult,
        )
        usage = aggregate_token_usage(recorder.records)
        self.assertFalse(usage["token_metadata_complete"])
        self.assertIsNone(usage["total_tokens"])
        self.assertEqual(usage["known_total_tokens"], 0)
        self.assertEqual(usage["calls_with_token_usage"], 0)
        self.assertEqual(usage["calls_missing_token_usage"], 1)

    def test_unbounded_recording_removes_max_tokens_before_provider_call(self):
        class InspectingLLM(_TokenLLM):
            def __init__(self):
                super().__init__()
                self.forwarded_max_tokens = "unset"

            def complete_structured(self, request, output_schema):
                self.forwarded_max_tokens = request.max_tokens
                return super().complete_structured(request, output_schema)

        inner = InspectingLLM()
        recorder = RecordingLLMClient(inner, unbounded_output=True)
        from research_agent.llm.client import LLMMessage

        recorder.complete_structured(
            LLMRequest(
                messages=(LLMMessage(role="user", content="x"),),
                max_tokens=4000,
            ),
            InspirationResult,
        )
        self.assertIsNone(inner.forwarded_max_tokens)
        self.assertIsNone(recorder.records[0].max_tokens)


class M12PilotContractTests(unittest.TestCase):
    def _config(self, markdown: Path, output: Path, semantic: Path, *, allow: bool = True):
        return RealPilotConfig(
            markdown_path=markdown,
            output_dir=output,
            semantic_scholar_config=semantic,
            allow_real_services=allow,
        )

    def test_real_pilot_requires_opt_in_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "paper.md"
            markdown.write_text("# A paper\n\n## Abstract\nA.\n", encoding="utf-8")
            semantic = root / "semantic.json"
            semantic.write_text('{"base_url":"https://example.invalid"}', encoding="utf-8")
            pilot = RealSinglePaperPilot(self._config(markdown, root / "out", semantic, allow=False))
            with self.assertRaises(PilotConfigurationError):
                pilot.run()
            self.assertIsNone(pilot._llm)

    def test_missing_deepseek_config_makes_no_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "paper.md"
            markdown.write_text("# A paper\n\n## Abstract\nA.\n", encoding="utf-8")
            semantic = root / "semantic.json"
            semantic.write_text('{"base_url":"https://example.invalid"}', encoding="utf-8")
            pilot = RealSinglePaperPilot(self._config(markdown, root / "out", semantic))
            import os

            old = {name: os.environ.pop(name, None) for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")}
            try:
                with self.assertRaises(Exception):
                    pilot.run()
            finally:
                for name, value in old.items():
                    if value is not None:
                        os.environ[name] = value
            self.assertIsNone(pilot._llm)

    def test_pilot_manifest_is_closed_and_serializable(self):
        manifest = PilotManifest(
            pilot_name="generic",
            markdown_path="paper.md",
            status="stopped",
            real_services_enabled=True,
            max_steps=12,
            max_searches=2,
            llm_request_count=1,
            token_metadata_complete=False,
            semantic_scholar_http_requests=0,
            arxiv_http_requests=0,
            scientific_searches_used=0,
            trajectory_event_count=2,
            replay_record_count=1,
        )
        self.assertEqual(manifest.model_validate(manifest.model_dump()), manifest)
        with self.assertRaises(Exception):
            PilotManifest.model_validate({**manifest.model_dump(), "api_key": "secret"})

    def test_inspiration_trigger_is_single_use_direct_handoff(self):
        context = TrajectoryContext.initialize(_seed())
        _append_inspiration(context)
        self.assertEqual(TrajectoryRunner._latest_inspiration_trigger(context), "I1")
        context.trajectory.append_event(2, "solution_revised", action="TARGETED_REVISION")
        self.assertIsNone(TrajectoryRunner._latest_inspiration_trigger(context))
        context.trajectory.append_event(3, "verification_recorded", action="VERIFY_RELATED_WORK")
        self.assertIsNone(TrajectoryRunner._latest_inspiration_trigger(context))

    def test_no_useful_inspiration_and_new_record_rules(self):
        context = TrajectoryContext.initialize(_seed())
        context.trajectory.append_event(
            1,
            "inspiration_retrieved",
            action="SEEK_INSPIRATION",
            result_ref="I1",
            result_payload={"result": {"status": "no_useful_inspiration"}},
        )
        self.assertIsNone(TrajectoryRunner._latest_inspiration_trigger(context))
        context.memory_store.save_inspiration(_inspiration_record("I2"))
        context.trajectory.append_event(
            2,
            "inspiration_retrieved",
            action="SEEK_INSPIRATION",
            result_ref="I2",
            result_payload={"result": {"status": "found"}},
        )
        self.assertEqual(TrajectoryRunner._latest_inspiration_trigger(context), "I2")

    def test_arxiv_not_selected_has_zero_requests(self):
        class FakeArxiv:
            total_request_count = 0
            last_request_count = 0
            last_retry_count = 0
            last_rate_limit_count = 0
            provider_name = "arxiv"

            class Config:
                base_url = "https://example.invalid"

            config = Config()

        audited = _AuditedArxivProvider(FakeArxiv())
        self.assertEqual(audited.audit()["http_request_count"], 0)
        self.assertEqual(audited.audit()["query_count"], 0)

    def test_semantic_scholar_cache_metadata_is_safe(self):
        class FakeSemantic:
            provider_name = "semantic_scholar"
            total_request_count = 0
            last_global_cache_operations = ["paper_search", "recommendations"]
            last_global_cache_hit = True
            last_resolved_provider_id = "provider-id"
            last_retry_count = 0
            last_rate_limit_count = 0

            class Config:
                base_url = "https://example.invalid"
                auth_mode = "bearer"
                api_key = "secret"

            config = Config()

        audit = _AuditedSemanticScholarProvider(FakeSemantic()).audit()
        self.assertEqual(audit["auth_mode"], "bearer")
        self.assertTrue(audit["api_key_configured"])
        self.assertNotIn("secret", json.dumps(audit))


if __name__ == "__main__":
    unittest.main()
