"""Deterministic tests for P4-B structured telemetry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from research_agent.controller.budget import BudgetTracker
from research_agent.llm.client import (
    LLMCallMetadata,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from research_agent.errors import LLMTransportError
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.observability.instrumentation import (
    ObservedLLMClient,
    instrument_trajectory_modules,
)
from research_agent.observability.logging import (
    JSONFormatter,
    bind_log_context,
    configure_logging,
    current_log_context,
    emit_event,
)
from research_agent.observability.telemetry import Telemetry
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.retrieval.fake import FakeRelatedPaperProvider

from test_m11 import modules, reading, seed


ROOT = Path(__file__).resolve().parents[1]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _StubLLM:
    def __init__(
        self,
        *,
        token_usage: bool = False,
        error: Exception | None = None,
        transport_attempts: int | None = None,
    ) -> None:
        self.token_usage = token_usage
        self.error = error
        if transport_attempts is not None:
            self.last_transport_attempt_count = transport_attempts
        self.result = StructuredLLMResponse(
            output={"private_output": "fixture response"},
            metadata=LLMCallMetadata(
                provider="fake",
                model="fixture-model",
                prompt_tokens=11 if token_usage else None,
                completion_tokens=7 if token_usage else None,
                total_tokens=18 if token_usage else None,
                format_retry_count=1,
            ),
        )

    def complete_structured(self, request, output_schema):
        if self.error is not None:
            raise self.error
        return self.result


def _metrics_collector(rows: list[dict]) -> Telemetry:
    def collect(event: str, **fields) -> None:
        rows.append(
            {
                "event": event,
                "fields": fields,
                "context": current_log_context(),
            }
        )

    return Telemetry(collect)


def test_json_formatter_uses_allowlisted_fields_and_redacts_unrecognized_data():
    logger = logging.getLogger("rise.p4b.json-test")
    handler = _Capture()
    original_level = logger.level
    original_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        with bind_log_context(
            service="worker",
            task_id="ab55be3b-0d22-4cc8-b82c-c09f7ed8e2f0",
            thread_id="ab55be3b-0d22-4cc8-b82c-c09f7ed8e2f0",
            worker_id="worker-test-1",
            action="VERIFY_RELATED_WORK",
        ):
            emit_event(
                logger,
                logging.INFO,
                "test.observation",
                duration_ms=12.34567,
                success=True,
                prompt="PRIVATE PAPER TEXT",
                api_key="sk-test-private-token",
                error_message="PRIVATE EXCEPTION BODY",
            )
        payload = json.loads(JSONFormatter().format(handler.records[-1]))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    assert payload["event"] == "test.observation"
    assert payload["task_id"] == "ab55be3b-0d22-4cc8-b82c-c09f7ed8e2f0"
    assert payload["worker_id"] == "worker-test-1"
    assert payload["duration_ms"] == 12.346
    assert payload["success"] is True
    encoded = json.dumps(payload)
    assert "PRIVATE PAPER TEXT" not in encoded
    assert "PRIVATE EXCEPTION BODY" not in encoded
    assert "sk-test-private-token" not in encoded


def test_logging_configuration_is_idempotent_and_can_switch_format():
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    formatter_state = [(handler, handler.formatter, getattr(handler, "_rise_structured_logging_handler", None)) for handler in handlers]
    try:
        first = configure_logging(level="WARNING", json_output=True)
        count_after_first = len(root.handlers)
        second = configure_logging(level="INFO", json_output=False)
        assert first is second
        assert len(root.handlers) == count_after_first
        assert root.level == logging.INFO
        assert second.formatter.__class__.__name__ == "PlainEventFormatter"
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
        root.handlers[:] = handlers
        root.setLevel(level)
        for handler, formatter, marker in formatter_state:
            handler.setFormatter(formatter)
            if marker is None:
                if hasattr(handler, "_rise_structured_logging_handler"):
                    delattr(handler, "_rise_structured_logging_handler")
            else:
                setattr(handler, "_rise_structured_logging_handler", marker)


def test_context_isolated_between_threads_and_fresh_processes():
    def read_in_thread(task_id: str) -> str | None:
        with bind_log_context(task_id=task_id):
            return current_log_context().get("task_id")

    task_ids = ["a1", "b2", "c3", "d4"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read_in_thread, task_ids)) == task_ids
    assert current_log_context() == {}

    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from research_agent.observability.logging import current_log_context; print(json.dumps(current_log_context()))",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(child.stdout) == {}


def test_worker_heartbeat_thread_receives_task_context():
    from research_agent.service.worker import _LeaseHeartbeat

    observed: list[dict] = []

    class Repository:
        def heartbeat(self, **_kwargs):
            observed.append(current_log_context())
            return False

    task = SimpleNamespace(id=uuid4(), worker_id="worker-thread", lease_token=uuid4())
    telemetry = _metrics_collector([])
    heartbeat = _LeaseHeartbeat(
        Repository(),
        task,
        lease_seconds=1.0,
        interval=0.01,
        fail_stop_on_ownership_loss=False,
        telemetry=telemetry,
    )
    with bind_log_context(
        service="worker",
        task_id=str(task.id),
        thread_id=str(task.id),
        worker_id=task.worker_id,
    ):
        heartbeat.start()
    heartbeat._thread.join(timeout=1)
    heartbeat.stop()
    assert heartbeat.lost.is_set()
    assert observed
    assert observed[0]["task_id"] == str(task.id)
    assert observed[0]["worker_id"] == "worker-thread"


def test_llm_wrapper_records_unknown_tokens_retries_and_preserves_return_and_error():
    rows: list[dict] = []
    telemetry = _metrics_collector(rows)
    request = LLMRequest((LLMMessage(role="user", content="PRIVATE PROMPT"),))

    client = _StubLLM(token_usage=False, transport_attempts=3)
    wrapper = ObservedLLMClient(client, telemetry)
    response = wrapper.complete_structured(request, dict)
    assert response is client.result
    assert len(rows) == 1
    fields = rows[0]["fields"]
    assert fields["token_usage_known"] is False
    assert fields["prompt_tokens"] is None
    assert fields["format_retry_count"] == 1
    assert fields["provider_retry_count"] == 2
    assert rows[0]["event"] == "llm.call"
    assert "PRIVATE PROMPT" not in json.dumps(rows)
    assert "fixture response" not in json.dumps(rows)

    error = RuntimeError("sensitive exception details")
    broken = ObservedLLMClient(_StubLLM(error=error), telemetry)
    try:
        broken.complete_structured(request, dict)
    except RuntimeError as caught:
        assert caught is error
    else:
        raise AssertionError("LLM exception must propagate unchanged")
    assert rows[-1]["fields"]["status"] == "error"
    assert rows[-1]["fields"]["error_type"] == "RuntimeError"
    assert rows[-1]["fields"]["provider_retry_count"] is None
    assert "sensitive exception details" not in json.dumps(rows)

    not_started = ObservedLLMClient(
        _StubLLM(error=RuntimeError("configuration rejected"), transport_attempts=0),
        telemetry,
    )
    try:
        not_started.complete_structured(request, dict)
    except RuntimeError:
        pass
    else:
        raise AssertionError("configuration error must propagate unchanged")
    assert rows[-1]["fields"]["provider_retry_count"] is None


def test_deepseek_transport_retries_are_observed_without_changing_call_result():
    class RetryingTransport:
        calls = 0

        def post_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise LLMTransportError("sensitive transport message")
            return {
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {},
            }

    rows: list[dict] = []
    telemetry = _metrics_collector(rows)
    transport = RetryingTransport()
    client = DeepSeekClient(
        DeepSeekConfig(
            api_key="private-key",
            base_url="https://example.test",
            model="fixture-model",
            max_retries=2,
        ),
        transport,
    )
    result = ObservedLLMClient(client, telemetry).complete_structured(
        LLMRequest((LLMMessage(role="user", content="PRIVATE PROMPT"),)), dict
    )
    assert result.output == {"answer": "ok"}
    assert transport.calls == 3
    assert rows[0]["fields"]["provider_retry_count"] == 2
    assert rows[0]["fields"]["token_usage_known"] is False
    assert "PRIVATE PROMPT" not in json.dumps(rows)
    assert "sensitive transport message" not in json.dumps(rows)


def test_sse_stream_records_full_lifetime_with_task_context(monkeypatch):
    import asyncio

    from research_agent.service import api

    async def direct_call(function, *args):
        return function(*args)

    monkeypatch.setattr(api, "run_in_threadpool", direct_call)

    task_id = uuid4()
    event = SimpleNamespace(
        seq=1,
        event_id=uuid4(),
        task_id=task_id,
        event_type="rise_action_confirmed",
        action="REFINE_GAP",
        payload={"safe": True},
        created_at=datetime.now(timezone.utc),
    )

    class Repository:
        def list_events(self, _task_id, *, after_seq, limit):
            assert limit == 500
            return [event] if after_seq == 0 else []

        def get_task(self, _task_id):
            return SimpleNamespace(task_status="completed")

    class Request:
        state = SimpleNamespace(request_id="request-sse-1")

        async def is_disconnected(self):
            return False

    rows: list[dict] = []
    resources = SimpleNamespace(
        repository=Repository(),
        settings=SimpleNamespace(sse_heartbeat_interval=30, sse_poll_interval=0.001),
        telemetry=_metrics_collector(rows),
    )
    output = asyncio.run(
        _collect_stream(api._event_stream(Request(), resources, task_id, after_seq=0))
    )
    assert len(output) == 1 and output[0].startswith("id: 1\n")
    finish = next(row for row in rows if row["event"] == "api.sse.stream_finished")
    assert finish["fields"]["event_count"] == 1
    assert finish["fields"]["status"] == "completed"
    assert finish["context"]["request_id"] == "request-sse-1"
    assert finish["context"]["task_id"] == str(task_id)


async def _collect_stream(stream):
    values = []
    async for value in stream:
        values.append(value)
    return values


def test_module_assembly_wraps_shared_llm_and_retrieval_only_once():
    rows: list[dict] = []
    telemetry = _metrics_collector(rows)
    shared_llm = _StubLLM()
    shared_provider = FakeRelatedPaperProvider()
    modules_bundle = SimpleNamespace(
        first=SimpleNamespace(llm_client=shared_llm, related_paper_provider=shared_provider),
        second=SimpleNamespace(llm_client=shared_llm, related_paper_provider=shared_provider),
    )
    counts = instrument_trajectory_modules(modules_bundle, telemetry)
    assert counts == {"llm_clients": 1, "retrieval_providers": 1}
    assert modules_bundle.first.llm_client is modules_bundle.second.llm_client
    assert modules_bundle.first.related_paper_provider is modules_bundle.second.related_paper_provider


def test_observed_graph_matches_baseline_and_survives_telemetry_sink_failure():
    baseline = LangGraphTrajectoryRunner(
        modules(),
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id="p4b-equivalence",
    ).run(seed(), full_paper_reading_context=reading())

    rows: list[dict] = []
    telemetry = _metrics_collector(rows)
    observed_modules = modules()
    instrument_trajectory_modules(observed_modules, telemetry)
    observed = LangGraphTrajectoryRunner(
        observed_modules,
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id="p4b-equivalence",
        telemetry=telemetry,
    ).run(seed(), full_paper_reading_context=reading())

    assert observed.rise_status == baseline.rise_status == "committed"
    assert observed.research_state == baseline.research_state
    assert observed.research_memory == baseline.research_memory
    assert [
        event.model_dump(mode="python", exclude={"timestamp"})
        for event in observed.trajectory_events
    ] == [
        event.model_dump(mode="python", exclude={"timestamp"})
        for event in baseline.trajectory_events
    ]
    assert observed.budget == baseline.budget
    assert observed.replay_records == baseline.replay_records
    assert observed.commit_snapshot == baseline.commit_snapshot
    assert observed.final_proposal == baseline.final_proposal
    assert any(row["event"] == "graph.node.finished" for row in rows)
    assert any(row["event"] == "llm.call" for row in rows)
    assert any(row["event"] == "retrieval.call" for row in rows)
    assert all(row["context"].get("thread_id") == "p4b-equivalence" for row in rows)

    failing_telemetry = Telemetry(lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")))
    failing_modules = modules()
    instrument_trajectory_modules(failing_modules, failing_telemetry)
    still_committed = LangGraphTrajectoryRunner(
        failing_modules,
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id="p4b-telemetry-failure",
        telemetry=failing_telemetry,
    ).run(seed(), full_paper_reading_context=reading())
    assert still_committed.rise_status == "committed"
    assert still_committed.budget == baseline.budget
