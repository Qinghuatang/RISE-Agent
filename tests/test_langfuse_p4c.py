"""Deterministic tests for optional Langfuse tracing and privacy boundaries."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from uuid import uuid4

from research_agent.controller.budget import BudgetTracker
from research_agent.llm.client import LLMMessage, LLMRequest
from research_agent.observability.instrumentation import (
    ObservedLLMClient,
    instrument_trajectory_modules,
)
from research_agent.observability.langfuse import LangfuseTracer, task_trace_id
from research_agent.observability.logging import current_log_context
from research_agent.observability.telemetry import Telemetry
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner

from test_m11 import modules, reading, seed

ROOT = Path(__file__).resolve().parents[1]


class _FakeObservation:
    def __init__(self, client: "_FakeLangfuse", record: dict) -> None:
        self.client = client
        self.record = record

    def update(self, **kwargs) -> None:
        if self.client.fail_update:
            raise RuntimeError("secret exporter update detail")
        self.record["updates"].append(kwargs)


class _FakeObservationContext:
    def __init__(self, client: "_FakeLangfuse", kwargs: dict) -> None:
        self.client = client
        self.kwargs = kwargs
        self.record = None

    def __enter__(self):
        if self.client.fail_start:
            raise RuntimeError("secret exporter start detail")
        parent = self.client.stack[-1] if self.client.stack else None
        trace_context = self.kwargs.get("trace_context") or {}
        record = {
            "name": self.kwargs["name"],
            "as_type": self.kwargs["as_type"],
            "kwargs": dict(self.kwargs),
            "trace_id": trace_context.get("trace_id")
            or (parent["trace_id"] if parent else uuid4().hex),
            "span_id": uuid4().hex[:16],
            "parent_span_id": parent["span_id"] if parent else None,
            "updates": [],
        }
        self.client.records.append(record)
        self.client.stack.append(record)
        self.record = record
        return _FakeObservation(self.client, record)

    def __exit__(self, _exc_type, _exc, _tb):
        if self.record is not None:
            assert self.client.stack[-1] is self.record
            self.client.stack.pop()
        if self.client.fail_end:
            raise RuntimeError("secret exporter end detail")
        return False


class _FakeLangfuse:
    def __init__(
        self,
        *,
        fail_start=False,
        fail_update=False,
        fail_end=False,
        fail_flush=False,
        fail_shutdown=False,
    ) -> None:
        self.records: list[dict] = []
        self.stack: list[dict] = []
        self.fail_start = fail_start
        self.fail_update = fail_update
        self.fail_end = fail_end
        self.fail_flush = fail_flush
        self.fail_shutdown = fail_shutdown
        self.flush_count = 0
        self.shutdown_count = 0

    def start_as_current_observation(self, **kwargs):
        return _FakeObservationContext(self, kwargs)

    def get_current_trace_id(self):
        return self.stack[-1]["trace_id"] if self.stack else None

    def flush(self):
        self.flush_count += 1
        if self.fail_flush:
            raise RuntimeError("secret exporter flush detail")

    def shutdown(self):
        self.shutdown_count += 1
        if self.fail_shutdown:
            raise RuntimeError("secret exporter shutdown detail")


def _capture_telemetry(rows: list[dict]) -> Telemetry:
    def capture(event: str, **fields) -> None:
        rows.append({"event": event, "fields": fields, "context": current_log_context()})

    return Telemetry(capture)


def _scientific_fields(state):
    return {
        "rise_status": state.rise_status,
        "research_state": state.research_state,
        "research_memory": state.research_memory,
        "trajectory_events": [
            event.model_dump(mode="python", exclude={"timestamp"})
            for event in state.trajectory_events
        ],
        "budget": state.budget,
        "replay_records": state.replay_records,
        "commit_snapshot": state.commit_snapshot,
        "final_proposal": state.final_proposal,
        "unresolved_issues": state.unresolved_issues,
    }


def test_disabled_tracing_does_not_import_sdk_or_start_observations(monkeypatch):
    monkeypatch.setenv("RISE_LANGFUSE_ENABLED", "false")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langfuse" or name.startswith("langfuse."):
            raise AssertionError("disabled tracing must not import the SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    tracer = LangfuseTracer.from_env()
    assert tracer.enabled is False
    with tracer.span("rise.disabled_test"):
        pass
    assert tracer.flush() is True
    assert tracer.shutdown() is True


def test_enabled_factory_reads_credentials_and_uses_manual_sdk_client(monkeypatch):
    calls: list[dict] = []
    client = _FakeLangfuse()

    def create_client(**kwargs):
        calls.append(kwargs)
        return client

    sdk = ModuleType("langfuse")
    sdk.Langfuse = create_client
    monkeypatch.setitem(sys.modules, "langfuse", sdk)
    monkeypatch.setenv("RISE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-fixture")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-fixture")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.test")

    tracer = LangfuseTracer.from_env()
    assert tracer.enabled is True
    assert calls == [
        {
            "public_key": "public-fixture",
            "secret_key": "secret-fixture",
            "base_url": "https://langfuse.example.test",
            "tracing_enabled": True,
            "timeout": 5,
        }
    ]
    assert tracer.shutdown() is True
    assert client.shutdown_count == 1


def test_trace_id_is_deterministic_across_processes_and_attempts_are_distinct():
    task_id = str(uuid4())
    expected = task_trace_id(task_id)
    assert expected == task_id.replace("-", "")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from research_agent.observability.langfuse import task_trace_id; print(task_trace_id(sys.argv[1]))",
            task_id,
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.stdout.strip() == expected

    client = _FakeLangfuse()
    tracer = LangfuseTracer(client)
    with tracer.span(
        "rise.worker_attempt",
        observation_type="agent",
        trace_id=expected,
        metadata={"task_id": task_id, "attempt_count": 1, "recovery_count": 0},
    ):
        with tracer.span("rise.graph.action", trace_id=expected):
            pass
    with tracer.span(
        "rise.worker_attempt",
        observation_type="agent",
        trace_id=expected,
        metadata={"task_id": task_id, "attempt_count": 2, "recovery_count": 1},
    ):
        with tracer.span("rise.graph.action", trace_id=expected):
            pass

    attempts = [record for record in client.records if record["name"] == "rise.worker_attempt"]
    actions = [record for record in client.records if record["name"] == "rise.graph.action"]
    assert len(attempts) == len(actions) == 2
    assert {record["trace_id"] for record in client.records} == {expected}
    assert attempts[0]["span_id"] != attempts[1]["span_id"]
    assert actions[0]["parent_span_id"] == attempts[0]["span_id"]
    assert actions[1]["parent_span_id"] == attempts[1]["span_id"]
    assert attempts[0]["kwargs"]["metadata"]["attemptCount"] == "1"
    assert attempts[1]["kwargs"]["metadata"]["recoveryCount"] == "1"


def test_graph_and_llm_spans_are_safe_and_do_not_change_scientific_result():
    baseline = LangGraphTrajectoryRunner(
        modules(), budget=BudgetTracker(max_steps=5, max_searches=1), run_id="p4c-trace-test"
    ).run(seed(), full_paper_reading_context=reading())

    client = _FakeLangfuse()
    tracer = LangfuseTracer(client)
    metrics: list[dict] = []
    instrumented_modules = modules()
    instrument_trajectory_modules(
        instrumented_modules,
        _capture_telemetry(metrics),
        tracer=tracer,
    )
    observed = LangGraphTrajectoryRunner(
        instrumented_modules,
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id="p4c-trace-test",
        tracer=tracer,
    ).run(seed(), full_paper_reading_context=reading())

    assert _scientific_fields(observed) == _scientific_fields(baseline)
    assert len([record for record in client.records if record["as_type"] == "generation"]) == len(
        [item for item in metrics if item["event"] == "llm.call"]
    )
    assert any(record["name"] == "rise.graph.bootstrap" for record in client.records)
    assert any(record["name"] == "rise.retrieval.call" for record in client.records)
    assert {record["trace_id"] for record in client.records} == {
        task_trace_id("p4c-trace-test")
    }

    encoded = json.dumps(client.records)
    assert "abstract" not in encoded.lower()
    assert "PRIVATE PROMPT" not in encoded
    assert "fixture response" not in encoded
    for record in client.records:
        assert "input" not in record["kwargs"]
        assert "output" not in record["kwargs"]
        for update in record["updates"]:
            assert "input" not in update
            assert "output" not in update


def test_exporter_failures_preserve_llm_exception_and_graph_result():
    client = _FakeLangfuse(fail_start=True, fail_update=True, fail_end=True)
    tracer = LangfuseTracer(client)
    metrics: list[dict] = []
    original_error = RuntimeError("secret provider failure detail")

    class BrokenLLM:
        def complete_structured(self, _request, _schema):
            raise original_error

    wrapped = ObservedLLMClient(BrokenLLM(), _capture_telemetry(metrics), tracer)
    try:
        wrapped.complete_structured(LLMRequest((LLMMessage("user", "private prompt"),)), dict)
    except RuntimeError as caught:
        assert caught is original_error
    else:
        raise AssertionError("original LLM error must propagate")
    assert any(item["event"] == "llm.call" for item in metrics)
    assert all(
        "secret provider failure detail" not in json.dumps(item)
        for item in client.records
    )

    baseline = LangGraphTrajectoryRunner(
        modules(), budget=BudgetTracker(max_steps=5, max_searches=1), run_id="p4c-export-fail"
    ).run(seed(), full_paper_reading_context=reading())
    failed_tracing_modules = modules()
    instrument_trajectory_modules(failed_tracing_modules, tracer=tracer)
    with_tracing_failure = LangGraphTrajectoryRunner(
        failed_tracing_modules,
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id="p4c-export-fail",
        tracer=tracer,
    ).run(seed(), full_paper_reading_context=reading())
    assert _scientific_fields(with_tracing_failure) == _scientific_fields(baseline)


def test_flush_and_shutdown_failures_are_isolated():
    metrics: list[dict] = []
    client = _FakeLangfuse(fail_flush=True, fail_shutdown=True)
    tracer = LangfuseTracer(client, _capture_telemetry(metrics))
    assert tracer.flush() is False
    assert tracer.shutdown() is False
    assert tracer.shutdown() is True
    assert [item["event"] for item in metrics] == [
        "langfuse.flush_failed",
        "langfuse.shutdown_failed",
    ]
    assert "secret exporter" not in json.dumps(metrics)
