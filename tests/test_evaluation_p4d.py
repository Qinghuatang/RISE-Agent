"""Deterministic engineering report tests over the existing RISE data models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from research_agent.controller.budget import BudgetTracker
from research_agent.observability.instrumentation import instrument_trajectory_modules
from research_agent.observability.logging import bind_log_context, current_log_context
from research_agent.observability.telemetry import Telemetry
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.service.evaluation import (
    build_evaluation_report,
    read_structured_metric_logs,
    render_markdown_report,
)

from test_m11 import modules, reading, seed


def _projection_events(state):
    events = []
    for index, event in enumerate(state.trajectory_events, start=1):
        events.append(
            SimpleNamespace(
                event_type="rise_trajectory_event",
                action=event.action,
                payload={
                    "trajectory_index": index,
                    "trajectory_event_type": event.event_type,
                    "step": event.step,
                    "result_ref": event.result_ref,
                    "state_fields": sorted(event.state_diff),
                    "memory_writes": list(event.memory_writes),
                    "budget": event.budget.model_dump(mode="json") if event.budget else None,
                    "error": event.error,
                },
            )
        )
    event_types = {
        "bootstrap": "rise_bootstrap_confirmed",
        "action": "rise_action_confirmed",
        "internal": "rise_internal_step_confirmed",
    }
    for record in state.replay_records:
        events.append(
            SimpleNamespace(
                event_type=event_types[record.kind],
                action=record.action,
                payload={
                    "replay_sequence": record.sequence,
                    "kind": record.kind,
                    "step": record.step,
                    "budget": {
                        "max_steps": record.budget_after.max_steps,
                        "max_searches": record.budget_after.max_searches,
                        "steps_used": record.budget_after.steps_used,
                        "searches_used": record.budget_after.searches_used,
                    },
                },
            )
        )
    return [
        SimpleNamespace(seq=index, event_type=event.event_type, action=event.action, payload=event.payload)
        for index, event in enumerate(events, start=1)
    ]


def _deterministic_report_sample():
    task_id = UUID("5b40289e-0064-4d44-84d1-0b110d445d8a")
    task_id_text = str(task_id)
    rows: list[dict] = []

    def collect(event: str, **fields) -> None:
        rows.append(
            {
                **current_log_context(),
                **fields,
                "event": event,
                "event_id": f"fixture-{len(rows) + 1}",
            }
        )

    telemetry = Telemetry(collect)
    runtime_modules = modules()
    instrument_trajectory_modules(runtime_modules, telemetry)
    runner = LangGraphTrajectoryRunner(
        runtime_modules,
        budget=BudgetTracker(max_steps=5, max_searches=1),
        run_id=task_id_text,
        telemetry=telemetry,
    )
    with bind_log_context(task_id=task_id_text, thread_id=task_id_text):
        state = runner.run(seed(), full_paper_reading_context=reading())

    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    task = SimpleNamespace(
        id=task_id,
        task_status="completed",
        rise_outcome=state.rise_status,
        created_at=created_at,
        started_at=created_at + timedelta(seconds=12),
        finished_at=created_at + timedelta(seconds=72),
        attempt_count=1,
        recovery_count=0,
        failure_kind=None,
        error_type=None,
        last_event_seq=len(_projection_events(state)),
    )
    # Timing is deliberately unknown in this fixed FakeLLM sample. Real run
    # durations are reported from task timestamps and supplied JSON logs.
    metric_records = []
    for row in rows:
        if row["event"] in {"llm.call", "retrieval.call"}:
            metric_records.append({**row, "duration_ms": None})

    report = build_evaluation_report(
        [task],
        events_by_task={task_id_text: _projection_events(state)},
        states_by_task={task_id_text: state},
        metric_records=metric_records,
    )
    return report, state, metric_records


def test_deterministic_fake_run_produces_repeatable_safe_evaluation_report():
    report, state, _metrics = _deterministic_report_sample()
    repeated, repeated_state, _ = _deterministic_report_sample()

    assert report == repeated
    assert state.rise_status == repeated_state.rise_status == "committed"
    assert report["task_status"]["counts"]["completed"] == 1
    assert report["rise_outcomes"]["counts"]["committed"] == 1
    assert report["rise_outcomes"]["proposal"]["success_rate"] == 1.0
    assert report["actions_and_budget"]["persisted_action_attempt_count"] == state.budget.steps_used
    assert report["actions_and_budget"]["steps_used_total"] == state.budget.steps_used
    assert report["actions_and_budget"]["searches_used_total"] == state.budget.searches_used
    assert report["model_and_retrieval_calls"]["llm"]["observed_call_count"] > 0
    assert report["model_and_retrieval_calls"]["retrieval"]["observed_call_count"] > 0
    assert report["model_and_retrieval_calls"]["llm"]["token_usage"]["unknown_call_count"] > 0
    assert report["model_and_retrieval_calls"]["llm"]["token_usage"]["total_tokens_known_subtotal"] is None
    assert report["checkpoint_event_consistency"]["consistent"] is True
    assert report["duration"]["queue"]["mean_seconds"] == 12.0
    assert report["duration"]["execution"]["mean_seconds"] == 60.0

    encoded = json.dumps(report, ensure_ascii=False)
    markdown = render_markdown_report(report)
    assert "supplied abstract" not in encoded.lower()
    assert "supplied abstract" not in markdown.lower()
    assert "PRIVATE PROMPT" not in encoded
    assert "PRIVATE PROMPT" not in markdown
    assert "unknown" in markdown.lower()


def test_checkpoint_event_report_flags_missing_projection_and_budget_mismatch():
    report, state, metric_records = _deterministic_report_sample()
    task_id = str(UUID("5b40289e-0064-4d44-84d1-0b110d445d8a"))
    task = SimpleNamespace(
        id=UUID(task_id),
        task_status="completed",
        rise_outcome="committed",
        created_at=None,
        started_at=None,
        finished_at=None,
        attempt_count=1,
        recovery_count=0,
        failure_kind=None,
        error_type=None,
        last_event_seq=len(_projection_events(state)),
    )
    damaged_events = _projection_events(state)[:-1]
    damaged_trajectory = next(
        event for event in damaged_events if event.event_type == "rise_trajectory_event"
    )
    damaged_trajectory.payload["memory_writes"].append("corrupted")
    damaged_replay = next(
        event for event in damaged_events if event.event_type == "rise_bootstrap_confirmed"
    )
    damaged_replay.payload["budget"]["max_steps"] += 1
    # Rebuild a valid budget to exercise report-side mismatch checks without
    # changing the persisted state model's own validation rules.
    from research_agent.controller.budget import ControllerBudget

    damaged_state = state.evolve(
        budget=ControllerBudget(
            max_steps=state.budget.max_steps,
            max_searches=state.budget.max_searches,
            steps_used=max(0, state.budget.steps_used - 1),
            searches_used=state.budget.searches_used,
        )
    )
    corrupted = build_evaluation_report(
        [task],
        events_by_task={task_id: damaged_events},
        states_by_task={task_id: damaged_state},
        metric_records=metric_records,
    )
    assert corrupted["checkpoint_event_consistency"]["consistent"] is False
    assert corrupted["checkpoint_event_consistency"]["mismatch_count"] >= 4
    assert corrupted["checkpoint_event_consistency"]["replay_projection"]["mismatched"] == 1
    assert corrupted["checkpoint_event_consistency"]["trajectory_projection"]["mismatched"] == 1


def test_report_separates_scientific_infrastructure_and_cancelled_task_results():
    committed_report, committed_state, metric_records = _deterministic_report_sample()
    del committed_report
    committed_id = UUID("5b40289e-0064-4d44-84d1-0b110d445d8a")
    scientific_id = UUID("06a3dbe2-05e7-4ffa-8b21-512304c0f806")
    infrastructure_id = UUID("bd2f4a6c-205b-4583-a982-10763a45bcea")
    cancelled_id = UUID("b608e379-d84c-4b68-a914-67c567d9ca03")
    scientific_state = committed_state.evolve(
        run_id=str(scientific_id),
        rise_status="failed",
        final_proposal=None,
        error_type="ScientificActionError",
    )
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def task(task_id, status, *, outcome=None, failure_kind=None, error_type=None, recovery=0, started=True):
        return SimpleNamespace(
            id=task_id,
            task_status=status,
            rise_outcome=outcome,
            created_at=base_time,
            started_at=base_time if started else None,
            finished_at=base_time + timedelta(seconds=5) if status in {"completed", "failed", "cancelled"} else None,
            attempt_count=recovery + 1,
            recovery_count=recovery,
            failure_kind=failure_kind,
            error_type=error_type,
            last_event_seq=0,
        )

    committed_task = task(committed_id, "completed", outcome="committed")
    committed_events = _projection_events(committed_state)
    committed_task.last_event_seq = len(committed_events)
    scientific_task = task(
        scientific_id,
        "completed",
        outcome="failed",
        failure_kind="rise",
        error_type="ScientificActionError",
        recovery=1,
    )
    scientific_events = _projection_events(scientific_state)
    scientific_task.last_event_seq = len(scientific_events)
    infrastructure_task = task(
        infrastructure_id,
        "failed",
        failure_kind="infrastructure",
        error_type="DatabaseUnavailable",
        recovery=2,
        started=False,
    )
    cancelled_task = task(cancelled_id, "cancelled", started=False)

    report = build_evaluation_report(
        [committed_task, scientific_task, infrastructure_task, cancelled_task],
        events_by_task={
            str(committed_id): committed_events,
            str(scientific_id): scientific_events,
        },
        states_by_task={
            str(committed_id): committed_state,
            str(scientific_id): scientific_state,
        },
        metric_records=metric_records,
    )
    assert report["task_status"]["counts"]["completed"] == 2
    assert report["task_status"]["counts"]["failed"] == 1
    assert report["task_status"]["counts"]["cancelled"] == 1
    assert report["rise_outcomes"]["scientific_failure_count"] == 1
    assert report["rise_outcomes"]["infrastructure_failure_count"] == 1
    assert report["rise_outcomes"]["input_failure_count"] == 0
    assert report["worker_recovery"]["recovery_count_total"] == 3
    assert report["worker_recovery"]["recovery_task_outcomes"]["completed"] == 1
    assert report["worker_recovery"]["recovery_task_outcomes"]["infrastructure_failed"] == 1
    assert report["actions_and_budget"]["action_count_known_task_count"] == 2
    assert report["actions_and_budget"]["action_count_unknown_task_count"] == 2
    assert report["actions_and_budget"]["action_counts_complete"] is False


def test_log_reader_deduplicates_event_ids_and_does_not_retain_sensitive_fields(tmp_path: Path):
    log_path = tmp_path / "runtime.jsonl"
    row = {
        "event_id": "e1",
        "event": "llm.call",
        "task_id": "task-safe-id",
        "status": "completed",
        "duration_ms": 4.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "prompt": "PRIVATE PROMPT BODY",
        "model_output": "PRIVATE MODEL OUTPUT",
    }
    log_path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\nnot-json\n",
        encoding="utf-8",
    )
    records, stats = read_structured_metric_logs([log_path])
    assert len(records) == 1
    assert stats["duplicate_event_ids"] == 1
    assert stats["invalid_json_lines"] == 1
    assert "prompt" not in records[0]
    assert "model_output" not in records[0]


def test_missing_structured_logs_keep_model_call_metrics_unknown():
    report = build_evaluation_report([])
    calls = report["model_and_retrieval_calls"]
    assert calls["source_available"] is False
    assert calls["llm"] is None
    assert calls["retrieval"] is None
