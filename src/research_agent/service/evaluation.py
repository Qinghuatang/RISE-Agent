"""Read-only engineering evaluation over persisted RISE task data and logs.

Research execution state remains in LangGraph checkpoints. This module only
aggregates task metadata, checkpoint snapshots, replay/event projections, and
optional structured JSON logs into ephemeral JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence
from uuid import UUID

from .checkpoint_results import read_checkpoint_state
from .contracts import TaskPageCursor
from .db import create_business_engine, create_session_factory, database_url_from_env
from .repository import TaskRepository
from ..runtime.checkpointer import open_postgres_checkpointer


_TASK_STATUSES = (
    "queued",
    "running",
    "cancel_requested",
    "completed",
    "failed",
    "cancelled",
)
_RISE_OUTCOMES = ("committed", "stopped", "failed", "finalization_failed")
_REPLAY_EVENT_TYPES = {
    "bootstrap": "rise_bootstrap_confirmed",
    "action": "rise_action_confirmed",
    "internal": "rise_internal_step_confirmed",
}
_MODEL_CALL_EVENTS = frozenset({"llm.call", "retrieval.call"})


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _task_key(task: Any) -> str:
    return str(_value(task, "id"))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _duration_seconds(start: Any, end: Any) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    duration = (end - start).total_seconds()
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _duration_summary(values: Sequence[float], expected_count: int) -> dict[str, Any]:
    known = sorted(value for value in values if math.isfinite(value) and value >= 0)
    return {
        "known_count": len(known),
        "unknown_count": max(0, expected_count - len(known)),
        "mean_seconds": round(statistics.fmean(known), 3) if known else None,
        "median_seconds": round(statistics.median(known), 3) if known else None,
        "max_seconds": round(max(known), 3) if known else None,
    }


def _event_parts(event: Any) -> tuple[str | None, str | None, dict[str, Any]]:
    payload = _value(event, "payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    return (
        _value(event, "event_type"),
        _enum_value(_value(event, "action")),
        dict(payload),
    )


def _budget_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    fields = ("max_steps", "max_searches", "steps_used", "searches_used")
    if all(hasattr(value, field) for field in fields):
        return {field: getattr(value, field) for field in fields}
    return None


def _compare_projections(
    tasks: Sequence[Any],
    events_by_task: Mapping[str, Sequence[Any]],
    states_by_task: Mapping[str, Any],
    checkpoint_errors: Mapping[str, str],
) -> dict[str, Any]:
    counters = Counter()
    terminal_without_checkpoint = 0
    state_outcome_mismatch = 0
    terminal_checkpoint_task_mismatch = 0
    task_event_sequence_mismatch = 0
    budget_action_mismatch = 0
    budget_snapshot_mismatch = 0
    committed_without_proposal = 0
    committed_without_snapshot = 0
    finalization_without_snapshot = 0
    projection_without_checkpoint_tasks = 0
    tasks_with_state = 0
    tasks_without_state = 0
    tasks_with_read_error = 0

    for task in tasks:
        task_id = _task_key(task)
        state = states_by_task.get(task_id)
        task_events = events_by_task.get(task_id, ())
        event_sequences = [
            _value(event, "seq")
            for event in task_events
            if isinstance(_value(event, "seq"), int)
        ]
        expected_last_seq = _value(task, "last_event_seq")
        if isinstance(expected_last_seq, int) and sorted(event_sequences) != list(
            range(1, expected_last_seq + 1)
        ):
            task_event_sequence_mismatch += 1
        if state is None:
            if task_id in checkpoint_errors:
                tasks_with_read_error += 1
            else:
                tasks_without_state += 1
                if _value(task, "rise_outcome") is not None:
                    terminal_without_checkpoint += 1
            if any(
                _event_parts(event)[0] in _REPLAY_EVENT_TYPES.values()
                or _event_parts(event)[0] == "rise_trajectory_event"
                for event in task_events
            ):
                projection_without_checkpoint_tasks += 1
            continue
        tasks_with_state += 1

        rise_outcome = _value(task, "rise_outcome")
        state_outcome = _value(state, "rise_status")
        if rise_outcome is not None and state_outcome != rise_outcome:
            state_outcome_mismatch += 1
        if (
            state_outcome in _RISE_OUTCOMES
            and (
                _value(task, "task_status") != "completed"
                or rise_outcome != state_outcome
            )
        ):
            terminal_checkpoint_task_mismatch += 1
        budget = _value(state, "budget")
        replay_records = list(_value(state, "replay_records", ()))
        confirmed_action_count = sum(
            _value(record, "kind") == "action" for record in replay_records
        )
        if budget is not None and _value(budget, "steps_used") != confirmed_action_count:
            budget_action_mismatch += 1
        if budget is not None and replay_records:
            if _value(replay_records[-1], "budget_after") != budget:
                budget_snapshot_mismatch += 1
        if rise_outcome == "committed":
            if _value(state, "final_proposal") is None:
                committed_without_proposal += 1
            if _value(state, "commit_snapshot") is None:
                committed_without_snapshot += 1
        if rise_outcome == "finalization_failed" and _value(state, "commit_snapshot") is None:
            finalization_without_snapshot += 1

        replay_expected = {
            int(_value(record, "sequence")): record
            for record in _value(state, "replay_records", ())
        }
        replay_observed: dict[int, list[Any]] = {}
        for event in task_events:
            event_type, _action, payload = _event_parts(event)
            if event_type not in _REPLAY_EVENT_TYPES.values():
                continue
            sequence = payload.get("replay_sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                replay_observed.setdefault(sequence, []).append(event)

        counters["replay_expected"] += len(replay_expected)
        counters["replay_present"] += sum(
            min(1, len(replay_observed.get(sequence, ())))
            for sequence in replay_expected
        )
        counters["replay_missing"] += sum(
            sequence not in replay_observed for sequence in replay_expected
        )
        counters["replay_extra"] += sum(
            sequence not in replay_expected for sequence in replay_observed
        )
        counters["replay_duplicate"] += sum(
            max(0, len(rows) - 1) for rows in replay_observed.values()
        )
        for sequence, replay in replay_expected.items():
            matches = replay_observed.get(sequence, ())
            if not matches:
                continue
            event_type, action, payload = _event_parts(matches[0])
            if (
                event_type != _REPLAY_EVENT_TYPES.get(_value(replay, "kind"))
                or action != _enum_value(_value(replay, "action"))
                or payload.get("kind") != _value(replay, "kind")
                or payload.get("step") != _value(replay, "step")
                or payload.get("budget") != _budget_payload(_value(replay, "budget_after"))
            ):
                counters["replay_mismatch"] += 1

        trajectory_expected = {
            index: event
            for index, event in enumerate(_value(state, "trajectory_events", ()), start=1)
        }
        trajectory_observed: dict[int, list[Any]] = {}
        for event in task_events:
            event_type, _action, payload = _event_parts(event)
            if event_type != "rise_trajectory_event":
                continue
            index = payload.get("trajectory_index")
            if isinstance(index, int) and not isinstance(index, bool):
                trajectory_observed.setdefault(index, []).append(event)

        counters["trajectory_expected"] += len(trajectory_expected)
        counters["trajectory_present"] += sum(
            min(1, len(trajectory_observed.get(index, ())))
            for index in trajectory_expected
        )
        counters["trajectory_missing"] += sum(
            index not in trajectory_observed for index in trajectory_expected
        )
        counters["trajectory_extra"] += sum(
            index not in trajectory_expected for index in trajectory_observed
        )
        counters["trajectory_duplicate"] += sum(
            max(0, len(rows) - 1) for rows in trajectory_observed.values()
        )
        for index, trajectory in trajectory_expected.items():
            matches = trajectory_observed.get(index, ())
            if not matches:
                continue
            _event_type, action, payload = _event_parts(matches[0])
            if (
                action != _enum_value(_value(trajectory, "action"))
                or payload.get("trajectory_event_type")
                != _value(trajectory, "event_type")
                or payload.get("step") != _value(trajectory, "step")
                or payload.get("result_ref") != _value(trajectory, "result_ref")
                or payload.get("state_fields")
                != sorted(_value(trajectory, "state_diff", {}).keys())
                or payload.get("memory_writes")
                != list(_value(trajectory, "memory_writes", ()))
                or payload.get("budget")
                != (
                    _value(trajectory, "budget").model_dump(mode="json")
                    if _value(trajectory, "budget") is not None
                    else None
                )
                or payload.get("error") != _value(trajectory, "error")
            ):
                counters["trajectory_mismatch"] += 1

    mismatch_keys = (
        "replay_missing",
        "replay_extra",
        "replay_duplicate",
        "replay_mismatch",
        "trajectory_missing",
        "trajectory_extra",
        "trajectory_duplicate",
        "trajectory_mismatch",
    )
    mismatch_count = sum(counters[key] for key in mismatch_keys)
    checkpoint_state_mismatch_count = (
        terminal_without_checkpoint
        + state_outcome_mismatch
        + terminal_checkpoint_task_mismatch
        + task_event_sequence_mismatch
        + budget_action_mismatch
        + budget_snapshot_mismatch
        + committed_without_proposal
        + committed_without_snapshot
        + finalization_without_snapshot
        + projection_without_checkpoint_tasks
    )
    return {
        "tasks_with_checkpoint": tasks_with_state,
        "tasks_without_checkpoint": tasks_without_state,
        "checkpoint_read_error_tasks": tasks_with_read_error,
        "checkpoint_read_error_type_counts": dict(
            sorted(Counter(checkpoint_errors.values()).items())
        ),
        "checkpoint_state_consistency": {
            "terminal_tasks_without_checkpoint": terminal_without_checkpoint,
            "rise_outcome_mismatches": state_outcome_mismatch,
            "terminal_checkpoint_task_mismatches": terminal_checkpoint_task_mismatch,
            "task_event_sequence_mismatches": task_event_sequence_mismatch,
            "budget_action_count_mismatches": budget_action_mismatch,
            "budget_snapshot_mismatches": budget_snapshot_mismatch,
            "committed_without_proposal": committed_without_proposal,
            "committed_without_snapshot": committed_without_snapshot,
            "finalization_failed_without_snapshot": finalization_without_snapshot,
            "projection_without_checkpoint_tasks": projection_without_checkpoint_tasks,
            "mismatch_count": checkpoint_state_mismatch_count,
        },
        "replay_projection": {
            "expected": counters["replay_expected"],
            "present": counters["replay_present"],
            "missing": counters["replay_missing"],
            "extra": counters["replay_extra"],
            "duplicate": counters["replay_duplicate"],
            "mismatched": counters["replay_mismatch"],
        },
        "trajectory_projection": {
            "expected": counters["trajectory_expected"],
            "present": counters["trajectory_present"],
            "missing": counters["trajectory_missing"],
            "extra": counters["trajectory_extra"],
            "duplicate": counters["trajectory_duplicate"],
            "mismatched": counters["trajectory_mismatch"],
        },
        "mismatch_count": mismatch_count + checkpoint_state_mismatch_count,
        "consistent": (
            mismatch_count == 0
            and checkpoint_state_mismatch_count == 0
            and tasks_with_read_error == 0
        ),
    }


def _call_metrics(
    metric_records: Sequence[Mapping[str, Any]] | None,
    task_ids: set[str],
    *,
    log_parse: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    available = metric_records is not None
    records = list(metric_records or ())
    result: dict[str, Any] = {
        "source": "structured_json_logs",
        "source_available": available,
        "call_count_is_observed_only": True,
        "llm": None,
        "retrieval": None,
        "unattributed_call_count": None,
        "unmatched_task_call_count": None,
        "log_parse": dict(log_parse or {}) if available else None,
    }
    if not available:
        return result

    calls = [row for row in records if row.get("event") in _MODEL_CALL_EVENTS]
    associated = [row for row in calls if str(row.get("task_id", "")) in task_ids]
    result["unattributed_call_count"] = sum(not row.get("task_id") for row in calls)
    result["unmatched_task_call_count"] = sum(
        bool(row.get("task_id")) and str(row.get("task_id")) not in task_ids
        for row in calls
    )

    for event_name, key in (("llm.call", "llm"), ("retrieval.call", "retrieval")):
        rows = [row for row in associated if row.get("event") == event_name]
        statuses = Counter(
            row.get("status") if row.get("status") in {"completed", "error"} else "unknown"
            for row in rows
        )
        durations = [
            float(row["duration_ms"]) / 1000.0
            for row in rows
            if isinstance(row.get("duration_ms"), (int, float))
            and not isinstance(row.get("duration_ms"), bool)
            and math.isfinite(float(row["duration_ms"]))
            and float(row["duration_ms"]) >= 0
        ]
        metric_result: dict[str, Any] = {
            "observed_call_count": len(rows),
            "status_counts": {
                status: statuses.get(status, 0)
                for status in ("completed", "error", "unknown")
            },
            "duration": _duration_summary(durations, len(rows)),
        }
        if event_name == "llm.call":
            valid_token_rows = [
                row
                for row in rows
                if all(
                    isinstance(row.get(field), int)
                    and not isinstance(row.get(field), bool)
                    and row[field] >= 0
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            ]
            metric_result["token_usage"] = {
                "known_call_count": len(valid_token_rows),
                "unknown_call_count": len(rows) - len(valid_token_rows),
                "prompt_tokens_known_subtotal": (
                    sum(row["prompt_tokens"] for row in valid_token_rows)
                    if valid_token_rows
                    else None
                ),
                "completion_tokens_known_subtotal": (
                    sum(row["completion_tokens"] for row in valid_token_rows)
                    if valid_token_rows
                    else None
                ),
                "total_tokens_known_subtotal": (
                    sum(row["total_tokens"] for row in valid_token_rows)
                    if valid_token_rows
                    else None
                ),
                "subtotal_is_complete": len(valid_token_rows) == len(rows),
            }
        metric_result["retry_counts"] = {
            field: {
                "known_call_count": sum(
                    isinstance(row.get(field), int)
                    and not isinstance(row.get(field), bool)
                    and row[field] >= 0
                    for row in rows
                ),
                "unknown_call_count": sum(
                    not (
                        isinstance(row.get(field), int)
                        and not isinstance(row.get(field), bool)
                        and row[field] >= 0
                    )
                    for row in rows
                ),
                "known_subtotal": (
                    sum(
                        row[field]
                        for row in rows
                        if isinstance(row.get(field), int)
                        and not isinstance(row.get(field), bool)
                        and row[field] >= 0
                    )
                    if any(
                        isinstance(row.get(field), int)
                        and not isinstance(row.get(field), bool)
                        and row[field] >= 0
                        for row in rows
                    )
                    else None
                ),
            }
            for field in (
                ("provider_retry_count", "format_retry_count")
                if event_name == "llm.call"
                else ("provider_retry_count", "http_request_count")
            )
        }
        result[key] = metric_result
    return result


def build_evaluation_report(
    tasks: Sequence[Any],
    *,
    events_by_task: Mapping[str, Sequence[Any]] | None = None,
    states_by_task: Mapping[str, Any] | None = None,
    checkpoint_errors: Mapping[str, str] | None = None,
    metric_records: Sequence[Mapping[str, Any]] | None = None,
    log_parse: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate persisted execution data without creating mutable counters."""

    events_by_task = events_by_task or {}
    states_by_task = states_by_task or {}
    checkpoint_errors = checkpoint_errors or {}
    task_ids = {_task_key(task) for task in tasks}
    status_counts = Counter(_value(task, "task_status") for task in tasks)
    outcome_counts = Counter(
        _value(task, "rise_outcome") for task in tasks if _value(task, "rise_outcome")
    )
    rise_denominator = sum(outcome_counts.values())
    committed = [task for task in tasks if _value(task, "rise_outcome") == "committed"]
    proposal_count = sum(
        states_by_task.get(_task_key(task)) is not None
        and _value(states_by_task[_task_key(task)], "final_proposal") is not None
        for task in committed
    )

    queue_durations = [
        value
        for task in tasks
        if (value := _duration_seconds(_value(task, "created_at"), _value(task, "started_at"))) is not None
    ]
    execution_durations = [
        value
        for task in tasks
        if (value := _duration_seconds(_value(task, "started_at"), _value(task, "finished_at"))) is not None
    ]

    action_counts: Counter[str] = Counter()
    failed_action_counts: Counter[str] = Counter()
    budget_steps = 0
    budget_searches = 0
    budget_known_tasks = 0
    snapshot_count = 0
    proposal_checkpoint_count = 0
    for task_id, state in states_by_task.items():
        seen_sequences: set[int] = set()
        for record in _value(state, "replay_records", ()):
            sequence = _value(record, "sequence")
            if not isinstance(sequence, int) or sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            if _value(record, "kind") == "action":
                action = _enum_value(_value(record, "action"))
                if action:
                    action_counts[action] += 1
        for event in _value(state, "trajectory_events", ()):
            if _value(event, "event_type") == "runner_action_failed":
                action = _enum_value(_value(event, "action"))
                if action:
                    failed_action_counts[action] += 1
        budget = _value(state, "budget")
        if budget is not None:
            budget_known_tasks += 1
            budget_steps += int(_value(budget, "steps_used", 0))
            budget_searches += int(_value(budget, "searches_used", 0))
        snapshot_count += _value(state, "commit_snapshot") is not None
        proposal_checkpoint_count += _value(state, "final_proposal") is not None

    recovery_outcomes = Counter()
    recovery_count_total = sum(int(_value(task, "recovery_count", 0) or 0) for task in tasks)
    recovered_tasks = [task for task in tasks if int(_value(task, "recovery_count", 0) or 0) > 0]
    for task in recovered_tasks:
        status = _value(task, "task_status")
        if status == "completed":
            recovery_outcomes["completed"] += 1
        elif status == "cancelled":
            recovery_outcomes["cancelled"] += 1
        elif status == "failed" and _value(task, "failure_kind") == "infrastructure":
            recovery_outcomes["infrastructure_failed"] += 1
        elif status == "failed":
            recovery_outcomes["other_failed"] += 1
        else:
            recovery_outcomes["still_active_or_queued"] += 1

    failure_kinds = Counter(
        _value(task, "failure_kind")
        for task in tasks
        if _value(task, "failure_kind") is not None
    )
    error_types = Counter(
        _value(task, "error_type")
        for task in tasks
        if isinstance(_value(task, "error_type"), str)
    )
    scientific_failure_count = sum(
        _value(task, "task_status") == "completed"
        and _value(task, "rise_outcome") in {"failed", "finalization_failed"}
        for task in tasks
    )
    status_denominator = len(tasks)
    projection = _compare_projections(
        tasks, events_by_task, states_by_task, checkpoint_errors
    )
    action_checkpoint_count = sum(
        _task_key(task) in states_by_task for task in tasks
    )

    return {
        "schema_version": 1,
        "task_count": status_denominator,
        "task_status": {
            "denominator": status_denominator,
            "counts": {status: status_counts.get(status, 0) for status in _TASK_STATUSES},
            "proportions": {
                status: _ratio(status_counts.get(status, 0), status_denominator)
                for status in _TASK_STATUSES
            },
        },
        "rise_outcomes": {
            "denominator": rise_denominator,
            "counts": {outcome: outcome_counts.get(outcome, 0) for outcome in _RISE_OUTCOMES},
            "proportions": {
                outcome: _ratio(outcome_counts.get(outcome, 0), rise_denominator)
                for outcome in _RISE_OUTCOMES
            },
            "scientific_failure_count": scientific_failure_count,
            "infrastructure_failure_count": sum(
                _value(task, "task_status") == "failed"
                and _value(task, "failure_kind") == "infrastructure"
                for task in tasks
            ),
            "input_failure_count": sum(
                _value(task, "task_status") == "failed"
                and _value(task, "failure_kind") == "input"
                for task in tasks
            ),
            "proposal": {
                "committed_task_count": len(committed),
                "proposal_checkpoint_count": proposal_checkpoint_count,
                "proposal_found_for_committed_tasks": proposal_count,
                "success_rate": _ratio(proposal_count, len(committed)),
            },
        },
        "duration": {
            "queue": _duration_summary(queue_durations, status_denominator),
            "execution": _duration_summary(execution_durations, status_denominator),
            "queue_definition": "started_at - created_at; tasks without started_at are unknown",
            "execution_definition": "finished_at - started_at; tasks without both timestamps are unknown",
        },
        "actions_and_budget": {
            "persisted_action_attempt_count": sum(action_counts.values()),
            "action_counts": dict(sorted(action_counts.items())),
            "failed_action_counts": dict(sorted(failed_action_counts.items())),
            "action_source": "unique action records in the latest persisted Graph replay snapshot; failed actions remain counted as attempts",
            "action_count_known_task_count": action_checkpoint_count,
            "action_count_unknown_task_count": max(0, len(tasks) - action_checkpoint_count),
            "action_counts_complete": action_checkpoint_count == len(tasks),
            "budget_known_task_count": budget_known_tasks,
            "budget_unknown_task_count": max(0, len(tasks) - budget_known_tasks),
            "steps_used_total": budget_steps if budget_known_tasks else None,
            "searches_used_total": budget_searches if budget_known_tasks else None,
            "commit_snapshot_count": snapshot_count,
            "proposal_checkpoint_count": proposal_checkpoint_count,
            "budget_definition": "final persisted ControllerBudget snapshot per readable checkpoint",
        },
        "model_and_retrieval_calls": _call_metrics(
            metric_records, task_ids, log_parse=log_parse
        ),
        "worker_recovery": {
            "attempt_count_total": sum(int(_value(task, "attempt_count", 0) or 0) for task in tasks),
            "recovery_count_total": recovery_count_total,
            "recovered_task_count": len(recovered_tasks),
            "recovery_task_outcomes": {
                key: recovery_outcomes.get(key, 0)
                for key in (
                    "completed",
                    "cancelled",
                    "infrastructure_failed",
                    "other_failed",
                    "still_active_or_queued",
                )
            },
            "failure_kind_counts": dict(sorted(failure_kinds.items())),
            "error_type_counts": dict(sorted(error_types.items())),
            "definition": "Task recovery_count and final business status; completed includes any terminal RISE outcome",
        },
        "checkpoint_event_consistency": projection,
    }


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render the aggregate report without paper, prompt, or output content."""

    task_status = report["task_status"]
    rise = report["rise_outcomes"]
    actions = report["actions_and_budget"]
    calls = report["model_and_retrieval_calls"]
    recovery = report["worker_recovery"]
    consistency = report["checkpoint_event_consistency"]
    lines = [
        "# RISE Engineering Evaluation",
        "",
        f"Tasks in report: **{report['task_count']}**",
        "",
        "## Task and RISE Outcomes",
        "",
        "| Business status | Count | Proportion |",
        "|---|---:|---:|",
    ]
    for status, count in task_status["counts"].items():
        proportion = task_status["proportions"][status]
        lines.append(f"| `{status}` | {count} | {proportion if proportion is not None else 'unknown'} |")
    lines.extend(["", "RISE outcome counts:"])
    lines.extend(
        f"- `{outcome}`: {count}"
        for outcome, count in rise["counts"].items()
    )
    proposal = rise["proposal"]
    lines.extend(
        [
            "",
            f"Scientific failure outcomes: **{rise['scientific_failure_count']}**; "
            f"infrastructure failures: **{rise['infrastructure_failure_count']}**; "
            f"input failures: **{rise['input_failure_count']}**.",
            f"Proposal checkpoint success: **{proposal['proposal_found_for_committed_tasks']}/"
            f"{proposal['committed_task_count']}** committed tasks "
            f"(rate {proposal['success_rate'] if proposal['success_rate'] is not None else 'unknown'}).",
            "",
            "## Runtime and Budget",
            "",
            f"Persisted Action attempts: **{actions['persisted_action_attempt_count']}** "
            f"({sum(actions['failed_action_counts'].values())} failed).",
            f"Action checkpoint coverage: {actions['action_count_known_task_count']} known, "
            f"{actions['action_count_unknown_task_count']} unknown.",
            "Action types: "
            + (
                ", ".join(f"`{action}` {count}" for action, count in actions["action_counts"].items())
                or "unknown"
            )
            + ".",
            f"Budget usage: {actions['steps_used_total']} Action steps and "
            f"{actions['searches_used_total']} searches across "
            f"{actions['budget_known_task_count']} readable checkpoints.",
            f"Queue time: mean {report['duration']['queue']['mean_seconds'] if report['duration']['queue']['mean_seconds'] is not None else 'unknown'} s "
            f"({report['duration']['queue']['known_count']} known; "
            f"{report['duration']['queue']['unknown_count']} unknown).",
            f"Execution time: mean {report['duration']['execution']['mean_seconds'] if report['duration']['execution']['mean_seconds'] is not None else 'unknown'} s "
            f"({report['duration']['execution']['known_count']} known; "
            f"{report['duration']['execution']['unknown_count']} unknown).",
            "",
            "## Worker Recovery",
            "",
            f"Attempts: **{recovery['attempt_count_total']}**; recoveries: **{recovery['recovery_count_total']}** "
            f"across {recovery['recovered_task_count']} recovered tasks.",
            "Recovery task outcomes: "
            + ", ".join(
                f"`{name}` {count}"
                for name, count in recovery["recovery_task_outcomes"].items()
            )
            + ".",
            "Failure kinds: "
            + (
                ", ".join(f"`{name}` {count}" for name, count in recovery["failure_kind_counts"].items())
                or "none recorded"
            )
            + ".",
            "Error classes: "
            + (
                ", ".join(f"`{name}` {count}" for name, count in recovery["error_type_counts"].items())
                or "none recorded"
            )
            + ".",
            "",
            "## LLM and Retrieval Observations",
            "",
        ]
    )
    if not calls["source_available"]:
        lines.append("Structured JSON logs were not supplied; call counts and token usage are unknown.")
    else:
        for label, key in (("LLM", "llm"), ("Retrieval", "retrieval")):
            metric = calls[key]
            lines.append(
                f"- {label}: {metric['observed_call_count']} observed calls "
                f"(not guaranteed complete); mean duration "
                f"{metric['duration']['mean_seconds'] if metric['duration']['mean_seconds'] is not None else 'unknown'} s."
            )
            if key == "llm":
                usage = metric["token_usage"]
                lines.append(
                    f"  Known token subtotal: {usage['total_tokens_known_subtotal'] if usage['total_tokens_known_subtotal'] is not None else 'unknown'} "
                    f"across {usage['known_call_count']} calls; "
                    f"{usage['unknown_call_count']} calls have unknown usage."
                )
    lines.extend(
        [
            "",
            "## Checkpoint and Event Projection",
            "",
            f"Projection consistent: **{consistency['consistent']}**; "
            f"mismatches: **{consistency['mismatch_count']}**; "
            f"checkpoint read errors: **{consistency['checkpoint_read_error_tasks']}**.",
            f"Replay projection: {consistency['replay_projection']['present']}/"
            f"{consistency['replay_projection']['expected']} present.",
            f"Trajectory projection: {consistency['trajectory_projection']['present']}/"
            f"{consistency['trajectory_projection']['expected']} present.",
            "",
            "## Metric Definitions",
            "",
            "- Business status proportions use all tasks in this report as denominator.",
            "- RISE outcome proportions use tasks with a non-null `rise_outcome`.",
            "- Action attempts are counted once from each checkpoint's unique replay sequence; failed Action attempts are reported separately.",
            "- LLM/Retrieval observations come from supplied structured JSON logs; missing logs or fields stay unknown.",
            "- Proposal success is committed tasks with a persisted `final_proposal` divided by committed tasks.",
            "",
        ]
    )
    return "\n".join(lines)


def read_structured_metric_logs(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read safe call metrics from JSON logs and de-duplicate repeated event IDs."""

    rows: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    seen_paths: set[Path] = set()
    stats = Counter()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen_paths:
            stats["duplicate_file_arguments"] += 1
            continue
        seen_paths.add(resolved)
        with resolved.open("r", encoding="utf-8") as stream:
            for line in stream:
                stats["lines_read"] += 1
                try:
                    decoded = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stats["invalid_json_lines"] += 1
                    continue
                if not isinstance(decoded, dict) or decoded.get("event") not in _MODEL_CALL_EVENTS:
                    continue
                event_id = decoded.get("event_id")
                if isinstance(event_id, str) and event_id:
                    if event_id in seen_event_ids:
                        stats["duplicate_event_ids"] += 1
                        continue
                    seen_event_ids.add(event_id)
                else:
                    stats["events_without_event_id"] += 1
                # Keep only fields needed by the evaluator; never retain or
                # echo arbitrary log payloads in a report.
                row = {
                    key: decoded.get(key)
                    for key in (
                        "event",
                        "task_id",
                        "status",
                        "duration_ms",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "provider_retry_count",
                        "format_retry_count",
                        "http_request_count",
                    )
                }
                rows.append(row)
    return rows, dict(stats)


def _load_all_tasks(repository: TaskRepository, limit: int | None) -> list[Any]:
    tasks: list[Any] = []
    cursor = None
    while limit is None or len(tasks) < limit:
        batch_limit = min(100, limit - len(tasks)) if limit is not None else 100
        batch = repository.list_tasks(limit=batch_limit, cursor=cursor)
        if not batch:
            break
        tasks.extend(batch)
        if len(batch) < batch_limit:
            break
        last = batch[-1]
        cursor = TaskPageCursor(created_at=last.created_at, task_id=last.id)
    return tasks


def evaluate_postgres(
    business_database_url: str,
    *,
    checkpoint_database_url: str | None = None,
    log_paths: Sequence[Path] = (),
    limit: int | None = None,
) -> dict[str, Any]:
    """Load existing business/checkpoint state and produce one aggregate."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    engine = create_business_engine(business_database_url)
    try:
        repository = TaskRepository(create_session_factory(engine))
        tasks = _load_all_tasks(repository, limit)
        events_by_task: dict[str, list[Any]] = {}
        for task in tasks:
            events: list[Any] = []
            after_seq = 0
            while True:
                batch = repository.list_events(task.id, after_seq=after_seq, limit=500)
                if not batch:
                    break
                events.extend(batch)
                after_seq = batch[-1].seq
                if len(batch) < 500:
                    break
            events_by_task[_task_key(task)] = events

        checkpoint_url = (
            checkpoint_database_url
            or os.environ.get("RISE_CHECKPOINT_DATABASE_URL")
            or business_database_url
        )
        states: dict[str, Any] = {}
        errors: dict[str, str] = {}
        # Evaluation reads the existing Checkpointer schema and never runs
        # setup/migrations or writes to business/checkpoint tables.
        with open_postgres_checkpointer(checkpoint_url, setup=False) as saver:
            for task in tasks:
                key = _task_key(task)
                try:
                    state = read_checkpoint_state(saver, UUID(key))
                except Exception as exc:
                    errors[key] = type(exc).__name__
                else:
                    if state is not None:
                        states[key] = state

        metric_records, log_stats = read_structured_metric_logs(log_paths) if log_paths else (None, None)
        return build_evaluation_report(
            tasks,
            events_by_task=events_by_task,
            states_by_task=states,
            checkpoint_errors=errors,
            metric_records=metric_records,
            log_parse=log_stats,
        )
    finally:
        engine.dispose()


def _write_report(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a RISE engineering evaluation report")
    parser.add_argument("--database-url", default=None, help="business PostgreSQL URL (defaults to RISE_DATABASE_URL)")
    parser.add_argument("--checkpoint-database-url", default=None, help="checkpoint PostgreSQL URL (defaults to business URL)")
    parser.add_argument("--logs", action="append", default=[], help="JSON structured log file; repeat to read multiple files")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the most recently created N tasks")
    parser.add_argument("--json-out", default="rise-evaluation.json", help="JSON report path, or - for stdout")
    parser.add_argument("--markdown-out", default="rise-evaluation.md", help="Markdown report path, or - for stdout")
    args = parser.parse_args(argv)
    try:
        business_url = args.database_url or database_url_from_env()
        report = evaluate_postgres(
            business_url,
            checkpoint_database_url=args.checkpoint_database_url,
            log_paths=[Path(value) for value in args.logs],
            limit=args.limit,
        )
        _write_report(args.json_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        _write_report(args.markdown_out, render_markdown_report(report))
        return 0
    except Exception as exc:
        # The exception message can include a database URL or data; only expose
        # the safe class name in this developer-facing CLI.
        print(f"evaluation failed ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the console entry point
    raise SystemExit(main())


__all__ = [
    "build_evaluation_report",
    "evaluate_postgres",
    "main",
    "read_structured_metric_logs",
    "render_markdown_report",
]
