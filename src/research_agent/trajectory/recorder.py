"""Append-only JSONL trajectory recorder."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..errors import SerializationError
from ..models.gap import GapAgenda
from ..models.trajectory import BudgetMetadata, TrajectoryEvent
from ..serialization import read_jsonl, write_jsonl


TRAJECTORY_SCHEMA = "trajectory_event"


def _copy_event(event: TrajectoryEvent) -> TrajectoryEvent:
    payload = event.model_dump(mode="python") if isinstance(event, TrajectoryEvent) else event
    return TrajectoryEvent.model_validate(payload)


class TrajectoryRecorder:
    """Stores ordered audit events; it does not replay or interpret them."""

    def __init__(self, events: list[TrajectoryEvent] | None = None) -> None:
        self._events: list[TrajectoryEvent] = []
        for event in events or []:
            self.append(event)

    def append(self, event: TrajectoryEvent) -> TrajectoryEvent:
        try:
            validated = _copy_event(event)
        except ValidationError as exc:
            raise ValueError("invalid TrajectoryEvent") from exc
        self.validate_next_step(validated.step)
        self._events.append(validated)
        return _copy_event(validated)

    def validate_next_step(self, step: int) -> None:
        """Validate a future event step without appending or mutating state."""

        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("trajectory step must be an integer")
        if step < 0:
            raise ValueError("trajectory step must be non-negative")
        if self._events and step < self._events[-1].step:
            raise ValueError("trajectory event step cannot move backwards")

    def append_event(
        self,
        step: int,
        event_type: str,
        *,
        action=None,
        result_ref: str | None = None,
        result_payload: dict[str, Any] | None = None,
        state_diff: dict[str, Any] | None = None,
        memory_writes: list[str] | None = None,
        budget: BudgetMetadata | dict[str, Any] | None = None,
        error: str | None = None,
        timestamp: datetime | None = None,
    ) -> TrajectoryEvent:
        event = TrajectoryEvent(
            step=step,
            event_type=event_type,
            action=action,
            result_ref=result_ref,
            result_payload=result_payload,
            state_diff=state_diff or {},
            memory_writes=memory_writes or [],
            budget=budget,
            error=error,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        return self.append(event)

    def get_events(self) -> list[TrajectoryEvent]:
        return [_copy_event(event) for event in self._events]

    def record_gap_agenda(
        self,
        step: int,
        gap_agenda: GapAgenda,
        *,
        memory_writes: list[str] | None = None,
        budget: BudgetMetadata | dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> TrajectoryEvent:
        agenda = GapAgenda.model_validate(gap_agenda.model_dump(mode="python"))
        return self.append_event(
            step,
            "gap_agenda_updated",
            result_ref="gap_agenda",
            result_payload=agenda.model_dump(mode="json"),
            memory_writes=memory_writes,
            budget=budget,
            timestamp=timestamp,
        )

    def get_latest_gap_agenda(self) -> GapAgenda | None:
        for event in reversed(self._events):
            if event.event_type != "gap_agenda_updated":
                continue
            if not isinstance(event.result_payload, Mapping):
                raise SerializationError("gap_agenda_updated has no object payload")
            try:
                return GapAgenda.model_validate(event.result_payload)
            except ValidationError as exc:
                raise SerializationError("stored GapAgenda payload is invalid") from exc
        return None

    def record_gap_refined(
        self,
        step: int,
        gap_id: str,
        *,
        memory_writes: list[str] | None = None,
        budget: BudgetMetadata | dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> TrajectoryEvent:
        if not isinstance(gap_id, str) or not gap_id.strip():
            raise ValueError("gap_id must be non-empty")
        return self.append_event(
            step,
            "gap_refined",
            result_ref=gap_id.strip(),
            result_payload={"gap_id": gap_id.strip()},
            memory_writes=memory_writes,
            budget=budget,
            timestamp=timestamp,
        )

    def has_refined_gap(self, gap_id: str) -> bool:
        for event in self._events:
            if event.event_type != "gap_refined":
                continue
            payload = event.result_payload
            if not isinstance(payload, Mapping) or not isinstance(payload.get("gap_id"), str):
                raise SerializationError("gap_refined has an invalid payload")
            if payload["gap_id"] == gap_id:
                return True
        return False

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, TRAJECTORY_SCHEMA, self._events)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryRecorder":
        try:
            envelopes = read_jsonl(path, expected_schema=TRAJECTORY_SCHEMA)
            events = [TrajectoryEvent.model_validate(item["payload"]) for item in envelopes]
        except (SerializationError, ValidationError) as exc:
            raise SerializationError("stored trajectory is invalid") from exc
        try:
            return cls(events)
        except ValueError as exc:
            raise SerializationError("stored trajectory event order is invalid") from exc
