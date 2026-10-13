"""Engineering-only replay journal for offline trajectory reconstruction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from ..errors import SerializationError
from ..memory.store import ResearchMemoryStore
from ..models.memory import ResearchMemory
from ..models.runner import (
    REPLAY_MEMORY_CATEGORIES,
    TrajectoryReplayRecord,
)
from ..models.seed import SeedPaperContext
from ..runtime.state_store import ResearchStateStore
from ..trajectory.context import TrajectoryContext
from ..trajectory.recorder import TrajectoryRecorder
from .codec import read_jsonl, write_jsonl


TRAJECTORY_REPLAY_SCHEMA = "trajectory_replay_record"


def _validated_records(records: Iterable[TrajectoryReplayRecord]) -> list[TrajectoryReplayRecord]:
    result: list[TrajectoryReplayRecord] = []
    expected_sequence = 1
    previous_step = -1
    seed_paper_id: str | None = None
    for raw in records:
        try:
            record = (
                raw
                if isinstance(raw, TrajectoryReplayRecord)
                else TrajectoryReplayRecord.model_validate(raw)
            )
        except ValidationError as exc:
            raise SerializationError("trajectory replay record is invalid") from exc
        if record.sequence != expected_sequence:
            raise SerializationError("trajectory replay sequence must start at 1 and increase by 1")
        if record.step < previous_step:
            raise SerializationError("trajectory replay step cannot move backwards")
        if seed_paper_id is None:
            seed_paper_id = record.seed_paper_id
        elif record.seed_paper_id != seed_paper_id:
            raise SerializationError("trajectory replay records use different seed papers")
        result.append(record)
        expected_sequence += 1
        previous_step = record.step
    return result


def save_replay_journal(
    records: Iterable[TrajectoryReplayRecord],
    path: str | Path,
) -> Path:
    validated = _validated_records(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(target, TRAJECTORY_REPLAY_SCHEMA, validated)
    return target


def load_replay_journal(path: str | Path) -> list[TrajectoryReplayRecord]:
    try:
        envelopes = read_jsonl(path, expected_schema=TRAJECTORY_REPLAY_SCHEMA)
        records = [TrajectoryReplayRecord.model_validate(item["payload"]) for item in envelopes]
    except (SerializationError, ValidationError) as exc:
        raise SerializationError("stored trajectory replay journal is invalid") from exc
    return _validated_records(records)


def _apply_memory_patch(
    memory_store: ResearchMemoryStore,
    memory_patch: dict[str, object],
) -> None:
    unknown = set(memory_patch) - REPLAY_MEMORY_CATEGORIES
    if unknown:
        raise SerializationError(
            "trajectory replay contains unsupported memory categories: "
            + ", ".join(sorted(unknown))
        )
    payload = memory_store.get_memory().model_dump(mode="python")
    for category, value in memory_patch.items():
        payload[category] = deepcopy(value)
    try:
        memory_store.replace_memory(ResearchMemory.model_validate(payload))
    except (ValueError, ValidationError) as exc:
        raise SerializationError("trajectory replay produced invalid Research Memory") from exc


def reconstruct_context(
    seed_context: SeedPaperContext,
    trajectory_path: str | Path,
    replay_path: str | Path,
) -> TrajectoryContext:
    """Reconstruct State and Memory without executing any scientific module."""

    try:
        seed = SeedPaperContext.model_validate(seed_context.model_dump(mode="python"))
    except ValidationError as exc:
        raise SerializationError("replay seed context is invalid") from exc
    recorder = TrajectoryRecorder.load(trajectory_path)
    records = load_replay_journal(replay_path)
    state_store = ResearchStateStore()
    memory_store = ResearchMemoryStore(seed.paper_id)
    for record in records:
        if record.seed_paper_id != seed.paper_id:
            raise SerializationError("trajectory replay seed paper does not match the input seed")
        state_store.replace_state(record.state_after)
        _apply_memory_patch(memory_store, record.memory_patch)
    return TrajectoryContext(
        seed,
        state_store=state_store,
        memory_store=memory_store,
        recorder=recorder,
    )


__all__ = [
    "TRAJECTORY_REPLAY_SCHEMA",
    "load_replay_journal",
    "reconstruct_context",
    "save_replay_journal",
]
