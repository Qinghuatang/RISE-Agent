"""Small validated store for the current Research State."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..models.state import ResearchState
from ..serialization import read_json, write_json
from ..errors import SerializationError


STATE_SCHEMA = "research_state"


def _validated_copy(value: ResearchState | object) -> ResearchState:
    try:
        if isinstance(value, ResearchState):
            payload = value.model_dump(mode="python")
        else:
            payload = value
        return ResearchState.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid ResearchState") from exc


class ResearchStateStore:
    """Owns only the latest validated state; it applies no scientific rules."""

    def __init__(self, initial_state: ResearchState | None = None) -> None:
        self._state = _validated_copy(initial_state or ResearchState())

    def get_state(self) -> ResearchState:
        return _validated_copy(self._state)

    def snapshot_state(self) -> ResearchState:
        return self.get_state()

    def replace_state(self, new_state: ResearchState) -> ResearchState:
        self._state = _validated_copy(new_state)
        return self.get_state()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, STATE_SCHEMA, self._state)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ResearchStateStore":
        try:
            envelope = read_json(path, expected_schema=STATE_SCHEMA)
            state = ResearchState.model_validate(envelope["payload"])
        except (SerializationError, ValidationError) as exc:
            raise SerializationError("stored ResearchState is invalid") from exc
        return cls(state)
