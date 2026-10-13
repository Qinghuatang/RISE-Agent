"""Small deterministic State diff for trajectory auditing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models.state import ResearchState


def _flatten(value: Any, path: str, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        if not value:
            output[path] = {}
            return
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            _flatten(value[key], child_path, output)
        return
    if isinstance(value, list):
        if not value:
            output[path] = []
            return
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", output)
        return
    output[path] = value


def _state_payload(state: ResearchState) -> dict[str, Any]:
    if not isinstance(state, ResearchState):
        raise TypeError("compute_state_diff requires ResearchState values")
    return state.model_dump(mode="json")


def compute_state_diff(
    before: ResearchState,
    after: ResearchState,
) -> dict[str, dict[str, Any]]:
    """Return changed leaf paths without mutating either input state."""

    before_flat: dict[str, Any] = {}
    after_flat: dict[str, Any] = {}
    _flatten(_state_payload(before), "", before_flat)
    _flatten(_state_payload(after), "", after_flat)
    changed: dict[str, dict[str, Any]] = {}
    for path in sorted(set(before_flat) | set(after_flat)):
        old = before_flat.get(path)
        new = after_flat.get(path)
        if old != new:
            changed[path] = {"before": old, "after": new}
    return changed
