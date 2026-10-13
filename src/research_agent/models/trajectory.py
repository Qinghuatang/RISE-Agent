"""Minimal, append-only-friendly trajectory event contract."""

from datetime import datetime
from typing import Any

from pydantic import Field

from .base import NonEmptyStr, ScientificAction, StrictModel


class BudgetMetadata(StrictModel):
    step_budget_remaining: int | None = None
    search_budget_remaining: int | None = None


class TrajectoryEvent(StrictModel):
    step: int
    event_type: NonEmptyStr
    action: ScientificAction | None = None
    result_ref: str | None = None
    result_payload: dict[str, Any] | None = None
    state_diff: dict[str, Any] = Field(default_factory=dict)
    memory_writes: list[str] = Field(default_factory=list)
    budget: BudgetMetadata | None = None
    error: str | None = None
    timestamp: datetime
