"""Scientific Controller contracts.

``ControllerContext`` and ``ControllerRoutingResult`` are execution-time
artifacts.  They are deliberately not part of Research State.
"""

from typing import Any, Literal

from pydantic import Field

from .base import ScientificAction, StrictModel


class ControllerDecision(StrictModel):
    action: ScientificAction
    reason: str


class ControllerContext(StrictModel):
    """Compact deterministic input presented to a Controller LLM."""

    active_gap_summary: dict[str, Any] | None = None
    candidate_solutions: list[dict[str, Any]] = Field(default_factory=list)
    current_solution_summary: dict[str, Any] | None = None
    component_status: dict[str, str] = Field(default_factory=dict)
    novelty_status: str | None = None
    experiment_status: dict[str, str] = Field(default_factory=dict)
    last_action: ScientificAction | None = None
    last_result: dict[str, Any] | None = None
    relevant_failure_memory: list[dict[str, Any]] = Field(default_factory=list)
    commit_allowed: bool = False
    unmet_commit_conditions: list[str] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)
    allowed_actions: list[ScientificAction] = Field(default_factory=list)


class ControllerRoutingResult(StrictModel):
    """The controller's execution-only result; it is not Research State."""

    status: Literal["decision", "stopped"]
    decision: ControllerDecision | None = None
    allowed_actions: list[ScientificAction] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    used_llm: bool = False
