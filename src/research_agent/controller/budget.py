"""Small execution-time budgets used by the Scientific Controller."""

from __future__ import annotations

from dataclasses import dataclass


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ControllerBudget:
    """Immutable budget snapshot; it is not part of Research State."""

    max_steps: int
    max_searches: int
    steps_used: int = 0
    searches_used: int = 0

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_searches", "steps_used", "searches_used"):
            _nonnegative_int(name, getattr(self, name))
        if self.steps_used > self.max_steps:
            raise ValueError("steps_used cannot exceed max_steps")
        if self.searches_used > self.max_searches:
            raise ValueError("searches_used cannot exceed max_searches")

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.steps_used

    @property
    def remaining_searches(self) -> int:
        return self.max_searches - self.searches_used

    def snapshot(self) -> "ControllerBudget":
        return self


class BudgetTracker:
    """Mutable runtime counter updated only after an action executes."""

    def __init__(
        self,
        max_steps: int,
        max_searches: int,
        *,
        steps_used: int = 0,
        searches_used: int = 0,
    ) -> None:
        self._budget = ControllerBudget(
            max_steps=max_steps,
            max_searches=max_searches,
            steps_used=steps_used,
            searches_used=searches_used,
        )

    @property
    def max_steps(self) -> int:
        return self._budget.max_steps

    @property
    def max_searches(self) -> int:
        return self._budget.max_searches

    @property
    def steps_used(self) -> int:
        return self._budget.steps_used

    @property
    def searches_used(self) -> int:
        return self._budget.searches_used

    @property
    def remaining_steps(self) -> int:
        return self._budget.remaining_steps

    @property
    def remaining_searches(self) -> int:
        return self._budget.remaining_searches

    def snapshot(self) -> ControllerBudget:
        return self._budget

    def can_execute(
        self,
        action: str | None = None,
        *,
        scientific_searches: int = 0,
        mandatory_follow_up: bool = False,
        requires_search: bool = False,
    ) -> bool:
        """Return whether an action and optional reserved follow-up fit.

        ``action`` is intentionally not interpreted here.  Scientific routing
        rules decide which action needs a reserve; the tracker only checks
        accounting.  Provider HTTP request counts are not accepted as budget
        input.
        """

        del action
        _nonnegative_int("scientific_searches", scientific_searches)
        required_steps = 1 + (1 if mandatory_follow_up else 0)
        if required_steps > self.remaining_steps:
            return False
        if requires_search and self.remaining_searches == 0:
            return False
        return scientific_searches <= self.remaining_searches

    def record_execution(
        self,
        scientific_searches: int = 0,
        *,
        provider_http_requests: int | None = None,
    ) -> ControllerBudget:
        """Record one completed high-level action and its reported searches."""

        _nonnegative_int("scientific_searches", scientific_searches)
        # Accepted for audit-call-site clarity, but deliberately not used for
        # scientific budget accounting: HTTP requests and scientific searches
        # are different quantities.
        if provider_http_requests is not None:
            _nonnegative_int("provider_http_requests", provider_http_requests)
        if self.remaining_steps < 1:
            raise ValueError("step budget exhausted")
        if scientific_searches > self.remaining_searches:
            raise ValueError("search budget exhausted")
        self._budget = ControllerBudget(
            max_steps=self.max_steps,
            max_searches=self.max_searches,
            steps_used=self.steps_used + 1,
            searches_used=self.searches_used + scientific_searches,
        )
        return self._budget


__all__ = ["BudgetTracker", "ControllerBudget"]
