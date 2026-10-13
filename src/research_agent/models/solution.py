"""Solution ideation drafts and lightweight module results.

The draft models are LLM-only contracts.  They deliberately do not contain
Research State identifiers or lifecycle fields; those are assigned by
deterministic module code after validation.
"""

from typing import Literal

from pydantic import Field

from .base import NonEmptyStr, StrictModel
from .state import CurrentSolution


class CandidateSolutionDraft(StrictModel):
    core_idea: NonEmptyStr
    rationale: NonEmptyStr


class FormSolutionsDraft(StrictModel):
    candidates: list[CandidateSolutionDraft] = Field(min_length=3, max_length=3)


class SolutionDevelopmentDraft(StrictModel):
    selected_solution_id: NonEmptyStr
    selection_reason: NonEmptyStr
    core_idea: NonEmptyStr
    rationale: NonEmptyStr
    components: list[NonEmptyStr] = Field(min_length=2, max_length=5)


class SolutionIdeationResult(StrictModel):
    generated_new_candidates: bool
    selected_solution_id: NonEmptyStr | None = None
    selection_reason: str | None = None
    current_solution: CurrentSolution | None = None
    next_required_action: Literal["VERIFY_RELATED_WORK"] | None = None
