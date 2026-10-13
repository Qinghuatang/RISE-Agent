"""Targeted Revision patch contracts.

The operation field is an internal patch instruction. The canonical State
still contains only the existing CurrentSolution/Component structure.
"""

from typing import Literal

from pydantic import Field, model_validator

from .base import NonEmptyStr, StrictModel


class RevisionNeed(StrictModel):
    problem_to_fix: str
    affected_units: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class ComponentRevisionPatch(StrictModel):
    component_id: NonEmptyStr
    description: str | None = None
    operation: Literal["modify", "add", "remove", "replace"] = "modify"

    @model_validator(mode="after")
    def validate_operation_payload(self) -> "ComponentRevisionPatch":
        if self.operation == "remove":
            if self.description is not None and self.description.strip():
                raise ValueError("remove Component patches must not provide a description")
            return self
        if self.description is None or not self.description.strip():
            raise ValueError(f"{self.operation} Component patches require a description")
        return self


class RevisionResult(StrictModel):
    status: Literal["revised", "unable_to_revise"]
    revised_core_idea: str | None = None
    revised_rationale: str | None = None
    revised_components: list[ComponentRevisionPatch] = Field(default_factory=list)
    changed_units: list[str] = Field(default_factory=list)
    revision_summary: str
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_patch_consistency(self) -> "RevisionResult":
        has_patch = (
            self.revised_core_idea is not None
            or self.revised_rationale is not None
            or bool(self.revised_components)
        )
        if self.status == "revised":
            if not has_patch:
                raise ValueError("revised requires at least one non-empty patch field")
            if self.failure_reason is not None:
                raise ValueError("revised cannot provide failure_reason")
        else:
            if has_patch:
                raise ValueError("unable_to_revise cannot provide a revision patch")
            if self.failure_reason is None:
                raise ValueError("unable_to_revise requires failure_reason")
        return self


class TargetedRevisionDraft(StrictModel):
    """Internal one-call output for Targeted Revision."""

    selected_inspiration_ref: str | None = None
    result: RevisionResult
