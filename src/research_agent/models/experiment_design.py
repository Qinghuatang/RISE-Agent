"""Internal structured drafts used by M8 Experiment Design.

These models are LLM/audit artifacts only.  The canonical Experiment model
remains in ``models.state`` and intentionally has no claim or dependency
fields.
"""

from pydantic import Field

from .base import NonEmptyStr, StrictModel


class KeyClaimDraft(StrictModel):
    """One claim that the proposed experiment set must make testable."""

    id: NonEmptyStr
    statement: NonEmptyStr
    target_unit: NonEmptyStr


class ExperimentProposalDraft(StrictModel):
    """LLM proposal without canonical Experiment identity or status."""

    existing_experiment_id: NonEmptyStr | None = None
    objective: NonEmptyStr
    setup: NonEmptyStr
    baselines: list[NonEmptyStr] = Field(default_factory=list)
    metrics: list[NonEmptyStr] = Field(default_factory=list)
    what_it_validates: NonEmptyStr
    validation_criterion: NonEmptyStr
    validates_claims: list[NonEmptyStr] = Field(default_factory=list)


class ClaimIdentifiabilityDraft(StrictModel):
    """Internal evidence check for one Key Claim."""

    claim_id: NonEmptyStr
    required_observable: NonEmptyStr
    dataset_support: NonEmptyStr
    ground_truth_availability: NonEmptyStr
    control_or_baseline: NonEmptyStr
    falsification: NonEmptyStr
    identifiable: bool


class ExperimentDesignDraft(StrictModel):
    """One-call M8 draft containing claims and initial/patch experiments."""

    key_claims: list[KeyClaimDraft] = Field(min_length=1, max_length=4)
    # A stale-only redesign may legitimately contain one proposal; the final
    # canonical State count is checked against the existing experiment set.
    experiments: list[ExperimentProposalDraft] = Field(min_length=1, max_length=4)
    identifiability_checks: list[ClaimIdentifiabilityDraft] = Field(
        min_length=1, max_length=4
    )
    design_summary: NonEmptyStr


__all__ = [
    "ExperimentDesignDraft",
    "ExperimentProposalDraft",
    "ClaimIdentifiabilityDraft",
    "KeyClaimDraft",
]
