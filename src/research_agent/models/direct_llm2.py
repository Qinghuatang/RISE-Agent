"""Strict output contract for the direct_llm2 prompt variant."""

from .base import NonEmptyStr, StrictModel


class DirectLLM2Proposal(StrictModel):
    research_problem: NonEmptyStr
    proposed_method: NonEmptyStr
    method_rationale: NonEmptyStr
    experimental_plan: NonEmptyStr


__all__ = ["DirectLLM2Proposal"]
