"""Prompts for the Flat ablation's generic proposal and routing calls.

These prompts intentionally describe only the Flat baseline's generic
proposal/evidence loop.  They do not expose or recreate RISE typed state.
"""

from __future__ import annotations

import json

from research_agent.models.base import NonEmptyStr, StrictModel


class FlatProposal(StrictModel):
    research_problem: NonEmptyStr
    proposed_method: NonEmptyStr
    experimental_plan: NonEmptyStr


class FlatControllerDecision(StrictModel):
    action: str
    reason: NonEmptyStr


FLAT_ACTIONS = (
    "VERIFY",
    "SEEK_INSPIRATION",
    "REFINE_PROPOSAL",
    "DESIGN_EXPERIMENTS",
    "COMMIT",
)


CONTROLLER_PROMPT = """You are a research workflow controller for a flat, multi-step proposal process.

Choose exactly one action from the supplied allowed_actions. Base the choice on
the seed paper, the current complete proposal, accumulated scientific evidence,
the action history, and remaining budgets. Do not assume a fixed action order.
Use VERIFY or SEEK_INSPIRATION only when the corresponding evidence acquisition
has scientific information value. Use REFINE_PROPOSAL or DESIGN_EXPERIMENTS to
make a concrete evidence-based improvement. Use COMMIT only when the proposal is
ready enough to report. Return one JSON object with action and a concise reason.

Return JSON matching this schema:
{schema}
"""


PROPOSAL_PROMPT = """You are an expert NLP/ML researcher working in a flat proposal refinement process.

Read the supplied seed paper and current evidence. Produce one complete,
technically coherent research proposal. If a previous proposal exists, update
it using the supplied evidence while keeping the research problem, mechanism,
and experimental plan scientifically consistent. Do not invent results or
outside evidence. Return exactly one JSON object matching this schema:
{schema}

SEED PAPER
{paper}

CURRENT PROPOSAL
{proposal}

EVIDENCE HISTORY
{evidence}
"""


EXPERIMENT_PROMPT = """You are improving the experimental plan of a research proposal.

Using only the seed paper, current proposal, and accumulated evidence, return
the complete updated proposal. Make the experimental plan able to test the
proposal's central claim, while preserving consistency with the problem and
method. Do not invent results or add outside facts. Return exactly one JSON
object matching this schema:
{schema}

SEED PAPER
{paper}

CURRENT PROPOSAL
{proposal}

EVIDENCE HISTORY
{evidence}
"""


def schema_json(model: type[StrictModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))


def render(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


__all__ = [
    "CONTROLLER_PROMPT",
    "EXPERIMENT_PROMPT",
    "FLAT_ACTIONS",
    "FlatControllerDecision",
    "FlatProposal",
    "PROPOSAL_PROMPT",
    "render",
    "schema_json",
]
