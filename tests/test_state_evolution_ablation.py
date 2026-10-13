"""Offline contract tests for the independent Full-reference/Flat pilot."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from experiments.state_evolution_ablation.flat_controller import FlatBudgetView, FlatController
from experiments.state_evolution_ablation.flat_prompts import FLAT_ACTIONS, FlatControllerDecision, FlatProposal
from experiments.state_evolution_ablation.flat_runner import (
    FlatExecutionState,
    FlatRunConfig,
    FlatRunner,
    _ephemeral_gap,
    _ephemeral_solution,
)
from experiments.state_evolution_ablation.full_reference import inspect_full_reference
from experiments.state_evolution_ablation.run_pilot8 import _parser
from scripts.run_state_evolution_ablation_retry_p031_p040_parallel import (
    RETRY_SEED_IDS,
    _parser as _retry_parser,
)
from experiments.state_evolution_ablation.pilot8_config import (
    DEFAULT_FULL_OUTPUT_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_STEPS,
    PILOT_SEEDS,
    resolve_seed_paths,
)
from research_agent.llm.client import LLMCallMetadata, StructuredLLMResponse
from research_agent.models.seed import SeedPaperContext
from research_agent.models.state import Component, CurrentSolution


class _NoCall:
    def complete_structured(self, *args, **kwargs):  # pragma: no cover - failure is the assertion
        raise AssertionError("unique legal Flat action must not call an LLM")


class _OneCall:
    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, request, output_schema):
        self.calls += 1
        return StructuredLLMResponse(
            output=FlatControllerDecision(action="COMMIT", reason="the proposal is ready"),
            metadata=LLMCallMetadata(provider="fake", model="fake"),
        )


def test_pilot_config_has_eight_existing_seed_files_and_current_budgets():
    assert len(PILOT_SEEDS) == 8
    assert DEFAULT_MAX_STEPS == 12
    assert DEFAULT_MAX_SEARCHES == 2
    assert all(path.is_file() for path in resolve_seed_paths(DEFAULT_INPUT_DIR).values())


def test_flat_pilot_defaults_to_eight_paper_workers():
    assert _parser().parse_args([]).workers == 8


def test_flat_retry_targets_two_seeds_and_defaults_to_two_workers():
    assert RETRY_SEED_IDS == ("P031", "P040")
    assert _retry_parser().parse_args([]).workers == 2


def test_historical_full_references_are_readable_and_compatible():
    references = [inspect_full_reference(seed, DEFAULT_FULL_OUTPUT_DIR) for seed in PILOT_SEEDS]
    assert len(references) == 8
    assert all(item["read_only"] is True for item in references)
    assert all(item["api_calls_in_this_experiment"] == 0 for item in references)
    assert all(item["historical_config"]["max_steps"] == DEFAULT_MAX_STEPS for item in references)
    assert all(item["historical_config"]["max_searches"] == DEFAULT_MAX_SEARCHES for item in references)


def test_flat_controller_unique_action_is_deterministic():
    controller = FlatController(_NoCall())
    decision = controller.choose(
        seed_paper="seed",
        current_proposal=None,
        evidence_history=[],
        action_history=[],
        budget=FlatBudgetView(12, 2),
        allowed_actions=["REFINE_PROPOSAL"],
    )
    assert decision.action == "REFINE_PROPOSAL"
    assert controller.call_count == 0


def test_flat_controller_uses_llm_only_for_multiple_legal_actions():
    fake = _OneCall()
    controller = FlatController(fake)
    decision = controller.choose(
        seed_paper="seed",
        paper_id="P-test",
        current_proposal={"research_problem": "p", "proposed_method": "m", "experimental_plan": "e"},
        evidence_history=[],
        action_history=[],
        budget=FlatBudgetView(8, 1),
        allowed_actions=["VERIFY", "COMMIT"],
    )
    assert decision.action == "COMMIT"
    assert fake.calls == 1
    assert controller.call_count == 1


def test_flat_state_contains_only_flat_persistent_fields():
    state = FlatExecutionState(
        current_proposal={"research_problem": "p", "proposed_method": "m", "experimental_plan": "e"},
        remaining_budget={"steps_remaining": 12, "searches_remaining": 2},
    )
    assert set(asdict(state)) == {
        "current_proposal",
        "evidence_history",
        "action_history",
        "remaining_budget",
    }


def test_transient_rise_adapter_is_not_flat_state():
    proposal = type("Proposal", (), {
        "research_problem": "problem",
        "proposed_method": "method",
        "experimental_plan": "plan",
    })()
    gap = _ephemeral_gap(proposal)
    solution = _ephemeral_solution(proposal)
    assert isinstance(gap, object)
    assert isinstance(solution, CurrentSolution)
    assert "current_solution" not in asdict(FlatExecutionState())


def test_flat_runner_allows_initial_refine_without_existing_proposal(tmp_path):
    runner = FlatRunner(
        FlatRunConfig(
            markdown_path=tmp_path / "seed.md",
            output_dir=tmp_path / "output",
            semantic_scholar_config=tmp_path / "semantic.json",
            max_steps=12,
            max_searches=2,
        )
    )
    expected = FlatProposal(
        research_problem="A grounded research problem",
        proposed_method="A concrete proposed method",
        experimental_plan="A testable experimental plan",
    )
    runner._reading = object()
    calls = []

    def fake_call_proposal(*, reading, state, experiment_mode):
        calls.append((reading, state.current_proposal, experiment_mode))
        return expected

    runner._call_proposal = fake_call_proposal
    state = FlatExecutionState(
        remaining_budget={"steps_remaining": 12, "searches_remaining": 2}
    )

    after, tool_output, evidence, search_delta = runner._run_tool(
        action="REFINE_PROPOSAL",
        seed_context=object(),
        state=state,
        step=1,
    )

    assert after == expected.model_dump(mode="json")
    assert tool_output == {"operation": "refine_proposal"}
    assert evidence is None
    assert search_delta == 0
    assert len(calls) == 1
    assert calls[0][1] is None
    assert calls[0][2] is False
