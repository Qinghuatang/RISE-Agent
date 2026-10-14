from __future__ import annotations

import unittest

from pydantic import ValidationError

from research_agent.controller.budget import ControllerBudget
from research_agent.models.seed import ExperimentContext, SeedPaperContext
from research_agent.runtime.graph_state import (
    GraphStateValidationError,
    PaperInputReference,
    RISEGraphState,
)


def seed() -> SeedPaperContext:
    return SeedPaperContext(
        paper_id="graph-state-seed",
        title="Graph state seed",
        experiment_context=ExperimentContext(),
    )


class GraphStateTests(unittest.TestCase):
    def test_json_round_trip_and_context_reconstruction(self):
        paper_input = PaperInputReference(
            locator="data/papers/example.md",
            sha256="a" * 64,
        )
        state = RISEGraphState.initialize(
            seed(),
            ControllerBudget(max_steps=4, max_searches=2),
            run_id="test-run",
            paper_input=paper_input,
        )

        restored = RISEGraphState.from_json(state.to_json())
        self.assertEqual(restored, state)
        self.assertEqual(restored.schema_version, 1)
        self.assertEqual(restored.paper_input, paper_input)

        context = restored.to_trajectory_context()
        self.assertEqual(context.get_state(), restored.research_state)
        self.assertEqual(context.get_memory(), restored.research_memory)
        self.assertEqual(context.trajectory.get_events(), restored.trajectory_events)
        self.assertEqual(context.paper_id, "graph-state-seed")

    def test_schema_and_cross_field_validation_fail_clearly(self):
        state = RISEGraphState.initialize(
            seed(),
            ControllerBudget(max_steps=2, max_searches=1),
            run_id="test-run",
        )
        payload = state.to_payload()
        payload["schema_version"] = 2
        with self.assertRaises(GraphStateValidationError):
            RISEGraphState.from_payload(payload)

        payload = state.to_payload()
        payload["unexpected_runtime_client"] = "must not be checkpointed"
        with self.assertRaises(GraphStateValidationError):
            RISEGraphState.from_payload(payload)

        with self.assertRaises(ValidationError):
            PaperInputReference(locator="paper.md", sha256="not-a-digest")

    def test_replacement_updates_do_not_merge_or_duplicate_history(self):
        state = RISEGraphState.initialize(
            seed(),
            ControllerBudget(max_steps=3, max_searches=1),
            run_id="test-run",
        )
        first_event_count = len(state.trajectory_events)
        restored = RISEGraphState.from_json(state.to_json())
        next_state = restored.evolve(unresolved_issues=["waiting"])

        self.assertEqual(len(next_state.trajectory_events), first_event_count)
        self.assertEqual(next_state.trajectory_events, state.trajectory_events)
        self.assertEqual(next_state.unresolved_issues, ["waiting"])

    def test_langgraph_1215_accepts_pydantic_state_schema(self):
        try:
            from langgraph.graph import END, START, StateGraph
            from importlib.metadata import version
        except ImportError:
            self.skipTest("install the optional agent extra to test LangGraph integration")

        self.assertEqual(version("langgraph"), "1.2.15")
        state = RISEGraphState.initialize(
            seed(),
            ControllerBudget(max_steps=2, max_searches=1),
            run_id="langgraph-smoke",
        )
        graph = StateGraph(RISEGraphState)
        graph.add_node("identity", lambda value: value)
        graph.add_edge(START, "identity")
        graph.add_edge("identity", END)

        result = graph.compile().invoke(state)
        self.assertEqual(result["run_id"], "langgraph-smoke")
        self.assertEqual(result["trajectory_events"], state.trajectory_events)


if __name__ == "__main__":
    unittest.main()
