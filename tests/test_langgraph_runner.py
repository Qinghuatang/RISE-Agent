from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from research_agent.controller.budget import BudgetTracker
from research_agent.models.controller import ControllerDecision, ControllerRoutingResult
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner

from test_m11 import modules, reading, seed


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "install the optional agent extra")
class LangGraphRunnerTests(unittest.TestCase):
    def test_full_run_commits_and_routes_saved_decisions_through_action_nodes(self):
        modules_bundle = modules()
        runner = LangGraphTrajectoryRunner(
            modules_bundle,
            budget=BudgetTracker(max_steps=5, max_searches=1),
            run_id="graph-commit",
        )

        with patch.object(
            modules_bundle.controller,
            "decide",
            wraps=modules_bundle.controller.decide,
        ) as decide:
            state = runner.run(seed(), full_paper_reading_context=reading())

        actions = [record.action for record in state.replay_records if record.action]
        self.assertEqual(
            actions,
            [
                "VERIFY_RELATED_WORK",
                "SOLUTION_IDEATION",
                "VERIFY_RELATED_WORK",
                "DESIGN_EXPERIMENT",
                "COMMIT",
            ],
        )
        self.assertEqual(decide.call_count, len(actions))
        self.assertEqual(state.rise_status, "committed")
        self.assertIsNotNone(state.commit_snapshot)
        self.assertIsNotNone(state.final_proposal)
        self.assertEqual(state.budget.steps_used, 5)
        self.assertEqual(state.budget.searches_used, 1)
        self.assertEqual(state.replay_records[0].kind, "bootstrap")
        self.assertEqual(len(state.replay_records), state.budget.steps_used + 1)

    def test_budget_exhaustion_stops_normally_after_one_action(self):
        modules_bundle = modules()
        runner = LangGraphTrajectoryRunner(
            modules_bundle,
            budget=BudgetTracker(max_steps=1, max_searches=1),
            run_id="graph-budget-stop",
        )

        with patch.object(
            modules_bundle.finalizer,
            "finalize",
            wraps=modules_bundle.finalizer.finalize,
        ) as finalize:
            state = runner.run(seed(), full_paper_reading_context=reading())

        self.assertEqual(state.rise_status, "stopped")
        self.assertEqual(state.budget.steps_used, 1)
        self.assertIsNone(state.commit_snapshot)
        self.assertIsNone(state.final_proposal)
        self.assertEqual(finalize.call_count, 0)
        self.assertTrue(
            any(event.event_type == "controller_stopped" for event in state.trajectory_events)
        )

    def test_action_failure_keeps_legacy_failure_accounting_and_terminates(self):
        modules_bundle = modules()
        runner = LangGraphTrajectoryRunner(
            modules_bundle,
            budget=BudgetTracker(max_steps=3, max_searches=2),
            run_id="graph-action-failure",
        )
        route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="SEEK_INSPIRATION",
                reason="Use the supplied deterministic route.",
            ),
            allowed_actions=["SEEK_INSPIRATION"],
            used_llm=False,
        )

        def fail_after_one_search(*args, **kwargs):
            modules_bundle.inspiration_retrieval.scientific_search_count += 1
            raise RuntimeError("synthetic retrieval action failure")

        with (
            patch.object(modules_bundle.controller, "decide", return_value=route) as decide,
            patch.object(
                modules_bundle.inspiration_retrieval,
                "retrieve",
                side_effect=fail_after_one_search,
            ),
        ):
            state = runner.run(seed(), full_paper_reading_context=reading())

        self.assertEqual(state.rise_status, "failed")
        self.assertEqual(state.error_type, "RuntimeError")
        self.assertEqual(state.budget.steps_used, 1)
        self.assertEqual(state.budget.searches_used, 1)
        self.assertEqual(len(state.replay_records), 2)
        self.assertEqual(state.replay_records[-1].action, "SEEK_INSPIRATION")
        self.assertEqual(decide.call_count, 1)
        self.assertTrue(
            any(event.event_type == "runner_action_failed" for event in state.trajectory_events)
        )

    def test_finalizer_failure_preserves_commit_snapshot_without_an_action_retry(self):
        modules_bundle = modules()
        runner = LangGraphTrajectoryRunner(
            modules_bundle,
            budget=BudgetTracker(max_steps=5, max_searches=1),
            run_id="graph-finalizer-failure",
        )

        with patch.object(
            modules_bundle.finalizer,
            "finalize",
            side_effect=RuntimeError("synthetic finalizer failure"),
        ):
            state = runner.run(seed(), full_paper_reading_context=reading())

        self.assertEqual(state.rise_status, "finalization_failed")
        self.assertEqual(state.error_type, "RuntimeError")
        self.assertIsNotNone(state.commit_snapshot)
        self.assertIsNone(state.final_proposal)
        self.assertEqual(state.budget.steps_used, 5)
        self.assertEqual(state.budget.searches_used, 1)

    def test_recursion_limit_scales_with_remaining_action_budget(self):
        runner = LangGraphTrajectoryRunner(
            modules(),
            budget=BudgetTracker(max_steps=40, max_searches=0),
            run_id="graph-recursion-limit",
        )
        self.assertEqual(runner.recursion_limit, 83)


if __name__ == "__main__":
    unittest.main()
