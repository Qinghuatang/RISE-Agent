from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.controller.budget import BudgetTracker
from research_agent.errors import RunnerInvariantError
from research_agent.models.controller import ControllerDecision, ControllerRoutingResult
from research_agent.runner import TrajectoryRunner
from research_agent.runtime.action_adapter import RISEActionAdapter
from research_agent.runtime.graph_state import RISEGraphState
from research_agent.serialization.trajectory import load_replay_journal
from research_agent.trajectory.context import TrajectoryContext

from test_m11 import modules, reading, seed


def _events_without_timestamps(events):
    return [
        {
            key: value
            for key, value in event.model_dump(mode="json").items()
            if key != "timestamp"
        }
        for event in events
    ]


def _run_adapter_to_terminal(adapter, state):
    state = adapter.bootstrap(state, full_paper_reading_context=reading())
    while state.rise_status == "running":
        state = RISEGraphState.from_json(state.to_json())
        state = adapter.decide(state)
        if state.rise_status != "running" or not state.decision_pending:
            break
        state = adapter.execute_selected_action(state)
        if state.rise_status != "running":
            break
        if state.commit_snapshot is not None:
            state = adapter.finalize(state)
    return RISEGraphState.from_json(state.to_json())


def _run_adapter_to_snapshot(adapter, state):
    state = adapter.bootstrap(state, full_paper_reading_context=reading())
    while state.rise_status == "running":
        state = adapter.decide(state)
        if not state.decision_pending:
            break
        state = adapter.execute_selected_action(state)
        if state.commit_snapshot is not None:
            return state
        if state.rise_status != "running":
            break
    raise AssertionError("fixture did not reach Commit")


class ActionAdapterTests(unittest.TestCase):
    def test_full_deterministic_run_matches_legacy_runner(self):
        seed_context = seed()
        legacy_modules = modules()
        with tempfile.TemporaryDirectory() as output_dir:
            legacy = TrajectoryRunner(
                legacy_modules,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                output_dir=output_dir,
            ).run(seed_context, full_paper_reading_context=reading())
            legacy_context = TrajectoryContext.load(seed_context, output_dir)
            legacy_replay = load_replay_journal(
                Path(output_dir, "trajectory_replay.jsonl")
            )

        adapter = RISEActionAdapter(modules())
        state = adapter.initialize(
            seed_context,
            max_steps=5,
            max_searches=1,
            run_id="parity-run",
        )
        state = _run_adapter_to_terminal(adapter, state)

        self.assertEqual(legacy.status, state.rise_status)
        self.assertEqual(legacy.final_state, state.research_state)
        self.assertEqual(legacy.budget, state.budget)
        self.assertEqual(legacy.commit_snapshot, state.commit_snapshot)
        self.assertEqual(legacy.final_proposal, state.final_proposal)
        self.assertEqual(legacy_context.get_memory(), state.research_memory)
        self.assertEqual(
            _events_without_timestamps(legacy_context.trajectory.get_events()),
            _events_without_timestamps(state.trajectory_events),
        )
        self.assertEqual(legacy_replay, state.replay_records)

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
        self.assertEqual(state.budget.steps_used, 5)
        self.assertEqual(state.budget.searches_used, 1)
        self.assertEqual(state.replay_records[0].kind, "bootstrap")
        self.assertEqual(len(state.replay_records), state.budget.steps_used + 1)

    def test_saved_controller_decision_is_dispatched_without_a_second_call(self):
        modules_bundle = modules()
        adapter = RISEActionAdapter(modules_bundle)
        state = adapter.initialize(
            seed(), max_steps=5, max_searches=1, run_id="single-decision"
        )
        state = adapter.bootstrap(state, full_paper_reading_context=reading())
        self.assertEqual(state.budget.steps_used, 0)

        with patch.object(
            modules_bundle.controller,
            "decide",
            wraps=modules_bundle.controller.decide,
        ) as decide:
            state = adapter.decide(state)
            self.assertTrue(state.decision_pending)
            self.assertEqual(decide.call_count, 1)

            state = RISEGraphState.from_json(state.to_json())
            state = adapter.execute_selected_action(state)
            self.assertEqual(decide.call_count, 1)

        self.assertFalse(state.decision_pending)
        with self.assertRaises(ValueError):
            adapter.execute_selected_action(state)

    def test_action_failure_counts_one_step_and_reported_search_delta(self):
        modules_bundle = modules()
        adapter = RISEActionAdapter(modules_bundle)
        state = adapter.initialize(
            seed(), max_steps=3, max_searches=2, run_id="action-failure"
        )
        state = adapter.bootstrap(state, full_paper_reading_context=reading())

        route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="SEEK_INSPIRATION",
                reason="Use the supplied deterministic route.",
            ),
            allowed_actions=["SEEK_INSPIRATION"],
            used_llm=False,
        )
        with patch.object(modules_bundle.controller, "decide", return_value=route):
            state = adapter.decide(state)

        def fail_after_one_search(*args, **kwargs):
            modules_bundle.inspiration_retrieval.scientific_search_count += 1
            raise RuntimeError("synthetic retrieval action failure")

        with patch.object(
            modules_bundle.inspiration_retrieval,
            "retrieve",
            side_effect=fail_after_one_search,
        ):
            state = adapter.execute_selected_action(state)

        self.assertEqual(state.rise_status, "failed")
        self.assertEqual(state.error_type, "RuntimeError")
        self.assertEqual(state.budget.steps_used, 1)
        self.assertEqual(state.budget.searches_used, 1)
        self.assertEqual(len(state.replay_records), 2)
        self.assertEqual(state.replay_records[-1].action, "SEEK_INSPIRATION")
        failed = [
            event for event in state.trajectory_events
            if event.event_type == "runner_action_failed"
        ]
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0].result_payload["attempt_started"])
        self.assertEqual(state.research_memory.decision_log[-1].action, "SEEK_INSPIRATION")
        self.assertTrue(state.research_memory.decision_log[-1].result_summary.startswith("failed:"))

        legacy_modules = modules()

        def legacy_fail_after_one_search(*args, **kwargs):
            legacy_modules.inspiration_retrieval.scientific_search_count += 1
            raise RuntimeError("synthetic retrieval action failure")

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.object(legacy_modules.controller, "decide", return_value=route),
                patch.object(
                    legacy_modules.inspiration_retrieval,
                    "retrieve",
                    side_effect=legacy_fail_after_one_search,
                ),
            ):
                legacy_result = TrajectoryRunner(
                    legacy_modules,
                    budget=BudgetTracker(max_steps=3, max_searches=2),
                    output_dir=output_dir,
                ).run(seed(), full_paper_reading_context=reading())
            legacy_context = TrajectoryContext.load(seed(), output_dir)
            legacy_replay = load_replay_journal(
                Path(output_dir, "trajectory_replay.jsonl")
            )

        self.assertEqual(state.rise_status, legacy_result.status)
        self.assertEqual(state.error_type, legacy_result.error_type)
        self.assertEqual(state.research_state, legacy_result.final_state)
        self.assertEqual(state.budget, legacy_result.budget)
        self.assertEqual(state.research_memory, legacy_context.get_memory())
        self.assertEqual(
            _events_without_timestamps(state.trajectory_events),
            _events_without_timestamps(legacy_context.trajectory.get_events()),
        )
        self.assertEqual(state.replay_records, legacy_replay)

    def test_dispatch_invariant_failure_does_not_consume_action_budget(self):
        adapter = RISEActionAdapter(modules())
        state = adapter.initialize(
            seed(), max_steps=3, max_searches=1, run_id="dispatch-invariant"
        )
        route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="VERIFY_RELATED_WORK",
                reason="The target is intentionally absent.",
            ),
            allowed_actions=["VERIFY_RELATED_WORK"],
        )
        state = state.evolve(
            bootstrap_complete=True,
            rise_status="running",
            controller_routing=route,
            decision_pending=True,
        )

        state = adapter.execute_selected_action(state)

        self.assertEqual(state.rise_status, "failed")
        self.assertEqual(state.error_type, RunnerInvariantError.__name__)
        self.assertEqual(state.budget.steps_used, 0)
        self.assertEqual(state.replay_records, [])
        event_types = [event.event_type for event in state.trajectory_events]
        self.assertIn("runner_invariant_failed", event_types)
        self.assertNotIn("runner_action_failed", event_types)

    def test_finalizer_failure_preserves_commit_snapshot_and_budget(self):
        modules_bundle = modules()
        adapter = RISEActionAdapter(modules_bundle)
        state = adapter.initialize(
            seed(), max_steps=5, max_searches=1, run_id="finalizer-failure"
        )
        state = _run_adapter_to_snapshot(adapter, state)
        snapshot = state.commit_snapshot
        budget = state.budget
        trajectory_count = len(state.trajectory_events)

        with patch.object(
            modules_bundle.finalizer,
            "finalize",
            side_effect=RuntimeError("synthetic finalizer failure"),
        ):
            state = adapter.finalize(state)

        self.assertEqual(state.rise_status, "finalization_failed")
        self.assertEqual(state.error_type, "RuntimeError")
        self.assertEqual(state.commit_snapshot, snapshot)
        self.assertIsNone(state.final_proposal)
        self.assertEqual(state.budget, budget)
        self.assertEqual(len(state.trajectory_events), trajectory_count)

        legacy_modules = modules()
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(
                legacy_modules.finalizer,
                "finalize",
                side_effect=RuntimeError("synthetic finalizer failure"),
            ):
                legacy_result = TrajectoryRunner(
                    legacy_modules,
                    budget=BudgetTracker(max_steps=5, max_searches=1),
                    output_dir=output_dir,
                ).run(seed(), full_paper_reading_context=reading())
            legacy_context = TrajectoryContext.load(seed(), output_dir)
            legacy_replay = load_replay_journal(
                Path(output_dir, "trajectory_replay.jsonl")
            )

        self.assertEqual(state.rise_status, legacy_result.status)
        self.assertEqual(state.error_type, legacy_result.error_type)
        self.assertEqual(state.commit_snapshot, legacy_result.commit_snapshot)
        self.assertEqual(state.budget, legacy_result.budget)
        self.assertEqual(state.research_state, legacy_result.final_state)
        self.assertEqual(state.research_memory, legacy_context.get_memory())
        self.assertEqual(state.replay_records, legacy_replay)
        self.assertEqual(
            _events_without_timestamps(state.trajectory_events),
            _events_without_timestamps(legacy_context.trajectory.get_events()),
        )


if __name__ == "__main__":
    unittest.main()
