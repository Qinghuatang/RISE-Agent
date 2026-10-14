from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from research_agent.controller.budget import BudgetTracker
from research_agent.models.controller import ControllerDecision, ControllerRoutingResult
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import (
    ExistingThreadError,
    LangGraphTrajectoryRunner,
    MissingCheckpointError,
)
from research_agent.runtime.graph_state import RISEGraphState
from research_agent.runtime.paper_store import PaperInputIntegrityError, PaperInputStore

from test_m11 import modules, reading, seed


PAPER_MARKDOWN = "\n".join(reading().parsed.lines) + "\n"


class CheckpointSerializerTests(unittest.TestCase):
    def test_restricted_serializer_roundtrips_complete_json_graph_snapshot(self):
        state = RISEGraphState.initialize(
            seed(),
            BudgetTracker(max_steps=5, max_searches=1).snapshot(),
            run_id=str(uuid4()),
        )
        serializer = JsonPlusSerializer(allowed_msgpack_modules=[])
        restored = serializer.loads_typed(serializer.dumps_typed(state.to_payload()))
        self.assertEqual(RISEGraphState.from_payload(restored), state)


class PaperInputStoreTests(unittest.TestCase):
    def test_content_addressed_write_preserves_bytes_and_checks_digest(self):
        raw = "# 研究标题\r\n\r\n## Method\r\n原始内容\r\n".encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            store = PaperInputStore(temporary)
            source = Path(temporary) / "source.md"
            source.write_bytes(raw)

            reference = store.persist(source)
            stored_path = Path(temporary) / reference.locator
            self.assertEqual(stored_path.read_bytes(), raw)
            self.assertEqual(reference.sha256, hashlib.sha256(raw).hexdigest())
            self.assertEqual(store.read(reference), raw)
            self.assertEqual(stored_path.stat().st_mode & 0o222, 0)
            self.assertEqual(source.read_bytes(), raw)

            stored_path.chmod(0o600)
            stored_path.write_bytes(raw + b"tampered")
            with self.assertRaisesRegex(PaperInputIntegrityError, "SHA-256 mismatch"):
                store.read(reference)

    def test_missing_input_invalid_utf8_and_path_escape_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = PaperInputStore(temporary)
            with self.assertRaisesRegex(PaperInputIntegrityError, "UTF-8"):
                store.persist(b"\xff")

            reference = store.persist("# A paper")
            (Path(temporary) / reference.locator).unlink()
            with self.assertRaisesRegex(PaperInputIntegrityError, "missing"):
                store.read(reference)

            outside = PaperInputStore(temporary).persist("# Other")
            escaped = outside.model_copy(update={"locator": "../outside.md"})
            with self.assertRaisesRegex(PaperInputIntegrityError, "relative path"):
                store.read(escaped)


class PersistentGraphRunnerTests(unittest.TestCase):
    def test_interrupted_controller_resumes_from_bootstrap_checkpoint_and_terminal_is_idempotent(self):
        modules_bundle = modules()
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            store = PaperInputStore(temporary)
            first_runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=store,
            )
            with patch.object(
                modules_bundle.gap_discovery,
                "discover",
                wraps=modules_bundle.gap_discovery.discover,
            ) as bootstrap, patch.object(
                first_runner.adapter,
                "decide",
                side_effect=KeyboardInterrupt("synthetic process interruption"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first_runner.start_persistent(
                        seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                    )

                inspection = first_runner.build_graph(checkpointer=checkpointer)
                snapshot = inspection.get_state(first_runner._thread_config(thread_id))
                saved = RISEGraphState.from_payload(snapshot.values)
                self.assertTrue(saved.bootstrap_complete)
                self.assertEqual(saved.run_id, thread_id)
                self.assertEqual(snapshot.next, ("controller",))
                self.assertEqual(bootstrap.call_count, 1)

                resumed_runner = LangGraphTrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=99, max_searches=99),
                    checkpointer=checkpointer,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(
                    modules_bundle.controller,
                    "decide",
                    wraps=modules_bundle.controller.decide,
                ) as decide:
                    result = resumed_runner.resume_persistent(thread_id=thread_id)
                    calls_after_resume = decide.call_count
                    repeated = resumed_runner.resume_persistent(thread_id=thread_id)

                self.assertEqual(bootstrap.call_count, 1)
                self.assertGreater(calls_after_resume, 0)
                self.assertEqual(decide.call_count, calls_after_resume)
                self.assertEqual(result, repeated)
                self.assertEqual(result.run_id, thread_id)
                self.assertEqual(result.rise_status, "committed")
                self.assertEqual(result.budget.max_steps, 5)
                self.assertEqual(result.budget.steps_used, 5)
                self.assertEqual(len(result.replay_records), result.budget.steps_used + 1)
                serializer = JsonPlusSerializer(allowed_msgpack_modules=[])
                serialized = serializer.dumps_typed(result.to_payload())
                self.assertEqual(
                    RISEGraphState.from_payload(serializer.loads_typed(serialized)),
                    result,
                )
                with self.assertRaises(ExistingThreadError):
                    resumed_runner.start_persistent(
                        seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                    )

    def test_saved_controller_decision_resumes_at_selected_action(self):
        modules_bundle = modules()
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=PaperInputStore(temporary),
            )
            with patch.object(
                modules_bundle.controller,
                "decide",
                wraps=modules_bundle.controller.decide,
            ) as decide, patch.object(
                runner.adapter,
                "execute_selected_action",
                side_effect=KeyboardInterrupt("interrupted before action execution"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.start_persistent(
                        seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                    )

                snapshot = runner.build_graph(checkpointer=checkpointer).get_state(
                    runner._thread_config(thread_id)
                )
                saved = RISEGraphState.from_payload(snapshot.values)
                self.assertTrue(saved.decision_pending)
                self.assertEqual(
                    saved.controller_routing.decision.action, "VERIFY_RELATED_WORK"
                )
                self.assertEqual(snapshot.next, ("verify_related_work",))
                self.assertEqual(decide.call_count, 1)

                resumed = LangGraphTrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=checkpointer,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)
                self.assertEqual(resumed.rise_status, "committed")
                self.assertEqual(decide.call_count, len(resumed.replay_records) - 1)
                self.assertEqual(resumed.budget.steps_used, 5)

    def test_completed_action_checkpoint_is_not_replayed_on_resume(self):
        modules_bundle = modules()
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=PaperInputStore(temporary),
            )
            saved_decide = runner.adapter.decide
            adapter_decision_calls = 0

            def interrupt_at_next_controller(state):
                nonlocal adapter_decision_calls
                adapter_decision_calls += 1
                if adapter_decision_calls == 2:
                    raise KeyboardInterrupt("interrupted after first action checkpoint")
                return saved_decide(state)

            with patch.object(
                modules_bundle.controller,
                "decide",
                wraps=modules_bundle.controller.decide,
            ) as decide, patch.object(
                runner.adapter, "decide", side_effect=interrupt_at_next_controller
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.start_persistent(
                        seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                    )
                snapshot = runner.build_graph(checkpointer=checkpointer).get_state(
                    runner._thread_config(thread_id)
                )
                saved_state = RISEGraphState.from_payload(snapshot.values)
                self.assertEqual(snapshot.next, ("controller",))
                self.assertEqual(saved_state.budget.steps_used, 1)
                self.assertEqual(
                    [record.action for record in saved_state.replay_records if record.action],
                    ["VERIFY_RELATED_WORK"],
                )

                resumed = LangGraphTrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=checkpointer,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)
                self.assertEqual(resumed.rise_status, "committed")
                self.assertEqual(decide.call_count, len(resumed.replay_records) - 1)
                self.assertEqual(resumed.budget.steps_used, 5)
                self.assertEqual(len(resumed.replay_records), 6)

    def test_missing_checkpoint_and_invalid_thread_id_do_not_start_a_run(self):
        runner = LangGraphTrajectoryRunner(
            modules(),
            budget=BudgetTracker(max_steps=2, max_searches=1),
            checkpointer=InMemorySaver(),
            paper_store=PaperInputStore(tempfile.gettempdir()),
        )
        with self.assertRaises(ValueError):
            runner.resume_persistent(thread_id="not-a-uuid")
        with self.assertRaises(MissingCheckpointError):
            runner.resume_persistent(thread_id=str(uuid4()))

    def test_scientific_action_failure_is_a_saved_terminal_and_is_not_retried(self):
        modules_bundle = modules()
        route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="SEEK_INSPIRATION",
                reason="Deterministic failure path for recovery test.",
            ),
            allowed_actions=["SEEK_INSPIRATION"],
            used_llm=False,
        )
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=3, max_searches=2),
                checkpointer=checkpointer,
                paper_store=PaperInputStore(temporary),
            )
            with patch.object(
                modules_bundle.controller, "decide", return_value=route
            ) as decide, patch.object(
                modules_bundle.inspiration_retrieval,
                "retrieve",
                side_effect=RuntimeError("synthetic retrieval failure"),
            ) as retrieve:
                failed = runner.start_persistent(
                    seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                )
                resumed = runner.resume_persistent(thread_id=thread_id)

            self.assertEqual(failed.rise_status, "failed")
            self.assertEqual(failed, resumed)
            self.assertEqual(failed.error_type, "RuntimeError")
            self.assertEqual(failed.budget.steps_used, 1)
            self.assertEqual(decide.call_count, 1)
            self.assertEqual(retrieve.call_count, 1)

    def test_finalization_failure_is_a_saved_terminal_and_is_not_retried(self):
        modules_bundle = modules()
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=PaperInputStore(temporary),
            )
            with patch.object(
                modules_bundle.finalizer,
                "finalize",
                side_effect=RuntimeError("synthetic finalizer failure"),
            ) as finalize:
                failed = runner.start_persistent(
                    seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                )
                resumed = runner.resume_persistent(thread_id=thread_id)

            self.assertEqual(failed.rise_status, "finalization_failed")
            self.assertEqual(failed, resumed)
            self.assertIsNotNone(failed.commit_snapshot)
            self.assertIsNone(failed.final_proposal)
            self.assertEqual(failed.budget.steps_used, 5)
            self.assertEqual(finalize.call_count, 1)

    def test_paper_integrity_is_checked_before_resuming_nonterminal_state(self):
        modules_bundle = modules()
        checkpointer = InMemorySaver()
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            store = PaperInputStore(temporary)
            runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=store,
            )
            with patch.object(runner.adapter, "decide", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.start_persistent(
                        seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                    )
            snapshot = runner.build_graph(checkpointer=checkpointer).get_state(
                runner._thread_config(thread_id)
            )
            state = RISEGraphState.from_payload(snapshot.values)
            assert state.paper_input is not None
            stored_path = Path(temporary) / state.paper_input.locator
            stored_path.chmod(0o600)
            stored_path.write_text("tampered", encoding="utf-8")

            resumed = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=5, max_searches=1),
                checkpointer=checkpointer,
                paper_store=PaperInputStore(temporary),
            )
            with patch.object(modules_bundle.controller, "decide") as decide:
                with self.assertRaisesRegex(PaperInputIntegrityError, "SHA-256 mismatch"):
                    resumed.resume_persistent(thread_id=thread_id)
                decide.assert_not_called()


POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")


@unittest.skipUnless(
    POSTGRES_URL,
    "set RISE_TEST_POSTGRES_URL to run real PostgreSQL checkpoint integration",
)
class PostgresCheckpointIntegrationTests(unittest.TestCase):
    def test_rebuild_runner_resume_across_saver_lifecycles_and_terminal_repeat(self):
        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            first_modules = modules()
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                runner = LangGraphTrajectoryRunner(
                    first_modules,
                    budget=BudgetTracker(max_steps=5, max_searches=1),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(runner.adapter, "decide", side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.start_persistent(
                            seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                        )

            second_modules = modules()
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                resumed = LangGraphTrajectoryRunner(
                    second_modules,
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)
                self.assertEqual(resumed.rise_status, "committed")
                self.assertEqual(resumed.run_id, thread_id)

            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                repeated = LangGraphTrajectoryRunner(
                    modules(),
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)
                self.assertEqual(repeated, resumed)


if __name__ == "__main__":
    unittest.main()
