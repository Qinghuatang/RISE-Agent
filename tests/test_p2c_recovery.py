from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from research_agent.controller.budget import BudgetTracker
from research_agent.models.controller import ControllerDecision, ControllerRoutingResult
from research_agent.llm.fake import FakeLLMClient
from research_agent.runtime.graph_runner import (
    LangGraphTrajectoryRunner,
    PersistentExecutionError,
)
from research_agent.runtime.graph_state import PaperInputReference, RISEGraphState
from research_agent.runtime.paper_store import PaperInputStore

from test_m11 import modules, reading, seed, solution_verification


PAPER_MARKDOWN = "\n".join(reading().parsed.lines) + "\n"


class _ExistingCheckpoint:
    def get_tuple(self, config):
        return object()


class ResumePreflightTests(unittest.TestCase):
    def _runner_with_snapshot(self, snapshot):
        runner = LangGraphTrajectoryRunner(
            modules(),
            budget=BudgetTracker(max_steps=5, max_searches=1),
            checkpointer=_ExistingCheckpoint(),
            paper_store=PaperInputStore(tempfile.gettempdir()),
        )
        fake_graph = SimpleNamespace(get_state=lambda config: snapshot)
        return runner, fake_graph

    def _base_snapshot(self):
        thread_id = str(uuid4())
        state = RISEGraphState.initialize(
            seed(),
            BudgetTracker(max_steps=5, max_searches=1).snapshot(),
            run_id=thread_id,
            paper_input=PaperInputReference(
                locator="papers/" + "a" * 64 + ".md", sha256="a" * 64
            ),
        )
        return thread_id, state

    def test_schema_mismatch_is_rejected_before_state_reconstruction(self):
        thread_id, state = self._base_snapshot()
        values = state.to_payload()
        values["schema_version"] = 77
        snapshot = SimpleNamespace(values=values, next=("bootstrap",))
        runner, graph = self._runner_with_snapshot(snapshot)
        with patch.object(runner, "build_graph", return_value=graph):
            with self.assertRaisesRegex(PersistentExecutionError, "schema_version"):
                runner.resume_persistent(thread_id=thread_id)

    def test_nonterminal_checkpoint_without_pending_node_is_rejected(self):
        thread_id, state = self._base_snapshot()
        snapshot = SimpleNamespace(values=state.to_payload(), next=())
        runner, graph = self._runner_with_snapshot(snapshot)
        with patch.object(runner, "build_graph", return_value=graph):
            with self.assertRaisesRegex(PersistentExecutionError, "no pending"):
                runner.resume_persistent(thread_id=thread_id)

    def test_pending_node_must_match_saved_controller_decision(self):
        thread_id, state = self._base_snapshot()
        running = state.evolve(rise_status="running", bootstrap_complete=True)
        route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="VERIFY_RELATED_WORK", reason="saved deterministic route"
            ),
            allowed_actions=["VERIFY_RELATED_WORK"],
            used_llm=False,
        )
        pending = running.evolve(controller_routing=route, decision_pending=True)
        snapshot = SimpleNamespace(
            values=pending.to_payload(), next=("solution_ideation",)
        )
        runner, graph = self._runner_with_snapshot(snapshot)
        with patch.object(runner, "build_graph", return_value=graph):
            with self.assertRaisesRegex(PersistentExecutionError, "does not match"):
                runner.resume_persistent(thread_id=thread_id)


POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")


@unittest.skipUnless(
    POSTGRES_URL,
    "set RISE_TEST_POSTGRES_URL to run real PostgreSQL fault-injection tests",
)
class PostgresFaultInjectionTests(unittest.TestCase):
    def _run_child(
        self,
        thread_id: str,
        paper_root: str,
        boundary: str,
        phase: str,
        event_log: str,
    ) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        child_code = r'''
import json
import os
import sys
from unittest.mock import patch
from research_agent.controller.budget import BudgetTracker
from research_agent.llm.fake import FakeLLMClient
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.graph_state import RISEGraphState
from research_agent.runtime.paper_store import PaperInputStore
from test_m11 import modules, reading, seed, solution_verification

thread_id, paper_root, boundary, phase, event_log = sys.argv[1:6]
module_bundle = modules()
if phase == "resume" and boundary == "action_checkpoint":
    # The saved checkpoint already includes the first gap verification.
    module_bundle.related_work_verification.llm_client = FakeLLMClient([solution_verification()])
runner_budget = BudgetTracker(max_steps=5, max_searches=1)
paper_markdown = "\n".join(reading().parsed.lines) + "\n"

def record(value):
    with open(event_log, "a", encoding="utf-8") as stream:
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())

def trace_related_external_calls():
    verification = module_bundle.related_work_verification
    client = verification.llm_client
    original_llm = client.complete_structured
    def record_related_llm(request, output_schema):
        record("external:related_llm")
        return original_llm(request, output_schema)
    client.complete_structured = record_related_llm

    provider = verification.related_paper_provider
    original_retrieval = provider.get_related_paper_pool
    def record_related_retrieval(seed_context, *, limit=20):
        record("external:related_retrieval")
        return original_retrieval(seed_context, limit=limit)
    provider.get_related_paper_pool = record_related_retrieval

def trace_finalizer_llm():
    client = module_bundle.finalizer.llm_client
    original_llm = client.complete_structured
    def record_finalizer_llm(request, output_schema):
        record("external:finalizer_llm")
        return original_llm(request, output_schema)
    client.complete_structured = record_finalizer_llm

with open_postgres_checkpointer(os.environ["RISE_TEST_POSTGRES_URL"]) as saver:
    runner = LangGraphTrajectoryRunner(
        module_bundle,
        budget=runner_budget,
        checkpointer=saver,
        paper_store=PaperInputStore(paper_root),
    )
    if phase == "crash":
        if boundary == "bootstrap_checkpoint":
            with patch.object(runner.adapter, "decide", side_effect=lambda state: os._exit(86)):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        elif boundary == "controller_checkpoint":
            with patch.object(runner.adapter, "execute_selected_action", side_effect=lambda state: os._exit(86)):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        elif boundary == "action_checkpoint":
            original = runner.adapter.decide
            calls = 0
            def stop_before_second_controller(state):
                global calls
                calls += 1
                if calls == 2:
                    os._exit(86)
                return original(state)
            with patch.object(runner.adapter, "decide", side_effect=stop_before_second_controller):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        elif boundary == "action_running":
            trace_related_external_calls()
            original = runner.adapter.execute_selected_action
            def stop_after_action_body(state):
                result = original(state)
                action = state.controller_routing.decision.action
                record("action:" + action)
                os._exit(86)
                return result
            with patch.object(runner.adapter, "execute_selected_action", side_effect=stop_after_action_body):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        elif boundary == "commit_checkpoint":
            with patch.object(module_bundle.finalizer, "finalize", side_effect=lambda *args: os._exit(86)):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        elif boundary == "finalizer_running":
            trace_finalizer_llm()
            original = module_bundle.finalizer.finalize
            def stop_after_finalizer_body(*args, **kwargs):
                result = original(*args, **kwargs)
                record("finalizer")
                os._exit(86)
                return result
            with patch.object(module_bundle.finalizer, "finalize", side_effect=stop_after_finalizer_body):
                runner.start_persistent(seed(), thread_id=thread_id, markdown=paper_markdown)
        raise SystemExit("fault injection point was not reached")

    if phase == "resume":
        if boundary == "action_running":
            trace_related_external_calls()
            original = runner.adapter.execute_selected_action
            def record_action(state):
                result = original(state)
                record("action:" + state.controller_routing.decision.action)
                return result
            runner.adapter.execute_selected_action = record_action
        elif boundary == "finalizer_running":
            trace_finalizer_llm()
            original = module_bundle.finalizer.finalize
            def record_finalizer(*args, **kwargs):
                result = original(*args, **kwargs)
                record("finalizer")
                return result
            module_bundle.finalizer.finalize = record_finalizer
        state = runner.resume_persistent(thread_id=thread_id)
        print("RISE_RESULT=" + json.dumps(state.to_payload(), ensure_ascii=False))
        raise SystemExit(0)
raise SystemExit("unknown phase")
'''
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root / "src"), str(repo_root / "tests"), child_env.get("PYTHONPATH", "")]
        )
        return subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                thread_id,
                paper_root,
                boundary,
                phase,
                event_log,
            ],
            cwd=repo_root,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    @staticmethod
    def _normalized(payload: dict) -> dict:
        payload = json.loads(json.dumps(payload))
        payload.pop("run_id", None)
        payload.pop("paper_input", None)
        for event in payload["trajectory_events"]:
            event.pop("timestamp", None)
        return payload

    def test_abrupt_process_exit_and_restart_at_all_scientific_boundaries(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer

        boundaries = {
            "bootstrap_checkpoint": ("controller", "bootstrap"),
            "controller_checkpoint": ("verify_related_work", "decision"),
            "action_checkpoint": ("controller", "action"),
            "action_running": ("verify_related_work", "uncommitted_action"),
            "commit_checkpoint": ("finalizer", "commit"),
            "finalizer_running": ("finalizer", "commit"),
        }
        baseline = LangGraphTrajectoryRunner(
            modules(), budget=BudgetTracker(max_steps=5, max_searches=1)
        ).run(seed(), full_paper_reading_context=reading())

        with tempfile.TemporaryDirectory() as temporary:
            for boundary, (expected_next, saved_kind) in boundaries.items():
                with self.subTest(boundary=boundary):
                    thread_id = str(uuid4())
                    event_log = str(Path(temporary) / f"{boundary}.jsonl")
                    crashed = self._run_child(
                        thread_id, temporary, boundary, "crash", event_log
                    )
                    self.assertEqual(
                        crashed.returncode,
                        86,
                        msg=f"child did not stop at {boundary}: {crashed.stderr}",
                    )

                    with open_postgres_checkpointer(POSTGRES_URL) as saver:
                        inspect_runner = LangGraphTrajectoryRunner(
                            modules(),
                            budget=BudgetTracker(max_steps=1, max_searches=0),
                            checkpointer=saver,
                            paper_store=PaperInputStore(temporary),
                        )
                        snapshot = inspect_runner.build_graph(checkpointer=saver).get_state(
                            inspect_runner._thread_config(thread_id)
                        )
                        saved = RISEGraphState.from_payload(snapshot.values)
                        self.assertEqual(snapshot.next, (expected_next,))
                        self.assertEqual(saved.run_id, thread_id)
                        if saved_kind == "bootstrap":
                            self.assertTrue(saved.bootstrap_complete)
                            self.assertFalse(saved.decision_pending)
                        elif saved_kind == "decision":
                            self.assertTrue(saved.decision_pending)
                            self.assertEqual(saved.budget.steps_used, 0)
                        elif saved_kind == "action":
                            self.assertEqual(saved.budget.steps_used, 1)
                            self.assertEqual(
                                [r.action for r in saved.replay_records if r.action],
                                ["VERIFY_RELATED_WORK"],
                            )
                        elif saved_kind == "uncommitted_action":
                            self.assertTrue(saved.decision_pending)
                            self.assertEqual(saved.budget.steps_used, 0)
                            self.assertEqual(
                                [r.action for r in saved.replay_records if r.action], []
                            )
                        elif saved_kind == "commit":
                            self.assertIsNotNone(saved.commit_snapshot)
                            self.assertEqual(saved.budget.steps_used, 5)
                            self.assertEqual(
                                [r.action for r in saved.replay_records if r.action].count("COMMIT"),
                                1,
                            )

                    resumed = self._run_child(
                        thread_id, temporary, boundary, "resume", event_log
                    )
                    self.assertEqual(
                        resumed.returncode,
                        0,
                        msg=f"resume failed at {boundary}: {resumed.stderr}\n{resumed.stdout}",
                    )
                    marker = next(
                        line for line in resumed.stdout.splitlines()
                        if line.startswith("RISE_RESULT=")
                    )
                    payload = json.loads(marker.removeprefix("RISE_RESULT="))
                    result = RISEGraphState.from_payload(payload)
                    self.assertEqual(result.rise_status, "committed")
                    self.assertEqual(result.budget.steps_used, 5)
                    self.assertEqual(result.budget.searches_used, 1)
                    self.assertEqual(self._normalized(payload), self._normalized(baseline.to_payload()))

                    if boundary == "action_running":
                        events = Path(event_log).read_text(encoding="utf-8").splitlines()
                        self.assertEqual(
                            [event for event in events if event.startswith("action:")],
                            [
                                "action:VERIFY_RELATED_WORK",
                                "action:VERIFY_RELATED_WORK",
                                "action:SOLUTION_IDEATION",
                                "action:VERIFY_RELATED_WORK",
                                "action:DESIGN_EXPERIMENT",
                                "action:COMMIT",
                            ],
                        )
                        self.assertEqual(events.count("external:related_retrieval"), 2)
                        self.assertEqual(events.count("external:related_llm"), 3)
                    if boundary == "finalizer_running":
                        events = Path(event_log).read_text(encoding="utf-8").splitlines()
                        self.assertEqual(events.count("external:finalizer_llm"), 2)
                        self.assertEqual(events.count("finalizer"), 2)

    def test_persisted_paper_missing_or_corrupt_blocks_postgres_resume(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer
        from research_agent.runtime.paper_store import PaperInputIntegrityError

        for damage in ("missing", "corrupt"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                thread_id = str(uuid4())
                store = PaperInputStore(temporary)
                with open_postgres_checkpointer(POSTGRES_URL) as saver:
                    runner = LangGraphTrajectoryRunner(
                        modules(),
                        budget=BudgetTracker(max_steps=5, max_searches=1),
                        checkpointer=saver,
                        paper_store=store,
                    )
                    with patch.object(runner.adapter, "decide", side_effect=KeyboardInterrupt("save bootstrap")):
                        with self.assertRaises(KeyboardInterrupt):
                            runner.start_persistent(
                                seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                            )
                    snapshot = runner.build_graph(checkpointer=saver).get_state(
                        runner._thread_config(thread_id)
                    )
                    saved = RISEGraphState.from_payload(snapshot.values)
                    assert saved.paper_input is not None
                    persisted_path = Path(temporary) / saved.paper_input.locator

                if damage == "missing":
                    persisted_path.unlink()
                else:
                    persisted_path.chmod(0o600)
                    persisted_path.write_text("damaged paper input", encoding="utf-8")

                with open_postgres_checkpointer(POSTGRES_URL) as saver:
                    modules_bundle = modules()
                    runner = LangGraphTrajectoryRunner(
                        modules_bundle,
                        budget=BudgetTracker(max_steps=5, max_searches=1),
                        checkpointer=saver,
                        paper_store=PaperInputStore(temporary),
                    )
                    with patch.object(modules_bundle.controller, "decide") as decide:
                        with self.assertRaises(PaperInputIntegrityError):
                            runner.resume_persistent(thread_id=thread_id)
                        decide.assert_not_called()

    def test_saved_scientific_failure_and_finalization_failure_are_not_retried(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer

        # Scientific Action failure remains a persisted RISE terminal outcome.
        modules_bundle = modules()
        action_route = ControllerRoutingResult(
            status="decision",
            decision=ControllerDecision(
                action="SEEK_INSPIRATION", reason="deterministic terminal failure"
            ),
            allowed_actions=["SEEK_INSPIRATION"],
            used_llm=False,
        )
        action_thread = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                runner = LangGraphTrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=3, max_searches=2),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(modules_bundle.controller, "decide", return_value=action_route), patch.object(
                    modules_bundle.inspiration_retrieval,
                    "retrieve",
                    side_effect=RuntimeError("synthetic retrieval failure"),
                ) as retrieve:
                    failed = runner.start_persistent(
                        seed(), thread_id=action_thread, markdown=PAPER_MARKDOWN
                    )
                    repeated = runner.resume_persistent(thread_id=action_thread)
                self.assertEqual(failed.rise_status, "failed")
                self.assertEqual(repeated, failed)
                self.assertEqual(failed.error_type, "RuntimeError")
                self.assertEqual(retrieve.call_count, 1)

        # Finalizer failure preserves the committed snapshot and also terminates.
        modules_bundle = modules()
        finalizer_thread = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                runner = LangGraphTrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=5, max_searches=1),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(
                    modules_bundle.finalizer,
                    "finalize",
                    side_effect=RuntimeError("synthetic finalizer failure"),
                ) as finalize:
                    failed = runner.start_persistent(
                        seed(), thread_id=finalizer_thread, markdown=PAPER_MARKDOWN
                    )
                    repeated = runner.resume_persistent(thread_id=finalizer_thread)
                self.assertEqual(failed.rise_status, "finalization_failed")
                self.assertEqual(repeated, failed)
                self.assertIsNotNone(failed.commit_snapshot)
                self.assertIsNone(failed.final_proposal)
                self.assertEqual(finalize.call_count, 1)

    def test_postgres_checkpoint_write_failure_propagates_as_infrastructure_error(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer

        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                runner = LangGraphTrajectoryRunner(
                    modules(),
                    budget=BudgetTracker(max_steps=5, max_searches=1),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(saver, "put", side_effect=OSError("injected checkpoint write outage")):
                    with self.assertRaisesRegex(OSError, "checkpoint write outage"):
                        runner.start_persistent(
                            seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                        )

                # The write failed before any RISE terminal result could be saved.
                self.assertIsNone(saver.get_tuple(runner._thread_config(thread_id)))

    def test_checkpoint_write_failure_resumes_durable_action_result(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer

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
                original_execute = runner.adapter.execute_selected_action
                original_put = saver.put
                action_returned = False

                def execute_then_fail_checkpoint(state):
                    nonlocal action_returned
                    result = original_execute(state)
                    if state.controller_routing.decision.action == "VERIFY_RELATED_WORK":
                        action_returned = True
                    return result

                def fail_after_action_checkpoint(config, checkpoint, metadata, new_versions):
                    nonlocal action_returned
                    if action_returned:
                        action_returned = False
                        raise OSError("injected Action checkpoint write failure")
                    return original_put(config, checkpoint, metadata, new_versions)

                with patch.object(
                    runner.adapter,
                    "execute_selected_action",
                    side_effect=execute_then_fail_checkpoint,
                ), patch.object(saver, "put", side_effect=fail_after_action_checkpoint):
                    with self.assertRaisesRegex(OSError, "Action checkpoint write failure"):
                        runner.start_persistent(
                            seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                        )

            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                inspection = LangGraphTrajectoryRunner(
                    modules(),
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                snapshot = inspection.build_graph(checkpointer=saver).get_state(
                    inspection._thread_config(thread_id)
                )
                committed = RISEGraphState.from_payload(snapshot.values)
                self.assertEqual(snapshot.next, ())
                self.assertEqual(len(snapshot.tasks), 1)
                self.assertEqual(snapshot.tasks[0].name, "verify_related_work")
                self.assertIsNotNone(snapshot.tasks[0].result)
                self.assertFalse(committed.decision_pending)
                self.assertEqual(committed.budget.steps_used, 1)
                self.assertEqual(
                    [record.action for record in committed.replay_records if record.action],
                    ["VERIFY_RELATED_WORK"],
                )
                self.assertEqual(
                    len(first_modules.related_work_verification.llm_client.calls), 1
                )
                self.assertEqual(
                    len(first_modules.related_work_verification.related_paper_provider.calls),
                    1,
                )
                resumed_modules = modules()
                resumed_modules.related_work_verification.llm_client = FakeLLMClient(
                    [solution_verification()]
                )
                recovered = LangGraphTrajectoryRunner(
                    resumed_modules,
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)

            baseline = LangGraphTrajectoryRunner(
                modules(), budget=BudgetTracker(max_steps=5, max_searches=1)
            ).run(seed(), full_paper_reading_context=reading())
            self.assertEqual(recovered.rise_status, "committed")
            self.assertEqual(recovered.budget.steps_used, 5)
            self.assertEqual(
                self._normalized(recovered.to_payload()),
                self._normalized(baseline.to_payload()),
            )

    def test_closed_postgres_connection_fails_resume_then_new_connection_recovers(self):
        from research_agent.runtime.checkpointer import open_postgres_checkpointer

        thread_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            with open_postgres_checkpointer(POSTGRES_URL) as saver:
                runner = LangGraphTrajectoryRunner(
                    modules(),
                    budget=BudgetTracker(max_steps=5, max_searches=1),
                    checkpointer=saver,
                    paper_store=PaperInputStore(temporary),
                )
                with patch.object(runner.adapter, "decide", side_effect=KeyboardInterrupt("save bootstrap")):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.start_persistent(
                            seed(), thread_id=thread_id, markdown=PAPER_MARKDOWN
                        )
                saver.conn.close()
                from psycopg import Error as PsycopgError

                with self.assertRaises(PsycopgError):
                    runner.resume_persistent(thread_id=thread_id)

            with open_postgres_checkpointer(POSTGRES_URL) as recovered_saver:
                recovered = LangGraphTrajectoryRunner(
                    modules(),
                    budget=BudgetTracker(max_steps=1, max_searches=0),
                    checkpointer=recovered_saver,
                    paper_store=PaperInputStore(temporary),
                ).resume_persistent(thread_id=thread_id)
            self.assertEqual(recovered.rise_status, "committed")
            self.assertEqual(recovered.run_id, thread_id)


if __name__ == "__main__":
    unittest.main()
