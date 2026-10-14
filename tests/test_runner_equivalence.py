from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

from research_agent.controller.budget import BudgetTracker
from research_agent.controller.scientific_controller import ScientificController
from research_agent.errors import RunnerInvariantError
from research_agent.llm.fake import FakeLLMClient
from research_agent.modules.commit import CommitModule
from research_agent.modules.experiment_design import ExperimentDesignModule
from research_agent.modules.finalizer import FinalizerModule
from research_agent.modules.gap_discovery import GapDiscoveryModule
from research_agent.modules.inspiration_retrieval import InspirationRetrievalModule
from research_agent.modules.related_work_verification import RelatedWorkVerificationModule
from research_agent.modules.solution_ideation import SolutionIdeationModule
from research_agent.modules.targeted_revision import TargetedRevisionModule
from research_agent.retrieval.fake import FakeInspirationSearchProvider, FakeRelatedPaperProvider
from research_agent.runner import TrajectoryModules, TrajectoryRunner
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.serialization.trajectory import load_replay_journal
from research_agent.trajectory.context import TrajectoryContext

from test_m11 import (
    experiment_design,
    finalizer_draft,
    gap_draft,
    gap_evaluation,
    gap_verification,
    inspiration_entities,
    inspiration_result,
    modules,
    paper,
    reading,
    revision_draft,
    seed,
    solution_development,
    solution_form,
    solution_partial_verification,
    solution_verification,
    two_gap_draft,
)


ModuleFactory = Callable[[], TrajectoryModules]


def _gap_route():
    from research_agent.models.controller import ControllerDecision, ControllerRoutingResult

    return ControllerRoutingResult(
        status="decision",
        decision=ControllerDecision(
            action="VERIFY_RELATED_WORK",
            reason="Exercise the saved verification route.",
        ),
        allowed_actions=["VERIFY_RELATED_WORK"],
        used_llm=False,
    )


def _inspiration_route():
    from research_agent.models.controller import ControllerDecision, ControllerRoutingResult

    return ControllerRoutingResult(
        status="decision",
        decision=ControllerDecision(
            action="SEEK_INSPIRATION",
            reason="Exercise the saved inspiration route.",
        ),
        allowed_actions=["SEEK_INSPIRATION"],
        used_llm=False,
    )


def _gap_path_modules(*, two_gaps: bool, verification_payloads: list[dict]) -> TrajectoryModules:
    discovery_responses: list[dict] = [
        two_gap_draft() if two_gaps else gap_draft(),
        gap_evaluation(count=2 if two_gaps else 1),
        {
            "refined_statement": "A narrower residual mechanism question.",
            "refinement_reason": "The verification covered one aspect.",
        },
    ]
    return TrajectoryModules(
        gap_discovery=GapDiscoveryModule(FakeLLMClient(discovery_responses)),
        solution_ideation=SolutionIdeationModule(FakeLLMClient([])),
        related_work_verification=RelatedWorkVerificationModule(
            FakeLLMClient(verification_payloads), FakeRelatedPaperProvider([paper()])
        ),
        inspiration_retrieval=InspirationRetrievalModule(
            FakeLLMClient([]), FakeInspirationSearchProvider([])
        ),
        targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
        experiment_design=ExperimentDesignModule(FakeLLMClient([])),
        controller=ScientificController(FakeLLMClient([])),
        commit=CommitModule(),
        finalizer=FinalizerModule(FakeLLMClient([])),
    )


def _inspiration_revision_modules() -> TrajectoryModules:
    inspiration_paper = paper().model_copy(update={"paper_id": "inspiration-1"})
    return TrajectoryModules(
        gap_discovery=GapDiscoveryModule(FakeLLMClient([gap_draft(), gap_evaluation()])),
        solution_ideation=SolutionIdeationModule(
            FakeLLMClient([solution_form(), solution_development()])
        ),
        related_work_verification=RelatedWorkVerificationModule(
            FakeLLMClient(
                [gap_verification(), solution_partial_verification(), solution_verification()]
            ),
            FakeRelatedPaperProvider([paper()]),
        ),
        inspiration_retrieval=InspirationRetrievalModule(
            FakeLLMClient([inspiration_entities(), inspiration_result()]),
            FakeInspirationSearchProvider([inspiration_paper]),
        ),
        targeted_revision=TargetedRevisionModule(FakeLLMClient([revision_draft()])),
        experiment_design=ExperimentDesignModule(FakeLLMClient([experiment_design()])),
        controller=ScientificController(
            FakeLLMClient(
                [{"action": "SEEK_INSPIRATION", "reason": "Use a transferable mechanism."}]
            )
        ),
        commit=CommitModule(),
        finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
    )


def _five_gap_commit_modules() -> TrajectoryModules:
    gap_template = gap_draft()["gaps"][0]
    candidates = []
    for index in range(1, 6):
        candidate = dict(gap_template)
        candidate["statement"] = (
            f"The supplied method leaves a distinct mechanism question {index} unresolved."
        )
        candidates.append(candidate)
    multi_gap_draft = {"gaps": candidates}
    resolved = {**gap_verification(), "status": "resolved", "remaining_gap": None}
    return TrajectoryModules(
        gap_discovery=GapDiscoveryModule(
            FakeLLMClient([multi_gap_draft, gap_evaluation(count=5)])
        ),
        solution_ideation=SolutionIdeationModule(
            FakeLLMClient([solution_form(), solution_development()])
        ),
        related_work_verification=RelatedWorkVerificationModule(
            FakeLLMClient(
                [resolved, resolved, resolved, resolved, gap_verification(), solution_verification()]
            ),
            FakeRelatedPaperProvider([paper()]),
        ),
        inspiration_retrieval=InspirationRetrievalModule(
            FakeLLMClient([]), FakeInspirationSearchProvider([])
        ),
        targeted_revision=TargetedRevisionModule(FakeLLMClient([])),
        experiment_design=ExperimentDesignModule(FakeLLMClient([experiment_design()])),
        controller=ScientificController(
            FakeLLMClient(
                [{"action": "DESIGN_EXPERIMENT", "reason": "Complete the validated proposal."}]
            )
        ),
        commit=CommitModule(),
        finalizer=FinalizerModule(FakeLLMClient([finalizer_draft()])),
    )


def _bootstrap_failure_modules() -> TrajectoryModules:
    modules_bundle = modules()
    modules_bundle.gap_discovery.llm_client = FakeLLMClient(
        [RuntimeError("synthetic bootstrap failure")]
    )
    return modules_bundle


def _events_without_timestamps(events: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in event.model_dump(mode="json").items()
            if key != "timestamp"
        }
        for event in events
    ]


def _module_observations(modules_bundle: TrajectoryModules) -> dict[str, Any]:
    component_names = (
        "gap_discovery",
        "solution_ideation",
        "related_work_verification",
        "inspiration_retrieval",
        "targeted_revision",
        "experiment_design",
        "controller",
        "commit",
        "finalizer",
    )
    call_counts = {
        name: getattr(getattr(modules_bundle, name), "call_count", None)
        for name in component_names
    }
    llm_requests = {}
    for name in component_names:
        client = getattr(getattr(modules_bundle, name), "llm_client", None)
        requests = getattr(client, "calls", None)
        if requests is not None:
            llm_requests[name] = [
                request.model_dump(mode="json")
                if hasattr(request, "model_dump")
                else request
                for request in requests
            ]
    related_provider = modules_bundle.related_work_verification.related_paper_provider
    inspiration_provider = modules_bundle.inspiration_retrieval.search_provider
    return {
        "component_call_counts": call_counts,
        "llm_requests": llm_requests,
        "related_provider_calls": list(getattr(related_provider, "calls", [])),
        "inspiration_provider_queries": list(getattr(inspiration_provider, "queries", [])),
        "scientific_search_counts": {
            "related_work": modules_bundle.related_work_verification.scientific_search_count,
            "inspiration": modules_bundle.inspiration_retrieval.scientific_search_count,
        },
    }


def _capture_run(
    engine: str,
    module_factory: ModuleFactory,
    *,
    max_steps: int,
    max_searches: int,
    controller_responses: list[Any] | None = None,
    action_failure: bool = False,
    finalizer_failure: bool = False,
    commit_failure: bool = False,
) -> dict[str, Any]:
    modules_bundle = module_factory()
    controller_spy = Mock(wraps=modules_bundle.controller.decide)
    if controller_responses is not None:
        controller_spy.side_effect = list(controller_responses)
    finalizer_spy = Mock(wraps=modules_bundle.finalizer.finalize)
    if finalizer_failure:
        finalizer_spy.side_effect = RuntimeError("synthetic finalizer failure")
    commit_spy = Mock(wraps=modules_bundle.commit.commit)
    if commit_failure:
        commit_spy.side_effect = RuntimeError("synthetic commit failure")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(modules_bundle.controller, "decide", new=controller_spy)
        )
        stack.enter_context(
            patch.object(modules_bundle.finalizer, "finalize", new=finalizer_spy)
        )
        stack.enter_context(patch.object(modules_bundle.commit, "commit", new=commit_spy))
        if action_failure:
            def fail_after_one_search(*args: Any, **kwargs: Any) -> None:
                modules_bundle.inspiration_retrieval.scientific_search_count += 1
                raise RuntimeError("synthetic retrieval action failure")

            retrieval_spy = Mock(
                wraps=modules_bundle.inspiration_retrieval.retrieve,
                side_effect=fail_after_one_search,
            )
            stack.enter_context(
                patch.object(
                    modules_bundle.inspiration_retrieval,
                    "retrieve",
                    new=retrieval_spy,
                )
            )
        else:
            retrieval_spy = None

        if engine == "legacy":
            with tempfile.TemporaryDirectory() as output_dir:
                result = TrajectoryRunner(
                    modules_bundle,
                    budget=BudgetTracker(max_steps=max_steps, max_searches=max_searches),
                    output_dir=output_dir,
                ).run(seed(), full_paper_reading_context=reading())
                context = TrajectoryContext.load(seed(), output_dir)
                replay = load_replay_journal(
                    Path(output_dir, "trajectory_replay.jsonl")
                )
                manifest = json.loads(
                    Path(output_dir, "run_manifest.json").read_text(encoding="utf-8")
                )["payload"]
            output = {
                "status": result.status,
                "unresolved_issues": result.unresolved_issues,
                "research_state": result.final_state,
                "research_memory": context.get_memory(),
                "trajectory_events": context.trajectory.get_events(),
                "budget": result.budget,
                "replay_records": replay,
                "commit_snapshot": result.commit_snapshot,
                "final_proposal": result.final_proposal,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "commit_gate_passed": manifest["commit_gate_passed"],
            }
        elif engine == "graph":
            graph_runner = LangGraphTrajectoryRunner(
                modules_bundle,
                budget=BudgetTracker(max_steps=max_steps, max_searches=max_searches),
                run_id="equivalence-run",
            )
            state = graph_runner.run(seed(), full_paper_reading_context=reading())
            output = {
                "status": state.rise_status,
                "unresolved_issues": state.unresolved_issues,
                "research_state": state.research_state,
                "research_memory": state.research_memory,
                "trajectory_events": state.trajectory_events,
                "budget": state.budget,
                "replay_records": state.replay_records,
                "commit_snapshot": state.commit_snapshot,
                "final_proposal": state.final_proposal,
                "error_type": state.error_type,
                "error_message": state.error_message,
                "commit_gate_passed": (
                    state.commit_check.commit_allowed
                    if state.commit_check is not None
                    else False
                ),
                "graph_recursion_limit": graph_runner.recursion_limit,
            }
        else:
            raise ValueError(f"unknown Runner kind: {engine}")

    output["controller_invocations"] = controller_spy.call_count
    output["finalizer_invocations"] = finalizer_spy.call_count
    output["commit_invocations"] = commit_spy.call_count
    output["retrieval_invocations"] = (
        retrieval_spy.call_count if retrieval_spy is not None else None
    )
    output["module_observations"] = _module_observations(modules_bundle)
    return output


class RunnerEquivalenceTests(unittest.TestCase):
    def _compare(
        self,
        module_factory: ModuleFactory,
        *,
        max_steps: int,
        max_searches: int,
        controller_responses: list[Any] | None = None,
        action_failure: bool = False,
        finalizer_failure: bool = False,
        commit_failure: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        arguments = {
            "max_steps": max_steps,
            "max_searches": max_searches,
            "controller_responses": controller_responses,
            "action_failure": action_failure,
            "finalizer_failure": finalizer_failure,
            "commit_failure": commit_failure,
        }
        legacy = _capture_run("legacy", module_factory, **arguments)
        graph = _capture_run("graph", module_factory, **arguments)

        for key in (
            "status",
            "unresolved_issues",
            "research_state",
            "research_memory",
            "budget",
            "replay_records",
            "commit_snapshot",
            "final_proposal",
            "error_type",
            "error_message",
            "commit_gate_passed",
            "controller_invocations",
            "finalizer_invocations",
            "commit_invocations",
            "retrieval_invocations",
            "module_observations",
        ):
            with self.subTest(field=key):
                self.assertEqual(legacy[key], graph[key])
        self.assertEqual(
            _events_without_timestamps(legacy["trajectory_events"]),
            _events_without_timestamps(graph["trajectory_events"]),
        )
        return legacy, graph

    @staticmethod
    def _actions(run: dict[str, Any]) -> list[str]:
        return [record.action for record in run["replay_records"] if record.action]

    def test_commit_and_both_verification_target_types_match(self):
        legacy, graph = self._compare(
            modules, max_steps=5, max_searches=1
        )
        self.assertEqual(legacy["status"], "committed")
        self.assertEqual(self._actions(legacy), self._actions(graph))
        self.assertEqual(
            self._actions(graph),
            [
                "VERIFY_RELATED_WORK",
                "SOLUTION_IDEATION",
                "VERIFY_RELATED_WORK",
                "DESIGN_EXPERIMENT",
                "COMMIT",
            ],
        )
        verifications = graph["research_memory"].verification_memory
        self.assertEqual([item.task for item in verifications], ["VERIFY_GAP", "VERIFY_SOLUTION"])
        self.assertTrue(graph["commit_gate_passed"])
        self.assertIsNotNone(graph["commit_snapshot"])
        self.assertIsNotNone(graph["final_proposal"])
        self.assertEqual(graph["replay_records"][0].kind, "bootstrap")
        self.assertEqual(graph["replay_records"][0].budget_after.steps_used, 0)
        self.assertEqual(graph["budget"].steps_used, 5)
        self.assertEqual(graph["budget"].searches_used, 1)
        self.assertEqual(graph["controller_invocations"], 5)

    def test_gap_refine_is_internal_and_second_partial_switches_gap(self):
        partial = {
            **gap_verification(),
            "status": "partially_resolved",
            "remaining_gap": "A residual mechanism question remains.",
        }
        single_gap = lambda: _gap_path_modules(
            two_gaps=False, verification_payloads=[partial, gap_verification()]
        )
        legacy, graph = self._compare(
            single_gap, max_steps=2, max_searches=1
        )
        self.assertEqual(legacy["status"], "stopped")
        self.assertEqual(self._actions(graph), ["VERIFY_RELATED_WORK"] * 2)
        self.assertEqual(graph["budget"].steps_used, 2)
        self.assertEqual(len(graph["replay_records"]), 3)
        refine_events = [
            event for event in graph["trajectory_events"] if event.event_type == "gap_refined"
        ]
        self.assertEqual(len(refine_events), 1)
        self.assertIsNone(refine_events[0].action)

        two_gaps = lambda: _gap_path_modules(
            two_gaps=True, verification_payloads=[partial, partial]
        )
        legacy, graph = self._compare(
            two_gaps, max_steps=3, max_searches=1
        )
        self.assertEqual(legacy["status"], "stopped")
        self.assertEqual(
            self._actions(graph),
            ["VERIFY_RELATED_WORK", "VERIFY_RELATED_WORK", "SWITCH_GAP"],
        )
        self.assertEqual(graph["research_state"].active_gap.id, "G2")
        self.assertEqual(graph["budget"].steps_used, 3)
        self.assertEqual(
            sum(event.event_type == "gap_refined" for event in graph["trajectory_events"]),
            1,
        )

    def test_inspiration_flows_to_targeted_revision_with_same_controller_context(self):
        legacy, graph = self._compare(
            _inspiration_revision_modules, max_steps=8, max_searches=2
        )
        self.assertEqual(legacy["status"], "committed")
        actions = self._actions(graph)
        inspiration_index = actions.index("SEEK_INSPIRATION")
        self.assertEqual(actions[inspiration_index + 1], "TARGETED_REVISION")
        revision = graph["research_memory"].revision_memory[0]
        self.assertEqual(revision.selected_inspiration_ref, "I1:1")
        self.assertEqual(graph["budget"].steps_used, 8)
        self.assertEqual(graph["budget"].searches_used, 2)
        self.assertEqual(graph["controller_invocations"], len(actions))

    def test_long_legal_gap_switch_path_exceeds_default_graph_recursion_limit(self):
        legacy, graph = self._compare(
            _five_gap_commit_modules, max_steps=20, max_searches=10
        )
        actions = self._actions(graph)
        self.assertEqual(legacy["status"], "committed")
        self.assertEqual(len(actions), 13)
        self.assertEqual(
            actions[:8],
            [
                "VERIFY_RELATED_WORK",
                "SWITCH_GAP",
                "VERIFY_RELATED_WORK",
                "SWITCH_GAP",
                "VERIFY_RELATED_WORK",
                "SWITCH_GAP",
                "VERIFY_RELATED_WORK",
                "SWITCH_GAP",
            ],
        )
        self.assertEqual(
            actions[8:],
            [
                "VERIFY_RELATED_WORK",
                "SOLUTION_IDEATION",
                "VERIFY_RELATED_WORK",
                "DESIGN_EXPERIMENT",
                "COMMIT",
            ],
        )
        graph_supersteps = (
            1
            + graph["controller_invocations"]
            + len(graph["replay_records"])
            - 1
            + graph["finalizer_invocations"]
        )
        self.assertGreater(graph_supersteps, 25)
        self.assertGreaterEqual(graph["graph_recursion_limit"], graph_supersteps)

    def test_action_and_search_budget_exhaustion_match_controller_stop(self):
        legacy, graph = self._compare(
            modules, max_steps=1, max_searches=1
        )
        self.assertEqual(legacy["status"], "stopped")
        self.assertEqual(graph["budget"].steps_used, 1)
        self.assertEqual(graph["finalizer_invocations"], 0)

        legacy, graph = self._compare(
            modules, max_steps=5, max_searches=0
        )
        self.assertEqual(legacy["status"], "stopped")
        self.assertEqual(graph["budget"].steps_used, 0)
        self.assertEqual(graph["budget"].searches_used, 0)
        self.assertEqual(self._actions(graph), [])
        self.assertTrue(
            any(event.event_type == "controller_stopped" for event in graph["trajectory_events"])
        )

    def test_bootstrap_and_controller_errors_match_without_action_retries(self):
        legacy, graph = self._compare(
            _bootstrap_failure_modules, max_steps=5, max_searches=1
        )
        self.assertEqual(legacy["status"], "failed")
        self.assertEqual(graph["error_type"], "RuntimeError")
        self.assertEqual(graph["budget"].steps_used, 0)
        self.assertEqual(graph["replay_records"], [])
        self.assertEqual(graph["controller_invocations"], 0)

        legacy, graph = self._compare(
            modules,
            max_steps=5,
            max_searches=1,
            controller_responses=[RuntimeError("synthetic Controller failure")],
        )
        self.assertEqual(legacy["status"], "failed")
        self.assertEqual(graph["error_message"], "synthetic Controller failure")
        self.assertEqual(graph["budget"].steps_used, 0)
        self.assertEqual(len(graph["replay_records"]), 1)
        self.assertEqual(graph["controller_invocations"], 1)

    def test_dispatch_invariant_and_action_execution_failure_match(self):
        route_twice = [_gap_route(), _gap_route()]
        legacy, graph = self._compare(
            modules,
            max_steps=3,
            max_searches=2,
            controller_responses=route_twice,
        )
        self.assertEqual(legacy["status"], "failed")
        self.assertEqual(graph["error_type"], RunnerInvariantError.__name__)
        self.assertEqual(graph["budget"].steps_used, 1)
        self.assertEqual(graph["module_observations"]["component_call_counts"]["related_work_verification"], 1)
        self.assertEqual(graph["controller_invocations"], 2)
        self.assertTrue(
            any(
                event.event_type == "runner_invariant_failed"
                for event in graph["trajectory_events"]
            )
        )

        legacy, graph = self._compare(
            modules,
            max_steps=3,
            max_searches=2,
            controller_responses=[_inspiration_route()],
            action_failure=True,
        )
        self.assertEqual(legacy["status"], "failed")
        self.assertEqual(graph["error_type"], "RuntimeError")
        self.assertEqual(graph["budget"].steps_used, 1)
        self.assertEqual(graph["budget"].searches_used, 1)
        self.assertEqual(graph["retrieval_invocations"], 1)
        self.assertEqual(graph["controller_invocations"], 1)

    def test_finalizer_failure_preserves_commit_and_unable_design_stops_cleanly(self):
        legacy, graph = self._compare(
            modules,
            max_steps=5,
            max_searches=1,
            finalizer_failure=True,
        )
        self.assertEqual(legacy["status"], "finalization_failed")
        self.assertIsNotNone(graph["commit_snapshot"])
        self.assertIsNone(graph["final_proposal"])
        self.assertEqual(graph["budget"].steps_used, 5)
        self.assertEqual(graph["finalizer_invocations"], 1)

        legacy, graph = self._compare(
            modules,
            max_steps=5,
            max_searches=1,
            commit_failure=True,
        )
        self.assertEqual(legacy["status"], "failed")
        self.assertEqual(graph["error_message"], "synthetic commit failure")
        self.assertIsNone(graph["commit_snapshot"])
        self.assertIsNone(graph["final_proposal"])
        self.assertEqual(graph["budget"].steps_used, 5)
        self.assertEqual(self._actions(graph)[-1], "COMMIT")
        self.assertEqual(graph["commit_invocations"], 1)
        self.assertEqual(graph["finalizer_invocations"], 0)

        payload = experiment_design()
        payload["identifiability_checks"][1]["identifiable"] = False
        legacy, graph = self._compare(
            lambda: modules(experiment_design_payload=payload),
            max_steps=4,
            max_searches=1,
        )
        self.assertEqual(legacy["status"], "stopped")
        self.assertIsNone(graph["error_type"])
        self.assertIsNone(graph["final_proposal"])
        self.assertEqual(graph["budget"].steps_used, 4)


if __name__ == "__main__":
    unittest.main()
