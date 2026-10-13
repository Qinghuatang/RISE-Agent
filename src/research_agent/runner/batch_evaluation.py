"""M13 batch evaluation orchestration and artifact aggregation.

This module iterates over independent Markdown inputs and delegates each paper
to an injected single-paper executor.  It does not implement or reinterpret any
scientific Action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..errors import SerializationError
from ..ingestion.markdown import parse_markdown_file
from ..models.evaluation import EvaluationGap, EvaluationReport, PaperEvaluation
from ..models.seed import SeedPaperContext
from ..serialization import read_json, read_jsonl, write_json
from ..serialization.trajectory import reconstruct_context
from ..trajectory.context import TrajectoryContext


class BatchEvaluationError(RuntimeError):
    """Raised for invalid batch inputs or unsafe output isolation."""


@dataclass(frozen=True)
class BatchEvaluationConfig:
    input_dir: Path
    output_dir: Path = Path("outputs/batch_eval")
    batch_name: str = "m13_batch_eval"
    max_papers: int | None = None
    selection_seed: int | None = 1301
    allow_overwrite: bool = False

    def __post_init__(self) -> None:
        if self.max_papers is not None and self.max_papers <= 0:
            raise ValueError("max_papers must be positive when provided")
        if not self.batch_name.strip():
            raise ValueError("batch_name must be non-empty")


PaperExecutor = Callable[[Path, Path], Any]


def _payload(path: Path, schema: str) -> dict[str, Any] | None:
    try:
        return read_json(path, expected_schema=schema)["payload"]
    except (OSError, SerializationError, KeyError, TypeError):
        return None


def _events(path: Path) -> list[dict[str, Any]]:
    try:
        return [envelope["payload"] for envelope in read_jsonl(path, expected_schema="trajectory_event")]
    except (OSError, SerializationError, KeyError, TypeError):
        return []


def _latest_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") == event_type:
            return event
    return None


def _event_payload(event: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(event, Mapping):
        return {}
    payload = event.get("result_payload")
    return payload if isinstance(payload, Mapping) else {}


def _nested_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, Mapping) else {}


def _inspiration_status(event: Mapping[str, Any]) -> str | None:
    payload = _event_payload(event)
    result = _nested_result(payload)
    status = result.get("status")
    return status if isinstance(status, str) else None


def _selected_inspiration_ref(event: Mapping[str, Any]) -> str | None:
    payload = _event_payload(event)
    candidates = [payload]
    for key in ("revision_result", "result"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        value = candidate.get("selected_inspiration_ref")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _manifest(output_dir: Path) -> dict[str, Any]:
    run = _payload(output_dir / "run_manifest.json", "run_manifest")
    if run is not None:
        return run
    pilot = _payload(output_dir / "pilot_manifest.json", "pilot_manifest")
    return pilot or {}


def _seed_context(output_dir: Path) -> SeedPaperContext | None:
    for schema in ("seed_context", "seed_paper_context"):
        payload = _payload(output_dir / "seed_context.json", schema)
        if payload is not None:
            try:
                return SeedPaperContext.model_validate(payload)
            except Exception:
                return None
    return None


def _extract_gaps(events: list[dict[str, Any]]) -> list[EvaluationGap]:
    event = _latest_event(events, "gap_agenda_updated")
    raw_gaps = _event_payload(event).get("gaps", [])
    if not isinstance(raw_gaps, list):
        return []
    result: list[EvaluationGap] = []
    for raw in raw_gaps:
        if not isinstance(raw, Mapping):
            continue
        try:
            result.append(
                EvaluationGap(
                    id=raw["id"],
                    confidence=raw["confidence"],
                    priority=raw["priority"],
                    evidence_refs=raw.get("evidence_refs", []),
                )
            )
        except Exception:
            continue
    return result


def _candidate_solution_count(
    state: Mapping[str, Any] | None,
    events: list[dict[str, Any]],
) -> int:
    counts: list[int] = []
    if isinstance(state, Mapping) and isinstance(state.get("candidate_solutions"), list):
        counts.append(len(state["candidate_solutions"]))
    for event in events:
        payload = _event_payload(event)
        for key in ("candidate_solutions", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                counts.append(len(value))
    return max(counts, default=0)


def _selected_components(
    state: Mapping[str, Any] | None,
    events: list[dict[str, Any]],
) -> int | None:
    if isinstance(state, Mapping) and isinstance(state.get("current_solution"), Mapping):
        components = state["current_solution"].get("components")
        if isinstance(components, list):
            return len(components)
    for event in reversed(events):
        solution = _event_payload(event).get("current_solution")
        if isinstance(solution, Mapping) and isinstance(solution.get("components"), list):
            return len(solution["components"])
    return None


def _novelty_status(
    state: Mapping[str, Any] | None,
    events: list[dict[str, Any]],
) -> str | None:
    if isinstance(state, Mapping) and isinstance(state.get("current_solution"), Mapping):
        value = state["current_solution"].get("novelty_status")
        if isinstance(value, str) and value != "unchecked":
            return value
    for event in reversed(events):
        payload = _event_payload(event)
        candidates = [payload]
        result = payload.get("result")
        if isinstance(result, Mapping):
            candidates.append(result)
        for candidate in candidates:
            value = candidate.get("novelty_status")
            if isinstance(value, str) and value != "unchecked":
                return value
    return None


def _action_sequence(memory: Mapping[str, Any] | None, events: list[dict[str, Any]]) -> list[str]:
    if isinstance(memory, Mapping) and isinstance(memory.get("decision_log"), list):
        actions = [
            item.get("action")
            for item in memory["decision_log"]
            if isinstance(item, Mapping) and isinstance(item.get("action"), str)
        ]
        if actions:
            return actions
    actions: list[str] = []
    for event in events:
        action = event.get("action")
        if isinstance(action, str) and action not in actions:
            actions.append(action)
    return actions


def _replay_consistency(output_dir: Path) -> tuple[bool, bool, bool]:
    state_payload = _payload(output_dir / "research_state.json", "research_state")
    memory_payload = _payload(output_dir / "research_memory.json", "research_memory")
    if state_payload is None or memory_payload is None:
        return False, False, False
    seed = _seed_context(output_dir)
    if seed is None:
        return False, False, False
    try:
        loaded = TrajectoryContext.load(seed, output_dir)
        replayed = reconstruct_context(
            seed,
            output_dir / "trajectory.jsonl",
            output_dir / "trajectory_replay.jsonl",
        )
    except Exception:
        return False, False, False
    state_equal = replayed.get_state() == loaded.get_state()
    memory_equal = replayed.get_memory() == loaded.get_memory()
    trajectory_equal = replayed.trajectory.get_events() == loaded.trajectory.get_events()
    return state_equal and memory_equal and trajectory_equal, state_equal, memory_equal


def summarize_paper_output(
    markdown_path: Path,
    output_dir: Path,
    *,
    executor_error: Exception | None = None,
) -> PaperEvaluation:
    """Read one pilot's artifacts without executing scientific logic."""

    events = _events(output_dir / "trajectory.jsonl")
    state = _payload(output_dir / "research_state.json", "research_state")
    memory = _payload(output_dir / "research_memory.json", "research_memory")
    manifest = _manifest(output_dir)
    paper_id = str(manifest.get("paper_id") or parse_markdown_file(markdown_path).paper_id)
    status = manifest.get("status")
    if status not in {"committed", "stopped", "failed", "finalization_failed"}:
        status = "failed"

    gaps = _extract_gaps(events)
    action_sequence = _action_sequence(memory, events)
    controller_events = [event for event in events if event.get("event_type") == "controller_decided"]
    inspiration_available = any(
        "SEEK_INSPIRATION" in (_event_payload(event).get("allowed_actions") or [])
        for event in controller_events
    )
    inspiration_triggered = any(
        _event_payload(event).get("decision", {}).get("action") == "SEEK_INSPIRATION"
        for event in controller_events
        if isinstance(_event_payload(event).get("decision"), Mapping)
    )
    inspiration_found = any(
        event.get("event_type") == "inspiration_retrieved"
        and _inspiration_status(event) == "found"
        for event in events
    )
    inspiration_used = any(
        event.get("action") == "TARGETED_REVISION"
        and _selected_inspiration_ref(event) is not None
        for event in events
    )
    revision_triggered = "TARGETED_REVISION" in action_sequence
    commit_status = "committed" if manifest.get("commit_gate_passed") else "not_committed"
    finalizer_status = "written" if manifest.get("final_proposal_written") else "not_written"
    solution_generated = bool(
        isinstance(state, Mapping) and state.get("current_solution") is not None
    ) or any(
        event.get("event_type") in {"solution_formed", "solution_developed"}
        for event in events
    )
    replay_verified, state_consistent, memory_consistent = _replay_consistency(output_dir)

    failure_type = None
    error_message = manifest.get("error_message")
    if executor_error is not None:
        failure_type = executor_error.__class__.__name__
        error_message = str(executor_error) or failure_type
    elif status == "stopped":
        failure_type = "budget_exhausted" if any(
            "budget" in issue.lower()
            for issue in (manifest.get("unresolved_issues") or [])
            if isinstance(issue, str)
        ) else "stopped"
    elif status in {"failed", "finalization_failed"}:
        failure_type = manifest.get("error_type") or (
            "finalization_failure" if status == "finalization_failed" else "workflow_failure"
        )

    pilot = _payload(output_dir / "pilot_manifest.json", "pilot_manifest") or {}
    prompt_tokens = pilot.get("prompt_tokens_total")
    completion_tokens = pilot.get("completion_tokens_total")
    total_tokens = pilot.get("total_tokens")
    known_prompt_tokens = int(pilot.get("known_prompt_tokens", 0) or 0)
    known_completion_tokens = int(pilot.get("known_completion_tokens", 0) or 0)
    known_total_tokens = int(pilot.get("known_total_tokens", 0) or 0)
    calls_with_token_usage = int(pilot.get("calls_with_token_usage", 0) or 0)
    calls_missing_token_usage = int(pilot.get("calls_missing_token_usage", 0) or 0)
    return PaperEvaluation(
        paper_id=paper_id,
        markdown_path=str(markdown_path),
        output_dir=str(output_dir),
        status=status,
        action_sequence=action_sequence,
        commit_status=commit_status,
        finalizer_status=finalizer_status,
        gap_found=bool(gaps),
        candidate_gap_count=len(gaps),
        gaps=gaps,
        solution_generated=solution_generated,
        candidate_solution_count=_candidate_solution_count(state, events),
        selected_solution_components=_selected_components(state, events),
        novelty_status=_novelty_status(state, events),
        inspiration_available=inspiration_available,
        inspiration_triggered=inspiration_triggered,
        inspiration_found=inspiration_found,
        inspiration_used=inspiration_used,
        revision_triggered=revision_triggered,
        llm_calls=int(pilot.get("llm_request_count", 0) or 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        known_prompt_tokens=known_prompt_tokens,
        known_completion_tokens=known_completion_tokens,
        known_total_tokens=known_total_tokens,
        calls_with_token_usage=calls_with_token_usage,
        calls_missing_token_usage=calls_missing_token_usage,
        token_metadata_complete=bool(pilot.get("token_metadata_complete", False)),
        semantic_scholar_requests=int(pilot.get("semantic_scholar_http_requests", 0) or 0),
        arxiv_requests=int(pilot.get("arxiv_http_requests", 0) or 0),
        replay_verified=replay_verified,
        state_consistent=state_consistent,
        memory_consistent=memory_consistent,
        failure_type=failure_type,
        error_message=error_message,
    )


def _counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _sum_or_none(values: list[int | None]) -> int | None:
    return sum(value for value in values if value is not None) if all(
        value is not None for value in values
    ) else None


def _sum_known(papers: list[PaperEvaluation], field: str) -> int:
    return sum(int(getattr(paper, field, 0) or 0) for paper in papers)


def build_evaluation_report(
    config: BatchEvaluationConfig,
    papers: list[PaperEvaluation],
    selected_paper_ids: list[str],
) -> EvaluationReport:
    statuses = [paper.status for paper in papers]
    committed = sum(status == "committed" for status in statuses)
    failed = sum(status in {"failed", "finalization_failed"} for status in statuses)
    stopped = sum(status == "stopped" for status in statuses)
    return EvaluationReport(
        batch_name=config.batch_name,
        input_dir=str(config.input_dir),
        output_dir=str(config.output_dir),
        selection_seed=config.selection_seed,
        selected_paper_ids=selected_paper_ids,
        paper_count=len(papers),
        completed_count=committed,
        failed_count=failed,
        stopped_count=stopped,
        papers=papers,
        commit_status_counts=_counts([paper.commit_status for paper in papers]),
        finalizer_status_counts=_counts([paper.finalizer_status for paper in papers]),
        gap_found_counts=_counts(["yes" if paper.gap_found else "no" for paper in papers]),
        novelty_status_counts=_counts(
            [paper.novelty_status for paper in papers if paper.novelty_status is not None]
        ),
        inspiration_available_count=sum(paper.inspiration_available for paper in papers),
        inspiration_triggered_count=sum(paper.inspiration_triggered for paper in papers),
        inspiration_found_count=sum(paper.inspiration_found for paper in papers),
        inspiration_used_count=sum(paper.inspiration_used for paper in papers),
        revision_triggered_count=sum(paper.revision_triggered for paper in papers),
        llm_calls_total=sum(paper.llm_calls for paper in papers),
        prompt_tokens_total=_sum_or_none([paper.prompt_tokens for paper in papers]),
        completion_tokens_total=_sum_or_none([paper.completion_tokens for paper in papers]),
        total_tokens=_sum_or_none([paper.total_tokens for paper in papers]),
        known_prompt_tokens=_sum_known(papers, "known_prompt_tokens"),
        known_completion_tokens=_sum_known(papers, "known_completion_tokens"),
        known_total_tokens=_sum_known(papers, "known_total_tokens"),
        calls_with_token_usage=_sum_known(papers, "calls_with_token_usage"),
        calls_missing_token_usage=_sum_known(papers, "calls_missing_token_usage"),
        token_metadata_complete=all(paper.token_metadata_complete for paper in papers),
        semantic_scholar_requests_total=sum(
            paper.semantic_scholar_requests for paper in papers
        ),
        arxiv_requests_total=sum(paper.arxiv_requests for paper in papers),
        replay_verified_count=sum(paper.replay_verified for paper in papers),
        state_consistent_count=sum(paper.state_consistent for paper in papers),
        memory_consistent_count=sum(paper.memory_consistent for paper in papers),
        failure_type_counts=_counts(
            [paper.failure_type for paper in papers if paper.failure_type is not None]
        ),
    )


class BatchEvaluationRunner:
    """Iterate papers, isolate outputs, delegate execution, and aggregate metrics."""

    def __init__(self, config: BatchEvaluationConfig, executor: PaperExecutor) -> None:
        self.config = config
        self.executor = executor

    def discover_papers(self) -> list[Path]:
        input_dir = self.config.input_dir
        if not input_dir.is_dir():
            raise BatchEvaluationError(f"batch input directory does not exist: {input_dir}")
        papers = sorted(
            path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
        )
        if not papers:
            raise BatchEvaluationError(f"batch input directory contains no Markdown files: {input_dir}")
        if self.config.max_papers is not None and self.config.max_papers < len(papers):
            rng = random.Random(self.config.selection_seed)
            papers = rng.sample(papers, self.config.max_papers)
        return papers

    @staticmethod
    def _paper_id(path: Path) -> str:
        paper_id = parse_markdown_file(path).paper_id
        if not paper_id or Path(paper_id).name != paper_id:
            raise BatchEvaluationError(f"unsafe paper id derived from Markdown filename: {paper_id!r}")
        return paper_id

    def run(self) -> EvaluationReport:
        papers = self.discover_papers()
        paper_ids = [self._paper_id(path) for path in papers]
        if len(set(paper_ids)) != len(paper_ids):
            raise BatchEvaluationError("selected Markdown files contain duplicate paper IDs")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        paper_outputs = [self.config.output_dir / paper_id for paper_id in paper_ids]
        if not self.config.allow_overwrite:
            existing = [path for path in paper_outputs if path.exists()]
            if existing:
                raise BatchEvaluationError(
                    "batch paper output already exists; choose a fresh output directory or "
                    "enable allow_overwrite: " + ", ".join(map(str, existing))
                )

        results: list[PaperEvaluation] = []
        for markdown_path, paper_id, paper_output in zip(
            papers, paper_ids, paper_outputs
        ):
            error: Exception | None = None
            try:
                self.executor(markdown_path, paper_output)
            except Exception as exc:  # isolate one paper failure from the batch
                error = exc
                paper_output.mkdir(parents=True, exist_ok=True)
                write_json(
                    paper_output / "batch_error.json",
                    "batch_error",
                    {"paper_id": paper_id, "error_type": type(exc).__name__, "error": str(exc)},
                )
            results.append(
                summarize_paper_output(
                    markdown_path,
                    paper_output,
                    executor_error=error,
                )
            )

        report = build_evaluation_report(self.config, results, paper_ids)
        write_json(self.config.output_dir / "evaluation_report.json", "evaluation_report", report)
        return report


def make_real_paper_executor(
    *,
    semantic_scholar_config: Path,
    max_steps: int = 12,
    max_searches: int = 2,
    allow_real_services: bool = False,
    allow_overwrite: bool = False,
) -> PaperExecutor:
    """Build an explicitly opted-in executor for the existing M12 pilot.

    The import is lazy so importing the batch framework never initializes a
    network client.  Scientific execution remains entirely delegated to
    ``RealSinglePaperPilot``; this adapter only maps each isolated output
    directory to the pilot configuration.
    """

    from .pilot import RealPilotConfig, RealSinglePaperPilot

    if not allow_real_services:
        raise BatchEvaluationError(
            "real batch execution requires explicit allow_real_services=True"
        )

    def execute(markdown_path: Path, output_dir: Path) -> Any:
        return RealSinglePaperPilot(
            RealPilotConfig(
                markdown_path=markdown_path,
                output_dir=output_dir,
                semantic_scholar_config=semantic_scholar_config,
                max_steps=max_steps,
                max_searches=max_searches,
                pilot_name="m13_batch_paper",
                allow_real_services=True,
                allow_overwrite=allow_overwrite,
            )
        ).run()

    return execute


__all__ = [
    "BatchEvaluationConfig",
    "BatchEvaluationError",
    "BatchEvaluationRunner",
    "PaperExecutor",
    "build_evaluation_report",
    "make_real_paper_executor",
    "summarize_paper_output",
]
