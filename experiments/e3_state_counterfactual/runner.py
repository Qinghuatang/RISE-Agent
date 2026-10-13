"""Execution harness for E3 global-revision counterfactuals."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.errors import SerializationError
from research_agent.llm.client import LLMClient
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.llm.recording import RecordingLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models.memory import ResearchMemory
from research_agent.models.state import ResearchState
from research_agent.models.verification import VerificationFocus
from research_agent.modules.related_work_verification import (
    RelatedWorkVerificationModule,
)
from research_agent.serialization import read_json, write_json
from research_agent.serialization.codec import write_jsonl
from research_agent.trajectory.context import TrajectoryContext
from research_agent.trajectory.recorder import TrajectoryRecorder
from research_agent.runtime.state_store import ResearchStateStore

from .checkpoint import (
    discover_revision_cases,
    extract_historical_local_branch,
    validate_checkpoint,
)
from .global_revision import GlobalRevisionModule
from .models import E3Checkpoint, RevisionCase


class CacheOnlyRelatedPaperProvider:
    """Guard provider: E3 must never perform a new related-paper retrieval."""

    provider_name = "e3_cache_only"
    last_request_count = 0

    def get_related_paper_pool(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "E3 attempted a new related-paper retrieval; the checkpoint pool is required"
        )

    get_related_papers = get_related_paper_pool


@dataclass(frozen=True)
class E3RunConfig:
    source_dir: Path
    output_dir: Path
    seed: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        source = self.source_dir.resolve()
        output = self.output_dir.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(
                "E3 output_dir must be separate from and outside the read-only source_dir"
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(root: Path, message: str) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with (root / "logs" / "e3.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{_now()} {message}\n")


_LOG_LOCK = threading.Lock()


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _paper_short_id(paper_id: str) -> str | None:
    match = re.search(r"(?:^|_)P(\d{3})(?:_|$)", paper_id)
    return f"P{match.group(1)}" if match else None


def _case_output(root: Path, paper_id: str) -> Path:
    return root / "cases" / paper_id


def _write_case_discovery(root: Path, case: RevisionCase) -> None:
    output = _case_output(root, case.paper_id)
    output.mkdir(parents=True, exist_ok=True)
    if case.checkpoint is not None:
        write_json(output / "checkpoint.json", "e3_checkpoint", case.checkpoint)
        local = extract_historical_local_branch(case.source_dir, case.checkpoint)
        write_json(output / "historical_local.json", "e3_historical_local", local)
    else:
        write_json(
            output / "error.json",
            "e3_case_validation_error",
            {
                "paper_id": case.paper_id,
                "errors": list(case.validation_errors),
            },
        )


def _compact_case(case: RevisionCase) -> dict[str, Any]:
    return {
        "paper_id": case.paper_id,
        "source_dir": case.source_dir,
        "first_revision_step": case.first_revision_step,
        "first_revision_replay_sequence": case.first_revision_replay_sequence,
        "trigger_type": case.trigger_type,
        "valid": case.valid,
        "validation_errors": list(case.validation_errors),
    }


def _select_case(cases: list[RevisionCase], requested: str) -> RevisionCase:
    exact = [case for case in cases if case.paper_id == requested]
    if len(exact) == 1:
        return exact[0]
    short = [case for case in cases if _paper_short_id(case.paper_id) == requested]
    if len(short) == 1:
        return short[0]
    available = ", ".join(case.paper_id for case in cases)
    raise ValueError(f"unknown or ambiguous --paper-id {requested!r}; available: {available}")


def _selected_inspiration(
    checkpoint: E3Checkpoint,
    historical_local: Any,
) -> dict[str, Any] | None:
    """Select the historical inspiration candidate without passing local output text.

    The reference is used only to identify the same retrieved candidate for the
    paired branch.  The global LLM receives the candidate's checkpoint-time
    retrieval record, never the historical local revision text or outcome.
    """

    record_payload = historical_local.inspiration_before
    if not isinstance(record_payload, dict):
        return None
    inspirations = record_payload.get("inspirations")
    if not isinstance(inspirations, list) or not inspirations:
        return None
    selected_ref = historical_local.selected_inspiration_ref
    index = 0
    if isinstance(selected_ref, str) and ":" in selected_ref:
        suffix = selected_ref.rsplit(":", 1)[-1]
        try:
            candidate_index = int(suffix) - 1
            if 0 <= candidate_index < len(inspirations):
                index = candidate_index
        except ValueError:
            pass
    candidate = inspirations[index]
    return {
        "record_id": record_payload.get("id"),
        "candidate_index": index + 1,
        "candidate": candidate,
        "inspiration_need": record_payload.get("inspiration_need"),
    }


def _verification_record_for_checkpoint(checkpoint: E3Checkpoint) -> Any | None:
    solution = checkpoint.state.current_solution
    if solution is None:
        return None
    records = [
        record
        for record in checkpoint.memory.verification_memory
        if record.task == "VERIFY_SOLUTION"
        and record.target_ref == solution.id
        and record.created_at_step <= checkpoint.checkpoint_step
    ]
    return max(records, key=lambda record: record.created_at_step) if records else None


def _failure_records(checkpoint: E3Checkpoint) -> list[Any]:
    return list(checkpoint.memory.failure_memory)


def _metadata_payload(metadata: Any, input_chars: int | None) -> dict[str, Any]:
    return {
        "provider": getattr(metadata, "provider", None),
        "model": getattr(metadata, "model", None),
        "request_id": getattr(metadata, "request_id", None),
        "prompt_tokens": getattr(metadata, "prompt_tokens", None),
        "completion_tokens": getattr(metadata, "completion_tokens", None),
        "total_tokens": getattr(metadata, "total_tokens", None),
        "format_retry_count": getattr(metadata, "format_retry_count", 0),
        "retry_reason": getattr(metadata, "retry_reason", None),
        "input_chars": input_chars,
    }


def _run_case(
    case: RevisionCase,
    output_root: Path,
    llm_client: LLMClient,
) -> dict[str, Any]:
    if not case.valid or case.checkpoint is None:
        raise ValueError("cannot execute an invalid E3 checkpoint")
    checkpoint = case.checkpoint
    source_dir = Path(case.source_dir)
    local = extract_historical_local_branch(source_dir, checkpoint)
    seed_payload = read_seed_context(source_dir)
    # The revision need is a checkpoint-time input captured in the source
    # trajectory.  It is passed to the global branch, while the historical
    # revision output itself is never included in the global prompt.
    revision_event_need = _find_revision_need(source_dir, checkpoint)
    verification = _verification_record_for_checkpoint(checkpoint)
    inspiration = _selected_inspiration(checkpoint, local)
    failures = _failure_records(checkpoint)
    output = _case_output(output_root, case.paper_id)
    output.mkdir(parents=True, exist_ok=True)

    revision_module = GlobalRevisionModule(llm_client)
    revision_execution = revision_module.revise(
        seed_context=seed_payload,
        active_gap=checkpoint.state.active_gap,
        current_solution=checkpoint.state.current_solution,
        revision_need=revision_event_need,
        verification_evidence=verification,
        selected_inspiration=inspiration,
        relevant_failure_memory=failures,
    )
    write_json(
        output / "global_revision.json",
        "e3_global_revision",
        {
            "solution_before": checkpoint.state.current_solution,
            "global_revision": revision_execution.draft,
            "global_revised_solution": revision_execution.revised_solution,
            "model_metadata": _metadata_payload(
                revision_execution.metadata, revision_execution.input_chars
            ),
            "call_count": revision_execution.call_count,
            "revision_input": {
                "active_gap": checkpoint.state.active_gap,
                "revision_need": revision_event_need,
                "verification_evidence": verification,
                "selected_inspiration": inspiration,
                "failure_memory": failures,
            },
        },
    )

    if revision_execution.revised_solution is None:
        metadata = {
            "paper_id": case.paper_id,
            "status": "unable_to_revise",
            "updated_at": _now(),
            "global_revision_call_count": revision_execution.call_count,
            "global_verification_call_count": 0,
            "note": revision_execution.draft.failure_reason,
        }
        write_json(output / "run_metadata.json", "e3_run_metadata", metadata)
        return metadata

    state = _copy_state(checkpoint.state)
    state.current_solution = revision_execution.revised_solution
    # A whole-solution rewrite invalidates any pre-existing experiment plan;
    # the E3 experiment ends after re-verification and does not redesign it.
    state.experiments = []
    context = TrajectoryContext(
        seed_payload,
        state_store=ResearchStateStore(state),
        memory_store=ResearchMemoryStore(
            checkpoint.paper_id,
            initial_memory=_copy_memory(checkpoint.memory),
        ),
        recorder=TrajectoryRecorder(checkpoint.trajectory_prefix),
    )
    global_step = checkpoint.checkpoint_step + 1
    context.record_transition(
        global_step,
        "e3_global_solution_revised",
        checkpoint.state,
        state,
        action="TARGETED_REVISION",
        result_ref=state.current_solution.id,
        result_payload={
            "status": "revised",
            "revision_mode": "whole_solution_rewrite",
            "solution": state.current_solution.model_dump(mode="json"),
        },
        memory_writes=[],
    )
    pool = checkpoint.memory.paper_memory.related_paper_pool
    if pool is None:
        raise SerializationError("valid E3 checkpoint has no related-paper pool")
    verifier = RelatedWorkVerificationModule(
        llm_client,
        CacheOnlyRelatedPaperProvider(),
    )
    verification_result = verifier.verify_solution(
        active_gap=state.active_gap,
        current_solution=state.current_solution,
        context=context,
        related_paper_pool=pool,
        verification_focus=VerificationFocus(mode="full_solution"),
        step=global_step + 1,
    )
    write_json(
        output / "global_verification.json",
        "e3_global_verification",
        {
            "verification_focus": {"mode": "full_solution", "changed_units": []},
            "compared_papers": [
                comparison.model_dump(mode="json")
                for comparison in verifier.last_solution_comparisons
            ],
            "verification_result": verification_result,
            "model_metadata": _metadata_payload(
                verifier.last_call_metadata, verifier.last_input_chars
            ),
            "call_count": verifier.call_count,
            "related_paper_pool_reused": True,
            "new_related_paper_retrievals": 0,
        },
    )
    llm_records = getattr(llm_client, "records", None)
    if isinstance(llm_records, list):
        write_jsonl(output / "llm_calls.jsonl", "llm_call_audit", llm_records)
    metadata = {
        "paper_id": case.paper_id,
        "status": "completed",
        "updated_at": _now(),
        "checkpoint_step": checkpoint.checkpoint_step,
        "global_revision_step": global_step,
        "global_verification_step": global_step + 1,
        "global_revision_call_count": revision_execution.call_count,
        "global_verification_call_count": verifier.call_count,
        "related_paper_pool_reused": True,
        "new_related_paper_retrievals": 0,
        "global_verification_novelty_status": verification_result.novelty_status,
    }
    write_json(output / "run_metadata.json", "e3_run_metadata", metadata)
    return metadata


def read_seed_context(source_dir: Path) -> Any:
    from research_agent.models.seed import SeedPaperContext

    return SeedPaperContext.model_validate(
        read_json(source_dir / "seed_context.json", expected_schema="seed_context")[
            "payload"
        ]
    )


def _find_revision_need(source_dir: Path, checkpoint: E3Checkpoint) -> dict[str, Any]:
    recorder = TrajectoryRecorder.load(source_dir / "trajectory.jsonl")
    for event in recorder.get_events():
        if (
            event.event_type == "solution_revised"
            and event.action == "TARGETED_REVISION"
            and event.step > checkpoint.checkpoint_step
        ):
            payload = event.result_payload or {}
            need = payload.get("revision_need")
            if isinstance(need, dict):
                return need
            break
    raise SerializationError("first revision has no checkpoint-time revision_need")


def _write_top_artifacts(
    root: Path,
    config: E3RunConfig,
    cases: list[RevisionCase],
    validations: list[Any],
    *,
    real_run: bool,
    deepseek: DeepSeekConfig | None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    write_json(
        root / "revision_cases.json",
        "e3_revision_cases",
        {
            "cases": [_compact_case(case) for case in cases],
            "total_revision_cases": len(cases),
            "valid_cases": sum(1 for case in cases if case.valid),
        },
    )
    write_json(
        root / "validation_report.json",
        "e3_validation_report",
        {
            "source_dir": str(config.source_dir),
            "total_revision_cases": len(cases),
            "valid_cases": sum(1 for case in cases if case.valid),
            "invalid_cases": sum(1 for case in cases if not case.valid),
            "trigger_counts": {
                "overlap_repair": sum(
                    1 for case in cases if case.trigger_type == "overlap_repair" and case.valid
                ),
                "inspiration_strengthening": sum(
                    1
                    for case in cases
                    if case.trigger_type == "inspiration_strengthening" and case.valid
                ),
            },
            "checkpoint_fidelity_all_passed": all(
                result.valid for result in validations
            ),
            "validation_results": [result.model_dump(mode="json") for result in validations],
        },
    )
    write_json(
        root / "execution_manifest.json",
        "e3_execution_manifest",
        {
            "experiment": "E3_state_evolution_counterfactual",
            "source_dir": str(config.source_dir),
            "output_dir": str(config.output_dir),
            "seed": config.seed,
            "real_run": real_run,
            "max_global_revision_calls_per_case": 1,
            "max_global_verification_calls_per_case": 1,
            "new_related_paper_retrievals_per_case": 0,
            "deepseek": deepseek.public_dict() if deepseek is not None else None,
            "created_at": _now(),
        },
    )


def validate_only(config: E3RunConfig) -> list[RevisionCase]:
    cases = discover_revision_cases(config.source_dir)
    validations = [validate_checkpoint(case.source_dir) for case in cases if case.valid]
    _write_top_artifacts(
        config.output_dir,
        config,
        cases,
        validations,
        real_run=False,
        deepseek=None,
    )
    for case in cases:
        _write_case_discovery(config.output_dir, case)
    return cases


def run_cases(
    config: E3RunConfig,
    cases: list[RevisionCase],
    llm_client: LLMClient | None = None,
    *,
    deepseek: DeepSeekConfig | None = None,
    llm_factory: Any | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if llm_client is not None and llm_factory is not None:
        raise ValueError("provide llm_client or llm_factory, not both")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and llm_factory is None:
        raise ValueError("parallel E3 execution requires an llm_factory")
    validations = [validate_checkpoint(case.source_dir) for case in cases if case.valid]
    _write_top_artifacts(
        config.output_dir,
        config,
        cases,
        validations,
        real_run=True,
        deepseek=deepseek,
    )
    for case in cases:
        _write_case_discovery(config.output_dir, case)
        _log(config.output_dir, f"case={case.paper_id} discovered valid={case.valid}")
    def execute(case: RevisionCase) -> dict[str, Any]:
        if not case.valid:
            print(f"[E3] {case.paper_id}: invalid checkpoint; skipped")
            return {"paper_id": case.paper_id, "status": "invalid"}
        output = _case_output(config.output_dir, case.paper_id)
        metadata_path = output / "run_metadata.json"
        if metadata_path.exists() and not config.overwrite:
            try:
                existing = read_json(metadata_path, expected_schema="e3_run_metadata")["payload"]
            except Exception:
                existing = None
            if isinstance(existing, dict) and existing.get("status") in {
                "completed",
                "unable_to_revise",
            }:
                print(f"[E3] {case.paper_id}: already complete; skipped")
                return existing
        case_client = None
        try:
            print(f"[E3] {case.paper_id}: running global revision + full verification")
            _log(config.output_dir, f"case={case.paper_id} execution_started")
            case_client = llm_factory() if llm_factory is not None else llm_client
            if case_client is None:
                raise ValueError("E3 execution requires an LLM client")
            result = _run_case(case, config.output_dir, case_client)
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=True)
            error = {
                "paper_id": case.paper_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2_000],
                "updated_at": _now(),
            }
            if isinstance(getattr(case_client, "records", None), list):
                write_jsonl(output / "llm_calls.jsonl", "llm_call_audit", case_client.records)
            write_json(output / "error.json", "e3_case_error", error)
            write_json(output / "run_metadata.json", "e3_run_metadata", error)
            print(f"[E3] {case.paper_id}: failed ({type(exc).__name__})")
            _log(config.output_dir, f"case={case.paper_id} failed={type(exc).__name__}")
            return error
        print(f"[E3] {case.paper_id}: {result.get('status')}")
        _log(config.output_dir, f"case={case.paper_id} status={result.get('status')}")
        return result

    results_by_index: dict[int, dict[str, Any]] = {}
    if workers == 1 or len(cases) <= 1:
        for index, case in enumerate(cases):
            results_by_index[index] = execute(case)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="e3-paper")
        futures = {
            executor.submit(execute, case): index for index, case in enumerate(cases)
        }
        try:
            for future in as_completed(futures):
                results_by_index[futures[future]] = future.result()
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    results = [results_by_index[index] for index in range(len(cases))]
    write_json(
        config.output_dir / "execution_summary.json",
        "e3_execution_summary",
        {
            "paper_count": len(cases),
            "completed": sum(item.get("status") == "completed" for item in results),
            "unable_to_revise": sum(
                item.get("status") == "unable_to_revise" for item in results
            ),
            "failed": sum(item.get("status") == "failed" for item in results),
            "results": sorted(results, key=lambda item: item.get("paper_id", "")),
        },
    )
    return results


def build_real_llm() -> tuple[RecordingLLMClient, DeepSeekConfig]:
    config = DeepSeekConfig.from_env()
    config.validate_for_request()
    client = RecordingLLMClient(DeepSeekClient(config), unbounded_output=True)
    return client, config


__all__ = [
    "E3RunConfig",
    "build_real_llm",
    "discover_revision_cases",
    "run_cases",
    "validate_only",
]
