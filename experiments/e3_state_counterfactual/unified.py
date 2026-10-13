"""E3-only loading and unified re-verification helpers.

This module reads the historical E3 inputs and runs the existing solution
verifier.  It does not execute Revision, Inspiration, or paper retrieval.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from research_agent.llm.client import LLMClient
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient
from research_agent.llm.recording import RecordingLLMClient
from research_agent.memory.store import ResearchMemoryStore
from research_agent.models.memory import RelatedPaperPool, ResearchMemory
from research_agent.models.revision import RevisionResult
from research_agent.models.seed import SeedPaperContext
from research_agent.models.state import CurrentSolution, ResearchState
from research_agent.models.verification import SolutionVerificationResult, VerificationFocus
from research_agent.runtime.state_store import ResearchStateStore
from research_agent.serialization import read_json, write_json
from research_agent.serialization.codec import write_jsonl
from research_agent.serialization.trajectory import load_replay_journal
from research_agent.trajectory.recorder import TrajectoryRecorder
from research_agent.modules.related_work_verification import (
    RelatedWorkVerificationModule,
    VerificationConfig,
)
from research_agent.errors import SerializationError

from .checkpoint import discover_revision_cases, validate_checkpoint
from .models import E3Checkpoint, HistoricalLocalBranch, RevisionCase
from .runner import CacheOnlyRelatedPaperProvider
from research_agent.modules.targeted_revision import apply_revision


BranchName = Literal["targeted", "global"]


@dataclass(frozen=True)
class UnifiedVerificationCase:
    """Validated, immutable inputs for one paired E3 re-verification case."""

    paper_id: str
    source_dir: Path
    e3_case_dir: Path
    seed_context: SeedPaperContext
    checkpoint: E3Checkpoint
    historical_local: HistoricalLocalBranch
    targeted_solution: CurrentSolution
    global_solution: CurrentSolution
    related_paper_pool: RelatedPaperPool
    before_outcome: str


@dataclass(frozen=True)
class CaseDiscovery:
    cases: list[UnifiedVerificationCase]
    errors: list[dict[str, Any]]
    discovered_revision_cases: int


def _payload(path: Path, schema: str) -> Any:
    return read_json(path, expected_schema=schema)["payload"]


def _copy_memory(memory: ResearchMemory) -> ResearchMemory:
    return ResearchMemory.model_validate(memory.model_dump(mode="python"))


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_solution(solution: CurrentSolution) -> CurrentSolution:
    return CurrentSolution.model_validate(solution.model_dump(mode="python"))


def _copy_pool(pool: RelatedPaperPool) -> RelatedPaperPool:
    return RelatedPaperPool.model_validate(pool.model_dump(mode="python"))


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible scientific object deterministically."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_revision_solution(source_dir: Path, expected: CurrentSolution) -> CurrentSolution:
    recorder = TrajectoryRecorder.load(source_dir / "trajectory.jsonl")
    revision_event = next(
        (
            event
            for event in recorder.get_events()
            if event.event_type == "solution_revised"
            and event.action == "TARGETED_REVISION"
        ),
        None,
    )
    if revision_event is None:
        raise SerializationError("source trajectory has no Targeted Revision event")
    replay = load_replay_journal(source_dir / "trajectory_replay.jsonl")
    revision_record = next(
        (
            record
            for record in replay
            if record.action == "TARGETED_REVISION"
            and record.step == revision_event.step
        ),
        None,
    )
    if revision_record is None or revision_record.state_after.current_solution is None:
        raise SerializationError("Targeted Revision replay has no revised solution")
    replay_solution = revision_record.state_after.current_solution
    revision_payload = revision_event.result_payload or {}
    raw_result = revision_payload.get("revision_result")
    if not isinstance(raw_result, dict):
        raw_result = revision_payload.get("result")
    if not isinstance(raw_result, dict):
        raise SerializationError("Targeted Revision event has no revision result")
    revision_result = RevisionResult.model_validate(raw_result)
    derived_solution = apply_revision(expected, revision_result)
    if derived_solution != replay_solution:
        raise SerializationError(
            "historical Targeted Revision patch does not match replay state_after"
        )
    return _copy_solution(replay_solution)


def _load_one_case(
    source_case: RevisionCase,
    e3_output_dir: Path,
) -> UnifiedVerificationCase:
    if not source_case.valid or source_case.checkpoint is None:
        raise SerializationError(
            "checkpoint validation failed: "
            + "; ".join(source_case.validation_errors)
        )
    source_dir = Path(source_case.source_dir)
    checkpoint_dir = e3_output_dir / "cases" / source_case.paper_id
    checkpoint = E3Checkpoint.model_validate(
        _payload(checkpoint_dir / "checkpoint.json", "e3_checkpoint")
    )
    if checkpoint != source_case.checkpoint:
        raise SerializationError("stored E3 checkpoint differs from reconstructed checkpoint")
    historical = HistoricalLocalBranch.model_validate(
        _payload(checkpoint_dir / "historical_local.json", "e3_historical_local")
    )
    if historical.trigger_type != source_case.trigger_type:
        raise SerializationError("historical trigger_type differs from source case")
    seed = SeedPaperContext.model_validate(_payload(source_dir / "seed_context.json", "seed_context"))
    if seed.paper_id != source_case.paper_id:
        raise SerializationError("source seed paper does not match E3 case")
    if checkpoint.state.current_solution is None or checkpoint.state.active_gap is None:
        raise SerializationError("checkpoint lacks active gap or current solution")
    if historical.solution_before != checkpoint.state.current_solution:
        raise SerializationError("historical solution_before differs from checkpoint state")

    targeted_solution = _first_revision_solution(
        source_dir,
        checkpoint.state.current_solution,
    )
    global_payload = _payload(checkpoint_dir / "global_revision.json", "e3_global_revision")
    if global_payload.get("global_revision", {}).get("status") != "revised":
        raise SerializationError("Global Revision case is not a completed rewrite")
    global_solution_raw = global_payload.get("global_revised_solution")
    if not isinstance(global_solution_raw, dict):
        raise SerializationError("Global Revision has no materialized solution")
    global_solution = CurrentSolution.model_validate(global_solution_raw)
    if global_payload.get("solution_before") != checkpoint.state.current_solution.model_dump(mode="json"):
        raise SerializationError("Global Revision solution_before differs from checkpoint")
    pool = checkpoint.memory.paper_memory.related_paper_pool
    if pool is None:
        raise SerializationError("checkpoint has no related-paper pool")
    before_outcome = checkpoint.state.current_solution.novelty_status
    if before_outcome not in {"partial_overlap", "distinct"}:
        raise SerializationError("checkpoint solution has no E3-supported novelty outcome")
    return UnifiedVerificationCase(
        paper_id=source_case.paper_id,
        source_dir=source_dir,
        e3_case_dir=checkpoint_dir,
        seed_context=seed,
        checkpoint=checkpoint,
        historical_local=historical,
        targeted_solution=targeted_solution,
        global_solution=global_solution,
        related_paper_pool=_copy_pool(pool),
        before_outcome=before_outcome,
    )


def discover_unified_cases(
    source_dir: str | Path,
    e3_output_dir: str | Path,
) -> CaseDiscovery:
    """Discover and validate the complete set without LLM or network calls."""

    source = Path(source_dir)
    e3_output = Path(e3_output_dir)
    source_cases = discover_revision_cases(source)
    valid: list[UnifiedVerificationCase] = []
    errors: list[dict[str, Any]] = []
    for source_case in source_cases:
        checkpoint_validation = validate_checkpoint(source_case.source_dir)
        if not checkpoint_validation.valid:
            errors.append(
                {
                    "paper_id": source_case.paper_id,
                    "stage": "checkpoint_validation",
                    "errors": checkpoint_validation.errors,
                }
            )
            continue
        try:
            valid.append(_load_one_case(source_case, e3_output))
        except Exception as exc:
            errors.append(
                {
                    "paper_id": source_case.paper_id,
                    "stage": "e3_input_validation",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1_000],
                }
            )
    return CaseDiscovery(
        cases=valid,
        errors=errors,
        discovered_revision_cases=len(source_cases),
    )


def _metadata_payload(metadata: Any, *, verifier: RelatedWorkVerificationModule) -> dict[str, Any]:
    return {
        "provider": getattr(metadata, "provider", None),
        "model": getattr(metadata, "model", None),
        "request_id": getattr(metadata, "request_id", None),
        "prompt_tokens": getattr(metadata, "prompt_tokens", None),
        "completion_tokens": getattr(metadata, "completion_tokens", None),
        "total_tokens": getattr(metadata, "total_tokens", None),
        "format_retry_count": getattr(metadata, "format_retry_count", 0),
        "retry_reason": getattr(metadata, "retry_reason", None),
        "input_chars": verifier.last_input_chars,
        "temperature": 0.0,
        "max_tokens": verifier.config.max_tokens,
        "new_related_paper_retrievals": verifier.scientific_search_count,
    }


def _run_branch(
    case: UnifiedVerificationCase,
    branch: BranchName,
    llm_client: LLMClient,
    verification_config: VerificationConfig,
) -> dict[str, Any]:
    solution = (
        case.targeted_solution if branch == "targeted" else case.global_solution
    )
    state = _copy_state(case.checkpoint.state)
    state.current_solution = _copy_solution(solution)
    memory_store = ResearchMemoryStore(
        case.paper_id,
        initial_memory=_copy_memory(case.checkpoint.memory),
    )
    state_store = ResearchStateStore(state)
    verifier = RelatedWorkVerificationModule(
        llm_client,
        CacheOnlyRelatedPaperProvider(),
        config=verification_config,
    )
    result = verifier.verify_solution(
        active_gap=state.active_gap,
        current_solution=state.current_solution,
        seed_context=case.seed_context,
        related_paper_pool=_copy_pool(case.related_paper_pool),
        verification_focus=VerificationFocus(mode="full_solution"),
        memory_store=memory_store,
        state_store=state_store,
        step=case.checkpoint.checkpoint_step + 2,
    )
    if not isinstance(result, SolutionVerificationResult):
        raise SerializationError("unified verification returned an invalid result")
    records = getattr(llm_client, "records", [])
    return {
        "paper_id": case.paper_id,
        "branch": branch,
        "revised_solution_id": solution.id,
        "revised_solution_hash": stable_hash(solution),
        "related_paper_pool_hash": stable_hash(case.related_paper_pool),
        "verification_focus": {"mode": "full_solution", "changed_units": []},
        "outcome": result.novelty_status,
        "novelty_relevant_overlaps": [
            overlap.model_dump(mode="json")
            for overlap in result.novelty_relevant_overlaps
        ],
        "matched_paper_refs": [
            overlap.matched_paper_ref
            for overlap in result.novelty_relevant_overlaps
        ],
        "verification_summary": result.summary,
        "related_papers": list(result.related_papers),
        "compared_papers": [
            comparison.model_dump(mode="json")
            for comparison in verifier.last_solution_comparisons
        ],
        "verification_result": result.model_dump(mode="json"),
        "model_config": _metadata_payload(
            verifier.last_call_metadata,
            verifier=verifier,
        ),
        "logical_verification_call_count": verifier.call_count,
        "physical_llm_records": len(records),
        "llm_call_audit": [
            record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            for record in records
        ],
        "related_paper_pool_reused": True,
        "new_related_paper_retrievals": verifier.scientific_search_count,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_unified_manifest(
    output_dir: Path,
    *,
    source_dir: Path,
    e3_output_dir: Path,
    deepseek: DeepSeekConfig | None,
    verification_config: VerificationConfig,
    case_count: int,
) -> None:
    write_json(
        output_dir / "execution_manifest.json",
        "e3_unified_reverification_manifest",
        {
            "experiment": "E3_unified_reverification",
            "source_dir": str(source_dir),
            "e3_output_dir": str(e3_output_dir),
            "output_dir": str(output_dir),
            "case_count": case_count,
            "branch_count": case_count * 2,
            "verification_focus": "full_solution",
            "new_related_paper_retrievals": 0,
            "inspiration_reruns": 0,
            "revision_reruns": 0,
            "verification_config": {
                "temperature": 0.0,
                "max_tokens": verification_config.max_tokens,
            },
            "deepseek": deepseek.public_dict() if deepseek is not None else None,
            "created_at": _now(),
        },
    )


def run_unified_case(
    case: UnifiedVerificationCase,
    output_dir: str | Path,
    *,
    deepseek: DeepSeekConfig,
    verification_config: VerificationConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run missing branch verifications for one case; never rerun revision."""

    root = Path(output_dir)
    case_output = root / "unified_verification" / case.paper_id
    case_output.mkdir(parents=True, exist_ok=True)
    branch_results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for branch in ("targeted", "global"):
        artifact_path = case_output / f"unified_verification_{branch}.json"
        if artifact_path.exists() and not overwrite:
            branch_results[branch] = _payload(
                artifact_path,
                f"e3_unified_verification_{branch}",
            )
            continue
        client = RecordingLLMClient(
            DeepSeekClient(deepseek),
            unbounded_output=True,
        )
        try:
            artifact = _run_branch(
                case,
                branch,  # type: ignore[arg-type]
                client,
                verification_config,
            )
            write_json(
                artifact_path,
                f"e3_unified_verification_{branch}",
                artifact,
            )
            branch_results[branch] = artifact
        except Exception as exc:
            error = {
                "paper_id": case.paper_id,
                "branch": branch,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2_000],
                "updated_at": _now(),
                "llm_call_audit": [
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else record
                    for record in getattr(client, "records", [])
                ],
            }
            failures.append(error)
            write_json(case_output / f"error_{branch}.json", "e3_unified_verification_error", error)
        else:
            write_jsonl(
                case_output / f"llm_calls_{branch}.jsonl",
                "llm_call_audit",
                getattr(client, "records", []),
            )
    status = "completed" if not failures and len(branch_results) == 2 else "failed"
    metadata = {
        "paper_id": case.paper_id,
        "status": status,
        "trigger_type": case.historical_local.trigger_type,
        "targeted_completed": "targeted" in branch_results,
        "global_completed": "global" in branch_results,
        "updated_at": _now(),
        "new_related_paper_retrievals": 0,
        "revision_reruns": 0,
        "failures": failures,
    }
    write_json(case_output / "run_metadata.json", "e3_unified_case_metadata", metadata)
    if failures:
        write_json(case_output / "error.json", "e3_unified_verification_error", {"failures": failures})
    return metadata


__all__ = [
    "CaseDiscovery",
    "UnifiedVerificationCase",
    "discover_unified_cases",
    "run_unified_case",
    "stable_hash",
    "write_unified_manifest",
]
