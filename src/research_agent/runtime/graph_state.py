"""Validated, serializable snapshot for the LangGraph RISE runtime.

The snapshot stores the existing scientific models. Runtime clients, module
instances, stores, and parsed paper contents deliberately remain outside it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from ..controller.budget import ControllerBudget
from ..memory.store import ResearchMemoryStore
from ..models.base import NonEmptyStr, StrictModel
from ..models.commit import CommitCheck, CommitSnapshot, FinalResearchProposal
from ..models.controller import ControllerRoutingResult
from ..models.memory import ResearchMemory
from ..models.runner import TrajectoryReplayRecord
from ..models.seed import SeedPaperContext
from ..models.state import ResearchState
from ..models.trajectory import TrajectoryEvent
from ..runtime.state_store import ResearchStateStore
from ..trajectory.context import TrajectoryContext
from ..trajectory.recorder import TrajectoryRecorder


class GraphStateValidationError(ValueError):
    """Raised when an external Graph State payload cannot be restored."""


class PaperInputReference(StrictModel):
    """Reference to immutable Markdown input held outside a graph checkpoint."""

    locator: NonEmptyStr
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RISEGraphState(StrictModel):
    """Single validated snapshot of one RISE trajectory.

    Values are replaced as a whole after each node. In particular, trajectory
    events and replay records are snapshots, not append reducers, so replaying
    a state update cannot duplicate their entries.
    """

    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    seed_context: SeedPaperContext
    paper_input: PaperInputReference | None = None

    research_state: ResearchState = Field(default_factory=ResearchState)
    research_memory: ResearchMemory = Field(default_factory=ResearchMemory)
    trajectory_events: list[TrajectoryEvent] = Field(default_factory=list)
    budget: ControllerBudget
    replay_records: list[TrajectoryReplayRecord] = Field(default_factory=list)

    controller_routing: ControllerRoutingResult | None = None
    commit_check: CommitCheck | None = None
    decision_pending: bool = False
    bootstrap_complete: bool = False

    commit_snapshot: CommitSnapshot | None = None
    final_proposal: FinalResearchProposal | None = None
    rise_status: Literal[
        "initialized",
        "running",
        "committed",
        "stopped",
        "failed",
        "finalization_failed",
    ] = "initialized"
    unresolved_issues: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_snapshot_consistency(self) -> "RISEGraphState":
        paper_id = self.seed_context.paper_id
        related_pool = self.research_memory.paper_memory.related_paper_pool
        if related_pool is not None and related_pool.seed_paper_id != paper_id:
            raise ValueError("ResearchMemory related-paper pool has a different seed")

        expected_sequences = list(range(1, len(self.replay_records) + 1))
        actual_sequences = [record.sequence for record in self.replay_records]
        if actual_sequences != expected_sequences:
            raise ValueError("replay record sequence must be contiguous from 1")
        if any(record.seed_paper_id != paper_id for record in self.replay_records):
            raise ValueError("replay record seed does not match SeedPaperContext")

        if self.decision_pending:
            if not self.bootstrap_complete or self.rise_status != "running":
                raise ValueError(
                    "a pending Controller decision requires a running trajectory"
                )
            if (
                self.controller_routing is None
                or self.controller_routing.status != "decision"
                or self.controller_routing.decision is None
            ):
                raise ValueError(
                    "a pending decision requires a complete Controller decision"
                )

        if self.rise_status == "initialized" and self.bootstrap_complete:
            raise ValueError("an initialized trajectory cannot have completed Bootstrap")
        if self.rise_status == "committed":
            if self.commit_snapshot is None or self.final_proposal is None:
                raise ValueError("a committed trajectory requires Snapshot and Proposal")
        elif self.final_proposal is not None:
            raise ValueError("Final Proposal is only valid for a committed trajectory")
        if self.rise_status == "finalization_failed" and self.commit_snapshot is None:
            raise ValueError("finalization_failed requires a preserved CommitSnapshot")
        if self.rise_status in {"committed", "stopped", "failed", "finalization_failed"}:
            if self.decision_pending:
                raise ValueError("a terminal trajectory cannot have a pending decision")
        return self

    @classmethod
    def initialize(
        cls,
        seed_context: SeedPaperContext,
        budget: ControllerBudget,
        *,
        run_id: str,
        paper_input: PaperInputReference | None = None,
    ) -> "RISEGraphState":
        seed = SeedPaperContext.model_validate(seed_context.model_dump(mode="python"))
        context = TrajectoryContext.initialize(seed)
        return cls.from_runtime(
            run_id=run_id,
            seed_context=seed,
            context=context,
            budget=budget,
            paper_input=paper_input,
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        run_id: str,
        seed_context: SeedPaperContext,
        context: TrajectoryContext,
        budget: ControllerBudget,
        paper_input: PaperInputReference | None = None,
        replay_records: list[TrajectoryReplayRecord] | None = None,
        **updates: Any,
    ) -> "RISEGraphState":
        """Capture complete runtime snapshots without merging old list fields."""

        payload: dict[str, Any] = {
            "run_id": run_id,
            "seed_context": seed_context,
            "paper_input": paper_input,
            "research_state": context.get_state(),
            "research_memory": context.get_memory(),
            "trajectory_events": context.trajectory.get_events(),
            "budget": budget,
            "replay_records": list(replay_records or []),
        }
        payload.update(updates)
        return cls.model_validate(payload)

    def to_trajectory_context(self) -> TrajectoryContext:
        """Rebuild the legacy mutable context as a temporary execution object."""

        try:
            state_store = ResearchStateStore(self.research_state)
            memory_store = ResearchMemoryStore(
                self.seed_context.paper_id,
                self.research_memory,
            )
            recorder = TrajectoryRecorder(self.trajectory_events)
            return TrajectoryContext(
                self.seed_context,
                state_store=state_store,
                memory_store=memory_store,
                recorder=recorder,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise GraphStateValidationError(
                "Graph State cannot reconstruct a valid TrajectoryContext"
            ) from exc

    def evolve(self, **updates: Any) -> "RISEGraphState":
        """Return a fully revalidated replacement snapshot."""

        payload = self.model_dump(mode="python")
        payload.update(updates)
        return type(self).model_validate(payload)

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible values suitable for a graph/checkpointer."""

        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_payload(cls, payload: Any) -> "RISEGraphState":
        try:
            return cls.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            raise GraphStateValidationError("invalid RISEGraphState payload") from exc

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RISEGraphState":
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise GraphStateValidationError("RISEGraphState is not valid JSON") from exc
        return cls.from_payload(decoded)


__all__ = [
    "GraphStateValidationError",
    "PaperInputReference",
    "RISEGraphState",
]
