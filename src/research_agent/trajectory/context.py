"""Lightweight runtime container for one Seed Paper trajectory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..memory.store import ResearchMemoryStore
from ..models.memory import ResearchMemory
from ..models.seed import SeedPaperContext
from ..models.state import ResearchState
from ..models.trajectory import BudgetMetadata, TrajectoryEvent
from ..runtime.state_store import ResearchStateStore
from .diff import compute_state_diff
from .recorder import TrajectoryRecorder


class TrajectoryContext:
    """Aggregates deterministic runtime resources without scientific behavior."""

    def __init__(
        self,
        seed_context: SeedPaperContext,
        *,
        state_store: ResearchStateStore | None = None,
        memory_store: ResearchMemoryStore | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.seed_context = SeedPaperContext.model_validate(
            seed_context.model_dump(mode="python")
        )
        self.paper_id = self.seed_context.paper_id
        self.state_store = state_store or ResearchStateStore()
        self.memory_store = memory_store or ResearchMemoryStore(self.paper_id)
        if self.memory_store.seed_paper_id != self.paper_id:
            raise ValueError("Memory store seed paper does not match SeedPaperContext")
        self.trajectory = recorder or TrajectoryRecorder()

    @classmethod
    def initialize(cls, seed_context: SeedPaperContext) -> "TrajectoryContext":
        context = cls(seed_context)
        context.trajectory.append_event(
            0,
            "trajectory_initialized",
            result_ref=context.paper_id,
            result_payload={"paper_id": context.paper_id},
        )
        return context

    def get_state(self) -> ResearchState:
        return self.state_store.get_state()

    def get_memory(self) -> ResearchMemory:
        return self.memory_store.get_memory()

    def record_transition(
        self,
        step: int,
        event_type: str,
        before_state: ResearchState,
        after_state: ResearchState,
        *,
        action=None,
        result_ref: str | None = None,
        result_payload: dict[str, Any] | None = None,
        memory_writes: list[str] | None = None,
        budget: BudgetMetadata | dict[str, Any] | None = None,
        error: str | None = None,
        timestamp: datetime | None = None,
    ) -> TrajectoryEvent:
        before = ResearchState.model_validate(before_state.model_dump(mode="python"))
        after = ResearchState.model_validate(after_state.model_dump(mode="python"))
        event = TrajectoryEvent(
            step=step,
            event_type=event_type,
            action=action,
            result_ref=result_ref,
            result_payload=result_payload,
            state_diff=compute_state_diff(before, after),
            memory_writes=memory_writes or [],
            budget=budget,
            error=error,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        recorded = self.trajectory.append(event)
        self.state_store.replace_state(after)
        return recorded

    def record_gap_agenda(self, step: int, gap_agenda, **kwargs) -> TrajectoryEvent:
        return self.trajectory.record_gap_agenda(step, gap_agenda, **kwargs)

    def get_latest_gap_agenda(self):
        return self.trajectory.get_latest_gap_agenda()

    def record_gap_refined(self, step: int, gap_id: str, **kwargs) -> TrajectoryEvent:
        return self.trajectory.record_gap_refined(step, gap_id, **kwargs)

    def has_refined_gap(self, gap_id: str) -> bool:
        return self.trajectory.has_refined_gap(gap_id)

    def persist(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return {
            "research_state": self.state_store.save(root / "research_state.json"),
            "research_memory": self.memory_store.save(root / "research_memory.json"),
            "trajectory": self.trajectory.save(root / "trajectory.jsonl"),
        }

    @classmethod
    def load(
        cls,
        seed_context: SeedPaperContext,
        output_dir: str | Path,
    ) -> "TrajectoryContext":
        root = Path(output_dir)
        state_store = ResearchStateStore.load(root / "research_state.json")
        memory_store = ResearchMemoryStore.load(
            root / "research_memory.json", seed_context.paper_id
        )
        recorder = TrajectoryRecorder.load(root / "trajectory.jsonl")
        return cls(
            seed_context,
            state_store=state_store,
            memory_store=memory_store,
            recorder=recorder,
        )
