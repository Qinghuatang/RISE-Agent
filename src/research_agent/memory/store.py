"""Simple trajectory-local Research Memory store.

The store persists records and applies only structural checks and deterministic
filters. It never decides whether an observation is scientifically a failure.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..errors import SerializationError
from ..models.memory import (
    DecisionRecord,
    FailureRecord,
    InspirationRecord,
    InspirationSearchRecord,
    PaperRecord,
    RelatedPaperPool,
    ResearchMemory,
    RevisionRecord,
    SearchRecord,
    VerificationRecord,
)
from ..serialization import read_json, write_json


MEMORY_SCHEMA = "research_memory"


def _copy_model(model, model_type):
    payload = model.model_dump(mode="python") if isinstance(model, model_type) else model
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid {model_type.__name__}") from exc


class ResearchMemoryStore:
    """Owns one seed-paper trajectory's ResearchMemory and nothing else."""

    def __init__(
        self,
        seed_paper_id: str,
        initial_memory: ResearchMemory | None = None,
    ) -> None:
        if not isinstance(seed_paper_id, str) or not seed_paper_id.strip():
            raise ValueError("seed_paper_id must be non-empty")
        self.seed_paper_id = seed_paper_id.strip()
        memory = initial_memory or ResearchMemory()
        self._memory = self._validated_memory(memory)
        self._check_pool_seed(self._memory)

    @staticmethod
    def _validated_memory(value: ResearchMemory | object) -> ResearchMemory:
        try:
            if isinstance(value, ResearchMemory):
                payload = value.model_dump(mode="python")
            else:
                payload = value
            return ResearchMemory.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid ResearchMemory") from exc

    def _check_pool_seed(self, memory: ResearchMemory) -> None:
        pool = memory.paper_memory.related_paper_pool
        if pool is not None and pool.seed_paper_id != self.seed_paper_id:
            raise ValueError(
                "RelatedPaperPool seed_paper_id does not match this trajectory"
            )

    def _commit(self, memory: ResearchMemory) -> None:
        validated = self._validated_memory(memory)
        self._check_pool_seed(validated)
        self._memory = validated

    def get_memory(self) -> ResearchMemory:
        return self._validated_memory(self._memory)

    def replace_memory(self, new_memory: ResearchMemory) -> ResearchMemory:
        """Replace the validated runtime snapshot for engineering replay.

        Scientific modules continue to use the narrow record-level methods
        below.  This whole-snapshot operation is intentionally exposed for
        replay reconstruction and does not infer or merge scientific data.
        """

        validated = self._validated_memory(new_memory)
        self._check_pool_seed(validated)
        self._commit(validated)
        return self.get_memory()

    def save_paper(self, paper: PaperRecord) -> None:
        paper_copy = _copy_model(paper, PaperRecord)
        memory = self.get_memory()
        papers = memory.paper_memory.papers
        for index, existing in enumerate(papers):
            if existing.paper_id == paper_copy.paper_id:
                if existing != paper_copy:
                    papers[index] = paper_copy
                self._commit(memory)
                return
        papers.append(paper_copy)
        self._commit(memory)

    def set_related_paper_pool(self, pool: RelatedPaperPool) -> None:
        pool_copy = _copy_model(pool, RelatedPaperPool)
        if pool_copy.seed_paper_id != self.seed_paper_id:
            raise ValueError(
                "RelatedPaperPool seed_paper_id does not match this trajectory"
            )
        memory = self.get_memory()
        memory.paper_memory.related_paper_pool = pool_copy
        self._commit(memory)

    def save_inspiration_search(self, record: InspirationSearchRecord) -> None:
        memory = self.get_memory()
        memory.paper_memory.inspiration_search_records.append(
            _copy_model(record, InspirationSearchRecord)
        )
        self._commit(memory)

    def save_search_record(self, record: SearchRecord) -> None:
        memory = self.get_memory()
        memory.paper_memory.search_records.append(_copy_model(record, SearchRecord))
        self._commit(memory)

    def save_verification(self, record: VerificationRecord) -> None:
        memory = self.get_memory()
        memory.verification_memory.append(_copy_model(record, VerificationRecord))
        self._commit(memory)

    def save_inspiration(self, record: InspirationRecord) -> None:
        memory = self.get_memory()
        memory.inspiration_memory.append(_copy_model(record, InspirationRecord))
        self._commit(memory)

    def save_revision(self, record: RevisionRecord) -> None:
        memory = self.get_memory()
        memory.revision_memory.append(_copy_model(record, RevisionRecord))
        self._commit(memory)

    def save_failure(self, record: FailureRecord) -> None:
        """Persist an explicitly classified failure; never infer one here."""

        memory = self.get_memory()
        memory.failure_memory.append(_copy_model(record, FailureRecord))
        self._commit(memory)

    def save_decision(self, record: DecisionRecord) -> None:
        memory = self.get_memory()
        memory.decision_log.append(_copy_model(record, DecisionRecord))
        self._commit(memory)

    def get_related_papers(self) -> list[PaperRecord]:
        pool = self._memory.paper_memory.related_paper_pool
        if pool is None:
            return []
        return [_copy_model(paper, PaperRecord) for paper in pool.papers]

    def get_latest_verification(
        self,
        target_ref: str,
        task: str | None = None,
    ) -> VerificationRecord | None:
        matches = [
            (index, record)
            for index, record in enumerate(self._memory.verification_memory)
            if record.target_ref == target_ref and (task is None or record.task == task)
        ]
        if not matches:
            return None
        _, record = max(matches, key=lambda item: (item[1].created_at_step, item[0]))
        return _copy_model(record, VerificationRecord)

    def get_latest_gap_verification(
        self,
        active_gap_id: str,
    ) -> VerificationRecord | None:
        return self.get_latest_verification(active_gap_id, task="VERIFY_GAP")

    def get_relevant_failures(
        self,
        failure_type: str | None = None,
        object_ref: str | None = None,
        recent_n: int | None = None,
    ) -> list[FailureRecord]:
        records = [
            record
            for record in self._memory.failure_memory
            if (failure_type is None or record.type == failure_type)
            and (object_ref is None or record.object_ref == object_ref)
        ]
        if recent_n is not None:
            if recent_n < 0:
                raise ValueError("recent_n must be non-negative")
            records = records[-recent_n:] if recent_n else []
        return [_copy_model(record, FailureRecord) for record in records]

    def get_relevant_inspirations(
        self,
        record_id: str | None = None,
        latest_n: int | None = None,
    ) -> list[InspirationRecord]:
        records = [
            record
            for record in self._memory.inspiration_memory
            if record_id is None or record.id == record_id
        ]
        if latest_n is not None:
            if latest_n < 0:
                raise ValueError("latest_n must be non-negative")
            records = records[-latest_n:] if latest_n else []
        return [_copy_model(record, InspirationRecord) for record in records]

    def get_revision_history(self, recent_n: int | None = None) -> list[RevisionRecord]:
        records = self._memory.revision_memory
        if recent_n is not None:
            if recent_n < 0:
                raise ValueError("recent_n must be non-negative")
            records = records[-recent_n:] if recent_n else []
        return [_copy_model(record, RevisionRecord) for record in records]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, MEMORY_SCHEMA, self._memory)
        return target

    @classmethod
    def load(cls, path: str | Path, seed_paper_id: str) -> "ResearchMemoryStore":
        try:
            envelope = read_json(path, expected_schema=MEMORY_SCHEMA)
            memory = ResearchMemory.model_validate(envelope["payload"])
        except (SerializationError, ValidationError) as exc:
            raise SerializationError("stored ResearchMemory is invalid") from exc
        try:
            return cls(seed_paper_id, memory)
        except ValueError as exc:
            raise SerializationError("stored ResearchMemory has the wrong seed paper") from exc
