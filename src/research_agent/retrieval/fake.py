"""Deterministic related-paper provider for offline tests."""

from __future__ import annotations

from ..models.memory import PaperRecord, RelatedPaperPool
from ..models.seed import SeedPaperContext


def _copy_paper(paper: PaperRecord) -> PaperRecord:
    return PaperRecord.model_validate(paper.model_dump(mode="python"))


class FakeRelatedPaperProvider:
    """Returns a fixed normalized pool and never performs network I/O."""

    provider_name = "fake"

    def __init__(self, papers: list[PaperRecord] | None = None) -> None:
        self.papers = [_copy_paper(paper) for paper in (papers or [])]
        self.calls: list[str] = []
        self.last_request_count = 0

    def get_related_paper_pool(
        self,
        seed_context: SeedPaperContext,
        *,
        limit: int = 20,
    ) -> RelatedPaperPool:
        if limit <= 0 or limit > 20:
            raise ValueError("related-paper limit must be in 1..20")
        self.calls.append(seed_context.paper_id)
        self.last_request_count = 0
        seen: set[str] = set()
        normalized: list[PaperRecord] = []
        for paper in self.papers:
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            normalized.append(_copy_paper(paper))
            if len(normalized) == limit:
                break
        return RelatedPaperPool(seed_paper_id=seed_context.paper_id, papers=normalized)

    fetch_related_papers = get_related_paper_pool
    get_related_papers = get_related_paper_pool


class FakeInspirationSearchProvider:
    """Fixed paper provider for M6 tests; it never performs network I/O."""

    provider_name = "fake_inspiration"

    def __init__(
        self,
        papers: list[PaperRecord] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.papers = [_copy_paper(paper) for paper in (papers or [])]
        self.error = error
        self.queries: list[str] = []
        self.limits: list[int] = []
        self.last_request_count = 0
        self.last_retry_count = 0

    def search(self, query: str, limit: int = 20) -> list[PaperRecord]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("inspiration search query must be non-empty")
        if limit <= 0 or limit > 20:
            raise ValueError("inspiration search limit must be in 1..20")
        self.queries.append(query)
        self.limits.append(limit)
        self.last_request_count = 1
        if self.error is not None:
            raise self.error
        return [_copy_paper(paper) for paper in self.papers[:limit]]


__all__ = ["FakeRelatedPaperProvider", "FakeInspirationSearchProvider"]
