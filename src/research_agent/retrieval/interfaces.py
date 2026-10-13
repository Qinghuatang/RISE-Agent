"""Small provider interfaces used by scientific modules."""

from __future__ import annotations

from typing import Protocol

from ..models.memory import PaperRecord, RelatedPaperPool
from ..models.seed import SeedPaperContext


class RelatedPaperProvider(Protocol):
    """Acquire one seed-centric related-paper pool.

    Implementations may perform provider-specific HTTP requests, but the
    scientific module only sees normalized canonical paper records.
    """

    last_request_count: int

    def get_related_paper_pool(
        self,
        seed_context: SeedPaperContext,
        *,
        limit: int = 20,
    ) -> RelatedPaperPool:
        ...

    def get_related_papers(
        self,
        seed_context: SeedPaperContext,
        *,
        limit: int = 20,
    ) -> RelatedPaperPool:
        ...


class InspirationSearchProvider(Protocol):
    """Search transferable inspiration papers for one combined query."""

    last_request_count: int

    def search(self, query: str, limit: int = 20) -> list[PaperRecord]:
        ...
