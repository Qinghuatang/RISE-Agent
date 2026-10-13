"""Deterministic normalization of paper references returned by verification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..errors import ModuleContractError


def canonicalize_paper_ref(value: str, available_papers: Iterable[Any]) -> str:
    """Return the unique canonical paper id for one allowed pool entry.

    Verification models are instructed to emit ``paper_ref`` identifiers, but
    a model may copy an exact title instead.  Exact title normalization is the
    only permitted repair: no fuzzy, substring, author, URL, or external
    lookup is performed.
    """

    papers = list(available_papers)
    paper_ids = [getattr(paper, "paper_id", None) for paper in papers]
    if value in paper_ids:
        return value

    matches = [
        paper
        for paper in papers
        if getattr(paper, "title", None) == value
    ]
    if len(matches) == 1:
        paper_id = getattr(matches[0], "paper_id", None)
        if isinstance(paper_id, str) and paper_id:
            return paper_id

    if len(matches) > 1:
        raise ModuleContractError(
            "paper reference matches multiple available paper titles exactly: "
            f"{value!r}"
        )
    raise ModuleContractError(
        "paper reference is neither an available canonical paper_id nor a "
        f"unique exact available title: {value!r}"
    )


def canonicalize_paper_refs(
    values: Iterable[str], available_papers: Iterable[Any]
) -> list[str]:
    """Canonicalize a sequence without changing order or deduplicating it."""

    papers = list(available_papers)
    return [canonicalize_paper_ref(value, papers) for value in values]


__all__ = ["canonicalize_paper_ref", "canonicalize_paper_refs"]
