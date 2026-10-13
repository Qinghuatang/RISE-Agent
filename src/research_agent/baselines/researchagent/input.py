"""Deterministic Seed Paper input adaptation for official ResearchAgent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...ingestion.markdown import ParsedMarkdown, parse_markdown_file


class SeedMarkdownInputError(ValueError):
    """Raised when the official baseline input cannot be extracted safely."""


@dataclass(frozen=True)
class SeedMarkdownInput:
    paper_id: str
    markdown_path: str
    title: str
    abstract: str
    title_section_id: str | None
    abstract_section_id: str | None
    parser_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_seed_markdown(path: str | Path) -> SeedMarkdownInput:
    """Extract only the title and abstract required by official ResearchAgent.

    The parser is shared with Ours, but no full-paper body is sent to the
    official pipeline.  Missing required fields are explicit errors rather
    than LLM-generated guesses.
    """

    source = Path(path)
    parsed: ParsedMarkdown = parse_markdown_file(source)
    if not parsed.title or not parsed.title.strip():
        raise SeedMarkdownInputError(f"title could not be determined: {source}")
    if not parsed.abstract or not parsed.abstract.strip():
        raise SeedMarkdownInputError(f"abstract could not be determined: {source}")
    return SeedMarkdownInput(
        paper_id=parsed.paper_id,
        markdown_path=str(source),
        title=parsed.title.strip(),
        abstract=parsed.abstract.strip(),
        title_section_id=parsed.title_section_id,
        abstract_section_id=parsed.abstract_section_id,
        parser_warnings=parsed.warnings,
    )


__all__ = ["SeedMarkdownInput", "SeedMarkdownInputError", "load_seed_markdown"]
