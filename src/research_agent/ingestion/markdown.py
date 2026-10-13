"""Deterministic parser and full-paper reading context for MinerU Markdown.

The parser deliberately preserves source text. It identifies layout boundaries
and removes only the references/appendix tail from the LLM reading context; it
does not select scientific sections or infer content from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_EQUATION_RE = re.compile(
    r"\$\$.*?\$\$|<eq\b.*?</eq\s*>|\\\[.*?\\\]",
    re.IGNORECASE | re.DOTALL,
)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\n)]*\)")


@dataclass(frozen=True)
class MarkdownBlock:
    kind: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class MarkdownSection:
    section_id: str
    heading: str
    level: int
    start_line: int
    end_line: int
    text: str
    parent_section_id: str | None = None
    is_appendix: bool = False
    is_references: bool = False


@dataclass(frozen=True)
class ParsedMarkdown:
    paper_id: str
    source_path: str | None
    lines: tuple[str, ...]
    title: str | None
    abstract: str | None
    title_section_id: str | None
    abstract_section_id: str | None
    sections: tuple[MarkdownSection, ...]
    references_start_line: int | None
    appendix_start_line: int | None
    html_tables: tuple[MarkdownBlock, ...]
    equation_blocks: tuple[MarkdownBlock, ...]
    image_placeholders: tuple[MarkdownBlock, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FullPaperReadingContext:
    parsed: ParsedMarkdown
    body_sections: tuple[MarkdownSection, ...]
    cut_line: int | None
    warnings: tuple[str, ...]

    @property
    def paper_id(self) -> str:
        return self.parsed.paper_id

    @property
    def title(self) -> str | None:
        return self.parsed.title

    @property
    def abstract(self) -> str | None:
        return self.parsed.abstract

    @property
    def body_section_ids(self) -> tuple[str, ...]:
        return tuple(section.section_id for section in self.body_sections)

    @property
    def body_end_line(self) -> int:
        return self.cut_line - 1 if self.cut_line is not None else len(self.parsed.lines)

    @property
    def body_text(self) -> str:
        return "\n".join(self.parsed.lines[: self.body_end_line])

    def render(self) -> str:
        parts = [f"PAPER_ID: {self.paper_id}"]
        parts.append("AVAILABLE MAIN-PAPER SOURCE SECTIONS:")
        for section in self.body_sections:
            parts.append(
                f"{section.section_id} LINES {section.start_line}-{section.end_line}: "
                f"{section.heading}"
            )
        parts.append(
            f"FULL MAIN-PAPER BODY LINES 1-{self.body_end_line}:\n{self.body_text}"
        )
        return "\n\n".join(parts)


def normalize_plain_text(text: str) -> str:
    """Only normalize whitespace; preserve paragraph and scientific content."""

    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))
    return "\n\n".join(paragraphs).strip()


def _heading_key(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip()


def _is_references_heading(heading: str) -> bool:
    # Only an exact normalized heading is a tail boundary.  In particular,
    # headings such as "References to Prior Work" remain part of the paper.
    return _heading_key(heading) in {"reference", "references", "bibliography"}


def _is_abstract_heading(heading: str) -> bool:
    return bool(re.match(r"^abstract(?:\b|\s|:)", _heading_key(heading)))


def _is_mineru_pseudo_heading(heading: str) -> bool:
    # MinerU can turn the first line of a pseudocode block into a level-one
    # heading (for example, "# param p, update u ..."). Keep it in the
    # surrounding section instead of treating it as a scientific section.
    return _heading_key(heading).startswith("param ")


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _find_blocks(text: str, pattern: re.Pattern[str], kind: str) -> tuple[MarkdownBlock, ...]:
    return tuple(
        MarkdownBlock(
            kind=kind,
            text=match.group(0),
            start_line=_line_for_offset(text, match.start()),
            end_line=_line_for_offset(text, match.end()),
        )
        for match in pattern.finditer(text)
    )


def parse_mineru_markdown(
    text: str,
    paper_id: str,
    source_path: str | None = None,
) -> ParsedMarkdown:
    if not paper_id or not paper_id.strip():
        raise ValueError("paper_id must be non-empty")
    lines = tuple(text.splitlines())
    headings: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match and not _is_mineru_pseudo_heading(match.group(2)):
            headings.append((line_number, len(match.group(1)), match.group(2).strip()))

    warnings: list[str] = []
    sections: list[MarkdownSection] = []
    parent_stack: list[MarkdownSection] = []
    references_start_line: int | None = None
    for index, (start_line, level, heading) in enumerate(headings):
        end_line = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        while parent_stack and parent_stack[-1].level >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1].section_id if parent_stack else None
        is_references = _is_references_heading(heading)
        if is_references and references_start_line is None:
            references_start_line = start_line
        section = MarkdownSection(
            section_id=f"S{index + 1}",
            heading=heading,
            level=level,
            start_line=start_line,
            end_line=end_line,
            text="\n".join(lines[start_line - 1 : end_line]),
            parent_section_id=parent_id,
            # Kept as a compatibility field; Appendix/Supplementary headings
            # no longer participate in main-body boundary detection.
            is_appendix=False,
            is_references=is_references,
        )
        sections.append(section)
        parent_stack.append(section)

    title_section = next(
        (
            section
            for section in sections
            if section.level == 1
            and not section.is_appendix
            and not section.is_references
            and not _is_abstract_heading(section.heading)
        ),
        None,
    )
    title = normalize_plain_text(title_section.heading) if title_section else None
    if title is None:
        warnings.append("title_not_found")

    abstract_section = next(
        (section for section in sections if _is_abstract_heading(section.heading)), None
    )
    abstract = None
    if abstract_section is not None:
        body_lines = lines[abstract_section.start_line : abstract_section.end_line]
        abstract = normalize_plain_text("\n".join(body_lines)) or None
    else:
        warnings.append("abstract_not_found")

    if _TABLE_RE.search(text) and text.lower().count("<table") != text.lower().count("</table"):
        warnings.append("unterminated_html_table")
    if text.count("$$") % 2:
        warnings.append("unterminated_dollar_equation")

    return ParsedMarkdown(
        paper_id=paper_id.strip(),
        source_path=source_path,
        lines=lines,
        title=title,
        abstract=abstract,
        title_section_id=title_section.section_id if title_section else None,
        abstract_section_id=abstract_section.section_id if abstract_section else None,
        sections=tuple(sections),
        references_start_line=references_start_line,
        # Kept as a compatibility field; it is intentionally never populated.
        appendix_start_line=None,
        html_tables=_find_blocks(text, _TABLE_RE, "html_table"),
        equation_blocks=_find_blocks(text, _EQUATION_RE, "equation"),
        image_placeholders=_find_blocks(text, _IMAGE_RE, "image_placeholder"),
        warnings=tuple(warnings),
    )


def parse_markdown_file(path: str | Path) -> ParsedMarkdown:
    source = Path(path)
    return parse_mineru_markdown(
        source.read_text(encoding="utf-8"),
        paper_id=source.stem,
        source_path=str(source),
    )


def build_full_paper_reading_context(parsed: ParsedMarkdown) -> FullPaperReadingContext:
    """Build the complete main-paper input without scientific section filtering.

    The first exact References/Reference/Bibliography heading is the only
    deterministic tail boundary. The body is never split or truncated based on
    Appendix/Supplementary headings or character count.
    """

    cut_line = parsed.references_start_line
    body_end_line = cut_line - 1 if cut_line is not None else len(parsed.lines)
    body_sections = tuple(
        section for section in parsed.sections if section.start_line <= body_end_line
    )
    warnings = list(parsed.warnings)
    if parsed.references_start_line is not None:
        warnings.append(f"references_removed_from_line:{parsed.references_start_line}")
    return FullPaperReadingContext(
        parsed=parsed,
        body_sections=body_sections,
        cut_line=cut_line,
        warnings=tuple(warnings),
    )
