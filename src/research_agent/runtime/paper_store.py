"""Immutable, content-addressed storage for paper Markdown used by graph runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from ..ingestion.markdown import (
    FullPaperReadingContext,
    build_full_paper_reading_context,
    parse_mineru_markdown,
)
from .graph_state import PaperInputReference


class PaperInputIntegrityError(RuntimeError):
    """Raised when persisted paper input is missing, unsafe, or has changed."""


class PaperInputStore:
    """Persist raw Markdown bytes outside checkpoints and verify every read.

    Files are addressed by SHA-256 and are never edited in place. ``root``
    should be a stable runtime data directory shared by API and worker
    processes; it must not point at the repository's source Markdown files.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def persist(self, markdown: str | bytes | Path) -> PaperInputReference:
        if isinstance(markdown, Path):
            content = markdown.read_bytes()
        elif isinstance(markdown, bytes):
            content = markdown
        elif isinstance(markdown, str):
            content = markdown.encode("utf-8")
        else:
            raise TypeError("markdown must be UTF-8 text, bytes, or a Path")

        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PaperInputIntegrityError("paper Markdown is not valid UTF-8") from exc

        digest = hashlib.sha256(content).hexdigest()
        locator = f"papers/{digest}.md"
        target = self._resolve_locator(locator)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        if target.exists():
            if self._sha256(target.read_bytes()) != digest:
                raise PaperInputIntegrityError(
                    "content-addressed paper input exists with a mismatched digest"
                )
            return PaperInputReference(locator=locator, sha256=digest)

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o400)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if self._sha256(target.read_bytes()) != digest:
                    raise PaperInputIntegrityError(
                        "concurrent paper write produced a mismatched digest"
                    )
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

        return PaperInputReference(locator=locator, sha256=digest)

    def read(self, reference: PaperInputReference) -> bytes:
        path = self._resolve_locator(reference.locator)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise PaperInputIntegrityError(
                f"persisted paper input is missing: {reference.locator}"
            ) from exc
        actual_digest = self._sha256(content)
        if actual_digest != reference.sha256:
            raise PaperInputIntegrityError(
                f"persisted paper input SHA-256 mismatch: {reference.locator}"
            )
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PaperInputIntegrityError(
                f"persisted paper input is not valid UTF-8: {reference.locator}"
            ) from exc
        return content

    def build_reading_context(
        self,
        reference: PaperInputReference,
        *,
        paper_id: str,
    ) -> FullPaperReadingContext:
        raw = self.read(reference).decode("utf-8")
        parsed = parse_mineru_markdown(
            raw,
            paper_id=paper_id,
            source_path=str(self._resolve_locator(reference.locator)),
        )
        return build_full_paper_reading_context(parsed)

    def _resolve_locator(self, locator: str) -> Path:
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts:
            raise PaperInputIntegrityError("paper locator must be a safe relative path")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PaperInputIntegrityError("paper locator escapes the storage root") from exc
        return candidate

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


__all__ = ["PaperInputIntegrityError", "PaperInputStore"]
