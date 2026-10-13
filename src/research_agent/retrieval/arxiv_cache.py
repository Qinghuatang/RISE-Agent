"""Persistent, key-free cache for arXiv search response bodies."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from ..errors import RetrievalConfigurationError

try:  # pragma: no cover - available on the Linux execution target
    import fcntl
except ImportError:  # pragma: no cover - keeps the adapter importable elsewhere
    fcntl = None


class ArxivCache:
    """Content-addressed cache shared by parallel inspiration searches."""

    schema = "arxiv_cache_entry"
    version = 1

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)
        if not str(self.root).strip():
            raise RetrievalConfigurationError("arXiv cache_dir must be non-empty")

    @staticmethod
    def _request_key(operation: str, request: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {"operation": operation, "request": request},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def entry_path(self, operation: str, request: Mapping[str, Any]) -> Path:
        return self.root / operation / f"{self._request_key(operation, request)}.json"

    def get(self, operation: str, request: Mapping[str, Any]) -> str | None:
        path = self.entry_path(operation, request)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalConfigurationError(
                f"arXiv cache entry is unreadable: {path}"
            ) from exc
        expected_key = self._request_key(operation, request)
        if not isinstance(payload, Mapping) or any(
            payload.get(field) != expected
            for field, expected in (
                ("schema", self.schema),
                ("version", self.version),
                ("operation", operation),
                ("cache_key", expected_key),
            )
        ):
            raise RetrievalConfigurationError(f"arXiv cache entry metadata mismatch: {path}")
        response = payload.get("response_text")
        if not isinstance(response, str):
            raise RetrievalConfigurationError(f"arXiv cache response must be text: {path}")
        return response

    def put(self, operation: str, request: Mapping[str, Any], response: str) -> Path:
        if not isinstance(response, str):
            raise RetrievalConfigurationError("arXiv cache can store only text responses")
        path = self.entry_path(operation, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        payload = {
            "schema": self.schema,
            "version": self.version,
            "provider": "arxiv",
            "operation": operation,
            "cache_key": self._request_key(operation, request),
            "request": dict(request),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "response_text": response,
        }
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except (OSError, TypeError, ValueError) as exc:
            raise RetrievalConfigurationError(
                f"could not persist arXiv cache entry: {path}"
            ) from exc
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        return path

    def get_or_fetch(
        self,
        operation: str,
        request: Mapping[str, Any],
        fetcher: Callable[[], str],
    ) -> tuple[str, bool]:
        path = self.entry_path(operation, request)
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                cached = self.get(operation, request)
                if cached is not None:
                    return cached, True
                response = fetcher()
                self.put(operation, request, response)
                return response, False
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


__all__ = ["ArxivCache"]
