"""Persistent, provider-specific cache for Semantic Scholar JSON responses.

The cache is intentionally independent from trajectory-local Research Memory.
It stores only successful, key-free HTTP response bodies and a safe request
descriptor.  Entries are permanent until manually removed, so a later
trajectory can reuse an earlier Semantic Scholar result without another API
request.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

try:  # pragma: no cover - available on the Linux execution target
    import fcntl
except ImportError:  # pragma: no cover - keeps the adapter importable elsewhere
    fcntl = None

from ..errors import RetrievalConfigurationError


class SemanticScholarCache:
    """A small content-addressed JSON cache shared by provider instances."""

    schema = "semantic_scholar_cache_entry"
    version = 1

    def __init__(self, cache_dir: str | Path) -> None:
        root = Path(cache_dir)
        if not str(root).strip():
            raise RetrievalConfigurationError("Semantic Scholar cache_dir must be non-empty")
        self.root = root

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
        key = self._request_key(operation, request)
        return self.root / operation / f"{key}.json"

    def get(
        self,
        operation: str,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        path = self.entry_path(operation, request)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalConfigurationError(
                f"Semantic Scholar cache entry is unreadable: {path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RetrievalConfigurationError(
                f"Semantic Scholar cache entry must be a JSON object: {path}"
            )
        expected_key = self._request_key(operation, request)
        if (
            payload.get("schema") != self.schema
            or payload.get("version") != self.version
            or payload.get("operation") != operation
            or payload.get("cache_key") != expected_key
        ):
            raise RetrievalConfigurationError(
                f"Semantic Scholar cache entry metadata mismatch: {path}"
            )
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise RetrievalConfigurationError(
                f"Semantic Scholar cache response must be a JSON object: {path}"
            )
        return dict(response)

    def put(
        self,
        operation: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> Path:
        if not isinstance(response, Mapping):
            raise RetrievalConfigurationError(
                "Semantic Scholar cache can store only JSON object responses"
            )
        path = self.entry_path(operation, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = self._request_key(operation, request)
        payload = {
            "schema": self.schema,
            "version": self.version,
            "provider": "semantic_scholar",
            "operation": operation,
            "cache_key": cache_key,
            "request": dict(request),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "response": dict(response),
        }
        temporary_path: str | None = None
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
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except (OSError, TypeError, ValueError) as exc:
            raise RetrievalConfigurationError(
                f"could not persist Semantic Scholar cache entry: {path}"
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
        fetcher: Callable[[], Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], bool]:
        """Read or populate one entry while holding its process-safe lock.

        The lock covers the cache miss check and the provider request.  This
        matters for parallel batch evaluation: two papers resolving the same
        provider request must not both pay for the same uncached call.
        """

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
