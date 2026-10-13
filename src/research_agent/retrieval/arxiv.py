"""Small provider-neutral adapter for the official arXiv Atom search API.

The adapter only retrieves and normalizes metadata.  It does not download
papers, scrape HTML, or perform any scientific interpretation.
"""

from __future__ import annotations

import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..errors import (
    RetrievalConfigurationError,
    RetrievalHTTPError,
    RetrievalTransportError,
)
from ..models.memory import PaperRecord
from .arxiv_cache import ArxivCache


class TextGetTransport(Protocol):
    def get_text(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> str:
        ...


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value.strip())
        return delay if math.isfinite(delay) and delay >= 0 else None
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class UrllibTextGetTransport:
    """Injectable standard-library GET transport for Atom XML."""

    def get_text(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> str:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured adapter URL
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise RetrievalHTTPError(
                exc.code,
                f"arXiv HTTP error {exc.code}: {detail[:512]}",
                retry_after_seconds=_parse_retry_after(exc.headers.get("Retry-After")),
            ) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RetrievalTransportError(
                f"arXiv transport error: {reason}", retryable=True
            ) from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RetrievalTransportError(
                "arXiv response was not valid UTF-8", retryable=False
            ) from exc


def _string_setting(
    values: Mapping[str, object], name: str, default: str
) -> str:
    raw = values.get(name)
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        raise RetrievalConfigurationError(f"{name} must be a non-empty string")
    return raw.strip().rstrip("/")


def _float_setting(
    values: Mapping[str, object], name: str, default: float, *, nonnegative: bool = False
) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RetrievalConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(value) or (value < 0 if nonnegative else value <= 0):
        qualifier = "non-negative" if nonnegative else "positive"
        raise RetrievalConfigurationError(f"{name} must be finite and {qualifier}")
    return value


def _int_setting(values: Mapping[str, object], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer") from exc
    if isinstance(raw, float) and raw != value:
        raise RetrievalConfigurationError(f"{name} must be an integer")
    if value < 0:
        raise RetrievalConfigurationError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ArxivConfig:
    base_url: str = "https://export.arxiv.org/api/query"
    timeout_seconds: float = 30.0
    max_retries: int = 1
    min_request_interval_seconds: float = 3.0
    # Direct test/injected configurations stay cache-free unless requested;
    # environment/file-backed runtime configuration defaults to cache/arxiv.
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/"))
        if not self.base_url:
            raise RetrievalConfigurationError("arXiv base_url must be non-empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise RetrievalConfigurationError("arXiv timeout_seconds must be positive")
        if self.max_retries < 0:
            raise RetrievalConfigurationError("arXiv max_retries must be non-negative")
        if (
            not math.isfinite(self.min_request_interval_seconds)
            or self.min_request_interval_seconds < 0
        ):
            raise RetrievalConfigurationError(
                "arXiv min_request_interval_seconds must be non-negative"
            )
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", self.cache_dir.strip() or None)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ArxivConfig":
        allowed = {
            "base_url",
            "timeout_seconds",
            "max_retries",
            "min_request_interval_seconds",
            "cache_dir",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise RetrievalConfigurationError(
                "unknown arXiv config fields: " + ", ".join(unknown)
            )
        return cls(
            base_url=_string_setting(
                values, "base_url", "https://export.arxiv.org/api/query"
            ),
            timeout_seconds=_float_setting(values, "timeout_seconds", 30.0),
            max_retries=_int_setting(values, "max_retries", 1),
            min_request_interval_seconds=_float_setting(
                values,
                "min_request_interval_seconds",
                3.0,
                nonnegative=True,
            ),
            cache_dir=(
                _string_setting(values, "cache_dir", "cache/arxiv")
                if "cache_dir" not in values or values.get("cache_dir") is not None
                else None
            ),
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ArxivConfig":
        values = os.environ if environ is None else environ
        mapping: dict[str, object] = {}
        env_map = {
            "ARXIV_BASE_URL": "base_url",
            "ARXIV_TIMEOUT_SECONDS": "timeout_seconds",
            "ARXIV_MAX_RETRIES": "max_retries",
            "ARXIV_MIN_REQUEST_INTERVAL_SECONDS": "min_request_interval_seconds",
            "ARXIV_CACHE_DIR": "cache_dir",
        }
        for env_name, config_name in env_map.items():
            if env_name in values and values[env_name].strip():
                mapping[config_name] = values[env_name]
        return cls.from_mapping(mapping)


ATOM_NS = "http://www.w3.org/2005/Atom"
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = _WHITESPACE.sub(" ", "".join(element.itertext())).strip()
    return value or None


def canonical_arxiv_id(value: str) -> str:
    """Return a stable arXiv work identifier without a version suffix."""

    candidate = value.strip().split("?", 1)[0].split("#", 1)[0]
    if "/abs/" in candidate:
        candidate = candidate.rsplit("/abs/", 1)[1]
    elif candidate.startswith("arxiv:"):
        candidate = candidate.split(":", 1)[1]
    candidate = candidate.rstrip("/")
    return _VERSION_SUFFIX.sub("", candidate)


def _paper_from_entry(entry: ET.Element) -> PaperRecord:
    ident = _text(entry.find(f"{{{ATOM_NS}}}id"))
    title = _text(entry.find(f"{{{ATOM_NS}}}title"))
    if not ident or not title:
        raise RetrievalTransportError(
            "arXiv entry is missing id or title", retryable=False
        )
    canonical = canonical_arxiv_id(ident)
    if not canonical:
        raise RetrievalTransportError(
            "arXiv entry has an empty canonical id", retryable=False
        )
    authors: list[str] = []
    for author in entry.findall(f"{{{ATOM_NS}}}author"):
        name = _text(author.find(f"{{{ATOM_NS}}}name"))
        if name:
            authors.append(name)
    published = _text(entry.find(f"{{{ATOM_NS}}}published"))
    year = None
    if published and re.match(r"^\d{4}", published):
        year = int(published[:4])
    abstract = _text(entry.find(f"{{{ATOM_NS}}}summary"))
    journal_ref = _text(entry.find(f"{{{ATOM_NS}}}journal_ref"))
    return PaperRecord(
        paper_id=f"arxiv:{canonical}",
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue=journal_ref,
        source="arxiv",
        url=ident,
    )


def _search_expression(query: str) -> str:
    terms = [part.strip() for part in re.split(r"\s+OR\s+", query) if part.strip()]
    if not terms:
        raise ValueError("inspiration search query must be non-empty")
    return " OR ".join(
        f'all:"{term.replace(chr(34), "")}"' for term in terms
    )


class ArxivInspirationSearchProvider:
    """Search arXiv once per module query and return at most 20 papers."""

    provider_name = "arxiv"

    def __init__(
        self,
        config: ArxivConfig | None = None,
        transport: TextGetTransport | None = None,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or ArxivConfig.from_env()
        self._transport = transport or UrllibTextGetTransport()
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request_at: float | None = None
        self.last_request_count = 0
        self.total_request_count = 0
        self.last_retry_count = 0
        self.last_rate_limit_count = 0
        self.last_pacing_waits: list[float] = []
        self.last_retry_waits: list[float] = []
        self._cache = ArxivCache(self.config.cache_dir) if self.config.cache_dir else None
        self.last_cache_hit = False
        self.last_cache_scope = "none"

    def _pace(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        wait = max(0.0, self.config.min_request_interval_seconds - elapsed)
        if wait:
            self.last_pacing_waits.append(wait)
            self._sleep(wait)

    def _get(self, url: str) -> str:
        attempts = 0
        while True:
            self._pace()
            self._last_request_at = self._clock()
            attempts += 1
            self.last_request_count += 1
            self.total_request_count += 1
            try:
                return self._transport.get_text(
                    url,
                    {"Accept": "application/atom+xml", "User-Agent": "research-agent/0.1"},
                    self.config.timeout_seconds,
                )
            except (RetrievalHTTPError, RetrievalTransportError) as exc:
                retryable = getattr(exc, "retryable", False)
                if not retryable or attempts > self.config.max_retries:
                    raise
                self.last_retry_count += 1
                if isinstance(exc, RetrievalHTTPError) and exc.status_code == 429:
                    self.last_rate_limit_count += 1
                wait = (
                    exc.retry_after_seconds
                    if isinstance(exc, RetrievalHTTPError)
                    else None
                )
                if wait is None:
                    wait = 2.0 ** (attempts - 1)
                self.last_retry_waits.append(wait)
                self._sleep(wait)

    def search(self, query: str, limit: int = 20) -> list[PaperRecord]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("inspiration search query must be non-empty")
        if limit <= 0 or limit > 20:
            raise ValueError("inspiration search limit must be in 1..20")
        self.last_request_count = 0
        self.last_retry_count = 0
        self.last_rate_limit_count = 0
        self.last_pacing_waits = []
        self.last_retry_waits = []
        self.last_cache_hit = False
        self.last_cache_scope = "none"
        params = urlencode(
            {
                "search_query": _search_expression(query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        request_descriptor = {
            "base_url": self.config.base_url,
            "endpoint": "/api/query",
            "params": {
                "search_query": _search_expression(query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        }
        if self._cache is not None:
            raw, self.last_cache_hit = self._cache.get_or_fetch(
                "search", request_descriptor, lambda: self._get(f"{self.config.base_url}?{params}")
            )
            self.last_cache_scope = "global" if self.last_cache_hit else "none"
        else:
            raw = self._get(f"{self.config.base_url}?{params}")
        try:
            root = ET.fromstring(raw)
        except (ET.ParseError, TypeError) as exc:
            raise RetrievalTransportError(
                "arXiv response was not valid Atom XML", retryable=False
            ) from exc
        papers: list[PaperRecord] = []
        seen: set[str] = set()
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            paper = _paper_from_entry(entry)
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            papers.append(paper)
            if len(papers) == limit:
                break
        return papers


__all__ = [
    "ArxivConfig",
    "ArxivInspirationSearchProvider",
    "TextGetTransport",
    "UrllibTextGetTransport",
    "canonical_arxiv_id",
]
