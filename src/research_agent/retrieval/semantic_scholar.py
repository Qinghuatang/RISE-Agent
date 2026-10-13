"""Official Semantic Scholar adapter for seed-centric recommendations.

The adapter owns HTTP, provider response normalization, and deterministic seed
identity resolution.  No scientific comparison or LLM reasoning belongs here.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Literal, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import json

from ..errors import (
    RetrievalConfigurationError,
    RetrievalHTTPError,
    RetrievalTransportError,
    SeedPaperResolutionError,
)
from ..models.memory import PaperRecord, RelatedPaperPool
from ..models.seed import SeedPaperContext
from .semantic_scholar_cache import SemanticScholarCache


class JsonGetTransport(Protocol):
    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        ...


class UrllibJsonGetTransport:
    """Injectable standard-library GET transport."""

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured adapter URL
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
            raise RetrievalHTTPError(
                exc.code,
                f"Semantic Scholar HTTP error {exc.code}: {detail[:512]}",
                retry_after_seconds=retry_after,
            ) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RetrievalTransportError(
                f"Semantic Scholar transport error: {reason}", retryable=True
            ) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalTransportError(
                "Semantic Scholar response was not valid JSON", retryable=False
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RetrievalTransportError(
                "Semantic Scholar response JSON must be an object", retryable=False
            )
        return decoded


def _optional_string(values: Mapping[str, Any], name: str) -> str | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RetrievalConfigurationError(f"{name} must be a string or null")
    value = value.strip()
    return value or None


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
        delay = (target - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delay)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_float(values: Mapping[str, Any], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = float(raw) if isinstance(raw, (int, float, str)) else None
    except (TypeError, ValueError) as exc:
        raise RetrievalConfigurationError(f"{name} must be a number") from exc
    if value is None:
        raise RetrievalConfigurationError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise RetrievalConfigurationError(f"{name} must be finite and positive")
    return value


def _nonnegative_float(values: Mapping[str, Any], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = float(raw) if isinstance(raw, (int, float, str)) else None
    except (TypeError, ValueError) as exc:
        raise RetrievalConfigurationError(f"{name} must be a number") from exc
    if value is None:
        raise RetrievalConfigurationError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise RetrievalConfigurationError(f"{name} must be finite and non-negative")
    return value


def _nonnegative_int(values: Mapping[str, Any], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        value = int(raw) if isinstance(raw, (int, float, str)) else None
    except (TypeError, ValueError) as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer") from exc
    if value is None:
        raise RetrievalConfigurationError(f"{name} must be an integer")
    if isinstance(raw, float) and raw != value:
        raise RetrievalConfigurationError(f"{name} must be an integer")
    if value < 0:
        raise RetrievalConfigurationError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class SemanticScholarConfig:
    """Runtime settings loaded from a local JSON configuration file."""

    api_key_environment_variable = "SEMANTIC_SCHOLAR_API_KEY"

    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    min_request_interval_seconds: float = 1.0
    cache_dir: str | None = None
    auth_mode: Literal["x-api-key", "bearer"] = "x-api-key"

    def __post_init__(self) -> None:
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip() or None)
        if self.base_url is not None:
            object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/") or None)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", self.cache_dir.strip() or None)
        if self.auth_mode not in {"x-api-key", "bearer"}:
            raise RetrievalConfigurationError(
                "auth_mode must be either 'x-api-key' or 'bearer'"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise RetrievalConfigurationError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise RetrievalConfigurationError("max_retries must be non-negative")
        if not math.isfinite(self.min_request_interval_seconds) or self.min_request_interval_seconds < 0:
            raise RetrievalConfigurationError(
                "min_request_interval_seconds must be finite and non-negative"
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "SemanticScholarConfig":
        allowed = {
            "api_key",
            "base_url",
            "timeout_seconds",
            "max_retries",
            "min_request_interval_seconds",
            "cache_dir",
            "auth_mode",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise RetrievalConfigurationError(
                "unknown Semantic Scholar config fields: " + ", ".join(unknown)
            )
        configured_api_key = _optional_string(values, "api_key")
        if configured_api_key is None:
            environment = os.environ if environment is None else environment
            configured_api_key = _optional_string(
                environment, cls.api_key_environment_variable
            )
        return cls(
            api_key=configured_api_key,
            base_url=_optional_string(values, "base_url"),
            timeout_seconds=_positive_float(
                values, "timeout_seconds", 30.0
            ),
            max_retries=_nonnegative_int(values, "max_retries", 1),
            min_request_interval_seconds=_nonnegative_float(
                values, "min_request_interval_seconds", 1.0
            ),
            cache_dir=(
                _optional_string(values, "cache_dir")
                if "cache_dir" in values
                else "cache/semantic_scholar"
            ),
            auth_mode=_optional_string(values, "auth_mode") or "x-api-key",
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "SemanticScholarConfig":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RetrievalConfigurationError(
                f"Semantic Scholar config file not found: {config_path}"
            ) from exc
        except OSError as exc:
            raise RetrievalConfigurationError(
                f"could not read Semantic Scholar config file: {config_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RetrievalConfigurationError(
                f"Semantic Scholar config file is not valid JSON: {config_path}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise RetrievalConfigurationError(
                "Semantic Scholar config file must contain a JSON object"
            )
        return cls.from_mapping(raw, environment=environment)

    def validate_for_request(self) -> None:
        if not self.base_url:
            raise RetrievalConfigurationError(
                "missing Semantic Scholar request configuration: "
                "base_url in the Semantic Scholar JSON config file"
            )

    def public_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "cache_dir": self.cache_dir,
            "auth_mode": self.auth_mode,
            "api_key_configured": self.api_key is not None,
        }


def normalize_title(value: str) -> str:
    """Normalize only presentation differences for deterministic matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _seed_version_rank(paper: PaperRecord) -> tuple[int, str]:
    """Return the deterministic provider-record preference tier."""

    types = {normalize_title(value).replace(" ", "") for value in paper.publication_types}
    venue = normalize_title(paper.venue or "")
    source = normalize_title(paper.source)
    if "journalarticle" in types or "journal" in venue:
        return 3, "journal_preferred"
    if "conference" in types or any(
        token in venue for token in ("conference", "proceedings", "symposium", "workshop")
    ):
        return 2, "conference_preferred"
    if source == "arxiv" or "arxiv" in venue or "preprint" in venue:
        return 1, "arxiv_preferred"
    return 0, "other"


def _select_seed_candidate(
    candidates: list[PaperRecord],
) -> tuple[PaperRecord, str]:
    """Select one exact-title record without semantic guessing."""

    if len(candidates) == 1:
        return candidates[0], "exact_unique"
    ranked = sorted(
        candidates,
        key=lambda paper: (
            -_seed_version_rank(paper)[0],
            -(1 if paper.abstract else 0),
            paper.paper_id,
        ),
    )
    selected = ranked[0]
    selected_rank, rank_reason = _seed_version_rank(selected)
    same_rank = [paper for paper in candidates if _seed_version_rank(paper)[0] == selected_rank]
    if len(same_rank) == 1 and len(candidates) > 1:
        return selected, rank_reason
    same_rank_with_abstract = [paper for paper in same_rank if paper.abstract]
    if len(same_rank_with_abstract) == 1 and not all(paper.abstract for paper in same_rank):
        return selected, "has_abstract"
    return selected, "stable_tiebreak"


def _paper_from_api(item: Mapping[str, Any]) -> PaperRecord:
    paper_id = item.get("paperId")
    title = item.get("title")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise RetrievalTransportError(
            "Semantic Scholar paper is missing paperId", retryable=False
        )
    if not isinstance(title, str) or not title.strip():
        raise RetrievalTransportError(
            f"Semantic Scholar paper {paper_id!r} is missing title", retryable=False
        )
    authors_raw = item.get("authors")
    authors: list[str] = []
    if isinstance(authors_raw, list):
        for author in authors_raw:
            if isinstance(author, Mapping) and isinstance(author.get("name"), str):
                authors.append(author["name"])
    year = item.get("year") if isinstance(item.get("year"), int) else None
    venue = item.get("venue") if isinstance(item.get("venue"), str) else None
    abstract = item.get("abstract") if isinstance(item.get("abstract"), str) else None
    publication_types_raw = item.get("publicationTypes")
    publication_types = (
        [
            value.strip()
            for value in publication_types_raw
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(publication_types_raw, list)
        else []
    )
    return PaperRecord(
        paper_id=paper_id.strip(),
        title=title.strip(),
        abstract=abstract.strip() if abstract and abstract.strip() else None,
        authors=authors,
        year=year,
        venue=venue,
        publication_types=publication_types,
        source="semantic_scholar",
        url=None,
    )


class SemanticScholarProvider:
    """Resolve a seed by title and retrieve official recommendations."""

    provider_name = "semantic_scholar"

    def __init__(
        self,
        config: SemanticScholarConfig,
        transport: JsonGetTransport | None = None,
        *,
        cache: SemanticScholarCache | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibJsonGetTransport()
        self._cache = cache or (
            SemanticScholarCache(config.cache_dir) if config.cache_dir else None
        )
        self.last_request_count = 0
        self.total_request_count = 0
        self.last_resolved_provider_id: str | None = None
        self.last_seed_resolution_audit: dict[str, Any] = {}
        self.last_retry_count = 0
        self.last_rate_limit_count = 0
        self.last_pacing_waits: list[float] = []
        self.last_retry_waits: list[float] = []
        self.last_global_cache_hit = False
        self.last_global_cache_operations: list[str] = []
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request_at: float | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticScholarProvider":
        return cls(SemanticScholarConfig.from_file(path))

    def _get(self, url: str) -> Mapping[str, Any]:
        self.config.validate_for_request()
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            if self.config.auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            else:
                headers["x-api-key"] = self.config.api_key
        last_error: RetrievalTransportError | None = None
        for attempt in range(self.config.max_retries + 1):
            self._pace_before_request()
            self.last_request_count += 1
            self.total_request_count += 1
            try:
                response = self._transport.get_json(
                    url, headers, self.config.timeout_seconds
                )
                return response
            except RetrievalHTTPError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                if exc.status_code == 429:
                    self.last_rate_limit_count += 1
                if attempt < self.config.max_retries:
                    self.last_retry_count += 1
                    self._sleep_before_retry(attempt, exc.retry_after_seconds)
            except RetrievalTransportError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                if attempt < self.config.max_retries:
                    self.last_retry_count += 1
                    self._sleep_before_retry(attempt, None)
        assert last_error is not None
        raise last_error

    def _get_with_global_cache(
        self,
        operation: str,
        request_descriptor: Mapping[str, Any],
        url: str,
    ) -> tuple[Mapping[str, Any], bool]:
        if self._cache is not None:
            payload, from_cache = self._cache.get_or_fetch(
                operation,
                request_descriptor,
                lambda: self._get(url),
            )
            if from_cache:
                self.last_global_cache_operations.append(operation)
            return payload, from_cache
        return self._get(url), False

    def _pace_before_request(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = self._clock()
            return
        elapsed = self._clock() - self._last_request_at
        delay = max(0.0, self.config.min_request_interval_seconds - elapsed)
        if delay:
            self.last_pacing_waits.append(delay)
            self._sleep(delay)
        self._last_request_at = self._clock()

    def _sleep_before_retry(
        self,
        attempt: int,
        retry_after_seconds: float | None,
    ) -> None:
        bounded_backoff = min(8.0, 0.5 * (2**attempt))
        delay = (
            retry_after_seconds
            if retry_after_seconds is not None
            else bounded_backoff
        )
        if delay:
            self.last_retry_waits.append(delay)
            self._sleep(delay)

    def resolve_seed_paper_id(self, seed_context: SeedPaperContext) -> str:
        if not seed_context.title or not seed_context.title.strip():
            raise SeedPaperResolutionError(
                "SeedPaperContext.title is required for Semantic Scholar resolution"
            )
        self.last_request_count = 0
        self.last_retry_count = 0
        self.last_rate_limit_count = 0
        self.last_pacing_waits = []
        self.last_retry_waits = []
        self.last_global_cache_hit = False
        self.last_global_cache_operations = []
        self.last_resolved_provider_id = None
        self.last_seed_resolution_audit = {}
        params = {
            "query": seed_context.title,
            "limit": 20,
            "fields": "paperId,title,abstract,year,venue,authors,publicationTypes",
        }
        query = urlencode(params)
        request_descriptor = {
            "base_url": self.config.base_url,
            "endpoint": "/graph/v1/paper/search",
            "params": params,
        }
        payload, from_cache = self._get_with_global_cache(
            "paper_search",
            request_descriptor,
            f"{self.config.base_url}/graph/v1/paper/search?{query}",
        )
        raw_items = payload.get("data")
        if not isinstance(raw_items, list):
            raise RetrievalTransportError(
                "Semantic Scholar paper search response is missing data[]",
                retryable=False,
            )
        candidates: list[PaperRecord] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise RetrievalTransportError(
                    "Semantic Scholar paper search item is malformed", retryable=False
                )
            candidates.append(_paper_from_api(item))
        wanted = normalize_title(seed_context.title)
        exact = [paper for paper in candidates if normalize_title(paper.title) == wanted]
        if exact:
            selected, reason = _select_seed_candidate(exact)
            self.last_resolved_provider_id = selected.paper_id
            self.last_seed_resolution_audit = {
                "candidate_count": len(exact),
                "selected_paper_id": selected.paper_id,
                "selected_title": selected.title,
                "selected_year": selected.year,
                "selected_venue": selected.venue,
                "selection_reason": reason,
            }
            return selected.paper_id
        if not candidates:
            raise SeedPaperResolutionError(
                "Semantic Scholar returned no candidates for the Seed Paper title"
            )
        scored = sorted(
            [
                (
                    SequenceMatcher(None, wanted, normalize_title(paper.title)).ratio(),
                    paper,
                )
                for paper in candidates
            ],
            key=lambda item: item[0],
        )
        best_score, best = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else 0.0
        if best_score < 0.96 or best_score - second_score < 0.03:
            raise SeedPaperResolutionError(
                "Semantic Scholar seed resolution found no unique conservative title match"
            )
        self.last_resolved_provider_id = best.paper_id
        return best.paper_id

    def get_related_paper_pool(
        self,
        seed_context: SeedPaperContext,
        *,
        limit: int = 20,
    ) -> RelatedPaperPool:
        if limit <= 0 or limit > 20:
            raise ValueError("related-paper limit must be in 1..20")
        self.last_request_count = 0
        self.last_global_cache_hit = False
        self.last_global_cache_operations = []
        provider_seed_id = self.resolve_seed_paper_id(seed_context)
        params = {
            "limit": limit,
            "fields": "paperId,title,abstract,year,venue,authors",
        }
        query = urlencode(params)
        request_descriptor = {
            "base_url": self.config.base_url,
            "endpoint": "/recommendations/v1/papers/forpaper/{paper_id}",
            "paper_id": provider_seed_id,
            "params": params,
        }
        payload, from_cache = self._get_with_global_cache(
            "recommendations",
            request_descriptor,
            f"{self.config.base_url}/recommendations/v1/papers/forpaper/"
            f"{quote(provider_seed_id, safe='')}?{query}",
        )
        raw_items = payload.get("recommendedPapers")
        if raw_items is None:
            raw_items = payload.get("data")
        if not isinstance(raw_items, list):
            raise RetrievalTransportError(
                "Semantic Scholar recommendations response is missing paper list",
                retryable=False,
            )
        papers: list[PaperRecord] = []
        seen: set[str] = set()
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise RetrievalTransportError(
                    "Semantic Scholar recommendation item is malformed",
                    retryable=False,
                )
            paper = _paper_from_api(item)
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            papers.append(paper)
            if len(papers) == limit:
                break
        self.last_global_cache_hit = {
            "paper_search",
            "recommendations",
        }.issubset(set(self.last_global_cache_operations))
        return RelatedPaperPool(seed_paper_id=seed_context.paper_id, papers=papers)

    get_related_papers = get_related_paper_pool


SemanticScholarRelatedPaperProvider = SemanticScholarProvider


__all__ = [
    "JsonGetTransport",
    "SemanticScholarConfig",
    "SemanticScholarProvider",
    "SemanticScholarRelatedPaperProvider",
    "UrllibJsonGetTransport",
    "normalize_title",
]
