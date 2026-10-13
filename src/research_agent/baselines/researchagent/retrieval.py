"""Semantic Scholar adapter preserving the official ResearchAgent retrieval semantics."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ...errors import (
    RetrievalConfigurationError,
    RetrievalHTTPError,
    RetrievalTransportError,
    SeedPaperResolutionError,
)
from ...retrieval.semantic_scholar import (
    SemanticScholarConfig,
    SemanticScholarCache,
    normalize_title,
)


def _version_rank(item: Mapping[str, Any]) -> tuple[int, str]:
    publication_types = {
        normalize_title(value).replace(" ", "")
        for value in (item.get("publicationTypes") or [])
        if isinstance(value, str)
    }
    venue = normalize_title(item.get("venue") or "")
    source = normalize_title(item.get("journal", {}).get("name", "") if isinstance(item.get("journal"), Mapping) else "")
    if "journalarticle" in publication_types or "journal" in venue or source:
        return 3, "journal_preferred"
    if "conference" in publication_types or any(
        token in venue for token in ("conference", "proceedings", "symposium", "workshop")
    ):
        return 2, "conference_preferred"
    if "arxiv" in venue or "arxiv" in source or "preprint" in venue:
        return 1, "arxiv_preferred"
    return 0, "other"


def select_seed_candidate(candidates: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str]:
    """Select an exact-title provider record deterministically."""

    if len(candidates) == 1:
        return candidates[0], "exact_unique"
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_version_rank(item)[0],
            -(1 if isinstance(item.get("abstract"), str) and item["abstract"].strip() else 0),
            str(item.get("paperId", "")),
        ),
    )
    selected = ranked[0]
    rank, rank_reason = _version_rank(selected)
    same_rank = [item for item in candidates if _version_rank(item)[0] == rank]
    if len(same_rank) == 1:
        return selected, rank_reason
    same_rank_with_abstract = [
        item for item in same_rank
        if isinstance(item.get("abstract"), str) and item["abstract"].strip()
    ]
    if len(same_rank_with_abstract) == 1 and len(same_rank_with_abstract) != len(same_rank):
        return selected, "has_abstract"
    return selected, "stable_tiebreak"


class _ResearchAgentHttp:
    def get_json(self, url: str, headers: Mapping[str, str], timeout: float) -> Any:
        return self._request("GET", url, headers, None, timeout)

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Any:
        return self._request("POST", url, headers, payload, timeout)

    @staticmethod
    def _request(
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout: float,
    ) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=data, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured adapter URL
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise RetrievalHTTPError(
                exc.code,
                f"ResearchAgent Semantic Scholar HTTP error {exc.code}: {detail[:512]}",
                retry_after_seconds=None,
            ) from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RetrievalTransportError(
                f"ResearchAgent Semantic Scholar transport error: {reason}",
                retryable=True,
            ) from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalTransportError(
                "ResearchAgent Semantic Scholar response was not valid JSON",
                retryable=False,
            ) from exc


class ResearchAgentSemanticScholarAdapter:
    """HTTP/cache adapter for the official reference-and-embedding algorithm.

    The official repository's selection algorithm is retained:
    references endpoint -> reference embedding batch -> cosine similarity with
    the seed embedding -> top 10 references.  This class only replaces its
    hard-coded HTTP implementation and adds deterministic seed resolution.
    """

    def __init__(
        self,
        config: SemanticScholarConfig,
        transport: Any | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._transport = transport or _ResearchAgentHttp()
        self._cache = SemanticScholarCache(config.cache_dir) if config.cache_dir else None
        self._sleep = sleep
        self.http_attempt_count = 0
        self.cache_hits = 0
        self.cache_hit_operations: list[str] = []
        self.reference_request_count = 0
        self.last_resolution: dict[str, Any] = {}
        self.last_retrieval: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.api_key:
            if self.config.auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            else:
                headers["x-api-key"] = self.config.api_key
        return headers

    def _request(self, method: str, url: str, payload: Mapping[str, Any] | None = None) -> Any:
        self.config.validate_for_request()
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self.http_attempt_count += 1
            try:
                if method == "GET":
                    return self._transport.get_json(url, self._headers(), self.config.timeout_seconds)
                return self._transport.post_json(
                    url, self._headers(), payload or {}, self.config.timeout_seconds
                )
            except (RetrievalHTTPError, RetrievalTransportError) as exc:
                last_error = exc
                if not getattr(exc, "retryable", False) or attempt >= self.config.max_retries:
                    raise
                self._sleep(min(8.0, 0.5 * (2**attempt)))
        assert last_error is not None
        raise last_error

    def _cached_request(
        self,
        operation: str,
        descriptor: Mapping[str, Any],
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        def fetch() -> Mapping[str, Any]:
            response = self._request(method, url, payload)
            if isinstance(response, list):
                return {"items": response}
            if not isinstance(response, Mapping):
                raise RetrievalTransportError(
                    f"Semantic Scholar {operation} response must be an object or list",
                    retryable=False,
                )
            return response

        if self._cache is None:
            cached = fetch()
            return cached.get("items") if "items" in cached and len(cached) == 1 else cached
        response, hit = self._cache.get_or_fetch(operation, descriptor, fetch)
        if hit:
            self.cache_hits += 1
            self.cache_hit_operations.append(operation)
        if "items" in response and len(response) == 1:
            return response["items"]
        return response

    def _search_seed(self, title: str) -> Mapping[str, Any]:
        fields = (
            "paperId,corpusId,title,abstract,year,publicationDate,referenceCount,"
            "citationCount,embedding.specter_v2,venue,authors,publicationTypes"
        )
        params = {"query": title, "limit": 20, "fields": fields}
        descriptor = {"base_url": self.config.base_url, "endpoint": "/graph/v1/paper/search", "params": params}
        payload = self._cached_request(
            "paper_search",
            descriptor,
            "GET",
            f"{self.config.base_url}/graph/v1/paper/search?{urlencode(params)}",
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise RetrievalTransportError("Semantic Scholar paper search response is malformed", retryable=False)
        wanted = normalize_title(title)
        candidates = [
            item for item in payload["data"]
            if isinstance(item, Mapping)
            and isinstance(item.get("paperId"), str)
            and isinstance(item.get("title"), str)
            and normalize_title(item["title"]) == wanted
        ]
        if not candidates:
            raise SeedPaperResolutionError(
                "Semantic Scholar returned no exact normalized-title candidate"
            )
        selected, reason = select_seed_candidate(candidates)
        self.last_resolution = {
            "candidate_count": len(candidates),
            "selected_paper_id": selected.get("paperId"),
            "selected_title": selected.get("title"),
            "selected_year": selected.get("year"),
            "selected_venue": selected.get("venue"),
            "selection_reason": reason,
        }
        return selected

    def resolve_seed_paper(self, title: str, abstract: str | None = None) -> dict[str, Any]:
        selected = dict(self._search_seed(title))
        paper_id = selected.get("paperId")
        fields = (
            "paperId,corpusId,title,abstract,year,publicationDate,referenceCount,"
            "citationCount,embedding.specter_v2,venue,authors,publicationTypes"
        )
        needs_details = any(
            selected.get(field) is None
            for field in ("corpusId", "referenceCount", "embedding")
        )
        if needs_details:
            params = {"fields": fields}
            descriptor = {
                "base_url": self.config.base_url,
                "endpoint": "/graph/v1/paper/{paper_id}",
                "paper_id": paper_id,
                "params": params,
            }
            details = self._cached_request(
                "paper_details",
                descriptor,
                "GET",
                f"{self.config.base_url}/graph/v1/paper/{quote(str(paper_id), safe='')}?{urlencode(params)}",
            )
            if not isinstance(details, Mapping):
                raise RetrievalTransportError("Semantic Scholar paper details response is malformed", retryable=False)
            selected.update(details)
        if not isinstance(selected.get("paperId"), str) or not isinstance(selected.get("corpusId"), int):
            raise SeedPaperResolutionError("resolved provider seed is missing paperId/corpusId")
        if not isinstance(selected.get("referenceCount"), int):
            raise SeedPaperResolutionError("resolved provider seed is missing referenceCount")
        embedding = selected.get("embedding")
        if not isinstance(embedding, Mapping) or not isinstance(embedding.get("vector"), list):
            raise SeedPaperResolutionError("resolved provider seed is missing embedding vector")
        return selected

    @staticmethod
    def _cosine(left: list[Any], right: list[Any]) -> float:
        if len(left) != len(right) or not left:
            return float("-inf")
        try:
            numerator = sum(float(a) * float(b) for a, b in zip(left, right))
            left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
            right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
        except (TypeError, ValueError):
            return float("-inf")
        if left_norm == 0 or right_norm == 0:
            return float("-inf")
        return numerator / (left_norm * right_norm)

    def _reference_page(self, paper_id: str, offset: int, limit: int) -> list[Mapping[str, Any]]:
        fields = "paperId,corpusId,title,abstract,year,publicationDate"
        params = {"fields": fields, "offset": offset, "limit": limit}
        descriptor = {
            "base_url": self.config.base_url,
            "endpoint": "/graph/v1/paper/{paper_id}/references",
            "paper_id": paper_id,
            "params": params,
        }
        payload = self._cached_request(
            "references",
            descriptor,
            "GET",
            f"{self.config.base_url}/graph/v1/paper/{quote(paper_id, safe='')}/references?{urlencode(params)}",
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise RetrievalTransportError("Semantic Scholar references response is malformed", retryable=False)
        return [
            item["citedPaper"]
            for item in payload["data"]
            if isinstance(item, Mapping) and isinstance(item.get("citedPaper"), Mapping)
        ]

    def _batch_embeddings(self, paper_ids: list[str]) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        fields = "embedding.specter_v2"
        for start in range(0, len(paper_ids), 100):
            ids = paper_ids[start : start + 100]
            descriptor = {
                "base_url": self.config.base_url,
                "endpoint": "/graph/v1/paper/batch",
                "ids": ids,
                "fields": fields,
            }
            payload = self._cached_request(
                "paper_batch",
                descriptor,
                "POST",
                f"{self.config.base_url}/graph/v1/paper/batch",
                {"ids": ids, "fields": fields},
            )
            if not isinstance(payload, list):
                raise RetrievalTransportError("Semantic Scholar paper batch response is malformed", retryable=False)
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                paper_id = item.get("paperId")
                embedding = item.get("embedding")
                if isinstance(paper_id, str) and isinstance(embedding, Mapping) and isinstance(embedding.get("vector"), list):
                    result[paper_id] = embedding["vector"]
        return result

    def get_relevant_references(self, seed_paper: Mapping[str, Any], *, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.http_attempt_count = 0
        self.cache_hits = 0
        self.cache_hit_operations = []
        self.last_retrieval = {}
        paper_id = seed_paper["paperId"]
        reference_count = int(seed_paper["referenceCount"])
        references: list[Mapping[str, Any]] = []
        # This intentionally mirrors the official repository's pagination:
        # range(0, referenceCount // batch_size + 1).
        for index in range(0, reference_count // 1000 + 1):
            references.extend(self._reference_page(paper_id, index * 1000, 1000))
        filtered = [
            dict(reference)
            for reference in references
            if all(reference.get(field) is not None for field in ("paperId", "title", "abstract"))
        ]
        embeddings = self._batch_embeddings([str(item["paperId"]) for item in filtered])
        seed_embedding = seed_paper["embedding"]["vector"]
        scored = [
            (self._cosine(seed_embedding, embeddings[reference["paperId"]]), reference["paperId"])
            for reference in filtered
            if reference["paperId"] in embeddings
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_ids = {paper_id for _, paper_id in scored[:top_k]}
        selected = [reference for reference in filtered if reference["paperId"] in selected_ids]
        self.reference_request_count = len(references)
        self.last_retrieval = {
            "raw_reference_count": len(references),
            "filtered_reference_count": len(filtered),
            "embedding_reference_count": len(embeddings),
            "selected_reference_count": len(selected),
            "selected_paper_ids": [item["paperId"] for item in selected],
            "http_attempt_count": self.http_attempt_count,
            "cache_hits": self.cache_hits,
            "cache_hit_operations": list(self.cache_hit_operations),
        }
        return selected

    def audit_dict(self) -> dict[str, Any]:
        return {
            "provider": "semantic_scholar",
            "config": self.config.public_dict(),
            "seed_resolution": dict(self.last_resolution),
            "retrieval": dict(self.last_retrieval),
            "http_attempt_count": self.http_attempt_count,
            "cache_hits": self.cache_hits,
            "cache_hit_operations": list(self.cache_hit_operations),
        }


__all__ = ["ResearchAgentSemanticScholarAdapter", "select_seed_candidate"]
