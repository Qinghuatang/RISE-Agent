"""Transparent, provider-neutral wrappers for RISE runtime dependencies."""

from __future__ import annotations

from functools import wraps
import time
from typing import Any, TypeVar

from ..llm.client import LLMClient, LLMRequest, StructuredLLMResponse, T
from .telemetry import DEFAULT_TELEMETRY, Telemetry

class ObservedLLMClient(LLMClient):
    """Measure one provider-neutral completion call and return it unchanged.

    The metric counts calls through ``complete_structured``. Provider-internal
    HTTP retries are not visible at this interface and remain unknown; a
    format retry performed by ``RecordingLLMClient`` is reported as metadata.
    """

    _rise_observed_llm = True

    def __init__(self, inner: LLMClient, telemetry: Telemetry = DEFAULT_TELEMETRY) -> None:
        self.inner = inner
        self.telemetry = telemetry

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @staticmethod
    def _provider_model(client: object) -> tuple[str | None, str | None]:
        config = getattr(client, "config", None)
        provider = getattr(client, "provider_name", None)
        if not isinstance(provider, str) and config is not None:
            provider = "deepseek"
        model = getattr(config, "model", None) if config is not None else None
        return (
            provider if isinstance(provider, str) else None,
            model if isinstance(model, str) else None,
        )

    @staticmethod
    def _provider_retry_count(client: object) -> int | None:
        attempts = getattr(client, "last_transport_attempt_count", None)
        if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 1:
            return max(0, attempts - 1)
        return None

    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]:
        started = time.monotonic_ns()
        provider, model = self._provider_model(self.inner)
        try:
            response = self.inner.complete_structured(request, output_schema)
        except Exception as exc:
            self.telemetry.record(
                "llm.call",
                operation="complete_structured",
                provider=provider,
                model=model,
                duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                status="error",
                success=False,
                error_category="llm",
                error_type=type(exc).__name__,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                token_usage_known=False,
                provider_retry_count=self._provider_retry_count(self.inner),
                format_retry_count=None,
            )
            raise

        metadata = response.metadata
        prompt_tokens = getattr(metadata, "prompt_tokens", None)
        completion_tokens = getattr(metadata, "completion_tokens", None)
        total_tokens = getattr(metadata, "total_tokens", None)
        retry_count = getattr(metadata, "format_retry_count", None)
        self.telemetry.record(
            "llm.call",
            operation="complete_structured",
            provider=getattr(metadata, "provider", provider),
            model=getattr(metadata, "model", model),
            duration_ms=(time.monotonic_ns() - started) / 1_000_000,
            status="completed",
            success=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_usage_known=all(
                value is not None
                for value in (prompt_tokens, completion_tokens, total_tokens)
            ),
            provider_retry_count=self._provider_retry_count(self.inner),
            format_retry_count=retry_count,
        )
        return response


class ObservedRetrievalProvider:
    """Proxy retrieval operations while preserving results and exceptions."""

    _rise_observed_retrieval = True

    def __init__(self, inner: object, telemetry: Telemetry = DEFAULT_TELEMETRY) -> None:
        self.inner = inner
        self.telemetry = telemetry

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.inner, name)
        if name.startswith("_") or not callable(value):
            return value

        @wraps(value)
        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic_ns()
            provider = getattr(self.inner, "provider_name", None)
            provider = provider if isinstance(provider, str) else type(self.inner).__name__
            try:
                result = value(*args, **kwargs)
            except Exception as exc:
                self.telemetry.record(
                    "retrieval.call",
                    operation=name,
                    provider=provider,
                    duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    status="error",
                    success=False,
                    error_category="retrieval",
                    error_type=type(exc).__name__,
                    http_request_count=getattr(self.inner, "last_request_count", None),
                    provider_retry_count=getattr(self.inner, "last_retry_count", None),
                )
                raise

            cache_hit = getattr(self.inner, "last_cache_hit", None)
            if cache_hit is None:
                cache_hit = getattr(self.inner, "last_global_cache_hit", None)
            self.telemetry.record(
                "retrieval.call",
                operation=name,
                provider=provider,
                duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                status="completed",
                success=True,
                http_request_count=getattr(self.inner, "last_request_count", None),
                provider_retry_count=getattr(self.inner, "last_retry_count", None),
                cache_hit=cache_hit if isinstance(cache_hit, bool) else None,
            )
            return result

        return measured


def _wrap_llm(client: object, telemetry: Telemetry) -> object:
    if getattr(client, "_rise_observed_llm", False):
        client.telemetry = telemetry
        return client
    return ObservedLLMClient(client, telemetry)  # type: ignore[arg-type]


def _wrap_retrieval(provider: object, telemetry: Telemetry) -> object:
    if getattr(provider, "_rise_observed_retrieval", False):
        provider.telemetry = telemetry
        return provider
    return ObservedRetrievalProvider(provider, telemetry)


def instrument_trajectory_modules(
    modules: object,
    telemetry: Telemetry = DEFAULT_TELEMETRY,
) -> dict[str, int]:
    """Instrument shared clients once at the Worker runtime assembly boundary.

    Current RISE modules keep their injected clients on ``llm_client`` and
    retrieval providers on ``related_paper_provider`` / ``search_provider``.
    Identity-based reuse prevents a shared LLM client from being wrapped once
    per module and therefore avoids duplicate metric records.
    """

    llm_wrappers: dict[int, object] = {}
    retrieval_wrappers: dict[int, object] = {}
    wrapped_llm_ids: set[int] = set()
    wrapped_retrieval_ids: set[int] = set()
    visited_modules: set[int] = set()
    for module in vars(modules).values():
        if id(module) in visited_modules:
            continue
        visited_modules.add(id(module))
        client = getattr(module, "llm_client", None)
        if client is not None:
            wrapper = llm_wrappers.get(id(client))
            if wrapper is None:
                wrapper = _wrap_llm(client, telemetry)
                llm_wrappers[id(client)] = wrapper
            try:
                setattr(module, "llm_client", wrapper)
                wrapped_llm_ids.add(id(client))
            except (AttributeError, TypeError):
                # A custom immutable module can opt in by wrapping its client
                # before constructing TrajectoryModules.
                telemetry.record(
                    "observability.dependency_not_wrapped",
                    operation="llm",
                    status="unsupported_module",
                    success=False,
                    error_category="unknown",
                )

        for attribute in ("related_paper_provider", "search_provider"):
            provider = getattr(module, attribute, None)
            if provider is None:
                continue
            wrapper = retrieval_wrappers.get(id(provider))
            if wrapper is None:
                wrapper = _wrap_retrieval(provider, telemetry)
                retrieval_wrappers[id(provider)] = wrapper
            try:
                setattr(module, attribute, wrapper)
                wrapped_retrieval_ids.add(id(provider))
            except (AttributeError, TypeError):
                telemetry.record(
                    "observability.dependency_not_wrapped",
                    operation="retrieval",
                    status="unsupported_module",
                    success=False,
                    error_category="unknown",
                )

    return {
        "llm_clients": len(wrapped_llm_ids),
        "retrieval_providers": len(wrapped_retrieval_ids),
    }


__all__ = [
    "ObservedLLMClient",
    "ObservedRetrievalProvider",
    "instrument_trajectory_modules",
]
