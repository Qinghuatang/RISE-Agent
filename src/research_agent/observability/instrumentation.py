"""Transparent, provider-neutral wrappers for RISE runtime dependencies."""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
import time
from typing import Any, TypeVar

from ..llm.client import LLMClient, LLMRequest, StructuredLLMResponse, T
from .langfuse import LangfuseTracer
from .telemetry import DEFAULT_TELEMETRY, Telemetry

class ObservedLLMClient(LLMClient):
    """Measure one provider-neutral completion call and return it unchanged.

    The metric counts calls through ``complete_structured``. Provider-internal
    HTTP retries are not visible at this interface and remain unknown; a
    format retry performed by ``RecordingLLMClient`` is reported as metadata.
    """

    _rise_observed_llm = True

    def __init__(
        self,
        inner: LLMClient,
        telemetry: Telemetry = DEFAULT_TELEMETRY,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.inner = inner
        self.telemetry = telemetry
        self.tracer = tracer

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
        trace_scope = (
            self.tracer.span(
                "rise.llm.complete_structured",
                observation_type="generation",
                trace_id=self.tracer.current_trace_id(),
                metadata={"provider": provider, "operation": "complete_structured"},
                model=model,
            )
            if self.tracer is not None
            else nullcontext(None)
        )
        with trace_scope as trace_span:
            try:
                response = self.inner.complete_structured(request, output_schema)
            except Exception as exc:
                duration_ms = (time.monotonic_ns() - started) / 1_000_000
                if trace_span is not None:
                    trace_span.mark_error(error_type=type(exc).__name__, error_category="llm")
                    trace_span.update(
                        provider_retry_count=self._provider_retry_count(self.inner),
                        duration_ms=duration_ms,
                    )
                self.telemetry.record(
                    "llm.call",
                    operation="complete_structured",
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
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

            response_metadata = response.metadata
            prompt_tokens = getattr(response_metadata, "prompt_tokens", None)
            completion_tokens = getattr(response_metadata, "completion_tokens", None)
            total_tokens = getattr(response_metadata, "total_tokens", None)
            retry_count = getattr(response_metadata, "format_retry_count", None)
            resolved_provider = getattr(response_metadata, "provider", provider)
            resolved_model = getattr(response_metadata, "model", model)
            duration_ms = (time.monotonic_ns() - started) / 1_000_000
            if trace_span is not None:
                trace_span.update(
                    provider=resolved_provider,
                    model=resolved_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    provider_retry_count=self._provider_retry_count(self.inner),
                    format_retry_count=retry_count,
                    duration_ms=duration_ms,
                    status="completed",
                )
            self.telemetry.record(
                "llm.call",
                operation="complete_structured",
                provider=resolved_provider,
                model=resolved_model,
                duration_ms=duration_ms,
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

    def __init__(
        self,
        inner: object,
        telemetry: Telemetry = DEFAULT_TELEMETRY,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.inner = inner
        self.telemetry = telemetry
        self.tracer = tracer

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.inner, name)
        if name.startswith("_") or not callable(value):
            return value

        @wraps(value)
        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic_ns()
            provider = getattr(self.inner, "provider_name", None)
            provider = provider if isinstance(provider, str) else type(self.inner).__name__
            trace_scope = (
                self.tracer.span(
                    "rise.retrieval.call",
                    observation_type="retriever",
                    trace_id=self.tracer.current_trace_id(),
                    metadata={"provider": provider, "operation": name},
                )
                if self.tracer is not None
                else nullcontext(None)
            )
            with trace_scope as trace_span:
                try:
                    result = value(*args, **kwargs)
                except Exception as exc:
                    duration_ms = (time.monotonic_ns() - started) / 1_000_000
                    request_count = getattr(self.inner, "last_request_count", None)
                    retry_count = getattr(self.inner, "last_retry_count", None)
                    if trace_span is not None:
                        trace_span.mark_error(error_type=type(exc).__name__, error_category="retrieval")
                        trace_span.update(
                            http_request_count=request_count,
                            provider_retry_count=retry_count,
                            duration_ms=duration_ms,
                        )
                    self.telemetry.record(
                        "retrieval.call",
                        operation=name,
                        provider=provider,
                        duration_ms=duration_ms,
                        status="error",
                        success=False,
                        error_category="retrieval",
                        error_type=type(exc).__name__,
                        http_request_count=request_count,
                        provider_retry_count=retry_count,
                    )
                    raise

                cache_hit = getattr(self.inner, "last_cache_hit", None)
                if cache_hit is None:
                    cache_hit = getattr(self.inner, "last_global_cache_hit", None)
                request_count = getattr(self.inner, "last_request_count", None)
                retry_count = getattr(self.inner, "last_retry_count", None)
                duration_ms = (time.monotonic_ns() - started) / 1_000_000
                if trace_span is not None:
                    trace_span.update(
                        http_request_count=request_count,
                        provider_retry_count=retry_count,
                        cache_hit=cache_hit if isinstance(cache_hit, bool) else None,
                        duration_ms=duration_ms,
                        status="completed",
                    )
                self.telemetry.record(
                    "retrieval.call",
                    operation=name,
                    provider=provider,
                    duration_ms=duration_ms,
                    status="completed",
                    success=True,
                    http_request_count=request_count,
                    provider_retry_count=retry_count,
                    cache_hit=cache_hit if isinstance(cache_hit, bool) else None,
                )
                return result

        return measured


def _wrap_llm(
    client: object,
    telemetry: Telemetry,
    tracer: LangfuseTracer | None,
) -> object:
    if getattr(client, "_rise_observed_llm", False):
        client.telemetry = telemetry
        if tracer is not None:
            client.tracer = tracer
        return client
    return ObservedLLMClient(client, telemetry, tracer)  # type: ignore[arg-type]


def _wrap_retrieval(
    provider: object,
    telemetry: Telemetry,
    tracer: LangfuseTracer | None,
) -> object:
    if getattr(provider, "_rise_observed_retrieval", False):
        provider.telemetry = telemetry
        if tracer is not None:
            provider.tracer = tracer
        return provider
    return ObservedRetrievalProvider(provider, telemetry, tracer)


def instrument_trajectory_modules(
    modules: object,
    telemetry: Telemetry = DEFAULT_TELEMETRY,
    tracer: LangfuseTracer | None = None,
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
                wrapper = _wrap_llm(client, telemetry, tracer)
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
                wrapper = _wrap_retrieval(provider, telemetry, tracer)
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
