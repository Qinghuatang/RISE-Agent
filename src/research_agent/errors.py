"""Shared, intentionally small error hierarchy."""

from typing import Any


class ResearchAgentError(Exception):
    """Base error for expected project failures."""


class ConfigurationError(ResearchAgentError):
    """Configuration is missing or invalid."""


class SerializationError(ResearchAgentError):
    """A JSON/JSONL payload is invalid or cannot be encoded."""


class LLMError(ResearchAgentError):
    """Base error for LLM infrastructure failures."""


class LLMTransportError(LLMError):
    """The LLM transport failed or returned an invalid API envelope."""


class StructuredOutputError(LLMError):
    """A structured output could not be decoded or validated."""

    def __init__(
        self,
        message: str,
        *,
        schema_name: str | None = None,
        validation_errors: list[dict[str, str]] | None = None,
        parsed_payload: dict[str, Any] | None = None,
        raw_response_excerpt: str | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.schema_name = schema_name
        self.validation_errors = list(validation_errors or [])
        self.parsed_payload = parsed_payload
        self.raw_response_excerpt = raw_response_excerpt
        self.retryable = retryable


class MalformedLLMResponseError(StructuredOutputError):
    """The provider response content was not valid structured JSON."""


class ExtractionError(ResearchAgentError):
    """A Markdown extraction or evidence-audit contract could not be satisfied."""


class ModuleContractError(ResearchAgentError, ValueError):
    """A scientific module received or produced an invalid structured contract."""


class PreconditionError(ResearchAgentError, ValueError):
    """A scientific module was invoked before its documented preconditions held."""


class ControllerRoutingError(ResearchAgentError, ValueError):
    """Research State/Memory cannot be routed without guessing or repairing it."""


class RunnerInvariantError(ResearchAgentError, ValueError):
    """The execution runner cannot safely dispatch the requested action."""


class RetrievalError(ResearchAgentError):
    """Base error for external paper-retrieval infrastructure."""


class RetrievalConfigurationError(RetrievalError, ConfigurationError):
    """Paper-provider configuration is missing or invalid."""


class RetrievalTransportError(RetrievalError):
    """A paper-provider request failed or returned malformed JSON."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class RetrievalHTTPError(RetrievalTransportError):
    """Structured HTTP failure from a paper provider."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        if retryable is None:
            retryable = status_code == 429 or 500 <= status_code <= 599
        super().__init__(message, retryable=retryable)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class SeedPaperResolutionError(RetrievalError):
    """A local Seed Paper could not be resolved to one provider paper."""
