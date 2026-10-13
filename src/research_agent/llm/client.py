"""Provider-neutral structured LLM contract."""

from __future__ import annotations

import json
import re
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Generic, Literal, Mapping, Protocol, TypeVar

from ..errors import MalformedLLMResponseError, StructuredOutputError

T = TypeVar("T")
MessageRole = Literal["system", "user", "assistant", "tool"]


def _sanitize_text(value: str, *, limit: int = 4_000) -> str:
    """Return a bounded, secret-safe diagnostic excerpt."""

    text = value[:limit]
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,}\"']+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|secret|token)(\s*[:=]\s*)[\"']?[^\"'\s,}]+",
        r"\1\2[REDACTED]",
        text,
    )
    return text


def sanitize_diagnostic_payload(value: Any) -> Any:
    """Recursively remove credential-like fields from audit payloads."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if re.search(r"(?i)(api[_-]?key|authorization|secret|token)", key_text):
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = sanitize_diagnostic_payload(item)
        return result
    if isinstance(value, list):
        return [sanitize_diagnostic_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_diagnostic_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _validation_diagnostics(error: Exception) -> list[dict[str, str]]:
    """Keep stable, secret-free Pydantic diagnostics for audit artifacts."""

    errors_method = getattr(error, "errors", None)
    if not callable(errors_method):
        return []
    try:
        raw_errors = errors_method()
    except Exception:
        return []
    diagnostics: list[dict[str, str]] = []
    if not isinstance(raw_errors, list):
        return diagnostics
    for item in raw_errors:
        if not isinstance(item, Mapping):
            continue
        location = item.get("loc", ())
        if isinstance(location, (list, tuple)):
            path = ".".join(str(part) for part in location) or "$"
        else:
            path = str(location) if location else "$"
        error_type = item.get("type")
        message = item.get("msg")
        diagnostics.append(
            {
                "field_path": path,
                "error_type": str(error_type) if error_type is not None else "validation_error",
                "message": str(message) if message is not None else "validation failed",
            }
        )
    return diagnostics


def _schema_validation_error(
    output_schema: type[Any],
    payload: Mapping[str, Any],
    error: Exception,
) -> StructuredOutputError:
    schema_name = getattr(output_schema, "__name__", str(output_schema))
    diagnostics = _validation_diagnostics(error)
    format_error_types = {
        "missing",
        "extra_forbidden",
        "string_type",
        "list_type",
        "dict_type",
        "set_type",
        "tuple_type",
        "iterable_type",
        "mapping_type",
        "int_type",
        "float_type",
        "bool_type",
        "literal_error",
        "json_invalid",
        "model_type",
        "too_long",
        "too_short",
    }
    retryable = bool(diagnostics) and all(
        item["error_type"] in format_error_types for item in diagnostics
    )
    detail = "; ".join(
        f"{item['field_path']}: {item['message']}" for item in diagnostics
    )
    message = f"{schema_name} validation failed"
    if detail:
        message += f": {detail}"
    return StructuredOutputError(
        message,
        schema_name=schema_name,
        validation_errors=diagnostics,
        parsed_payload=sanitize_diagnostic_payload(dict(payload)),
        retryable=retryable,
    )


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("LLMRequest requires at least one message")
        if not all(isinstance(message, LLMMessage) for message in messages):
            raise TypeError("LLMRequest.messages must contain LLMMessage values")
        object.__setattr__(self, "messages", messages)
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")


@dataclass(frozen=True)
class LLMCallMetadata:
    provider: str
    model: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    format_retry_count: int = 0
    retry_reason: str | None = None


@dataclass(frozen=True)
class StructuredLLMResponse(Generic[T]):
    output: T
    metadata: LLMCallMetadata


class LLMClient(Protocol):
    """Provider-neutral interface used by all future scientific modules."""

    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]:
        ...


def _json_payload(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise MalformedLLMResponseError("structured response has an unterminated code fence")
        text = "\n".join(lines[1:-1]).strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        # Some OpenAI-compatible gateways/models still wrap a JSON-mode
        # response in a short preamble or trailing explanation.  Tolerate
        # that transport formatting only when the entire response contains
        # exactly one parseable JSON object.  We never repair JSON, merge
        # objects, or bypass the canonical schema validation below.
        decoder = json.JSONDecoder()
        candidates: list[Mapping[str, Any]] = []
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        raise MalformedLLMResponseError(
            "structured response is not valid JSON and did not contain one unique JSON object",
            raw_response_excerpt=_sanitize_text(content),
        ) from exc
    if not isinstance(decoded, Mapping):
        raise StructuredOutputError(
            "structured response JSON must be an object",
            raw_response_excerpt=_sanitize_text(content),
        )
    return decoded


def _canonicalize_format_payload(
    payload: Mapping[str, Any], output_schema: type[Any]
) -> Mapping[str, Any]:
    """Apply only explicitly safe, schema-specific representation fixes."""

    schema_name = getattr(output_schema, "__name__", str(output_schema))
    if schema_name != "ExperimentDesignDraft":
        return payload
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        return payload
    canonical = dict(payload)
    canonical_experiments: list[Any] = []
    for index, item in enumerate(experiments):
        if not isinstance(item, Mapping):
            canonical_experiments.append(item)
            continue
        proposal = dict(item)
        value = proposal.get("what_it_validates")
        if isinstance(value, list):
            if len(value) != 1 or not isinstance(value[0], str) or not value[0].strip():
                raise StructuredOutputError(
                    "ExperimentDesignDraft canonicalization requires a singleton list[string] for what_it_validates",
                    schema_name=schema_name,
                    validation_errors=[
                        {
                            "field_path": f"experiments.{index}.what_it_validates",
                            "error_type": "safe_canonicalization_failed",
                            "message": "expected string or singleton list[string]",
                        }
                    ],
                    parsed_payload=sanitize_diagnostic_payload(dict(payload)),
                )
            proposal["what_it_validates"] = value[0]
        canonical_experiments.append(proposal)
    canonical["experiments"] = canonical_experiments
    return canonical


def coerce_structured_output(raw: Any, output_schema: type[T]) -> T:
    """Decode and validate one structured object without swallowing errors."""

    if not isinstance(output_schema, type):
        raise StructuredOutputError("output_schema must be a type")
    if isinstance(raw, output_schema):
        return raw
    schema_name = getattr(output_schema, "__name__", str(output_schema))
    if isinstance(raw, str):
        try:
            payload: Mapping[str, Any] = _json_payload(raw)
        except StructuredOutputError as exc:
            if exc.schema_name is not None:
                raise
            error_type = (
                MalformedLLMResponseError
                if isinstance(exc, MalformedLLMResponseError)
                else StructuredOutputError
            )
            raise error_type(
                f"{schema_name} structured output decoding failed: {exc}",
                schema_name=schema_name,
                validation_errors=list(getattr(exc, "validation_errors", []) or []),
                parsed_payload=getattr(exc, "parsed_payload", None),
                raw_response_excerpt=getattr(exc, "raw_response_excerpt", None),
                retryable=getattr(exc, "retryable", True),
            ) from exc
    elif isinstance(raw, Mapping):
        payload = raw
    else:
        raise StructuredOutputError("structured output must be a JSON object or mapping")

    payload = _canonicalize_format_payload(payload, output_schema)

    if output_schema is dict:
        return dict(payload)  # type: ignore[return-value]

    model_validate = getattr(output_schema, "model_validate", None)
    if callable(model_validate):
        try:
            return model_validate(payload)
        except Exception as exc:  # schema libraries expose different exception types
            raise _schema_validation_error(output_schema, payload, exc) from exc

    from_dict = getattr(output_schema, "from_dict", None)
    if callable(from_dict):
        try:
            return from_dict(payload)
        except Exception as exc:
            raise _schema_validation_error(output_schema, payload, exc) from exc

    if is_dataclass(output_schema):
        expected = {
            field.name
            for field in fields(output_schema)
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing = sorted(name for name in expected if name not in payload)
        if missing:
            raise StructuredOutputError(
                f"output missing required fields for {output_schema.__name__}: {', '.join(missing)}"
            )
        try:
            return output_schema(**dict(payload))
        except Exception as exc:
            raise StructuredOutputError(
                f"output failed dataclass construction for {output_schema.__name__}"
            ) from exc

    raise StructuredOutputError(
        f"output schema {output_schema.__name__} must provide from_dict/model_validate or be a dataclass"
    )
