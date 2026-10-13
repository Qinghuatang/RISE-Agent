"""Recording wrapper for one pilot-wide provider-neutral LLM client."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..errors import StructuredOutputError
from .client import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
    T,
    sanitize_diagnostic_payload,
)
from ..models.pilot import LLMCallAudit


class RecordingLLMClient:
    """Delegate every call and record safe, structured execution metadata."""

    def __init__(self, inner: LLMClient, *, unbounded_output: bool = False) -> None:
        self.inner = inner
        self.unbounded_output = unbounded_output
        self.records: list[LLMCallAudit] = []
        self.format_retry_enabled = True

    @staticmethod
    def _provider_model(inner: object) -> tuple[str | None, str | None]:
        config = getattr(inner, "config", None)
        provider = "deepseek" if config is not None else None
        model = getattr(config, "model", None) if config is not None else None
        return provider, model

    @staticmethod
    def _safe_error(inner: object, error: Exception) -> str:
        message = str(error).strip().replace("\n", " ")[:500]
        config = getattr(inner, "config", None)
        key = getattr(config, "api_key", None)
        if isinstance(key, str) and key:
            message = message.replace(key, "[REDACTED]")
        return message or type(error).__name__

    @staticmethod
    def _retry_request(request: LLMRequest, error: StructuredOutputError) -> LLMRequest:
        reason = str(error).strip().replace("\n", " ")[:500]
        retry_instruction = (
            "FORMAT-ONLY RETRY. Return the same scientific answer and use the "
            "supplied canonical JSON schema exactly. Correct only JSON, field "
            "types, required structural fields, and provenance identifiers. "
            "Do not add reasoning, change the scientific decision, or introduce "
            "new evidence. Output one JSON object only."
        )
        if error.schema_name == "GapEvaluationDraft":
            retry_instruction += (
                " For GapEvaluationDraft, the evaluations array must contain "
                "exactly one complete object per supplied Candidate Gap. Each "
                "object must contain candidate_id, confidence, "
                "confidence_reason, priority, and priority_reason together. "
                "Do not split one candidate into separate confidence and "
                "priority objects, and do not add extra array elements."
            )
        messages = list(request.messages)
        system_index = next(
            (index for index, message in enumerate(messages) if message.role == "system"),
            None,
        )
        if system_index is None:
            messages.insert(0, LLMMessage(role="system", content=retry_instruction))
        else:
            message = messages[system_index]
            messages[system_index] = LLMMessage(
                role="system",
                content=message.content + "\n\n" + retry_instruction,
            )
        metadata: dict[str, Any] = dict(request.metadata)
        metadata.update(
            {
                "format_retry_count": 1,
                "retry_reason": reason,
            }
        )
        return replace(request, messages=tuple(messages), metadata=metadata)

    def _record_failure(
        self,
        *,
        sequence: int,
        request: LLMRequest,
        output_schema: type[T],
        provider: str | None,
        model: str | None,
        error: Exception,
    ) -> None:
        self.records.append(
            LLMCallAudit(
                sequence=sequence,
                task=(request.metadata.get("task") if isinstance(request.metadata.get("task"), str) else None),
                paper_id=(request.metadata.get("paper_id") if isinstance(request.metadata.get("paper_id"), str) else None),
                output_schema=getattr(output_schema, "__name__", str(output_schema)),
                provider=provider,
                model=model,
                input_chars=sum(len(message.content) for message in request.messages),
                max_tokens=request.max_tokens,
                success=False,
                error_type=type(error).__name__,
                error_message=self._safe_error(self.inner, error),
                validation_errors=list(getattr(error, "validation_errors", []) or []),
                parsed_payload=sanitize_diagnostic_payload(
                    getattr(error, "parsed_payload", None)
                ),
                raw_response_excerpt=sanitize_diagnostic_payload(
                    getattr(error, "raw_response_excerpt", None)
                ),
                format_retry_count=int(request.metadata.get("format_retry_count", 0) or 0),
                retry_reason=(
                    request.metadata.get("retry_reason")
                    if isinstance(request.metadata.get("retry_reason"), str)
                    else None
                ),
            )
        )

    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]:
        sequence = len(self.records) + 1
        metadata = request.metadata
        task = metadata.get("task")
        paper_id = metadata.get("paper_id")
        task = task if isinstance(task, str) else None
        paper_id = paper_id if isinstance(paper_id, str) else None
        input_chars = sum(len(message.content) for message in request.messages)
        provider, model = self._provider_model(self.inner)
        forwarded_request = (
            replace(request, max_tokens=None) if self.unbounded_output else request
        )
        try:
            response = self.inner.complete_structured(
                forwarded_request, output_schema
            )
        except StructuredOutputError as exc:
            self._record_failure(
                sequence=sequence,
                request=forwarded_request,
                output_schema=output_schema,
                provider=provider,
                model=model,
                error=exc,
            )
            retry_count = int(forwarded_request.metadata.get("format_retry_count", 0) or 0)
            if (
                not self.format_retry_enabled
                or retry_count >= 1
                or not getattr(exc, "retryable", True)
            ):
                raise
            retry_request = self._retry_request(forwarded_request, exc)
            retry_forwarded = (
                replace(retry_request, max_tokens=None)
                if self.unbounded_output
                else retry_request
            )
            try:
                response = self.inner.complete_structured(
                    retry_forwarded, output_schema
                )
            except Exception as retry_error:
                self._record_failure(
                    sequence=len(self.records) + 1,
                    request=retry_forwarded,
                    output_schema=output_schema,
                    provider=provider,
                    model=model,
                    error=retry_error,
                )
                raise
            response = replace(
                response,
                metadata=replace(
                    response.metadata,
                    format_retry_count=1,
                    retry_reason=str(exc).strip().replace("\n", " ")[:500],
                ),
            )
            forwarded_request = retry_forwarded
            sequence = len(self.records) + 1
        except Exception as exc:
            self._record_failure(
                sequence=sequence,
                request=forwarded_request,
                output_schema=output_schema,
                provider=provider,
                model=model,
                error=exc,
            )
            raise
        request_retry_count = int(
            forwarded_request.metadata.get("format_retry_count", 0) or 0
        )
        request_retry_reason = forwarded_request.metadata.get("retry_reason")
        call_metadata = response.metadata
        if request_retry_count > call_metadata.format_retry_count:
            call_metadata = replace(
                call_metadata,
                format_retry_count=request_retry_count,
                retry_reason=(
                    request_retry_reason
                    if isinstance(request_retry_reason, str)
                    else call_metadata.retry_reason
                ),
            )
            response = replace(response, metadata=call_metadata)
        self.records.append(
            LLMCallAudit(
                sequence=sequence,
                task=task,
                paper_id=paper_id,
                output_schema=getattr(output_schema, "__name__", str(output_schema)),
                provider=call_metadata.provider,
                model=call_metadata.model,
                request_id=call_metadata.request_id,
                prompt_tokens=call_metadata.prompt_tokens,
                completion_tokens=call_metadata.completion_tokens,
                total_tokens=call_metadata.total_tokens,
                input_chars=sum(
                    len(message.content) for message in forwarded_request.messages
                ),
                max_tokens=forwarded_request.max_tokens,
                success=True,
                format_retry_count=call_metadata.format_retry_count,
                retry_reason=call_metadata.retry_reason,
            )
        )
        return response


def aggregate_token_usage(records: list[LLMCallAudit]) -> dict[str, object]:
    """Aggregate known usage without treating missing provider fields as zero."""

    complete_calls = [
        record
        for record in records
        if record.prompt_tokens is not None
        and record.completion_tokens is not None
        and record.total_tokens is not None
    ]
    complete = len(complete_calls) == len(records)
    known_prompt = sum(
        record.prompt_tokens for record in records if record.prompt_tokens is not None
    )
    known_completion = sum(
        record.completion_tokens
        for record in records
        if record.completion_tokens is not None
    )
    known_total = sum(
        record.total_tokens for record in records if record.total_tokens is not None
    )
    result: dict[str, object] = {
        "known_prompt_tokens": known_prompt,
        "known_completion_tokens": known_completion,
        "known_total_tokens": known_total,
        "calls_with_token_usage": len(complete_calls),
        "calls_missing_token_usage": len(records) - len(complete_calls),
        "token_metadata_complete": complete,
        "prompt_tokens_total": known_prompt if complete else None,
        "completion_tokens_total": known_completion if complete else None,
        "total_tokens": known_total if complete else None,
    }
    return result


__all__ = ["RecordingLLMClient", "aggregate_token_usage"]
