"""Deterministic fake LLM for offline tests and future module fixtures."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque

from ..errors import LLMError
from .client import (
    LLMCallMetadata,
    LLMClient,
    LLMRequest,
    StructuredLLMResponse,
    T,
    coerce_structured_output,
)


ResponseHandler = Callable[[LLMRequest, type[T]], Any]


class FakeLLMClient(LLMClient):
    """Returns queued or handler-produced outputs and never performs I/O."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        handler: ResponseHandler | None = None,
    ) -> None:
        if responses is not None and handler is not None:
            raise ValueError("provide responses or handler, not both")
        self._responses: Deque[Any] = deque(responses or [])
        self._handler = handler
        self.calls: list[LLMRequest] = []

    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]:
        self.calls.append(request)
        if self._handler is not None:
            raw = self._handler(request, output_schema)
        elif self._responses:
            raw = self._responses.popleft()
        else:
            raise LLMError("FakeLLMClient has no queued response")
        if isinstance(raw, Exception):
            raise raw
        output = coerce_structured_output(raw, output_schema)
        return StructuredLLMResponse(
            output=output,
            metadata=LLMCallMetadata(
                provider="fake",
                model="fake",
                request_id=f"fake-{len(self.calls)}",
            ),
        )

