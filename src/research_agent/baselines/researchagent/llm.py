"""Text-completion DeepSeek adapter for the official ResearchAgent client API."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...llm.config import DeepSeekConfig
from ...llm.deepseek import JsonTransport, UrllibJsonTransport
from ...errors import LLMTransportError


@dataclass(frozen=True)
class ResearchAgentLLMCall:
    sequence: int
    stage: str
    attempt: int
    success: bool
    provider: str
    model: str | None
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "attempt": self.attempt,
            "success": self.success,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def _safe_error(value: object) -> str:
    text = str(value).replace("\n", " ")
    return text[:1_000]


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _stage(messages: Sequence[Mapping[str, str]]) -> str:
    system = messages[0].get("content", "").lower() if messages else ""
    if "identify promising" in system:
        return "problem_stage"
    if "assess the quality and validity of scientific problems" in system:
        return "problem_reviewer"
    if "propose innovative, rigorous, and valid methodologies" in system:
        return "method_stage"
    if "assess the quality and soundness of scientific methods" in system:
        return "method_reviewer"
    if "design robust, feasible, and impactful experiments" in system:
        return "experiment_stage"
    if "evaluate the experimental designs" in system:
        return "experiment_reviewer"
    return "unknown_stage"


class ResearchAgentDeepSeekClient:
    """Implement the official ``OpenAIClient.call`` surface over DeepSeek.

    The official client sends plain text Chat Completions with
    ``temperature=1.25`` and ``max_tokens=1000``.  This adapter intentionally
    does not request JSON mode and does not alter any official prompt.
    """

    def __init__(
        self,
        config: DeepSeekConfig,
        transport: JsonTransport | None = None,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibJsonTransport()
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._calls: list[ResearchAgentLLMCall] = []

    def _allocate_sequence(self) -> int:
        with self._lock:
            self._next_sequence += 1
            return self._next_sequence

    def _record(self, record: ResearchAgentLLMCall) -> None:
        with self._lock:
            self._calls.append(record)

    @property
    def calls(self) -> list[ResearchAgentLLMCall]:
        with self._lock:
            return sorted(self._calls, key=lambda item: (item.sequence, item.attempt))

    def call(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        self.config.validate_for_request()
        assert self.config.api_key is not None
        assert self.config.base_url is not None
        assert self.config.model is not None
        sequence = self._allocate_sequence()
        stage = _stage(messages)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 1.25,
        }
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._transport.post_json(
                    f"{self.config.base_url}/chat/completions",
                    {
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    payload,
                    self.config.timeout_seconds,
                )
                content = self._extract_content(response)
                usage = response.get("usage")
                usage = usage if isinstance(usage, Mapping) else {}
                self._record(
                    ResearchAgentLLMCall(
                        sequence=sequence,
                        stage=stage,
                        attempt=attempt + 1,
                        success=True,
                        provider="deepseek",
                        model=self.config.model,
                        request_id=_optional_string(response.get("id")),
                        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                        completion_tokens=_optional_int(usage.get("completion_tokens")),
                        total_tokens=_optional_int(usage.get("total_tokens")),
                    )
                )
                return content.strip()
            except Exception as exc:  # preserve official retry/return behavior
                last_error = exc
                self._record(
                    ResearchAgentLLMCall(
                        sequence=sequence,
                        stage=stage,
                        attempt=attempt + 1,
                        success=False,
                        provider="deepseek",
                        model=self.config.model,
                        request_id=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        error_type=type(exc).__name__,
                        error_message=_safe_error(exc),
                    )
                )
                if attempt >= max_retries:
                    return _safe_error(last_error)
                self._sleep(4 ** (attempt + 1))
        return _safe_error(last_error)

    @staticmethod
    def _extract_content(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMTransportError("ResearchAgent DeepSeek response missing choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMTransportError("ResearchAgent DeepSeek response choice malformed")
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise LLMTransportError("ResearchAgent DeepSeek response content malformed")
        return message["content"]


__all__ = ["ResearchAgentDeepSeekClient", "ResearchAgentLLMCall"]
