"""DeepSeek HTTP adapter; no scientific logic belongs here."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..errors import LLMTransportError
from .client import (
    LLMCallMetadata,
    LLMClient,
    LLMRequest,
    StructuredLLMResponse,
    T,
    coerce_structured_output,
)
from .config import DeepSeekConfig


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        ...


class UrllibJsonTransport:
    """Small standard-library transport, injectable for offline tests."""

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - configured adapter URL
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(256).decode("utf-8", errors="replace")
            raise LLMTransportError(
                f"DeepSeek HTTP error {exc.code}: {detail[:256]}"
            ) from exc
        except URLError as exc:
            raise LLMTransportError(f"DeepSeek transport error: {exc.reason}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMTransportError("DeepSeek response was not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise LLMTransportError("DeepSeek response JSON must be an object")
        return decoded


class DeepSeekClient(LLMClient):
    """Infrastructure implementation of the provider-neutral LLMClient."""

    def __init__(self, config: DeepSeekConfig, transport: JsonTransport | None = None):
        self.config = config
        self._transport = transport or UrllibJsonTransport()

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(DeepSeekConfig.from_env())

    def complete_structured(
        self,
        request: LLMRequest,
        output_schema: type[T],
    ) -> StructuredLLMResponse[T]:
        self.config.validate_for_request()
        assert self.config.api_key is not None
        assert self.config.base_url is not None
        assert self.config.model is not None
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "response_format": {"type": "json_object"},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        response = self._post_with_retries(
            f"{self.config.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            payload,
        )
        content = self._extract_content(response)
        output = coerce_structured_output(content, output_schema)
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return StructuredLLMResponse(
            output=output,
            metadata=LLMCallMetadata(
                provider="deepseek",
                model=self.config.model,
                request_id=self._optional_string(response.get("id")),
                prompt_tokens=self._optional_int(usage.get("prompt_tokens")),
                completion_tokens=self._optional_int(usage.get("completion_tokens")),
                total_tokens=self._optional_int(usage.get("total_tokens")),
            ),
        )

    def _post_with_retries(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        last_error: LLMTransportError | None = None
        for _ in range(self.config.max_retries + 1):
            try:
                return self._transport.post_json(
                    url,
                    headers,
                    payload,
                    self.config.timeout_seconds,
                )
            except LLMTransportError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_content(response: Mapping[str, Any]) -> str | Mapping[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMTransportError("DeepSeek response did not contain choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMTransportError("DeepSeek response choice was malformed")
        message = first.get("message")
        if not isinstance(message, Mapping) or "content" not in message:
            raise LLMTransportError("DeepSeek response message was malformed")
        content = message["content"]
        if not isinstance(content, (str, Mapping)):
            raise LLMTransportError("DeepSeek response content was malformed")
        return content

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

