"""Configuration for the DeepSeek infrastructure adapter."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

from ..errors import ConfigurationError


def _optional_env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = _optional_env(environ, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return value


def _parse_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = _optional_env(environ, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class DeepSeekConfig:
    """DeepSeek settings, with credentials optional until a request is made."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.base_url is not None:
            normalized = self.base_url.strip().rstrip("/") or None
            object.__setattr__(self, "base_url", normalized)
        if self.model is not None:
            normalized = self.model.strip() or None
            object.__setattr__(self, "model", normalized)
        if self.api_key is not None:
            normalized = self.api_key.strip() or None
            object.__setattr__(self, "api_key", normalized)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be a finite positive number")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DeepSeekConfig":
        values = os.environ if environ is None else environ
        return cls(
            api_key=_optional_env(values, "DEEPSEEK_API_KEY"),
            base_url=_optional_env(values, "DEEPSEEK_BASE_URL"),
            model=_optional_env(values, "DEEPSEEK_MODEL"),
            timeout_seconds=_parse_float(values, "DEEPSEEK_TIMEOUT_SECONDS", 60.0),
            max_retries=_parse_int(values, "DEEPSEEK_MAX_RETRIES", 0),
        )

    def validate_for_request(self) -> None:
        missing = [
            name
            for name, value in (
                ("DEEPSEEK_API_KEY", self.api_key),
                ("DEEPSEEK_BASE_URL", self.base_url),
                ("DEEPSEEK_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(f"missing DeepSeek request configuration: {joined}")

    def public_dict(self) -> dict[str, object]:
        """Return safe metadata without exposing the API key."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "api_key_configured": self.api_key is not None,
        }

