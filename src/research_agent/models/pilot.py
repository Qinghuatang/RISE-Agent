"""Execution-only contracts for the M12 real single-paper pilot."""

from typing import Any, Literal

from pydantic import Field

from .base import StrictModel


class LLMCallAudit(StrictModel):
    """Safe per-call metadata; credentials and full prompts are excluded."""

    sequence: int = Field(ge=1)
    task: str | None = None
    paper_id: str | None = None
    output_schema: str
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    input_chars: int = Field(ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    validation_errors: list[dict[str, str]] = Field(default_factory=list)
    parsed_payload: dict[str, Any] | None = None
    raw_response_excerpt: str | None = None
    format_retry_count: int = Field(default=0, ge=0)
    retry_reason: str | None = None


class PilotManifest(StrictModel):
    """Execution manifest, deliberately separate from scientific state."""

    pilot_name: str
    markdown_path: str
    paper_id: str | None = None
    status: Literal[
        "preflight_failed",
        "running",
        "committed",
        "stopped",
        "failed",
        "finalization_failed",
    ]
    real_services_enabled: bool
    max_steps: int = Field(ge=0)
    max_searches: int = Field(ge=0)
    llm_request_count: int = Field(ge=0)
    prompt_tokens_total: int | None = Field(default=None, ge=0)
    completion_tokens_total: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    known_prompt_tokens: int = Field(default=0, ge=0)
    known_completion_tokens: int = Field(default=0, ge=0)
    known_total_tokens: int = Field(default=0, ge=0)
    calls_with_token_usage: int = Field(default=0, ge=0)
    calls_missing_token_usage: int = Field(default=0, ge=0)
    token_metadata_complete: bool
    semantic_scholar_http_requests: int = Field(ge=0)
    semantic_scholar_cache_scope: str = "none"
    semantic_scholar_cached_operations: list[str] = Field(default_factory=list)
    arxiv_http_requests: int = Field(ge=0)
    scientific_searches_used: int = Field(ge=0)
    trajectory_event_count: int = Field(ge=0)
    replay_record_count: int = Field(ge=0)
    commit_gate_passed: bool = False
    final_proposal_written: bool = False
    error_type: str | None = None
    error_message: str | None = None


class ProviderAudit(StrictModel):
    """Safe provider-level accounting for one pilot."""

    semantic_scholar: dict[str, object] = Field(default_factory=dict)
    arxiv: dict[str, object] = Field(default_factory=dict)


__all__ = ["LLMCallAudit", "PilotManifest", "ProviderAudit"]
