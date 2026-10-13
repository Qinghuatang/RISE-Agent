"""M12 real single-paper pilot wiring.

This module is deliberately an execution boundary.  It supplies real
adapters to the already-defined scientific modules, but never chooses a
scientific action or edits a scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..controller.budget import BudgetTracker
from ..controller.scientific_controller import ScientificController
from ..errors import ConfigurationError, RetrievalConfigurationError
from ..llm.config import DeepSeekConfig
from ..llm.deepseek import DeepSeekClient
from ..llm.recording import RecordingLLMClient, aggregate_token_usage
from ..memory.store import ResearchMemoryStore
from ..models.pilot import PilotManifest, ProviderAudit
from ..models.seed import SeedPaperContext
from ..modules.commit import CommitModule
from ..modules.experiment_design import ExperimentDesignModule
from ..modules.finalizer import FinalizerModule
from ..modules.gap_discovery import GapDiscoveryModule
from ..modules.inspiration_retrieval import InspirationRetrievalModule
from ..modules.related_work_verification import RelatedWorkVerificationModule
from ..modules.seed_context import SeedContextExtractor
from ..modules.solution_ideation import SolutionIdeationModule
from ..modules.targeted_revision import TargetedRevisionModule
from ..retrieval.arxiv import ArxivConfig, ArxivInspirationSearchProvider
from ..retrieval.semantic_scholar import (
    SemanticScholarConfig,
    SemanticScholarProvider,
)
from ..serialization import read_json, write_json
from ..serialization.codec import write_jsonl
from .trajectory_runner import TrajectoryModules, TrajectoryRunner


class PilotConfigurationError(ConfigurationError):
    """Raised before a real provider or LLM call is allowed."""


class _AuditedSemanticScholarProvider:
    """Transparent provider adapter that accumulates provider accounting."""

    provider_name = "semantic_scholar"

    def __init__(self, provider: SemanticScholarProvider) -> None:
        self.provider = provider
        self.http_requests = 0
        self.cached_operations: list[str] = []
        self.cache_scopes: list[str] = []
        self.related_pool_size = 0
        self.papers_with_abstract = 0
        self.resolved_provider_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def get_related_paper_pool(self, seed_context, *, limit: int = 20):
        before = self.provider.total_request_count
        pool = self.provider.get_related_paper_pool(seed_context, limit=limit)
        self.http_requests += self.provider.total_request_count - before
        for operation in self.provider.last_global_cache_operations:
            if operation not in self.cached_operations:
                self.cached_operations.append(operation)
        if self.provider.last_global_cache_hit:
            scope = "global"
        elif self.provider.last_global_cache_operations:
            scope = "global_partial"
        else:
            scope = "none"
        self.cache_scopes.append(scope)
        self.related_pool_size = len(pool.papers)
        self.papers_with_abstract = sum(1 for paper in pool.papers if paper.abstract)
        self.resolved_provider_id = self.provider.last_resolved_provider_id
        return pool

    get_related_papers = get_related_paper_pool

    def audit(self) -> dict[str, object]:
        return {
            "provider_name": self.provider.provider_name,
            "base_url": self.provider.config.base_url,
            "auth_mode": self.provider.config.auth_mode,
            "api_key_configured": self.provider.config.api_key is not None,
            "http_request_count": self.http_requests,
            "global_cache_operations": list(self.cached_operations),
            "cache_scopes": list(self.cache_scopes),
            "cache_scope": self.cache_scopes[-1] if self.cache_scopes else "none",
            "cache_scope_overall": (
                "global"
                if self.cache_scopes and all(scope == "global" for scope in self.cache_scopes)
                else (self.cache_scopes[-1] if self.cache_scopes else "none")
            ),
            "resolved_seed_provider_id": self.resolved_provider_id,
            "seed_resolution": dict(
                getattr(self.provider, "last_seed_resolution_audit", {})
            ),
            "related_pool_size": self.related_pool_size,
            "papers_with_abstract": self.papers_with_abstract,
            "retry_count": self.provider.last_retry_count,
            "rate_limit_count": self.provider.last_rate_limit_count,
        }


class _AuditedArxivProvider:
    """Transparent arXiv adapter; it performs no search unless called."""

    provider_name = "arxiv"

    def __init__(self, provider: ArxivInspirationSearchProvider) -> None:
        self.provider = provider
        self.http_requests = 0
        self.query_count = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self.queries: list[str] = []
        self.cache_hits = 0
        self.cache_scopes: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def search(self, query: str, limit: int = 20):
        before = self.provider.total_request_count
        papers = self.provider.search(query, limit=limit)
        self.http_requests += self.provider.total_request_count - before
        self.query_count += 1
        self.queries.append(query)
        if getattr(self.provider, "last_cache_hit", False):
            self.cache_hits += 1
        self.cache_scopes.append(getattr(self.provider, "last_cache_scope", "none"))
        self.retry_count += self.provider.last_retry_count
        self.rate_limit_count += self.provider.last_rate_limit_count
        return papers

    def audit(self) -> dict[str, object]:
        return {
            "provider_name": self.provider.provider_name,
            "base_url": self.provider.config.base_url,
            "http_request_count": self.http_requests,
            "retry_count": self.retry_count,
            "rate_limit_count": self.rate_limit_count,
            "query_count": self.query_count,
            "queries": list(self.queries),
            "cache_hits": self.cache_hits,
            "cache_scopes": list(self.cache_scopes),
            "cache_dir": getattr(self.provider.config, "cache_dir", None),
        }


@dataclass(frozen=True)
class RealPilotConfig:
    markdown_path: Path
    output_dir: Path | None
    semantic_scholar_config: Path
    max_steps: int = 12
    max_searches: int = 2
    pilot_name: str | None = None
    allow_real_services: bool = False
    allow_overwrite: bool = False


def _safe_error(error: Exception, inner: DeepSeekClient | None = None) -> str:
    message = str(error).strip().replace("\n", " ")[:500]
    key = inner.config.api_key if inner is not None else None
    if key:
        message = message.replace(key, "[REDACTED]")
    return message or type(error).__name__


class RealSinglePaperPilot:
    """Run exactly one explicitly opted-in real single-paper trajectory."""

    def __init__(self, config: RealPilotConfig) -> None:
        self.config = config
        self._llm: RecordingLLMClient | None = None
        self._deepseek: DeepSeekClient | None = None
        self._semantic: _AuditedSemanticScholarProvider | None = None
        self._arxiv: _AuditedArxivProvider | None = None
        self._output_dir: Path | None = None

    def _paper_id(self) -> str:
        from ..ingestion.markdown import parse_markdown_file

        return parse_markdown_file(self.config.markdown_path).paper_id

    def _resolve_output_dir(self, paper_id: str) -> Path:
        if self.config.output_dir is not None:
            return self.config.output_dir
        name = self.config.pilot_name or "m12_real_pilot"
        return Path("outputs") / "pilots" / paper_id / name

    def preflight(self) -> tuple[str, Path, DeepSeekConfig, SemanticScholarConfig, ArxivConfig]:
        if not self.config.allow_real_services:
            raise PilotConfigurationError(
                "real pilot requires the explicit --allow-real-services flag"
            )
        path = self.config.markdown_path
        if not path.is_file():
            raise PilotConfigurationError(f"Markdown input does not exist: {path}")
        if self.config.max_steps < 0 or self.config.max_searches < 0:
            raise PilotConfigurationError("pilot budgets must be non-negative")
        paper_id = self._paper_id()
        output_dir = self._resolve_output_dir(paper_id)
        existing = output_dir / "pilot_manifest.json"
        if existing.exists() and not self.config.allow_overwrite:
            try:
                status = read_json(existing, expected_schema="pilot_manifest")["payload"].get("status")
            except Exception:
                status = "unknown"
            if status not in {"preflight_failed"}:
                raise PilotConfigurationError(
                    f"output directory already contains a pilot ({status}); "
                    "choose a new directory or pass --allow-overwrite"
                )
        deepseek = DeepSeekConfig.from_env()
        deepseek.validate_for_request()
        semantic = SemanticScholarConfig.from_file(self.config.semantic_scholar_config)
        semantic.validate_for_request()
        arxiv = ArxivConfig.from_env()
        return paper_id, output_dir, deepseek, semantic, arxiv

    def _write_call_audit(self, output_dir: Path) -> None:
        assert self._llm is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "llm_calls.jsonl", "llm_call_audit", self._llm.records)

    def _manifest(
        self,
        *,
        paper_id: str | None,
        status: str,
        result: Any = None,
        error: Exception | None = None,
    ) -> PilotManifest:
        records = self._llm.records if self._llm is not None else []
        usage = aggregate_token_usage(records)
        budget = getattr(result, "budget", None)
        final_state = getattr(result, "final_state", None)
        event_count = 0
        replay_count = 0
        if self._output_dir is not None:
            trajectory = self._output_dir / "trajectory.jsonl"
            replay = self._output_dir / "trajectory_replay.jsonl"
            if trajectory.exists():
                event_count = sum(1 for line in trajectory.read_text(encoding="utf-8").splitlines() if line.strip())
            if replay.exists():
                replay_count = sum(1 for line in replay.read_text(encoding="utf-8").splitlines() if line.strip())
        semantic_audit = self._semantic.audit() if self._semantic else {}
        arxiv_audit = self._arxiv.audit() if self._arxiv else {}
        return PilotManifest(
            pilot_name=self.config.pilot_name or "m12_real_pilot",
            markdown_path=str(self.config.markdown_path),
            paper_id=paper_id,
            status=status,
            real_services_enabled=self.config.allow_real_services,
            max_steps=self.config.max_steps,
            max_searches=self.config.max_searches,
            llm_request_count=len(records),
            prompt_tokens_total=usage["prompt_tokens_total"],
            completion_tokens_total=usage["completion_tokens_total"],
            total_tokens=usage["total_tokens"],
            known_prompt_tokens=int(usage["known_prompt_tokens"]),
            known_completion_tokens=int(usage["known_completion_tokens"]),
            known_total_tokens=int(usage["known_total_tokens"]),
            calls_with_token_usage=int(usage["calls_with_token_usage"]),
            calls_missing_token_usage=int(usage["calls_missing_token_usage"]),
            token_metadata_complete=bool(usage["token_metadata_complete"]),
            semantic_scholar_http_requests=int(semantic_audit.get("http_request_count", 0)),
            semantic_scholar_cache_scope=str(semantic_audit.get("cache_scope_overall", "none")),
            semantic_scholar_cached_operations=list(semantic_audit.get("global_cache_operations", [])),
            arxiv_http_requests=int(arxiv_audit.get("http_request_count", 0)),
            scientific_searches_used=(budget.searches_used if budget is not None else 0),
            trajectory_event_count=event_count,
            replay_record_count=replay_count,
            commit_gate_passed=bool(result and getattr(result, "commit_snapshot", None) is not None),
            final_proposal_written=bool(result and getattr(result, "final_proposal", None) is not None),
            error_type=type(error).__name__ if error else getattr(result, "error_type", None),
            error_message=_safe_error(error, self._deepseek) if error else getattr(result, "error_message", None),
        )

    def _write_provider_audit(self, output_dir: Path) -> None:
        audit = ProviderAudit(
            semantic_scholar=self._semantic.audit() if self._semantic else {},
            arxiv=self._arxiv.audit() if self._arxiv else {},
        )
        write_json(output_dir / "provider_audit.json", "provider_audit", audit)

    def run(self):
        paper_id: str | None = None
        output_dir: Path | None = None
        try:
            paper_id, output_dir, deepseek_config, semantic_config, arxiv_config = self.preflight()
            self._output_dir = output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            self._deepseek = DeepSeekClient(deepseek_config)
            # The real pilot deliberately omits max_tokens from every
            # provider request.  The provider/model context window remains
            # the hard external limit; no local output truncation is added.
            self._llm = RecordingLLMClient(
                self._deepseek,
                unbounded_output=True,
            )

            # Seed extraction is intentionally fresh and is the only input to
            # the fresh trajectory.  It remains one full-paper call.
            seed_result, full_paper_reading_context = SeedContextExtractor(self._llm).extract_file_with_reading(
                self.config.markdown_path
            )
            write_json(output_dir / "seed_context.json", "seed_paper_context", seed_result.context)
            write_json(output_dir / "seed_extraction_audit.json", "seed_extraction_audit", seed_result.audit)
            paper_id = seed_result.context.paper_id

            semantic_provider = SemanticScholarProvider(semantic_config)
            arxiv_provider = ArxivInspirationSearchProvider(arxiv_config)
            self._semantic = _AuditedSemanticScholarProvider(semantic_provider)
            self._arxiv = _AuditedArxivProvider(arxiv_provider)
            modules = TrajectoryModules(
                gap_discovery=GapDiscoveryModule(self._llm),
                solution_ideation=SolutionIdeationModule(self._llm),
                related_work_verification=RelatedWorkVerificationModule(self._llm, self._semantic),
                inspiration_retrieval=InspirationRetrievalModule(self._llm, self._arxiv),
                targeted_revision=TargetedRevisionModule(self._llm),
                experiment_design=ExperimentDesignModule(self._llm),
                controller=ScientificController(self._llm),
                commit=CommitModule(),
                finalizer=FinalizerModule(self._llm),
            )
            runner = TrajectoryRunner(
                modules,
                budget=BudgetTracker(
                    max_steps=self.config.max_steps,
                    max_searches=self.config.max_searches,
                ),
                output_dir=output_dir,
            )
            result = runner.run(
                seed_result.context,
                full_paper_reading_context=full_paper_reading_context,
            )
            self._write_call_audit(output_dir)
            self._write_provider_audit(output_dir)
            write_json(
                output_dir / "pilot_manifest.json",
                "pilot_manifest",
                self._manifest(paper_id=paper_id, status=result.status, result=result),
            )
            return result
        except Exception as exc:
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                self._write_call_audit(output_dir)
                self._write_provider_audit(output_dir)
                write_json(
                    output_dir / "pilot_manifest.json",
                    "pilot_manifest",
                    self._manifest(
                        paper_id=paper_id,
                        status=("preflight_failed" if self._llm is None else "failed"),
                        error=exc,
                    ),
                )
            raise


__all__ = ["PilotConfigurationError", "RealPilotConfig", "RealSinglePaperPilot"]
