"""Minimal offline command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Scientific research ideation agent foundation CLI.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "offline-smoke",
        help="run a local smoke check without LLM or network access",
    )
    extract_parser = subparsers.add_parser(
        "extract-seed-context",
        help="extract one SeedPaperContext using the explicitly configured LLM",
    )
    extract_parser.add_argument("markdown_path", type=Path)
    extract_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for seed_context.json and seed_extraction_audit.json",
    )
    extract_parser.add_argument("--max-tokens", type=int, default=None)
    gap_parser = subparsers.add_parser(
        "discover-gaps",
        help=(
            "discover gaps from the full Markdown paper and an existing "
            "SeedPaperContext using the explicitly configured LLM"
        ),
    )
    gap_parser.add_argument("seed_context_path", type=Path)
    gap_parser.add_argument(
        "--markdown-path",
        type=Path,
        required=True,
        help="the original Markdown paper used as the complete Gap Discovery input",
    )
    gap_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_gap_smoke"),
        help="directory for Gap Discovery result, state, memory, and trajectory",
    )
    gap_parser.add_argument("--max-tokens", type=int, default=None)
    verify_parser = subparsers.add_parser(
        "verify-gap",
        help="verify an existing active gap against Semantic Scholar related work",
    )
    verify_parser.add_argument("seed_context_path", type=Path)
    verify_parser.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="existing trajectory directory containing the active M4 gap",
    )
    verify_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_verify_gap_smoke"),
        help="directory for verification result, audit, memory, and trajectory",
    )
    verify_parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=Path("config/semantic_scholar.json"),
        help="JSON file containing Semantic Scholar API configuration",
    )
    verify_parser.add_argument("--max-tokens", type=int, default=None)
    pilot_parser = subparsers.add_parser(
        "real-pilot",
        help="run one explicitly opted-in real single-paper trajectory",
    )
    pilot_parser.add_argument("markdown_path", type=Path)
    pilot_parser.add_argument(
        "--allow-real-services",
        action="store_true",
        help="required explicit opt-in for DeepSeek and external providers",
    )
    pilot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="fresh output directory for this pilot",
    )
    pilot_parser.add_argument("--max-steps", type=int, default=12)
    pilot_parser.add_argument("--max-searches", type=int, default=2)
    pilot_parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=Path("config/semantic_scholar.json"),
    )
    pilot_parser.add_argument("--pilot-name", default=None)
    pilot_parser.add_argument("--allow-overwrite", action="store_true")
    batch_parser = subparsers.add_parser(
        "batch-eval",
        help="run the existing single-paper pilot independently for a Markdown batch",
    )
    batch_parser.add_argument("input_dir", type=Path)
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_eval"),
    )
    batch_parser.add_argument("--batch-name", default="m13_batch_eval")
    batch_parser.add_argument("--max-papers", type=int, default=None)
    batch_parser.add_argument("--selection-seed", type=int, default=1301)
    batch_parser.add_argument("--max-steps", type=int, default=12)
    batch_parser.add_argument("--max-searches", type=int, default=2)
    batch_parser.add_argument(
        "--semantic-scholar-config",
        type=Path,
        default=Path("config/semantic_scholar.json"),
    )
    batch_parser.add_argument(
        "--allow-real-services",
        action="store_true",
        help="required explicit opt-in for DeepSeek and external providers",
    )
    batch_parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "offline-smoke":
        print("research-agent offline smoke: ok")
    elif args.command == "extract-seed-context":
        from .llm.deepseek import DeepSeekClient
        from .modules.seed_context import (
            SeedContextConfig,
            SeedContextExtractor,
            write_seed_context_outputs,
        )

        config = SeedContextConfig.from_env()
        if args.max_tokens is not None:
            config = SeedContextConfig(
                max_tokens=args.max_tokens,
                audit_excerpt_chars=config.audit_excerpt_chars,
            )
        result = SeedContextExtractor(DeepSeekClient.from_env(), config).extract_file(
            args.markdown_path
        )
        context_path, audit_path = write_seed_context_outputs(result, args.output_dir)
        print(f"seed context: {context_path}")
        print(f"extraction audit: {audit_path}")
    elif args.command == "discover-gaps":
        from .llm.deepseek import DeepSeekClient
        from .ingestion.markdown import build_full_paper_reading_context, parse_markdown_file
        from .models.seed import SeedPaperContext
        from .modules.gap_discovery import GapDiscoveryConfig, GapDiscoveryModule
        from .serialization import read_json, write_json
        from .trajectory.context import TrajectoryContext

        envelope = read_json(
            args.seed_context_path,
            expected_schema="seed_paper_context",
        )
        seed_context = SeedPaperContext.model_validate(envelope["payload"])
        parsed = parse_markdown_file(args.markdown_path)
        if parsed.paper_id != seed_context.paper_id:
            raise SystemExit(
                "--markdown-path paper_id does not match the SeedPaperContext: "
                f"{parsed.paper_id!r} != {seed_context.paper_id!r}"
            )
        reading = build_full_paper_reading_context(parsed)
        config = GapDiscoveryConfig.from_env()
        if args.max_tokens is not None:
            config = GapDiscoveryConfig(max_tokens=args.max_tokens)
        client = DeepSeekClient.from_env()
        context = TrajectoryContext.initialize(seed_context)
        module = GapDiscoveryModule(client, config)
        result = module.discover(
            seed_context,
            full_paper_reading_context=reading,
            context=context,
            step=1,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "gap_discovery_result.json", "gap_discovery_result", result)
        context.persist(output_dir)
        metadata = module.last_call_metadata
        report = {
            "provider": metadata.provider if metadata else None,
            "model": metadata.model if metadata else None,
            "request_count": module.call_count,
            "request_id": metadata.request_id if metadata else None,
            "prompt_tokens": metadata.prompt_tokens if metadata else None,
            "completion_tokens": metadata.completion_tokens if metadata else None,
            "total_tokens": metadata.total_tokens if metadata else None,
            "input_chars": module.last_input_chars,
            "candidate_gap_count": len(result.gap_agenda.gaps),
            "active_gap": result.active_gap.model_dump(mode="json") if result.active_gap else None,
            "warnings": list(reading.warnings),
        }
        write_json(output_dir / "gap_discovery_audit.json", "gap_discovery_audit", report)
        print(f"gap discovery result: {output_dir / 'gap_discovery_result.json'}")
        print(f"request_count: {module.call_count}")
        print(f"candidate_gap_count: {len(result.gap_agenda.gaps)}")
        print(f"active_gap: {result.active_gap.id if result.active_gap else None}")
    elif args.command == "verify-gap":
        from .errors import RetrievalError, RetrievalHTTPError
        from .llm.deepseek import DeepSeekClient
        from .models.seed import SeedPaperContext
        from .modules.related_work_verification import (
            RelatedWorkVerificationModule,
            VerificationConfig,
        )
        from .retrieval.semantic_scholar import SemanticScholarProvider
        from .serialization import read_json, write_json
        from .trajectory.context import TrajectoryContext

        envelope = read_json(
            args.seed_context_path,
            expected_schema="seed_paper_context",
        )
        seed_context = SeedPaperContext.model_validate(envelope["payload"])
        context = TrajectoryContext.load(seed_context, args.trajectory_dir)
        state = context.get_state()
        if state.active_gap is None:
            raise SystemExit("verify-gap requires an active gap in --trajectory-dir")
        step = max((event.step for event in context.trajectory.get_events()), default=0) + 1
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        provider = None
        module = None
        try:
            config = VerificationConfig.from_env()
            if args.max_tokens is not None:
                config = VerificationConfig(max_tokens=args.max_tokens)
            provider = SemanticScholarProvider.from_file(args.semantic_scholar_config)
            module = RelatedWorkVerificationModule(
                DeepSeekClient.from_env(), provider, config
            )
            result = module.verify_gap(state.active_gap, context=context, step=step)
        except RetrievalError as exc:
            http_status = exc.status_code if isinstance(exc, RetrievalHTTPError) else None
            scientific_search_count = module.scientific_search_count if module else 0
            provider_request_count = provider.last_request_count if provider else 0
            context.trajectory.append_event(
                step,
                "related_pool_acquisition_failed",
                action="VERIFY_RELATED_WORK",
                result_ref="related_paper_pool",
                result_payload={
                    "provider": "semantic_scholar",
                    "semantic_scholar_base_url": (
                        provider.config.base_url if provider else None
                    ),
                    "auth_mode": provider.config.auth_mode if provider else None,
                    "http_status": http_status,
                    "scientific_search_count": scientific_search_count,
                    "provider_http_request_count": provider_request_count,
                    "cache_hit": False,
                    "cache_scope": "none",
                    "cached_operations": (
                        module.last_global_cache_operations if module else []
                    ),
                },
                error=type(exc).__name__,
            )
            context.persist(output_dir)
            write_json(
                output_dir / "verification_error.json",
                "verification_error",
                {
                    "provider": "semantic_scholar",
                    "operation": "related_pool_acquisition",
                    "error_type": type(exc).__name__,
                    "http_status": http_status,
                    "api_key_configured": bool(provider and provider.config.api_key),
                    "config_path": str(args.semantic_scholar_config),
                    "semantic_scholar_base_url": (
                        provider.config.base_url if provider else None
                    ),
                    "auth_mode": provider.config.auth_mode if provider else None,
                    "provider_http_request_count": provider_request_count,
                    "scientific_search_count": scientific_search_count,
                    "cache_hit": False,
                    "cache_scope": module.last_cache_scope if module else "none",
                    "cached_operations": (
                        module.last_global_cache_operations if module else []
                    ),
                    "retry_count": provider.last_retry_count if provider else 0,
                    "rate_limit_count": provider.last_rate_limit_count if provider else 0,
                    "pacing_waits_seconds": provider.last_pacing_waits if provider else [],
                    "retry_waits_seconds": provider.last_retry_waits if provider else [],
                    "resolved_semantic_scholar_paper_id": (
                        provider.last_resolved_provider_id if provider else None
                    ),
                    "warnings": [
                        "Verification stopped before LLM call; no scientific result was fabricated."
                    ],
                },
            )
            print(f"verification error: {output_dir / 'verification_error.json'}")
            print(f"scientific_search_count: {scientific_search_count}")
            print(f"provider_http_request_count: {provider_request_count}")
            return 2
        write_json(output_dir / "gap_verification_result.json", "gap_verification_result", result)
        context.persist(output_dir)
        metadata = module.last_call_metadata
        pool = context.get_memory().paper_memory.related_paper_pool
        audit = {
            "provider": metadata.provider if metadata else None,
            "model": metadata.model if metadata else None,
            "request_count": module.call_count,
            "request_id": metadata.request_id if metadata else None,
            "prompt_tokens": metadata.prompt_tokens if metadata else None,
            "completion_tokens": metadata.completion_tokens if metadata else None,
            "total_tokens": metadata.total_tokens if metadata else None,
            "input_chars": module.last_input_chars,
            "provider_http_request_count": module.last_provider_request_count,
            "scientific_search_count": module.scientific_search_count,
            "cache_hit": module.last_cache_hit,
            "cache_scope": module.last_cache_scope,
            "cached_operations": module.last_global_cache_operations,
            "global_cache_hit": module.last_cache_scope == "global",
            "semantic_scholar_cache_dir": provider.config.cache_dir,
            "semantic_scholar_base_url": provider.config.base_url,
            "auth_mode": provider.config.auth_mode,
            "api_key_configured": provider.config.api_key is not None,
            "retry_count": provider.last_retry_count,
            "rate_limit_count": provider.last_rate_limit_count,
            "pacing_waits_seconds": provider.last_pacing_waits,
            "retry_waits_seconds": provider.last_retry_waits,
            "resolved_semantic_scholar_paper_id": provider.last_resolved_provider_id,
            "pool_size": len(pool.papers) if pool else 0,
            "pool_papers_with_abstract": sum(
                1 for paper in (pool.papers if pool else []) if paper.abstract
            ),
            "validation": "success",
            "warnings": [],
        }
        write_json(output_dir / "verification_audit.json", "verification_audit", audit)
        print(f"gap verification result: {output_dir / 'gap_verification_result.json'}")
        print(f"request_count: {module.call_count}")
        print(f"provider_http_request_count: {module.last_provider_request_count}")
        print(f"scientific_search_count: {module.scientific_search_count}")
        print(f"status: {result.status}")
    elif args.command == "real-pilot":
        from .runner.pilot import RealPilotConfig, RealSinglePaperPilot

        pilot = RealSinglePaperPilot(
            RealPilotConfig(
                markdown_path=args.markdown_path,
                output_dir=args.output_dir,
                semantic_scholar_config=args.semantic_scholar_config,
                max_steps=args.max_steps,
                max_searches=args.max_searches,
                pilot_name=args.pilot_name,
                allow_real_services=args.allow_real_services,
                allow_overwrite=args.allow_overwrite,
            )
        )
        try:
            result = pilot.run()
        except Exception as exc:
            print(f"real pilot failed: {type(exc).__name__}: {str(exc)[:500]}")
            return 2
        print(f"real pilot status: {result.status}")
        print(f"real pilot output: {pilot._output_dir}")
    elif args.command == "batch-eval":
        from .runner.batch_evaluation import (
            BatchEvaluationConfig,
            BatchEvaluationRunner,
            make_real_paper_executor,
        )

        try:
            executor = make_real_paper_executor(
                semantic_scholar_config=args.semantic_scholar_config,
                max_steps=args.max_steps,
                max_searches=args.max_searches,
                allow_real_services=args.allow_real_services,
                allow_overwrite=args.allow_overwrite,
            )
            report = BatchEvaluationRunner(
                BatchEvaluationConfig(
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    batch_name=args.batch_name,
                    max_papers=args.max_papers,
                    selection_seed=args.selection_seed,
                    allow_overwrite=args.allow_overwrite,
                ),
                executor,
            ).run()
        except Exception as exc:
            print(f"batch evaluation failed: {type(exc).__name__}: {str(exc)[:500]}")
            return 2
        print(f"batch evaluation report: {args.output_dir / 'evaluation_report.json'}")
        print(f"paper_count: {report.paper_count}")
        print(f"completed_count: {report.completed_count}")
        print(f"failed_count: {report.failed_count}")
        print(f"stopped_count: {report.stopped_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
