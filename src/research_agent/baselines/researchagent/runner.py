"""Execution and artifact adapter around the official ResearchAgent pipeline."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...llm.config import DeepSeekConfig
from ...retrieval.semantic_scholar import SemanticScholarConfig
from ...serialization import write_json, write_jsonl
from .input import SeedMarkdownInput, load_seed_markdown
from .llm import ResearchAgentDeepSeekClient
from .retrieval import ResearchAgentSemanticScholarAdapter


class ResearchAgentBaselineError(RuntimeError):
    """Expected adapter-level failure with artifacts still materialized."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_official_pipeline(official_repo: str | Path):
    root = Path(official_repo)
    code_root = root / "code"
    if not code_root.is_dir():
        raise ResearchAgentBaselineError(f"official ResearchAgent code directory not found: {code_root}")
    code_text = str(code_root)
    if code_text not in sys.path:
        sys.path.insert(0, code_text)
    module = importlib.import_module("pipelines.research_pipeline")
    return module.ResearchPipeline


def _normalized_proposal(context: Mapping[str, Any]) -> dict[str, str]:
    problem = context.get("problem")
    problem_rationale = context.get("problem_rationale")
    method = context.get("method")
    method_rationale = context.get("method_rationale")
    experiment = context.get("experiment")
    experiment_rationale = context.get("experiment_rationale")
    required = {
        "problem": problem,
        "problem_rationale": problem_rationale,
        "method": method,
        "method_rationale": method_rationale,
        "experiment": experiment,
        "experiment_rationale": experiment_rationale,
    }
    missing = [name for name, value in required.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ResearchAgentBaselineError(
            "official ResearchAgent final output is incomplete: " + ", ".join(missing)
        )
    return {
        "research_problem": f"{problem.strip()}\n\nRationale: {problem_rationale.strip()}",
        "proposed_method": f"{method.strip()}\n\nRationale: {method_rationale.strip()}",
        "technical_rationale": method_rationale.strip(),
        "experimental_plan": f"{experiment.strip()}\n\nRationale: {experiment_rationale.strip()}",
    }


def _call_statistics(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    known = {
        field: sum(item[field] for item in calls if isinstance(item.get(field), int))
        for field in token_fields
    }
    missing = sum(
        1
        for item in calls
        if any(not isinstance(item.get(field), int) for field in token_fields)
    )
    complete = bool(calls) and missing == 0
    return {
        "llm_call_count": len(calls),
        "logical_llm_call_count": len({item.get("sequence") for item in calls}),
        "problem_stage_calls": sum(item.get("stage") == "problem_stage" for item in calls),
        "method_stage_calls": sum(item.get("stage") == "method_stage" for item in calls),
        "experiment_stage_calls": sum(item.get("stage") == "experiment_stage" for item in calls),
        "reviewer_calls": sum("reviewer" in str(item.get("stage")) for item in calls),
        "prompt_tokens": known["prompt_tokens"] if complete else None,
        "completion_tokens": known["completion_tokens"] if complete else None,
        "total_tokens": known["total_tokens"] if complete else None,
        "known_prompt_tokens": known["prompt_tokens"],
        "known_completion_tokens": known["completion_tokens"],
        "known_total_tokens": known["total_tokens"],
        "calls_missing_token_metadata": missing,
        "token_metadata_complete": complete,
    }


class ResearchAgentBaselineRunner:
    """Run official ResearchAgent with only infrastructure adapters changed."""

    def __init__(
        self,
        *,
        official_repo: str | Path,
        llm_client: ResearchAgentDeepSeekClient,
        semantic_scholar: ResearchAgentSemanticScholarAdapter,
        knowledge_store: Any,
        iterations: int = 3,
    ) -> None:
        if iterations != 3:
            raise ValueError("official ResearchAgent baseline is frozen at iterations=3")
        self.official_repo = str(official_repo)
        self.llm_client = llm_client
        self.semantic_scholar = semantic_scholar
        self.knowledge_store = knowledge_store
        self.iterations = iterations

    def _write_llm_calls(self, destination: Path) -> list[dict[str, Any]]:
        calls = [call.to_dict() for call in self.llm_client.calls]
        write_jsonl(destination / "llm_calls.jsonl", "researchagent_llm_call", calls)
        return calls

    def run(self, markdown_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        source = Path(markdown_path)
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"ResearchAgent output directory already exists: {destination}")
        destination.mkdir(parents=True)
        seed: SeedMarkdownInput | None = None
        context: dict[str, Any] = {}
        error: Exception | None = None
        normalized: dict[str, str] | None = None
        seed_record: dict[str, Any] | None = None
        references: list[dict[str, Any]] = []
        entities: list[str] = []

        try:
            seed = load_seed_markdown(source)
            write_json(destination / "seed_input.json", "researchagent_seed_input", seed.to_dict())
            seed_record = self.semantic_scholar.resolve_seed_paper(seed.title, seed.abstract)
            references = self.semantic_scholar.get_relevant_references(seed_record, top_k=10)
            entity_ids = [seed_record["corpusId"]] + [
                item["corpusId"] for item in references if isinstance(item.get("corpusId"), int)
            ]
            entities = list(self.knowledge_store.get_relevant_entities(entity_ids))
            paper = dict(seed_record)
            # Official prompts consume only title/abstract. Keep these values
            # authoritative from deterministic Markdown extraction.
            paper["title"] = seed.title
            paper["abstract"] = seed.abstract
            context = {"paper": paper, "references": references, "entities": entities}
            pipeline_type = _load_official_pipeline(self.official_repo)
            pipeline = pipeline_type(api_client=self.llm_client, iterations=self.iterations)
            context = pipeline.run(context)

            history = context.get("history") if isinstance(context.get("history"), Mapping) else {}
            write_json(
                destination / "problem_output.json",
                "researchagent_problem_output",
                {"selected": {key: context.get(key) for key in ("problem", "problem_rationale", "problem_feedbacks")}, "iterations": history.get("problems", [])},
            )
            write_json(
                destination / "method_output.json",
                "researchagent_method_output",
                {"selected": {key: context.get(key) for key in ("method", "method_rationale", "method_feedbacks")}, "iterations": history.get("methods", [])},
            )
            write_json(
                destination / "experiment_output.json",
                "researchagent_experiment_output",
                {"selected": {key: context.get(key) for key in ("experiment", "experiment_rationale", "experiment_feedbacks")}, "iterations": history.get("experiments", [])},
            )
            normalized = _normalized_proposal(context)
            write_json(destination / "final_proposal.json", "researchagent_normalized_proposal", normalized)
            write_json(destination / "researchagent_raw_result.json", "researchagent_raw_result", context)
        except Exception as exc:
            error = exc

        calls = self._write_llm_calls(destination)
        retrieval_audit = self.semantic_scholar.audit_dict()
        retrieval_audit.update(
            {
                "seed_paper_id": seed.paper_id if seed else source.stem,
                "reference_count": len(references),
                "knowledge_covered_reference_count": sum(
                    1
                    for item in references
                    if isinstance(item.get("corpusId"), int)
                    and item["corpusId"] in self.knowledge_store.paper2entities
                ),
                "entity_count": len(entities),
            }
        )
        write_json(destination / "retrieval_audit.json", "researchagent_retrieval_audit", retrieval_audit)
        if error is not None:
            write_json(
                destination / "error.json",
                "researchagent_error",
                {"error_type": type(error).__name__, "error_message": str(error)[:2_000]},
            )

        stats = _call_statistics(calls)
        manifest = {
            "paper_id": seed.paper_id if seed else source.stem,
            "markdown_path": str(source),
            "status": "succeeded" if error is None else "failed",
            "official_repo": self.official_repo,
            "pipeline": "Problem → Review/Refine → Method → Review/Refine → Experiment → Review/Refine",
            "iterations": self.iterations,
            "reviewer_dimensions_per_stage": 5,
            "provider": self.llm_client.config.public_dict(),
            "semantic_scholar": self.semantic_scholar.config.public_dict(),
            "llm": stats,
            "retrieval": retrieval_audit,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error)[:2_000] if error else None,
            "normalized_proposal_saved": normalized is not None,
        }
        write_json(destination / "run_manifest.json", "researchagent_run_manifest", manifest)
        return manifest


def run_researchagent_batch(
    markdown_paths: Sequence[str | Path],
    output_dir: str | Path,
    runner_factory: Callable[[], ResearchAgentBaselineRunner],
    *,
    batch_name: str,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"ResearchAgent batch output directory already exists: {destination}")
    paths = sorted((Path(path) for path in markdown_paths), key=lambda path: path.name)
    if not paths:
        raise ValueError("ResearchAgent batch requires at least one Markdown input")
    destination.mkdir(parents=True)
    papers: list[dict[str, Any]] = []
    for source in paths:
        paper_output = destination / source.stem
        try:
            papers.append(runner_factory().run(source, paper_output))
        except Exception as exc:
            papers.append(
                {
                    "paper_id": source.stem,
                    "markdown_path": str(source),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2_000],
                    "llm": {},
                    "retrieval": {},
                }
            )

    def sum_nested(path: tuple[str, ...]) -> int:
        return sum(
            int(_nested_get(paper, path) or 0)
            for paper in papers
            if isinstance(_nested_get(paper, path), int)
        )

    report = {
        "batch_name": batch_name,
        "official_pipeline": "ResearchAgent official ResearchPipeline",
        "total_papers": len(paths),
        "succeeded": sum(paper.get("status") == "succeeded" for paper in papers),
        "failed": sum(paper.get("status") == "failed" for paper in papers),
        "total_llm_calls": sum_nested(("llm", "llm_call_count")),
        "logical_llm_calls": sum_nested(("llm", "logical_llm_call_count")),
        "problem_stage_calls": sum_nested(("llm", "problem_stage_calls")),
        "method_stage_calls": sum_nested(("llm", "method_stage_calls")),
        "experiment_stage_calls": sum_nested(("llm", "experiment_stage_calls")),
        "reviewer_calls": sum_nested(("llm", "reviewer_calls")),
        "known_prompt_tokens": sum_nested(("llm", "known_prompt_tokens")),
        "known_completion_tokens": sum_nested(("llm", "known_completion_tokens")),
        "known_total_tokens": sum_nested(("llm", "known_total_tokens")),
        "calls_missing_token_metadata": sum_nested(("llm", "calls_missing_token_metadata")),
        "token_metadata_complete": bool(papers)
        and all(paper.get("status") == "succeeded" for paper in papers)
        and all(
            paper.get("llm", {}).get("token_metadata_complete", False)
            for paper in papers
        ),
        "semantic_scholar_http_attempts": sum_nested(("retrieval", "http_attempt_count")),
        "semantic_scholar_cache_hits": sum_nested(("retrieval", "cache_hits")),
        "references_retrieved": sum_nested(("retrieval", "reference_count")),
        "knowledge_covered_references": sum_nested(("retrieval", "knowledge_covered_reference_count")),
        "entities_retrieved": sum_nested(("retrieval", "entity_count")),
        "papers": papers,
    }
    report["prompt_tokens"] = report["known_prompt_tokens"] if report["token_metadata_complete"] else None
    report["completion_tokens"] = report["known_completion_tokens"] if report["token_metadata_complete"] else None
    report["total_tokens"] = report["known_total_tokens"] if report["token_metadata_complete"] else None
    write_json(destination / "evaluation_report.json", "researchagent_evaluation_report", report)
    return report


def _nested_get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


__all__ = [
    "ResearchAgentBaselineError",
    "ResearchAgentBaselineRunner",
    "run_researchagent_batch",
]
