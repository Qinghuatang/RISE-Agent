#!/usr/bin/env python3
"""Run the frozen one-shot Direct LLM baseline for one or more papers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Allow this script to be run directly from the repository root without
# requiring the caller to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient, JsonTransport, UrllibJsonTransport
from research_agent.runner.direct_llm import DirectLLMRunner, run_direct_batch


class CapturingJsonTransport(JsonTransport):
    """Capture the provider response body without capturing request headers."""

    def __init__(self) -> None:
        self.inner = UrllibJsonTransport()
        self.last_response: Mapping[str, Any] | None = None

    def post_json(self, url, headers, payload, timeout):
        response = self.inner.post_json(url, headers, payload, timeout)
        self.last_response = response
        return response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot Direct LLM baseline runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, nargs="+", help="one or more Markdown files")
    group.add_argument("--input-dir", type=Path, help="directory containing Markdown files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-name", default="direct_llm_batch")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def _paths(args: argparse.Namespace) -> list[Path]:
    if args.input is not None:
        paths = list(args.input)
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"input directory does not exist: {args.input_dir}")
        paths = sorted(
            path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"
        )
    if not paths:
        raise SystemExit("no Markdown inputs supplied")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing Markdown input(s): " + ", ".join(missing))
    return paths


def main() -> int:
    args = _parser().parse_args()
    paths = _paths(args)
    config = DeepSeekConfig.from_env()
    config.validate_for_request()

    def factory() -> DirectLLMRunner:
        capture = CapturingJsonTransport()
        client = DeepSeekClient(config, transport=capture)
        return DirectLLMRunner(
            client,
            raw_response_source=capture,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if len(paths) == 1:
        result = factory().run(paths[0], args.output_dir)
        print(json.dumps(result.manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if result.manifest.status == "succeeded" else 1

    report = run_direct_batch(paths, args.output_dir, factory, batch_name=args.batch_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
