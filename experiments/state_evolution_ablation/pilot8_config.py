"""Frozen, deterministic configuration for the eight-paper pilot.

The seed choices are execution metadata for this ablation only.  They do not
change the benchmark and do not enter either scientific workflow's prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from research_agent.runner.pilot import RealPilotConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "nlp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "state_evolution_ablation_pilot8"
DEFAULT_FULL_OUTPUT_DIR = REPO_ROOT / "outputs" / "output_final_80" / "papers"
DEFAULT_SEMANTIC_SCHOLAR_CONFIG = REPO_ROOT / "config" / "semantic_scholar.json"

# Read the current Full runner defaults instead of duplicating a guessed value.
DEFAULT_MAX_STEPS = int(RealPilotConfig.__dataclass_fields__["max_steps"].default)
DEFAULT_MAX_SEARCHES = int(RealPilotConfig.__dataclass_fields__["max_searches"].default)


@dataclass(frozen=True)
class PilotSeed:
    category: str
    short_id: str
    filename: str
    trajectory_note: str

    @property
    def paper_id(self) -> str:
        return Path(self.filename).stem


# One seed per requested direction.  The choices are existing papers and were
# selected to cover both short/canonical and revision/inspiration histories.
PILOT_SEEDS: tuple[PilotSeed, ...] = (
    PilotSeed(
        "S1 Reasoning & Test-Time Computation",
        "P004",
        "MinerU_markdown_P004_Stepwise_Informativeness_Search_2091868130714607617.md",
        "canonical_short_path",
    ),
    PilotSeed(
        "S2 Agentic Planning & Tool Use",
        "P010",
        "MinerU_markdown_P010_Procedural_Environment_Generation_for_Tool-Use_Agents_2091868130714607626.md",
        "inspiration_and_targeted_revision",
    ),
    PilotSeed(
        "S3 Retrieval & Knowledge-Augmented Generation",
        "P015",
        "MinerU_markdown_P015_RAG_with_Estimation_of_Source_Reliability_2091868130714607631.md",
        "canonical_short_path",
    ),
    PilotSeed(
        "S4 Post-Training & Preference Optimization",
        "P019",
        "MinerU_markdown_P019_MPPO_2091868130714607635.md",
        "canonical_short_path",
    ),
    PilotSeed(
        "S5 Efficient Inference & Model Compression",
        "P021",
        "MinerU_markdown_P021_RefreshKV_2091868057381396482.md",
        "targeted_revision",
    ),
    PilotSeed(
        "S6 Multilingual & Cross-Lingual Adaptation",
        "P026",
        "MinerU_markdown_P026_Cross-Lingual_Optimization_2091868057381396487.md",
        "canonical_short_path",
    ),
    PilotSeed(
        "S7 Safety Alignment & Jailbreak Robustness",
        "P031",
        "MinerU_markdown_P031_Shaping_the_Safety_Boundaries_2091868057381396491.md",
        "targeted_revision",
    ),
    PilotSeed(
        "S8 Information Extraction & Structured NLP",
        "P040",
        "MinerU_markdown_P040_Temporal_Relation_Extraction_GraphTREx_2091868057381396499.md",
        "inspiration_and_targeted_revision",
    ),
)


def get_seed(identifier: str) -> PilotSeed:
    value = identifier.strip()
    for seed in PILOT_SEEDS:
        if value in {seed.short_id, seed.paper_id, seed.filename}:
            return seed
    raise KeyError(f"unknown pilot seed: {identifier}")


def resolve_seed_paths(input_dir: str | Path, seeds: Iterable[PilotSeed] = PILOT_SEEDS) -> dict[str, Path]:
    root = Path(input_dir)
    return {seed.short_id: root / seed.filename for seed in seeds}


def validate_seed_files(input_dir: str | Path, seeds: Iterable[PilotSeed] = PILOT_SEEDS) -> list[str]:
    paths = resolve_seed_paths(input_dir, seeds)
    return [short_id for short_id, path in paths.items() if not path.is_file()]


__all__ = [
    "DEFAULT_INPUT_DIR",
    "DEFAULT_MAX_SEARCHES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_FULL_OUTPUT_DIR",
    "DEFAULT_SEMANTIC_SCHOLAR_CONFIG",
    "PILOT_SEEDS",
    "PilotSeed",
    "get_seed",
    "resolve_seed_paths",
    "validate_seed_files",
]
