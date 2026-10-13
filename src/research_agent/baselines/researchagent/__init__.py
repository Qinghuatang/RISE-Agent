"""Compatibility runner for the official ResearchAgent repository.

The scientific pipeline is imported from the downloaded official repository;
this package contains only input, provider, artifact, and batch adapters.
"""

from .input import SeedMarkdownInput, load_seed_markdown
from .llm import ResearchAgentDeepSeekClient
from .retrieval import ResearchAgentSemanticScholarAdapter
from .runner import ResearchAgentBaselineRunner, run_researchagent_batch

__all__ = [
    "SeedMarkdownInput",
    "load_seed_markdown",
    "ResearchAgentDeepSeekClient",
    "ResearchAgentSemanticScholarAdapter",
    "ResearchAgentBaselineRunner",
    "run_researchagent_batch",
]
