"""Provider-neutral external paper retrieval infrastructure."""

from .arxiv import (
    ArxivConfig,
    ArxivInspirationSearchProvider,
    TextGetTransport,
    UrllibTextGetTransport,
)
from .fake import FakeInspirationSearchProvider, FakeRelatedPaperProvider
from .interfaces import InspirationSearchProvider, RelatedPaperProvider
from .semantic_scholar import (
    JsonGetTransport,
    SemanticScholarConfig,
    SemanticScholarProvider,
    SemanticScholarRelatedPaperProvider,
    UrllibJsonGetTransport,
)
from .semantic_scholar_cache import SemanticScholarCache
from .arxiv_cache import ArxivCache

__all__ = [
    "FakeRelatedPaperProvider",
    "FakeInspirationSearchProvider",
    "RelatedPaperProvider",
    "InspirationSearchProvider",
    "ArxivConfig",
    "ArxivInspirationSearchProvider",
    "TextGetTransport",
    "UrllibTextGetTransport",
    "SemanticScholarConfig",
    "SemanticScholarProvider",
    "SemanticScholarRelatedPaperProvider",
    "JsonGetTransport",
    "UrllibJsonGetTransport",
    "SemanticScholarCache",
    "ArxivCache",
]
