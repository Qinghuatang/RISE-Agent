"""Command-line entry point for the RISE FastAPI service."""

from __future__ import annotations

import uvicorn

from ..observability.logging import configure_logging
from .api import APISettings


def main() -> None:
    configure_logging()
    settings = APISettings.from_env()
    uvicorn.run(
        "research_agent.service.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
