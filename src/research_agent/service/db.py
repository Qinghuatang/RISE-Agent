"""SQLAlchemy engine/session lifecycle and Alembic entry point."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    """Business database configuration is missing or invalid."""


def normalize_database_url(database_url: str) -> URL:
    """Use psycopg 3 for PostgreSQL URLs with or without an explicit driver."""

    try:
        url = make_url(database_url)
    except Exception as exc:
        raise DatabaseConfigurationError("invalid PostgreSQL database URL") from exc
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    if url.drivername != "postgresql+psycopg":
        raise DatabaseConfigurationError(
            "business database must use PostgreSQL with the psycopg 3 driver"
        )
    return url


def database_url_from_env() -> str:
    """Read the application URL, allowing the dedicated test URL as fallback."""

    value = os.environ.get("RISE_DATABASE_URL") or os.environ.get(
        "RISE_TEST_POSTGRES_URL"
    )
    if not value:
        raise DatabaseConfigurationError(
            "set RISE_DATABASE_URL (or RISE_TEST_POSTGRES_URL for tests)"
        )
    return value


def create_business_engine(
    database_url: str | None = None,
    *,
    echo: bool = False,
    pool_pre_ping: bool = True,
) -> Engine:
    """Create the application-owned SQLAlchemy engine.

    LangGraph's ``PostgresSaver`` has its own connection lifecycle and tables;
    this engine is exclusively for ``tasks``, ``task_events`` and artifacts.
    """

    raw_url = database_url if database_url is not None else database_url_from_env()
    return create_engine(
        normalize_database_url(raw_url),
        echo=echo,
        pool_pre_ping=pool_pre_ping,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def upgrade_business_schema(engine: Engine, *, revision: str = "head") -> None:
    """Run Alembic migrations on the supplied connection without touching saver tables."""

    repository_root = Path(__file__).resolve().parents[3]
    config = Config(str(repository_root / "alembic.ini"))
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)
