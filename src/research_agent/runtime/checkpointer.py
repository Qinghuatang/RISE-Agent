"""Lifecycle helpers for the official synchronous PostgreSQL checkpointer."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def open_postgres_checkpointer(
    database_url: str,
    *,
    setup: bool = True,
) -> Iterator[object]:
    """Open ``PostgresSaver`` with restricted deserialization.

    The returned saver remains valid only inside this context. Checkpoint
    payloads are JSON-compatible Graph State snapshots, so no application
    modules need to be enabled in the MessagePack deserializer allowlist.
    Runtime owners initialize the schema by default. Read-only tooling can set
    ``setup=False`` to inspect an existing checkpoint database without schema
    writes.
    """

    try:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:  # pragma: no cover - optional persistence extra
        raise RuntimeError(
            "PostgreSQL checkpoints require `pip install -e '.[agent,postgres]'`."
        ) from exc

    connection = psycopg.connect(
        database_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    try:
        saver = PostgresSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=[]),
        )
        if setup:
            saver.setup()
        yield saver
    finally:
        connection.close()


__all__ = ["open_postgres_checkpointer"]
