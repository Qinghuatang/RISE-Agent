"""PostgreSQL session advisory locks for exclusive Graph Thread execution."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from threading import Event
from uuid import UUID


class ThreadLockUnavailable(RuntimeError):
    """Raised when an execution lock could not be acquired in the requested time."""


def advisory_keys(thread_id: str) -> tuple[int, int]:
    """Map a stable UUID to PostgreSQL's two-int advisory lock namespace."""

    try:
        value = UUID(thread_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("thread_id must be a valid UUID") from exc
    raw = hashlib.blake2b(value.bytes, digest_size=8).digest()
    first = int.from_bytes(raw[:4], byteorder="big", signed=True)
    second = int.from_bytes(raw[4:], byteorder="big", signed=True)
    return first, second


class ThreadAdvisoryLock:
    """Hold a session-level PostgreSQL lock on one Graph Thread.

    The connection is private to this object and stays open for the whole
    execution. A session-level advisory lock survives transaction boundaries,
    which lets repository heartbeats use independent connections.
    """

    def __init__(self, database_url: str, thread_id: str) -> None:
        if database_url.startswith("postgresql+psycopg://"):
            database_url = "postgresql://" + database_url.removeprefix(
                "postgresql+psycopg://"
            )
        self.database_url = database_url
        self.thread_id = str(UUID(thread_id))
        self.keys = advisory_keys(self.thread_id)
        self._connection = None
        self.acquired = False

    def acquire(
        self,
        *,
        timeout: float | None = 0,
        poll_interval: float = 0.1,
        should_abort: Event | Callable[[], bool] | None = None,
    ) -> bool:
        if self.acquired:
            return True
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional postgres extra
            raise RuntimeError("Thread locks require the project postgres extra") from exc

        self._connection = psycopg.connect(
            self.database_url,
            autocommit=True,
            prepare_threshold=0,
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                row = self._connection.execute(
                    "SELECT pg_try_advisory_lock(%s, %s)", self.keys
                ).fetchone()
                if row is not None and row[0]:
                    self.acquired = True
                    return True
                abort = False
                if isinstance(should_abort, Event):
                    abort = should_abort.is_set()
                elif should_abort is not None:
                    abort = should_abort()
                if abort:
                    self.close()
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.close()
                        return False
                    time.sleep(min(poll_interval, remaining))
                else:
                    time.sleep(poll_interval)
        except Exception:
            self.close()
            raise

    def check_connection(self) -> bool:
        """Verify that the lock-owning PostgreSQL session is still alive."""

        connection = self._connection
        if not self.acquired or connection is None or connection.closed:
            return False
        try:
            row = connection.execute("SELECT 1").fetchone()
        except Exception:
            return False
        return row is not None and row[0] == 1

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        try:
            if self.acquired:
                try:
                    connection.execute(
                        "SELECT pg_advisory_unlock(%s, %s)", self.keys
                    )
                except Exception:
                    # Closing the session releases its advisory lock even if
                    # the explicit unlock cannot reach PostgreSQL.
                    pass
        finally:
            self.acquired = False
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self) -> "ThreadAdvisoryLock":
        if not self.acquire(timeout=None):
            raise ThreadLockUnavailable(f"could not acquire Thread lock {self.thread_id}")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ThreadAdvisoryLock", "ThreadLockUnavailable", "advisory_keys"]
