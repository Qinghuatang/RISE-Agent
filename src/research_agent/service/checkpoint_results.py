"""Read-only access to a persisted RISE graph result."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..runtime.graph_state import GraphStateValidationError, RISEGraphState


class CheckpointResultError(RuntimeError):
    """A saved Graph checkpoint cannot be read as a valid RISE result."""


def read_checkpoint_state(checkpointer: Any, thread_id: UUID) -> RISEGraphState | None:
    """Read the latest state channels without constructing or invoking a graph."""

    config = {"configurable": {"thread_id": str(thread_id)}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    if checkpoint_tuple is None:
        return None
    channels = checkpoint_tuple.checkpoint.get("channel_values")
    if not isinstance(channels, dict) or not channels:
        raise CheckpointResultError("checkpoint does not contain Graph State channels")
    try:
        state = RISEGraphState.from_payload(channels)
    except GraphStateValidationError as exc:
        raise CheckpointResultError("checkpoint Graph State is invalid") from exc
    if state.run_id != str(thread_id):
        raise CheckpointResultError("checkpoint identity does not match its task")
    return state


__all__ = ["CheckpointResultError", "read_checkpoint_state"]
