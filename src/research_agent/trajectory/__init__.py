"""Append-only trajectory recording and deterministic audit helpers."""

from .diff import compute_state_diff
from .recorder import TrajectoryRecorder
from .context import TrajectoryContext

__all__ = ["TrajectoryContext", "TrajectoryRecorder", "compute_state_diff"]
