"""Versioned JSON and JSONL serialization helpers."""

from .codec import (
    CURRENT_ENVELOPE_VERSION,
    dumps_json,
    dumps_jsonl,
    loads_json,
    loads_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

TRAJECTORY_REPLAY_SCHEMA = "trajectory_replay_record"


def save_replay_journal(*args, **kwargs):
    from .trajectory import save_replay_journal as implementation

    return implementation(*args, **kwargs)


def load_replay_journal(*args, **kwargs):
    from .trajectory import load_replay_journal as implementation

    return implementation(*args, **kwargs)


def reconstruct_context(*args, **kwargs):
    from .trajectory import reconstruct_context as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "CURRENT_ENVELOPE_VERSION",
    "dumps_json",
    "dumps_jsonl",
    "loads_json",
    "loads_jsonl",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "TRAJECTORY_REPLAY_SCHEMA",
    "load_replay_journal",
    "reconstruct_context",
    "save_replay_journal",
]
