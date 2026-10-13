"""Small, strict, versioned JSON/JSONL envelopes."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from ..errors import SerializationError

CURRENT_ENVELOPE_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def make_envelope(schema: str, payload: Any, version: int = CURRENT_ENVELOPE_VERSION) -> dict[str, Any]:
    if not isinstance(schema, str) or not schema.strip():
        raise SerializationError("schema must be a non-empty string")
    if not isinstance(version, int) or version <= 0:
        raise SerializationError("version must be a positive integer")
    return {
        "schema": schema,
        "version": version,
        "payload": _jsonable(payload),
    }


def dumps_json(schema: str, payload: Any, version: int = CURRENT_ENVELOPE_VERSION) -> str:
    envelope = make_envelope(schema, payload, version)
    try:
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SerializationError("payload is not JSON serializable") from exc


def _validate_envelope(
    decoded: Any,
    expected_schema: str | None,
    expected_version: int | None,
) -> dict[str, Any]:
    if not isinstance(decoded, dict):
        raise SerializationError("serialized value must be an object envelope")
    for field in ("schema", "version", "payload"):
        if field not in decoded:
            raise SerializationError(f"envelope missing {field!r}")
    if not isinstance(decoded["schema"], str) or not decoded["schema"]:
        raise SerializationError("envelope schema must be a non-empty string")
    if not isinstance(decoded["version"], int) or isinstance(decoded["version"], bool):
        raise SerializationError("envelope version must be an integer")
    if expected_schema is not None and decoded["schema"] != expected_schema:
        raise SerializationError(
            f"unexpected schema {decoded['schema']!r}; expected {expected_schema!r}"
        )
    if expected_version is not None and decoded["version"] != expected_version:
        raise SerializationError(
            f"unexpected envelope version {decoded['version']!r}; expected {expected_version!r}"
        )
    return decoded


def loads_json(
    text: str,
    expected_schema: str | None = None,
    expected_version: int | None = CURRENT_ENVELOPE_VERSION,
) -> dict[str, Any]:
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SerializationError("input is not valid JSON") from exc
    return _validate_envelope(decoded, expected_schema, expected_version)


def dumps_jsonl(
    schema: str,
    records: Iterable[Any],
    version: int = CURRENT_ENVELOPE_VERSION,
) -> str:
    lines = [dumps_json(schema, record, version) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def loads_jsonl(
    text: str,
    expected_schema: str | None = None,
    expected_version: int | None = CURRENT_ENVELOPE_VERSION,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(loads_json(line, expected_schema, expected_version))
        except SerializationError as exc:
            raise SerializationError(f"invalid JSONL record at line {line_number}: {exc}") from exc
    return records


def write_json(path: str | Path, schema: str, payload: Any, version: int = CURRENT_ENVELOPE_VERSION) -> None:
    Path(path).write_text(dumps_json(schema, payload, version) + "\n", encoding="utf-8")


def read_json(
    path: str | Path,
    expected_schema: str | None = None,
    expected_version: int | None = CURRENT_ENVELOPE_VERSION,
) -> dict[str, Any]:
    return loads_json(Path(path).read_text(encoding="utf-8"), expected_schema, expected_version)


def write_jsonl(
    path: str | Path,
    schema: str,
    records: Iterable[Any],
    version: int = CURRENT_ENVELOPE_VERSION,
) -> None:
    Path(path).write_text(dumps_jsonl(schema, records, version), encoding="utf-8")


def read_jsonl(
    path: str | Path,
    expected_schema: str | None = None,
    expected_version: int | None = CURRENT_ENVELOPE_VERSION,
) -> list[dict[str, Any]]:
    return loads_jsonl(Path(path).read_text(encoding="utf-8"), expected_schema, expected_version)
