"""Canonical hashing and CSV/JSON helpers for Debye-MVP artifacts."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REGISTRY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "K",
    "template_family",
    "offsets_log10_tau",
    "span",
    "shift",
    "time_window_anchor",
    "candidate_hash",
)
OFFSETS_SEPARATOR = ";"


def _to_jsonable(value: Any, *, allow_nan: bool) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value), allow_nan=allow_nan)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item, allow_nan=allow_nan) for item in value.tolist()]
    if isinstance(value, np.generic):
        python_value = value.item()
        return _to_jsonable(python_value, allow_nan=allow_nan)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item, allow_nan=allow_nan) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, allow_nan=allow_nan) for item in value]
    if isinstance(value, float):
        if not allow_nan and not np.isfinite(value):
            raise ValueError("non-finite floats are not allowed in canonical JSON")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON type: {type(value)!r}")


def canonical_json(payload: Any) -> str:
    """Return a stable JSON encoding used for hashing."""

    return json.dumps(
        _to_jsonable(payload, allow_nan=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    """Return the hex SHA-256 digest of ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_offsets(offsets) -> str:
    """Format log10-tau offsets for the registry CSV."""

    values = np.asarray(offsets, dtype=float).reshape(-1)
    return OFFSETS_SEPARATOR.join(f"{float(value):.12f}" for value in values)


def parse_offsets(text: str) -> tuple[float, ...]:
    """Parse a ``;``-separated offset list."""

    parts = [part for part in str(text).split(OFFSETS_SEPARATOR) if part != ""]
    return tuple(float(part) for part in parts)


def write_csv(path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> Path:
    """Write ``rows`` with a fixed column order and Unix newlines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [str(column) for column in columns]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    return destination


def read_csv(path) -> list[dict[str, str]]:
    """Read a CSV file as a list of string dictionaries."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_candidate_registry_csv(path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write a candidate registry with the frozen column contract."""

    expected = set(REGISTRY_COLUMNS)
    for index, row in enumerate(rows):
        keys = set(row)
        if keys != expected:
            raise ValueError(
                f"registry row {index} keys {sorted(keys)} do not match {list(REGISTRY_COLUMNS)}"
            )
    return write_csv(path, REGISTRY_COLUMNS, rows)


def read_candidate_registry_csv(path) -> list[dict[str, str]]:
    """Read a candidate registry and reject unexpected headers."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != REGISTRY_COLUMNS:
            raise ValueError(
                f"registry header {header} does not match {REGISTRY_COLUMNS}"
            )
        return [dict(row) for row in reader]


def to_record(obj: Any) -> dict[str, object]:
    """Convert a dataclass or mapping into a JSON-friendly record."""

    converted = _to_jsonable(obj, allow_nan=True)
    if not isinstance(converted, dict):
        raise TypeError("to_record requires a dataclass or mapping")
    return converted


def write_json(path, payload) -> Path:
    """Write a JSON artifact with sorted keys."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = to_record(payload) if dataclasses.is_dataclass(payload) and not isinstance(payload, type) else _to_jsonable(payload, allow_nan=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return destination


def read_json(path) -> object:
    """Read a JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_records_csv(
    path,
    records: Sequence[Mapping[str, object]],
    columns: Sequence[str] | None = None,
) -> Path:
    """Write a flat table of records."""

    if columns is None:
        keys: set[str] = set()
        for record in records:
            keys.update(str(key) for key in record)
        fieldnames = tuple(sorted(keys))
    else:
        fieldnames = tuple(str(column) for column in columns)
    return write_csv(path, fieldnames, records)
