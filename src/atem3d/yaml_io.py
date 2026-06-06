"""Small YAML output helper with a PyYAML-free fallback."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


def safe_dump_yaml(data, *, sort_keys: bool = True) -> str:
    """Return YAML text, using PyYAML when available and a small fallback otherwise."""

    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=sort_keys)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
    return _fallback_dump(data, sort_keys=sort_keys)


def _fallback_dump(value, *, sort_keys: bool) -> str:
    lines = _dump_value(value, indent=0, sort_keys=sort_keys)
    return "\n".join(lines) + "\n"


def _dump_value(value, *, indent: int, sort_keys: bool) -> list[str]:
    if isinstance(value, Mapping):
        lines: list[str] = []
        items = list(value.items())
        if sort_keys:
            items = sorted(items, key=lambda item: str(item[0]))
        for key, item in items:
            key_text = _plain_key(str(key))
            if _is_scalar(item):
                lines.append(f"{' ' * indent}{key_text}: {_scalar(item)}")
            else:
                lines.append(f"{' ' * indent}{key_text}:")
                lines.extend(_dump_value(item, indent=indent + 2, sort_keys=sort_keys))
        return lines
    if _is_sequence(value):
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{' ' * indent}- {_scalar(item)}")
            else:
                lines.append(f"{' ' * indent}-")
                lines.extend(_dump_value(item, indent=indent + 2, sort_keys=sort_keys))
        return lines
    return [f"{' ' * indent}{_scalar(value)}"]


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        return repr(float(value))
    return _plain_string(str(value))


def _plain_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return value
    return _quoted(value)


def _plain_string(value: str) -> str:
    if value == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./\\:-]+", value) and not value.lower() in {
        "true",
        "false",
        "null",
        "nan",
        ".nan",
        ".inf",
        "-.inf",
    }:
        return value
    return _quoted(value)


def _quoted(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
