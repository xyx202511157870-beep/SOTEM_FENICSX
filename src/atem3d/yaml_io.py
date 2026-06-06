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


def safe_load_yaml(text: str):
    """Load simple YAML text, using PyYAML when available and a small fallback otherwise."""

    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
    return _fallback_load(text)


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


def _fallback_load(text: str):
    lines = []
    for raw in str(text).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("unsupported YAML structure")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int):
    if lines[index][1].startswith("-"):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[tuple[int, str]], index: int, indent: int):
    values = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or text.startswith("-"):
            break
        key, rest = _parse_mapping_pair(text)
        index += 1
        if rest:
            values[key] = _parse_scalar(rest)
        elif index < len(lines) and (
            lines[index][0] > indent
            or (lines[index][0] == indent and lines[index][1].startswith("-"))
        ):
            values[key], index = _parse_block(lines, index, lines[index][0])
        else:
            values[key] = None
    return values, index


def _parse_sequence(lines: list[tuple[int, str]], index: int, indent: int):
    values = []
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not text.startswith("-"):
            break
        rest = text[1:].strip()
        index += 1
        if rest:
            if _is_inline_mapping(rest):
                key, map_rest = _parse_mapping_pair(rest)
                item = {key: _parse_scalar(map_rest) if map_rest else None}
                if index < len(lines) and lines[index][0] > indent:
                    extra, index = _parse_mapping(lines, index, lines[index][0])
                    item.update(extra)
                values.append(item)
            else:
                values.append(_parse_scalar(rest))
        elif index < len(lines) and lines[index][0] > indent:
            item, index = _parse_block(lines, index, lines[index][0])
            values.append(item)
        else:
            values.append(None)
    return values, index


def _parse_mapping_pair(text: str):
    key, separator, rest = text.partition(":")
    if not separator:
        raise ValueError(f"unsupported YAML mapping line: {text!r}")
    return _parse_key(key.strip()), rest.strip()


def _is_inline_mapping(value: str) -> bool:
    key, separator, _ = value.partition(":")
    return bool(separator and key.strip() and re.fullmatch(r"[A-Za-z0-9_.-]+", key.strip()))


def _parse_key(value: str) -> str:
    return str(_parse_scalar(value))


def _parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in _split_inline_items(inner)]
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value in {".nan", ".NaN", ".NAN"}:
        return float("nan")
    if value in {".inf", ".Inf", ".INF"}:
        return float("inf")
    if value in {"-.inf", "-.Inf", "-.INF"}:
        return float("-inf")
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        inner = value[1:-1]
        if value.startswith('"'):
            return inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    try:
        if re.fullmatch(r"[-+]?[0-9]+", value):
            return int(value)
        if re.fullmatch(r"[-+]?([0-9]*\.[0-9]+|[0-9]+\.?)([eE][-+]?[0-9]+)?", value):
            return float(value)
    except ValueError:
        pass
    return value


def _split_inline_items(value: str) -> list[str]:
    items = []
    start = 0
    quote = ""
    escape = False
    depth = 0
    for index, char in enumerate(value):
        if escape:
            escape = False
            continue
        if char == "\\" and quote == '"':
            escape = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 0:
            items.append(value[start:index])
            start = index + 1
    items.append(value[start:])
    return items
