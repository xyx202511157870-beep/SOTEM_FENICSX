from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml


class _BenchmarkSafeLoader(yaml.SafeLoader):
    pass


_BenchmarkSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    coordinates: str
    source_start_down: tuple[float, float, float]
    source_end_down: tuple[float, float, float]
    receiver_down: tuple[float, float, float]
    current_a: float
    rho_air_ohm_m: float
    earth: Mapping[str, Any]
    polarization: Mapping[str, float] | None
    components: tuple[str, ...]
    observation_times: np.ndarray
    validation_role: str
    literature: Mapping[str, str]
    provenance_file: str | None
    surface_offsets_m: tuple[float, ...]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(payload: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in payload:
        raise ValueError(f"{name} is required")
    return payload[key]


def _real_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _vec3(values: Any, name: str) -> tuple[float, float, float]:
    message = f"{name} must contain three finite values"
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != 3
    ):
        raise ValueError(message)
    try:
        result = tuple(_real_scalar(value, name) for value in values)
    except ValueError as exc:
        raise ValueError(message) from exc
    return result


def _immutable_array(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _times(payload: Mapping[str, Any]) -> np.ndarray:
    kind = _required(payload, "kind", "times.kind")
    start = _real_scalar(
        _required(payload, "start_s", "times.start_s"), "times.start_s"
    )
    stop = _real_scalar(
        _required(payload, "stop_s", "times.stop_s"), "times.stop_s"
    )
    count = _required(payload, "count", "times.count")

    if (
        kind != "logspace"
        or start <= 0.0
        or stop <= start
        or isinstance(count, bool)
        or not isinstance(count, Integral)
        or count < 2
    ):
        raise ValueError("invalid benchmark time definition")
    return _immutable_array(np.geomspace(start, stop, int(count)))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _optional_text(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _literature(value: Any) -> Mapping[str, str]:
    if value is None:
        return _freeze({})
    payload = _mapping(value, "literature")
    normalized: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("literature keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"literature.{key} must be a non-empty string")
        normalized[key.strip()] = item.strip()
    return _freeze(normalized)


def _surface_offsets(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("surface_offsets_m must be a sequence")
    offsets = tuple(_real_scalar(item, "surface_offsets_m") for item in value)
    if any(item < 0.0 for item in offsets):
        raise ValueError("surface_offsets_m must be non-negative")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("surface_offsets_m must be strictly increasing")
    return offsets


def _earth(earth: Any) -> Mapping[str, Any]:
    earth = _mapping(earth, "earth")
    if set(earth) == {"rho_ohm_m"}:
        return _freeze(
            {
                "rho_ohm_m": _positive_finite(
                    earth["rho_ohm_m"], "earth.rho_ohm_m"
                )
            }
        )
    if set(earth) != {"layers"}:
        raise ValueError("earth must use exactly one supported schema")

    layers = earth["layers"]
    if (
        isinstance(layers, (str, bytes))
        or not isinstance(layers, Sequence)
        or not layers
    ):
        raise ValueError("earth.layers must be a non-empty sequence")

    normalized_layers = []
    previous_bottom = None
    required_fields = {"top_m", "bottom_m", "rho_ohm_m"}
    for index, raw_layer in enumerate(layers):
        layer_name = f"earth.layers[{index}]"
        layer = _mapping(raw_layer, layer_name)
        if set(layer) != required_fields:
            raise ValueError(f"{layer_name} must contain exactly {sorted(required_fields)}")

        top = _real_scalar(layer["top_m"], f"{layer_name}.top_m")
        if index == 0 and top != 0.0:
            raise ValueError("earth.layers[0].top_m must equal 0.0")

        raw_bottom = layer["bottom_m"]
        if raw_bottom is None:
            if index != len(layers) - 1:
                raise ValueError("only the final earth layer may have no bottom")
            bottom = None
        else:
            if index == len(layers) - 1:
                raise ValueError(f"{layer_name}.bottom_m must be null")
            bottom = _real_scalar(raw_bottom, f"{layer_name}.bottom_m")
            if bottom <= top:
                raise ValueError(f"{layer_name}.bottom_m must be greater than top_m")

        if index and top != previous_bottom:
            raise ValueError("earth layers must be ordered and contiguous")

        normalized_layers.append(
            {
                "top_m": top,
                "bottom_m": bottom,
                "rho_ohm_m": _positive_finite(
                    layer["rho_ohm_m"], f"{layer_name}.rho_ohm_m"
                ),
            }
        )
        previous_bottom = bottom

    return _freeze({"layers": normalized_layers})


def _polarization(polarization: Any) -> Mapping[str, float]:
    polarization = _mapping(polarization, "polarization")
    field_names = ("top_m", "bottom_m", "rho0_ohm_m", "m", "tau_s", "c")
    if set(polarization) != set(field_names):
        raise ValueError("invalid polarization definition")

    normalized = {
        name: _real_scalar(polarization[name], f"polarization.{name}")
        for name in field_names
    }
    if (
        normalized["bottom_m"] <= normalized["top_m"]
        or normalized["rho0_ohm_m"] <= 0.0
        or not 0.0 <= normalized["m"] < 1.0
        or normalized["tau_s"] <= 0.0
        or not 0.0 < normalized["c"] <= 1.0
    ):
        raise ValueError("invalid polarization definition")
    return _freeze(normalized)


def load_benchmark_case(path: str | Path) -> BenchmarkCase:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=_BenchmarkSafeLoader)

    if not isinstance(payload, Mapping):
        raise ValueError("benchmark root must be a mapping")

    case_id = _required(payload, "case_id", "case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")

    source = _mapping(_required(payload, "source", "source"), "source")
    receiver = _mapping(_required(payload, "receiver", "receiver"), "receiver")
    air = _mapping(_required(payload, "air", "air"), "air")
    times = _mapping(_required(payload, "times", "times"), "times")

    coordinates = _required(payload, "coordinates", "coordinates")
    if coordinates != "z_down":
        raise ValueError("coordinates must be z_down")
    if _required(source, "waveform", "source.waveform") != "ideal_step_off":
        raise ValueError("source waveform must be ideal_step_off")
    components = _required(payload, "components", "components")
    if components != ["Ex", "Ey", "Hz", "dBzdt"]:
        raise ValueError("components must be [Ex, Ey, Hz, dBzdt] in that order")

    raw_polarization = payload.get("polarization")
    polarization = (
        None if raw_polarization is None else _polarization(raw_polarization)
    )
    provenance_file = payload.get("provenance_file")
    if provenance_file is not None and (
        not isinstance(provenance_file, str)
        or not provenance_file.strip()
        or Path(provenance_file).name != provenance_file
    ):
        raise ValueError("provenance_file must be a local non-empty filename")

    return BenchmarkCase(
        case_id=case_id,
        coordinates=coordinates,
        source_start_down=_vec3(
            _required(source, "start_m", "source.start_m"), "source.start_m"
        ),
        source_end_down=_vec3(
            _required(source, "end_m", "source.end_m"), "source.end_m"
        ),
        receiver_down=_vec3(
            _required(receiver, "location_m", "receiver.location_m"),
            "receiver.location_m",
        ),
        current_a=_positive_finite(
            _required(source, "current_a", "source.current_a"), "source.current_a"
        ),
        rho_air_ohm_m=_positive_finite(
            _required(air, "rho_ohm_m", "air.rho_ohm_m"), "air.rho_ohm_m"
        ),
        earth=_earth(_required(payload, "earth", "earth")),
        polarization=polarization,
        components=tuple(components),
        observation_times=_times(times),
        validation_role=_optional_text(
            payload, "validation_role", "legacy_unspecified"
        ),
        literature=_literature(payload.get("literature")),
        provenance_file=None if provenance_file is None else provenance_file.strip(),
        surface_offsets_m=_surface_offsets(payload.get("surface_offsets_m")),
    )


def load_benchmark_provenance(
    path: str | Path,
    *,
    case_path: str | Path,
) -> Mapping[str, Any]:
    """Load parameter provenance and verify that it is bound to ``case_path``."""

    provenance_path = Path(path)
    case_file = Path(case_path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark provenance must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark provenance root must be a mapping")
    if payload.get("schema") != "atem3d.benchmark-parameter-provenance/v1":
        raise ValueError("unsupported benchmark provenance schema")

    case = load_benchmark_case(case_file)
    if payload.get("case_id") != case.case_id:
        raise ValueError("benchmark provenance case_id mismatch")
    case_bytes = case_file.read_bytes().replace(b"\r\n", b"\n")
    expected_hash = hashlib.sha256(case_bytes).hexdigest()
    if payload.get("case_file_sha256") != expected_hash:
        raise ValueError("benchmark provenance case-file hash mismatch")
    if payload.get("status") != "parameter_contract_verified":
        raise ValueError("benchmark provenance status is not verified")

    fields = payload.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("benchmark provenance fields must be a non-empty mapping")
    allowed_grades = {"A", "B", "C", "D"}
    for name, record in fields.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("benchmark provenance field names must be non-empty")
        if not isinstance(record, Mapping):
            raise ValueError(f"benchmark provenance field {name!r} must be a mapping")
        if record.get("evidence_grade") not in allowed_grades:
            raise ValueError(
                f"benchmark provenance field {name!r} has unsupported evidence grade"
            )
        source_url = record.get("source_url")
        if not isinstance(source_url, str) or not source_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError(
                f"benchmark provenance field {name!r} requires a source URL"
            )
        note = record.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(
                f"benchmark provenance field {name!r} requires a source note"
            )
    return _freeze(payload)
