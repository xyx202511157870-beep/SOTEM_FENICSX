from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    coordinates: str
    source_start_down: tuple[float, float, float]
    source_end_down: tuple[float, float, float]
    receiver_down: tuple[float, float, float]
    current_a: float
    rho_air_ohm_m: float
    earth: dict[str, Any]
    polarization: dict[str, Any] | None
    components: list[str]
    observation_times: np.ndarray


def _vec3(values: Any, name: str) -> tuple[float, float, float]:
    message = f"{name} must contain three finite values"
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if len(result) != 3 or not all(np.isfinite(value) for value in result):
        raise ValueError(message)
    return result


def _times(payload: Mapping[str, Any]) -> np.ndarray:
    try:
        start = float(payload["start_s"])
        stop = float(payload["stop_s"])
        count = payload["count"]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid benchmark time definition") from exc

    if (
        payload.get("kind") != "logspace"
        or not np.isfinite(start)
        or not np.isfinite(stop)
        or start <= 0.0
        or stop <= start
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 2
    ):
        raise ValueError("invalid benchmark time definition")
    return np.geomspace(start, stop, count)


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _validate_polarization(polarization: Any) -> None:
    if not isinstance(polarization, Mapping):
        raise ValueError("polarization must be a mapping")
    try:
        top = float(polarization["top_m"])
        bottom = float(polarization["bottom_m"])
        rho0 = float(polarization["rho0_ohm_m"])
        chargeability = float(polarization["m"])
        tau = float(polarization["tau_s"])
        exponent = float(polarization["c"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid polarization definition") from exc

    values = (top, bottom, rho0, chargeability, tau, exponent)
    if (
        not all(np.isfinite(value) for value in values)
        or bottom <= top
        or rho0 <= 0.0
        or not 0.0 <= chargeability < 1.0
        or tau <= 0.0
        or not 0.0 < exponent <= 1.0
    ):
        raise ValueError("invalid polarization definition")


def load_benchmark_case(path: str | Path) -> BenchmarkCase:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, Mapping):
        raise ValueError("benchmark root must be a mapping")

    source = payload["source"]
    if payload["coordinates"] != "z_down":
        raise ValueError("coordinates must be z_down")
    if source["waveform"] != "ideal_step_off":
        raise ValueError("source waveform must be ideal_step_off")
    if payload["components"] != ["Ex", "Ey", "Hz", "dBzdt"]:
        raise ValueError("components must be [Ex, Ey, Hz, dBzdt] in that order")

    polarization = payload.get("polarization")
    if polarization is not None:
        _validate_polarization(polarization)

    return BenchmarkCase(
        case_id=payload["case_id"],
        coordinates=payload["coordinates"],
        source_start_down=_vec3(source["start_m"], "source.start_m"),
        source_end_down=_vec3(source["end_m"], "source.end_m"),
        receiver_down=_vec3(payload["receiver"]["location_m"], "receiver.location_m"),
        current_a=_positive_finite(source["current_a"], "source.current_a"),
        rho_air_ohm_m=_positive_finite(
            payload["air"]["rho_ohm_m"], "air.rho_ohm_m"
        ),
        earth=payload["earth"],
        polarization=polarization,
        components=payload["components"],
        observation_times=_times(payload["times"]),
    )
