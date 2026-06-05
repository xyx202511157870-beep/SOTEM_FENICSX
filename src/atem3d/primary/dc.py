"""Analytic DC primary fields for simple grounded-source backgrounds."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import as_points


def analytic_halfspace_grounded_wire_dc_electric_field(
    points,
    *,
    source_start,
    source_end,
    current: float,
    resistivity: float,
) -> np.ndarray:
    """Return the static grounded-wire E field in a uniform conducting halfspace."""

    pts = as_points(points, "points")
    start = _point3(source_start, "source_start")
    end = _point3(source_end, "source_end")
    current = float(current)
    resistivity = float(resistivity)
    if resistivity <= 0.0:
        raise ValueError("resistivity must be positive")
    r_start = pts - start[None, :]
    r_end = pts - end[None, :]
    n_start = np.maximum(np.linalg.norm(r_start, axis=1), np.finfo(float).eps)
    n_end = np.maximum(np.linalg.norm(r_end, axis=1), np.finfo(float).eps)
    scale = resistivity * current / (2.0 * np.pi)
    return scale * (r_end / n_end[:, None] ** 3 - r_start / n_start[:, None] ** 3)


def analytic_halfspace_dc_runner(points, *, config: dict[str, Any], **kwargs) -> np.ndarray:
    """DC runner compatible with EmpymodPrimaryProvider.get_Ep_dc_on_V."""

    resistivity = kwargs.get("resistivity")
    if resistivity is None:
        resistivity = _background_resistivity(config)
    current = config.get("strength", config.get("current", 1.0))
    return analytic_halfspace_grounded_wire_dc_electric_field(
        points,
        source_start=_source_start(config),
        source_end=_source_end(config),
        current=float(current),
        resistivity=float(resistivity),
    )


def _source_start(config: dict[str, Any]):
    if "source_start" in config:
        return config["source_start"]
    return config["source"]["start"]


def _source_end(config: dict[str, Any]):
    if "source_end" in config:
        return config["source_end"]
    return config["source"]["end"]


def _background_resistivity(config: dict[str, Any]) -> float:
    if "rho_earth" in config:
        return float(config["rho_earth"])
    if "resistivity" in config:
        return float(config["resistivity"])
    if "resistivities" in config:
        values = [float(value) for value in config["resistivities"]]
        if not values:
            raise ValueError("resistivities must not be empty")
        return values[-1]
    raise ValueError("config must provide rho_earth, resistivity, or resistivities")


def _point3(values, name: str) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have length 3")
    return point
