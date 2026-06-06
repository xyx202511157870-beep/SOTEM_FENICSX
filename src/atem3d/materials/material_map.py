"""Cell-marker material maps and leakage-channel utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atem3d.materials.prony import PronyConductivity


@dataclass(frozen=True)
class CellMaterialMap:
    """Map integer cell markers to Prony conductivity materials."""

    markers: np.ndarray
    materials: dict[int, PronyConductivity]

    def __post_init__(self) -> None:
        markers = np.asarray(self.markers, dtype=int)
        if markers.ndim != 1 or markers.size == 0:
            raise ValueError("markers must be a non-empty 1D array")
        missing = sorted(set(int(marker) for marker in markers) - set(self.materials))
        if missing:
            raise ValueError(f"materials missing markers: {missing}")
        object.__setattr__(self, "markers", markers)
        object.__setattr__(self, "materials", dict(self.materials))

    @property
    def markers_present(self) -> tuple[int, ...]:
        return tuple(sorted(int(marker) for marker in np.unique(self.markers)))

    def sigma0(self) -> np.ndarray:
        return self._material_array("sigma0")

    def sigma_inf(self) -> np.ndarray:
        return self._material_array("sigma_inf")

    def material_for_cell(self, cell_index: int) -> PronyConductivity:
        return self.materials[int(self.markers[int(cell_index)])]

    def _material_array(self, attribute: str) -> np.ndarray:
        values = np.zeros(self.markers.size, dtype=float)
        for marker, material in self.materials.items():
            values[self.markers == int(marker)] = float(getattr(material, attribute))
        return values


@dataclass(frozen=True)
class LeakageMarkerResult:
    """Markers plus diagnostics for leakage-channel marking."""

    markers: np.ndarray
    diagnostics: dict[str, object]


def mark_leakage_channel(cell_centers, channel_points, radius: float) -> np.ndarray:
    """Return cells within ``radius`` of an irregular 3D channel polyline."""

    centers = _as_points(cell_centers, "cell_centers")
    channel = _as_points(channel_points, "channel_points")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if channel.shape[0] < 2:
        raise ValueError("channel_points must contain at least two points")
    distances = _distance_to_channel(centers, channel)
    return distances <= radius


def apply_leakage_channel_marker(
    markers,
    cell_centers,
    *,
    channel_points,
    radius: float,
    leakage_marker: int,
) -> np.ndarray:
    """Return markers with leakage-channel cells overwritten."""

    return apply_leakage_channel_marker_with_diagnostics(
        markers,
        cell_centers,
        channel_points=channel_points,
        radius=radius,
        leakage_marker=leakage_marker,
    ).markers


def apply_leakage_channel_marker_with_diagnostics(
    markers,
    cell_centers,
    *,
    channel_points,
    radius: float,
    leakage_marker: int,
    min_marked_cells: int = 0,
) -> LeakageMarkerResult:
    """Return markers and diagnostics, optionally marking nearest fallback cells."""

    values = np.asarray(markers, dtype=int).copy()
    if values.ndim != 1:
        raise ValueError("markers must be a 1D array")
    centers = _as_points(cell_centers, "cell_centers")
    if centers.shape[0] != values.size:
        raise ValueError("cell_centers length must match markers")
    channel = _as_points(channel_points, "channel_points")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    min_marked_cells = int(min_marked_cells)
    if min_marked_cells < 0:
        raise ValueError("min_marked_cells must be nonnegative")
    distances = _distance_to_channel(centers, channel)
    mask = distances <= radius
    initial_count = int(np.count_nonzero(mask))
    fallback_added = 0
    if initial_count < min_marked_cells:
        needed = min(min_marked_cells, values.size) - initial_count
        if needed > 0:
            candidates = np.argsort(distances)
            for index in candidates:
                if not mask[int(index)]:
                    mask[int(index)] = True
                    fallback_added += 1
                    if fallback_added >= needed:
                        break
    values[mask] = int(leakage_marker)
    leakage_count = int(np.count_nonzero(mask))
    diagnostics = {
        "cell_count": int(values.size),
        "radius_m": radius,
        "initial_leakage_cell_count": initial_count,
        "leakage_cell_count": leakage_count,
        "leakage_cell_fraction": float(leakage_count / values.size),
        "nearest_channel_distance_m": float(np.min(distances)),
        "farthest_channel_distance_m": float(np.max(distances)),
        "fallback_used": bool(fallback_added > 0),
        "fallback_added_cell_count": int(fallback_added),
        "min_marked_cells": int(min_marked_cells),
        "marked": bool(leakage_count > 0),
    }
    return LeakageMarkerResult(markers=values, diagnostics=diagnostics)


def structured_box_cell_centers(domain_min, domain_max, cells) -> np.ndarray:
    """Return cell centers for an axis-aligned structured box."""

    lower = np.asarray(domain_min, dtype=float)
    upper = np.asarray(domain_max, dtype=float)
    counts = np.asarray(cells, dtype=int)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("domain_min and domain_max must have length 3")
    if counts.shape != (3,) or np.any(counts <= 0):
        raise ValueError("cells must contain three positive integers")
    if np.any(upper <= lower):
        raise ValueError("domain_max must be greater than domain_min")
    axes = []
    for lo, hi, count in zip(lower, upper, counts):
        edges = np.linspace(float(lo), float(hi), int(count) + 1)
        axes.append(0.5 * (edges[:-1] + edges[1:]))
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([item.ravel() for item in grid])


def leakage_channel_marker_diagnostics(
    *,
    domain_min,
    domain_max,
    cells,
    channel_points,
    radius: float,
    min_marked_cells: int = 0,
) -> dict[str, object]:
    """Return pure preflight diagnostics for structured leakage markers."""

    centers = structured_box_cell_centers(domain_min, domain_max, cells)
    channel = _as_points(channel_points, "channel_points")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    distances = _distance_to_channel(centers, channel)
    mask = distances <= radius
    initial_count = int(np.count_nonzero(mask))
    min_marked_cells = int(min_marked_cells)
    if min_marked_cells < 0:
        raise ValueError("min_marked_cells must be nonnegative")
    fallback_added = 0
    if initial_count < min_marked_cells:
        needed = min(min_marked_cells, centers.shape[0]) - initial_count
        for index in np.argsort(distances):
            if not mask[int(index)]:
                mask[int(index)] = True
                fallback_added += 1
                if fallback_added >= needed:
                    break
    leakage_count = int(np.count_nonzero(mask))
    return {
        "cell_count": int(centers.shape[0]),
        "cells": [int(value) for value in np.asarray(cells, dtype=int)],
        "domain_min": [float(value) for value in np.asarray(domain_min, dtype=float)],
        "domain_max": [float(value) for value in np.asarray(domain_max, dtype=float)],
        "radius_m": radius,
        "initial_leakage_cell_count": initial_count,
        "leakage_cell_count": leakage_count,
        "leakage_cell_fraction": float(leakage_count / centers.shape[0]),
        "nearest_channel_distance_m": float(np.min(distances)),
        "farthest_channel_distance_m": float(np.max(distances)),
        "fallback_used": bool(fallback_added > 0),
        "fallback_added_cell_count": int(fallback_added),
        "min_marked_cells": int(min_marked_cells),
        "marked": bool(leakage_count > 0),
    }


def _distance_to_channel(centers: np.ndarray, channel: np.ndarray) -> np.ndarray:
    if channel.shape[0] < 2:
        raise ValueError("channel_points must contain at least two points")
    distances = np.full(centers.shape[0], np.inf, dtype=float)
    for start, end in zip(channel[:-1], channel[1:]):
        distances = np.minimum(distances, _distance_to_segment(centers, start, end))
    return distances


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom == 0.0:
        return np.linalg.norm(points - start[None, :], axis=1)
    weights = ((points - start[None, :]) @ segment) / denom
    weights = np.clip(weights, 0.0, 1.0)
    closest = start[None, :] + weights[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _as_points(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return array
