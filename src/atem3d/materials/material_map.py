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


def mark_leakage_channel(cell_centers, channel_points, radius: float) -> np.ndarray:
    """Return cells within ``radius`` of an irregular 3D channel polyline."""

    centers = _as_points(cell_centers, "cell_centers")
    channel = _as_points(channel_points, "channel_points")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if channel.shape[0] < 2:
        raise ValueError("channel_points must contain at least two points")
    distances = np.full(centers.shape[0], np.inf, dtype=float)
    for start, end in zip(channel[:-1], channel[1:]):
        distances = np.minimum(distances, _distance_to_segment(centers, start, end))
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

    values = np.asarray(markers, dtype=int).copy()
    if values.ndim != 1:
        raise ValueError("markers must be a 1D array")
    centers = _as_points(cell_centers, "cell_centers")
    if centers.shape[0] != values.size:
        raise ValueError("cell_centers length must match markers")
    mask = mark_leakage_channel(centers, channel_points, radius)
    values[mask] = int(leakage_marker)
    return values


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
