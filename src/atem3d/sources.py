"""Grounded-wire source definitions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
from simpeg.electromagnetics.time_domain.sources import (
    line_through_faces,
    segmented_line_current_source_term,
)


@dataclass(frozen=True)
class StepOffWaveform:
    """Ideal current that is on before ``off_time`` and zero afterwards."""

    off_time: float = 0.0
    on_value: float = 1.0

    @property
    def has_initial_fields(self) -> bool:
        return True

    def value(self, time: float) -> float:
        return float(self.on_value if time <= self.off_time else 0.0)

    def previous_value(self, time: float) -> float:
        if time == self.off_time:
            return float(self.on_value)
        return self.value(time)

    def initial_value(self) -> float:
        return float(self.on_value)


@dataclass(frozen=True)
class TabulatedWaveform:
    """Piecewise-linear current waveform."""

    times: np.ndarray
    values: np.ndarray
    initial_field_value: float = 0.0

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
            raise ValueError("times and values must be 1D arrays with the same length")
        if times.size < 2:
            raise ValueError("at least two waveform samples are required")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    @property
    def has_initial_fields(self) -> bool:
        return abs(self.initial_field_value) > 0.0

    def value(self, time: float) -> float:
        return float(np.interp(time, self.times, self.values, left=self.values[0], right=self.values[-1]))

    def previous_value(self, time: float) -> float:
        return self.value(time)

    def initial_value(self) -> float:
        return float(self.initial_field_value)


@dataclass(frozen=True)
class GroundedWireSource:
    """Finite grounded electric wire source projected onto mesh edges."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    current: float
    waveform: StepOffWaveform | TabulatedWaveform
    face_projection: str = "auto"

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("start and end must be 3D coordinates")
        if np.allclose(start, end):
            raise ValueError("a grounded wire must have distinct endpoints")
        if self.current == 0.0:
            raise ValueError("current must be nonzero")
        face_projection = str(self.face_projection).strip().lower()
        if face_projection not in {"auto", "axis_aligned"}:
            raise ValueError("face_projection must be 'auto' or 'axis_aligned'")
        object.__setattr__(self, "start", tuple(float(v) for v in start))
        object.__setattr__(self, "end", tuple(float(v) for v in end))
        object.__setattr__(self, "face_projection", face_projection)
        object.__setattr__(self, "_unit_cache", {})

    @property
    def locations(self) -> np.ndarray:
        return np.vstack([np.asarray(self.start, dtype=float), np.asarray(self.end, dtype=float)])

    def unit_edge_vector(self, mesh) -> np.ndarray:
        """Return the unit-current integrated edge source vector."""

        key = ("edge", id(mesh))
        cache = self._unit_cache
        if key not in cache:
            try:
                cache[key] = segmented_line_current_source_term(mesh, self.locations)
            except ValueError:
                cache[key] = _nearest_edge_line_source(mesh, self.locations)
        return cache[key]

    def unit_face_vector(self, mesh) -> np.ndarray:
        """Return the unit-current H/J face source vector."""

        key = ("face", id(mesh))
        cache = self._unit_cache
        if key not in cache:
            if self.face_projection == "axis_aligned":
                cache[key] = _nearest_face_line_source(mesh, self.locations)
            else:
                try:
                    cache[key] = line_through_faces(
                        mesh,
                        self.locations,
                        normalize_by_area=True,
                    )
                except (NotImplementedError, ValueError):
                    cache[key] = _nearest_face_line_source(mesh, self.locations)
        return cache[key]

    def edge_vector(self, mesh, current_scale: float | None = None) -> np.ndarray:
        """Return the source vector for ``current * current_scale``."""

        if current_scale is None:
            current_scale = 1.0
        return float(self.current * current_scale) * self.unit_edge_vector(mesh)

    def face_vector(self, mesh, current_scale: float | None = None) -> np.ndarray:
        """Return the H/J face source vector for ``current * current_scale``."""

        if current_scale is None:
            current_scale = 1.0
        return float(self.current * current_scale) * self.unit_face_vector(mesh)

    def edge_vector_at(self, mesh, time: float) -> np.ndarray:
        """Return the impressed electric source term at a time node."""

        return self.edge_vector(mesh, self.waveform.value(time))

    def face_vector_at(self, mesh, time: float) -> np.ndarray:
        """Return the H/J impressed electric source at a time node."""

        return self.face_vector(mesh, self.waveform.value(time))

    def previous_edge_vector_at(self, mesh, time: float) -> np.ndarray:
        """Return the left-limit source term at a time node."""

        return self.edge_vector(mesh, self.waveform.previous_value(time))

    def previous_face_vector_at(self, mesh, time: float) -> np.ndarray:
        """Return the H/J left-limit source term at a time node."""

        return self.face_vector(mesh, self.waveform.previous_value(time))

    def initial_edge_vector(self, mesh) -> np.ndarray:
        """Return the on-time source used for static initial fields."""

        if not self.waveform.has_initial_fields:
            return np.zeros(mesh.n_edges, dtype=float)
        return self.edge_vector(mesh, self.waveform.initial_value())

    def initial_face_vector(self, mesh) -> np.ndarray:
        """Return the on-time H/J face source used for static initial fields."""

        if not self.waveform.has_initial_fields:
            return np.zeros(mesh.n_faces, dtype=float)
        return self.face_vector(mesh, self.waveform.initial_value())


def _nearest_edge_line_source(mesh, locations: np.ndarray) -> np.ndarray:
    """Fallback line-source projection for straight grounded wires.

    SimPEG's exact tensor-mesh projector is preferred. This nearest-edge
    quadrature keeps examples and boundary-adjacent tests usable when that
    projector rejects an otherwise interior endpoint. It is intentionally
    conservative and should be replaced by mesh refinement for production
    near-source accuracy studies.
    """

    if not all(hasattr(mesh, name) for name in ("gridEx", "gridEy", "gridEz")):
        raise ValueError("fallback source projection requires a 3D TensorMesh-like mesh")

    trees = [
        cKDTree(mesh.gridEx),
        cKDTree(mesh.gridEy),
        cKDTree(mesh.gridEz),
    ]
    offsets = np.r_[0, mesh.n_edges_x, mesh.n_edges_x + mesh.n_edges_y]
    source = np.zeros(mesh.n_edges, dtype=float)
    min_h = min(float(np.min(widths)) for widths in mesh.h)

    for start, end in zip(locations[:-1], locations[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length == 0.0:
            continue
        n_quad = max(1, int(np.ceil(4.0 * length / min_h)))
        dl = delta / n_quad
        for i in range(n_quad):
            midpoint = start + (i + 0.5) / n_quad * delta
            for axis, value in enumerate(dl):
                if value == 0.0:
                    continue
                _, local_index = trees[axis].query(midpoint)
                source[offsets[axis] + local_index] += value

    return source


def _nearest_face_line_source(mesh, locations: np.ndarray) -> np.ndarray:
    """Fallback H/J face source for axis-aligned grounded wires.

    SimPEG's face projector snaps endpoints to cell centers before tracing the
    source path. If an endpoint lies on an ambiguous cell-center bisector, the
    snapped path can become slightly diagonal even when the physical source is
    axis-aligned. This fallback keeps the original source axis and distributes
    current between adjacent transverse face channels when the physical wire
    lies between them.
    """

    required = ("faces_x", "faces_y", "faces_z", "face_areas")
    if not all(hasattr(mesh, name) for name in required):
        raise ValueError("fallback face source projection requires a 3D TensorMesh-like mesh")

    face_grids = [mesh.faces_x, mesh.faces_y, mesh.faces_z]
    offsets = np.r_[0, mesh.n_faces_x, mesh.n_faces_x + mesh.n_faces_y]
    source = np.zeros(mesh.n_faces, dtype=float)
    mesh_scale = max(float(np.max(np.abs(mesh.nodes_x))), 1.0)
    tolerance = 1.0e-12 * mesh_scale

    for start, end in zip(locations[:-1], locations[1:]):
        delta = end - start
        active_axes = np.flatnonzero(np.abs(delta) > tolerance)
        if active_axes.size != 1:
            raise ValueError("fallback face source projection requires axis-aligned segments")

        axis = int(active_axes[0])
        direction = float(np.sign(start[axis] - end[axis]))
        grid = face_grids[axis]
        transverse_axes = [index for index in range(3) if index != axis]
        low = min(float(start[axis]), float(end[axis]))
        high = max(float(start[axis]), float(end[axis]))
        midpoint = 0.5 * (start + end)
        channel_weights = [
            _face_channel_weights(
                grid[:, transverse_axis],
                float(midpoint[transverse_axis]),
                tolerance,
            )
            for transverse_axis in transverse_axes
        ]

        found = False
        for channel_pair in product(*channel_weights):
            weight = float(np.prod([item[1] for item in channel_pair]))
            if weight == 0.0:
                continue
            mask = (grid[:, axis] >= low - tolerance) & (grid[:, axis] <= high + tolerance)
            for local_index, transverse_axis in enumerate(transverse_axes):
                mask &= np.isclose(
                    grid[:, transverse_axis],
                    channel_pair[local_index][0],
                    rtol=0.0,
                    atol=tolerance,
                )

            local_indices = np.flatnonzero(mask)
            if local_indices.size == 0:
                continue
            found = True
            global_indices = offsets[axis] + local_indices
            source[global_indices] += weight * direction / mesh.face_areas[global_indices]

        if not found:
            raise ValueError("fallback face source projection found no faces on the wire path")

    return source


def _face_channel_weights(
    coordinates: np.ndarray,
    value: float,
    tolerance: float,
) -> list[tuple[float, float]]:
    """Return linear interpolation weights on sorted unique face-channel coordinates."""

    unique = np.unique(np.asarray(coordinates, dtype=float))
    close = np.flatnonzero(np.isclose(unique, value, rtol=0.0, atol=tolerance))
    if close.size:
        return [(float(unique[int(close[0])]), 1.0)]

    upper_index = int(np.searchsorted(unique, value))
    if upper_index <= 0 or upper_index >= unique.size:
        tree = cKDTree(unique.reshape(-1, 1))
        distance, nearest = tree.query([[value]])
        if float(distance[0]) <= tolerance:
            return [(float(unique[int(nearest[0])]), 1.0)]
        raise ValueError("fallback face source projection is outside the mesh face channels")

    lower = float(unique[upper_index - 1])
    upper = float(unique[upper_index])
    width = upper - lower
    if width <= 0.0:
        raise ValueError("face channel coordinates must be strictly increasing")
    lower_weight = (upper - value) / width
    upper_weight = (value - lower) / width
    weights = [(lower, lower_weight), (upper, upper_weight)]
    return [(coordinate, weight) for coordinate, weight in weights if weight > tolerance]
