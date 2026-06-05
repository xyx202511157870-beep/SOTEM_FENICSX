"""Survey and layer model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class FiniteWireSurvey:
    """Finite grounded wire and one receiver."""

    source_start: Point3D = (-500.0, 200.0, -0.1)
    source_end: Point3D = (500.0, 200.0, -0.1)
    receiver: Point3D = (0.0, -300.0, -0.1)
    current: float = 10.0

    @property
    def source_length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.source_end) - np.asarray(self.source_start)))

    @property
    def parallel_offset(self) -> float:
        start = np.asarray(self.source_start[:2], dtype=float)
        end = np.asarray(self.source_end[:2], dtype=float)
        rec = np.asarray(self.receiver[:2], dtype=float)
        axis = end - start
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("source endpoints must be distinct")
        tangent = axis / norm
        normal = np.array([-tangent[1], tangent[0]])
        return float(abs(np.dot(rec - start, normal)))

    def validate(self, *, expected_length: float | None = None, expected_offset: float | None = None) -> None:
        if self.current == 0.0:
            raise ValueError("current must be non-zero")
        if self.source_length <= 0.0:
            raise ValueError("source endpoints must be distinct")
        if expected_length is not None and not np.isclose(self.source_length, expected_length):
            raise ValueError(f"source length {self.source_length:g} m != expected {expected_length:g} m")
        if expected_offset is not None and not np.isclose(self.parallel_offset, expected_offset):
            raise ValueError(f"parallel offset {self.parallel_offset:g} m != expected {expected_offset:g} m")


@dataclass(frozen=True)
class LayerModel:
    """1D air-earth layered resistivity model for empymod-style references."""

    depths: tuple[float, ...] = (350.0, 650.0)
    resistivities: tuple[float, ...] = (100.0, 100.0, 100.0)
    rho_air: float = 1.0e6

    def empymod_depth_res(self):
        if len(self.resistivities) != len(self.depths) + 1:
            raise ValueError("resistivities must have one more entry than depths")
        return [0.0, *map(float, self.depths)], [float(self.rho_air), *map(float, self.resistivities)]

