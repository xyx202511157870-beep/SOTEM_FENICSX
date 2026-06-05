"""Adapters from primary providers to FEM-space sample tables."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import PrimaryFieldProvider, as_points


Assembler = Callable[[np.ndarray, np.ndarray], Any]


@dataclass(frozen=True)
class PrimaryFEMInterpolator:
    """Sample a primary provider on FEM interpolation points.

    The optional ``assembler`` is the only FEM-specific hook. In DOLFINx wiring
    it can convert point/vector samples into a function vector; in pure tests the
    raw ``(n_points, 3)`` table is returned.
    """

    provider: PrimaryFieldProvider
    points: np.ndarray
    assembler: Assembler | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", as_points(self.points, "points"))

    def sample_Ep(self, t: float) -> np.ndarray:
        values = np.asarray(self.provider.get_Ep_on_V(float(t), self.points), dtype=float)
        return self._require_field_shape(values, "Ep")

    def sample_Ep_dc(self) -> np.ndarray:
        values = np.asarray(self.provider.get_Ep_dc_on_V(self.points), dtype=float)
        return self._require_field_shape(values, "Ep_dc")

    def sample_Ep_times(self, times: Iterable[float]) -> np.ndarray:
        samples = [self.sample_Ep(float(t)) for t in times]
        if not samples:
            return np.empty((0, self.points.shape[0], 3), dtype=float)
        return np.stack(samples, axis=0)

    def interpolate_Ep(self, t: float):
        return self._assemble(self.sample_Ep(t))

    def interpolate_Ep_dc(self):
        return self._assemble(self.sample_Ep_dc())

    def _assemble(self, values: np.ndarray):
        if self.assembler is None:
            return values.copy()
        return self.assembler(self.points.copy(), values.copy())

    def _require_field_shape(self, values: np.ndarray, name: str) -> np.ndarray:
        if values.shape != (self.points.shape[0], 3):
            raise ValueError(f"{name} must have shape (n_points, 3)")
        return values.copy()


@dataclass(frozen=True)
class TabulatedVectorField:
    """Callable vector field backed by exact tabulated point samples."""

    points: np.ndarray
    values: np.ndarray
    atol: float = 1.0e-10

    def __post_init__(self) -> None:
        points = as_points(self.points, "points")
        values = np.asarray(self.values, dtype=float)
        if values.shape != points.shape:
            raise ValueError("values must have shape (n_points, 3)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "atol", float(self.atol))

    def __call__(self, x) -> np.ndarray:
        query = np.asarray(x, dtype=float)
        if query.ndim != 2 or query.shape[0] != 3:
            raise ValueError("x must have shape (3, n_points)")
        output = np.empty((3, query.shape[1]), dtype=float)
        for column, point in enumerate(query.T):
            output[:, column] = self._lookup(point)
        return output

    def _lookup(self, point: np.ndarray) -> np.ndarray:
        matches = np.all(np.isclose(self.points, point[None, :], rtol=0.0, atol=self.atol), axis=1)
        indices = np.flatnonzero(matches)
        if indices.size == 0:
            raise ValueError("query point is not in tabulated primary field points")
        return self.values[int(indices[0])]


def make_tabulated_vector_assembler(*, atol: float = 1.0e-10) -> Assembler:
    """Return an assembler producing a DOLFINx-interpolate-compatible callable."""

    def assemble(points: np.ndarray, values: np.ndarray) -> TabulatedVectorField:
        return TabulatedVectorField(points=points, values=values, atol=atol)

    return assemble
