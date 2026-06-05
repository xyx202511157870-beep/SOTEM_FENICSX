"""Receiver interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import mu_0


_EDGE_COMPONENTS = {"Ex": "Ex", "Ey": "Ey", "Ez": "Ez"}
_HJ_ELECTRIC_COMPONENTS = {"Ex": "Fx", "Ey": "Fy", "Ez": "Fz"}
_HJ_MAGNETIC_COMPONENTS = {
    "Bx": "Ex",
    "By": "Ey",
    "Bz": "Ez",
    "Hx": "Ex",
    "Hy": "Ey",
    "Hz": "Ez",
}
_FACE_COMPONENTS = {
    "Bx": "Fx",
    "By": "Fy",
    "Bz": "Fz",
    "Hx": "Fx",
    "Hy": "Fy",
    "Hz": "Fz",
}
_DBDT_COMPONENTS = {
    "dBxdt": "Fx",
    "dBydt": "Fy",
    "dBzdt": "Fz",
}
_VECTOR_COMPONENT_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


@dataclass(frozen=True)
class PointReceiver:
    """Point receiver for electric, magnetic-flux, or magnetic-field components."""

    location: tuple[float, float, float]
    component: str

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=float)
        if location.shape != (3,):
            raise ValueError("location must be a 3D coordinate")
        if self.component not in {*_EDGE_COMPONENTS, *_FACE_COMPONENTS, *_DBDT_COMPONENTS}:
            raise ValueError("unsupported receiver component")
        object.__setattr__(self, "location", tuple(float(v) for v in location))

    def sample(self, mesh, e: np.ndarray, b: np.ndarray, mu: float = mu_0) -> float:
        """Sample a field state."""

        loc = np.asarray(self.location, dtype=float).reshape(1, 3)
        if self.component in _EDGE_COMPONENTS:
            matrix = mesh.get_interpolation_matrix(loc, _EDGE_COMPONENTS[self.component])
            return float((matrix @ e)[0])
        if self.component in _DBDT_COMPONENTS:
            return 0.0

        matrix = mesh.get_interpolation_matrix(loc, _FACE_COMPONENTS[self.component])
        value = float((matrix @ b)[0])
        if self.component.startswith("H"):
            value = value / mu
        return value

    def sample_hj(self, mesh, e: np.ndarray, h: np.ndarray, mu: float = mu_0) -> float:
        """Sample a field state from an H/J formulation."""

        loc = np.asarray(self.location, dtype=float).reshape(1, 3)
        if self.component in _HJ_ELECTRIC_COMPONENTS:
            matrix = mesh.get_interpolation_matrix(loc, _HJ_ELECTRIC_COMPONENTS[self.component])
            return float((matrix @ e)[0])
        if self.component in _DBDT_COMPONENTS:
            return 0.0

        matrix = mesh.get_interpolation_matrix(loc, _HJ_MAGNETIC_COMPONENTS[self.component])
        value = float((matrix @ h)[0])
        if self.component.startswith("B"):
            value *= mu
        return value

    @property
    def uses_magnetic_field_vector(self) -> bool:
        """Whether this receiver can be sampled from a recovered H vector."""

        return self.component in _FACE_COMPONENTS

    @property
    def vector_component_index(self) -> int:
        """Return x/y/z index for vector-valued receiver components."""

        suffix = self.component[-1].lower()
        if suffix not in _VECTOR_COMPONENT_INDEX:
            raise ValueError("receiver component does not map to a vector index")
        return _VECTOR_COMPONENT_INDEX[suffix]

    def sample_magnetic_field_vector(self, h_vector: np.ndarray, mu: float = mu_0) -> float:
        """Sample an H/B receiver from a recovered magnetic-field vector."""

        value = float(np.asarray(h_vector, dtype=float)[self.vector_component_index])
        if self.component.startswith("B"):
            value *= mu
        return value

    def sample_time_derivative(
        self,
        mesh,
        e: np.ndarray,
        b_new: np.ndarray,
        b_old: np.ndarray,
        dt: float,
        mu: float = mu_0,
    ) -> float:
        """Sample a time-derivative receiver using adjacent magnetic states."""

        if self.component not in _DBDT_COMPONENTS:
            return self.sample(mesh, e, b_new, mu)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        loc = np.asarray(self.location, dtype=float).reshape(1, 3)
        matrix = mesh.get_interpolation_matrix(loc, _DBDT_COMPONENTS[self.component])
        return float((matrix @ ((b_new - b_old) / dt))[0])

    def sample_hj_time_derivative(
        self,
        mesh,
        e: np.ndarray,
        h_new: np.ndarray,
        h_old: np.ndarray,
        dt: float,
        mu: float = mu_0,
    ) -> float:
        """Sample a time-derivative receiver from H/J magnetic states."""

        if self.component not in _DBDT_COMPONENTS:
            return self.sample_hj(mesh, e, h_new, mu)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        loc = np.asarray(self.location, dtype=float).reshape(1, 3)
        edge_component = {"dBxdt": "Ex", "dBydt": "Ey", "dBzdt": "Ez"}[self.component]
        matrix = mesh.get_interpolation_matrix(loc, edge_component)
        return float((matrix @ (mu * (h_new - h_old) / dt))[0])


@dataclass(frozen=True)
class AverageReceiver:
    """Average point receiver over deterministic disk or volume samples."""

    location: tuple[float, float, float]
    component: str
    receiver_type: str
    radius: float

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=float)
        if location.shape != (3,):
            raise ValueError("location must be a 3D coordinate")
        receiver_type = str(self.receiver_type).strip().lower()
        if receiver_type not in {"disk_average", "volume_average"}:
            raise ValueError("receiver_type must be 'disk_average' or 'volume_average'")
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")
        object.__setattr__(self, "location", tuple(float(v) for v in location))
        object.__setattr__(self, "receiver_type", receiver_type)
        object.__setattr__(self, "radius", radius)

    @property
    def sample_points(self) -> np.ndarray:
        center = np.asarray(self.location, dtype=float)
        r = float(self.radius)
        if self.receiver_type == "disk_average":
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [r, 0.0, 0.0],
                    [-r, 0.0, 0.0],
                    [0.0, r, 0.0],
                    [0.0, -r, 0.0],
                ],
                dtype=float,
            )
        else:
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [r, 0.0, 0.0],
                    [-r, 0.0, 0.0],
                    [0.0, r, 0.0],
                    [0.0, -r, 0.0],
                    [0.0, 0.0, r],
                    [0.0, 0.0, -r],
                ],
                dtype=float,
            )
        return center.reshape(1, 3) + offsets

    @property
    def sample_count(self) -> int:
        return int(self.sample_points.shape[0])

    def point_receivers(self) -> list[PointReceiver]:
        return [
            PointReceiver(location=tuple(float(value) for value in point), component=self.component)
            for point in self.sample_points
        ]

    def sample(self, mesh, e: np.ndarray, b: np.ndarray, mu: float = mu_0) -> float:
        values = [receiver.sample(mesh, e, b, mu) for receiver in self.point_receivers()]
        return float(np.mean(values))

    def sample_hj(self, mesh, e: np.ndarray, h: np.ndarray, mu: float = mu_0) -> float:
        values = [receiver.sample_hj(mesh, e, h, mu) for receiver in self.point_receivers()]
        return float(np.mean(values))

    def sample_time_derivative(
        self,
        mesh,
        e: np.ndarray,
        b_new: np.ndarray,
        b_old: np.ndarray,
        dt: float,
        mu: float = mu_0,
    ) -> float:
        values = [
            receiver.sample_time_derivative(mesh, e, b_new, b_old, dt, mu)
            for receiver in self.point_receivers()
        ]
        return float(np.mean(values))

    def sample_hj_time_derivative(
        self,
        mesh,
        e: np.ndarray,
        h_new: np.ndarray,
        h_old: np.ndarray,
        dt: float,
        mu: float = mu_0,
    ) -> float:
        values = [
            receiver.sample_hj_time_derivative(mesh, e, h_new, h_old, dt, mu)
            for receiver in self.point_receivers()
        ]
        return float(np.mean(values))
