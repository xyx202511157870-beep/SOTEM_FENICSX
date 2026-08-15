"""Receiver interpolation helpers.

中文说明：接收算子负责把离散场投影为仪器量。点电场、有限电偶极、H、
B 和 dB/dt 不是同一物理量；输出前必须同时核对接收方向、面积/长度、
mu0 换算、单位以及时间采样位置。
"""

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
_DISK_RADIAL_ORDER = 3
_DISK_AZIMUTH_COUNT = 12


def build_receiver(
    *,
    location: tuple[float, float, float],
    component: str,
    receiver_type: str = "point",
    radius: float | None = None,
    normal: tuple[float, float, float] | None = None,
) -> "PointReceiver | AverageReceiver":
    """Build a point or averaged receiver from configuration-style values."""

    receiver_type = str(receiver_type).strip().lower()
    if receiver_type == "point":
        return PointReceiver(location=location, component=component)
    if receiver_type in {"disk_average", "volume_average"}:
        if radius is None:
            raise ValueError("radius is required for average receiver types")
        return AverageReceiver(
            location=location,
            component=component,
            receiver_type=receiver_type,
            radius=float(radius),
            normal=normal,
        )
    raise ValueError("receiver_type must be 'point', 'disk_average', or 'volume_average'")


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

    @property
    def requires_previous_magnetic_state(self) -> bool:
        """Whether the receiver needs two magnetic states to produce data."""

        return self.component in _DBDT_COMPONENTS

    def sample(self, mesh, e: np.ndarray, b: np.ndarray, mu: float = mu_0) -> float:
        """Sample a field state.

        A dB/dt receiver returns the explicit initial-node convention ``0.0``;
        subsequent nodes must use :meth:`sample_time_derivative`.
        """

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
        """Sample a field state from an H/J formulation.

        A dB/dt receiver returns the explicit initial-node convention ``0.0``;
        subsequent nodes must use :meth:`sample_hj_time_derivative`.
        """

        loc = np.asarray(self.location, dtype=float).reshape(1, 3)
        if self.component in _HJ_ELECTRIC_COMPONENTS:
            matrix = mesh.get_interpolation_matrix(
                loc, _HJ_ELECTRIC_COMPONENTS[self.component]
            )
            return float((matrix @ e)[0])
        if self.component in _DBDT_COMPONENTS:
            return 0.0

        matrix = mesh.get_interpolation_matrix(
            loc, _HJ_MAGNETIC_COMPONENTS[self.component]
        )
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

    def sample_magnetic_field_vector(
        self, h_vector: np.ndarray, mu: float = mu_0
    ) -> float:
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
        edge_component = {
            "dBxdt": "Ex",
            "dBydt": "Ey",
            "dBzdt": "Ez",
        }[self.component]
        matrix = mesh.get_interpolation_matrix(loc, edge_component)
        return float((matrix @ (mu * (h_new - h_old) / dt))[0])


@dataclass(frozen=True)
class AverageReceiver:
    """Average a vector field over a finite disk or compact volume stencil.

    For ``disk_average`` the disk normal and measured vector direction default
    to the component axis: x-components use a y-z disk, y-components use an
    x-z disk, and z-components use an x-y disk. Supplying ``normal`` models a
    rotated coil and projects the vector field onto that physical normal.
    """

    location: tuple[float, float, float]
    component: str
    receiver_type: str
    radius: float
    normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=float)
        if location.shape != (3,):
            raise ValueError("location must be a 3D coordinate")
        if self.component not in {*_EDGE_COMPONENTS, *_FACE_COMPONENTS, *_DBDT_COMPONENTS}:
            raise ValueError("unsupported receiver component")
        receiver_type = str(self.receiver_type).strip().lower()
        if receiver_type not in {"disk_average", "volume_average"}:
            raise ValueError("receiver_type must be 'disk_average' or 'volume_average'")
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")

        normal = self.normal
        if normal is None and receiver_type == "disk_average":
            normal_array = _component_axis(self.component)
        elif normal is None:
            normal_array = None
        else:
            normal_array = _normalised_vector(normal, "normal")

        object.__setattr__(self, "location", tuple(float(v) for v in location))
        object.__setattr__(self, "receiver_type", receiver_type)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(
            self,
            "normal",
            None
            if normal_array is None
            else tuple(float(value) for value in normal_array),
        )

    @property
    def requires_previous_magnetic_state(self) -> bool:
        """Whether the receiver needs two magnetic states to produce data."""

        return self.component in _DBDT_COMPONENTS

    @property
    def measurement_axis(self) -> np.ndarray:
        if self.normal is not None:
            return _normalised_vector(self.normal, "normal")
        return _component_axis(self.component)

    @property
    def sample_points(self) -> np.ndarray:
        points, _weights = self._quadrature()
        return points

    @property
    def sample_weights(self) -> np.ndarray:
        _points, weights = self._quadrature()
        return weights

    @property
    def sample_count(self) -> int:
        return int(self.sample_points.shape[0])

    def point_receivers(self) -> list[PointReceiver]:
        """Return legacy scalar point receivers for axis-aligned diagnostics.

        Rotated receivers require vector interpolation and therefore cannot be
        represented as independent scalar point receivers without losing the
        projection onto the physical coil normal.
        """

        if not np.allclose(
            self.measurement_axis,
            _component_axis(self.component),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise RuntimeError(
                "rotated average receivers require vector sampling; "
                "legacy scalar point_receivers() is not valid"
            )
        return [
            PointReceiver(
                location=tuple(float(value) for value in point),
                component=self.component,
            )
            for point in self.sample_points
        ]

    @property
    def uses_magnetic_field_vector(self) -> bool:
        return self.component in _FACE_COMPONENTS

    @property
    def vector_component_index(self) -> int:
        suffix = self.component[-1].lower()
        if suffix not in _VECTOR_COMPONENT_INDEX:
            raise ValueError("receiver component does not map to a vector index")
        return _VECTOR_COMPONENT_INDEX[suffix]

    def sample_magnetic_field_vector(
        self, h_vectors: np.ndarray, mu: float = mu_0
    ) -> float:
        """Project recovered H vectors onto the physical receiver normal.

        A single ``(3,)`` vector is accepted for legacy Biot recovery paths that
        currently evaluate only the receiver centre. A ``(sample_count, 3)``
        array performs the full finite-area quadrature.
        """

        values = np.asarray(h_vectors, dtype=float)
        if values.shape == (3,):
            mean_vector = values
        elif values.shape == (self.sample_count, 3):
            mean_vector = self.sample_weights @ values
        else:
            raise ValueError(
                "h_vectors must have shape (3,) or (sample_count, 3)"
            )
        value = float(np.dot(mean_vector, self.measurement_axis))
        if self.component.startswith("B"):
            value *= mu
        return value

    def sample(self, mesh, e: np.ndarray, b: np.ndarray, mu: float = mu_0) -> float:
        if self.component in _DBDT_COMPONENTS:
            return 0.0
        if self.component in _EDGE_COMPONENTS:
            vectors = _sample_vector_field(
                mesh,
                self.sample_points,
                e,
                ("Ex", "Ey", "Ez"),
            )
            return self._weighted_projection(vectors)

        vectors = _sample_vector_field(
            mesh,
            self.sample_points,
            b,
            ("Fx", "Fy", "Fz"),
        )
        value = self._weighted_projection(vectors)
        if self.component.startswith("H"):
            value /= mu
        return value

    def sample_hj(self, mesh, e: np.ndarray, h: np.ndarray, mu: float = mu_0) -> float:
        if self.component in _DBDT_COMPONENTS:
            return 0.0
        if self.component in _HJ_ELECTRIC_COMPONENTS:
            vectors = _sample_vector_field(
                mesh,
                self.sample_points,
                e,
                ("Fx", "Fy", "Fz"),
            )
            return self._weighted_projection(vectors)

        vectors = _sample_vector_field(
            mesh,
            self.sample_points,
            h,
            ("Ex", "Ey", "Ez"),
        )
        value = self._weighted_projection(vectors)
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
        if self.component not in _DBDT_COMPONENTS:
            return self.sample(mesh, e, b_new, mu)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        vectors = _sample_vector_field(
            mesh,
            self.sample_points,
            (np.asarray(b_new) - np.asarray(b_old)) / dt,
            ("Fx", "Fy", "Fz"),
        )
        return self._weighted_projection(vectors)

    def sample_hj_time_derivative(
        self,
        mesh,
        e: np.ndarray,
        h_new: np.ndarray,
        h_old: np.ndarray,
        dt: float,
        mu: float = mu_0,
    ) -> float:
        if self.component not in _DBDT_COMPONENTS:
            return self.sample_hj(mesh, e, h_new, mu)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        vectors = _sample_vector_field(
            mesh,
            self.sample_points,
            mu * (np.asarray(h_new) - np.asarray(h_old)) / dt,
            ("Ex", "Ey", "Ez"),
        )
        return self._weighted_projection(vectors)

    def _weighted_projection(self, vectors: np.ndarray) -> float:
        projections = np.asarray(vectors, dtype=float) @ self.measurement_axis
        return float(np.dot(self.sample_weights, projections))

    def _quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(self.location, dtype=float)
        radius = float(self.radius)
        if self.receiver_type == "volume_average":
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [radius, 0.0, 0.0],
                    [-radius, 0.0, 0.0],
                    [0.0, radius, 0.0],
                    [0.0, -radius, 0.0],
                    [0.0, 0.0, radius],
                    [0.0, 0.0, -radius],
                ],
                dtype=float,
            )
            weights = np.full(offsets.shape[0], 1.0 / offsets.shape[0])
            return center.reshape(1, 3) + offsets, weights

        normal = _normalised_vector(self.normal, "normal")
        basis_u, basis_v = _disk_basis(normal)
        nodes, radial_weights = np.polynomial.legendre.leggauss(
            _DISK_RADIAL_ORDER
        )
        unit_area_nodes = 0.5 * (nodes + 1.0)
        unit_area_weights = 0.5 * radial_weights
        angles = 2.0 * np.pi * (
            np.arange(_DISK_AZIMUTH_COUNT, dtype=float) + 0.5
        ) / _DISK_AZIMUTH_COUNT

        points = []
        weights = []
        for unit_area, radial_weight in zip(unit_area_nodes, unit_area_weights):
            radial_distance = radius * np.sqrt(unit_area)
            for angle in angles:
                direction = np.cos(angle) * basis_u + np.sin(angle) * basis_v
                points.append(center + radial_distance * direction)
                weights.append(radial_weight / _DISK_AZIMUTH_COUNT)
        return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def _sample_vector_field(mesh, locations, values, components) -> np.ndarray:
    locations = np.asarray(locations, dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    sampled = [
        np.asarray(
            mesh.get_interpolation_matrix(locations, component) @ values,
            dtype=float,
        ).reshape(-1)
        for component in components
    ]
    return np.column_stack(sampled)


def _component_axis(component: str) -> np.ndarray:
    suffix = str(component)[-1].lower()
    if suffix not in _VECTOR_COMPONENT_INDEX:
        raise ValueError("receiver component does not define a vector axis")
    axis = np.zeros(3, dtype=float)
    axis[_VECTOR_COMPONENT_INDEX[suffix]] = 1.0
    return axis


def _normalised_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3D vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _disk_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalised_vector(normal, "normal")
    reference_axes = np.eye(3)
    reference = reference_axes[int(np.argmin(np.abs(reference_axes @ normal)))]
    basis_u = np.cross(normal, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v
