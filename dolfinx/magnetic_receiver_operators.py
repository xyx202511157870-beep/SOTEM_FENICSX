"""Magnetic receiver geometry and integration operators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HorizontalLoopRule:
    """Periodic quadrature rule for a horizontal circular receiver loop."""

    center: NDArray[np.float64]
    radius: float
    points: NDArray[np.float64]
    tangents: NDArray[np.float64]
    line_weights: NDArray[np.float64]
    area: float


def horizontal_loop_rule(
    center: ArrayLike,
    *,
    radius: float,
    point_count: int,
) -> HorizontalLoopRule:
    """Return pairwise-symmetric periodic quadrature around a horizontal loop."""

    center_array = np.asarray(center, dtype=float).reshape(3)
    radius_value = float(radius)
    count = int(point_count)
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("faraday loop radius must be positive")
    if count < 8 or count % 4:
        raise ValueError(
            "faraday loop point count must be a multiple of 4 and at least 8"
        )
    if not np.all(np.isfinite(center_array)):
        raise ValueError("faraday loop center must be finite")

    theta = 2.0 * np.pi * np.arange(count, dtype=float) / count
    offsets = np.column_stack(
        (
            radius_value * np.cos(theta),
            radius_value * np.sin(theta),
            np.zeros(count),
        )
    )
    tangents = np.column_stack(
        (-np.sin(theta), np.cos(theta), np.zeros(count))
    )
    return HorizontalLoopRule(
        center=center_array,
        radius=radius_value,
        points=center_array + offsets,
        tangents=tangents,
        line_weights=np.full(count, 2.0 * np.pi * radius_value / count),
        area=float(np.pi * radius_value**2),
    )


def faraday_loop_dbdt(
    electric_values: ArrayLike,
    rule: HorizontalLoopRule,
) -> float:
    """Return area-averaged dBz/dt from the closed-loop electric circulation."""

    electric = np.asarray(electric_values, dtype=float)
    if electric.shape != rule.points.shape or not np.all(np.isfinite(electric)):
        raise ValueError(
            "electric values must be finite and match the faraday loop points"
        )
    circulation = np.sum(
        rule.line_weights * np.einsum("ij,ij->i", electric, rule.tangents)
    )
    return -float(circulation) / rule.area


def tetra4_rule(
    vertices: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the symmetric four-point quadrature rule for one tetrahedron."""

    coordinates = np.asarray(vertices, dtype=float)
    if coordinates.shape != (4, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("tetrahedron vertices must be finite with shape (4, 3)")
    determinant = np.linalg.det(
        np.column_stack(
            (
                coordinates[1] - coordinates[0],
                coordinates[2] - coordinates[0],
                coordinates[3] - coordinates[0],
            )
        )
    )
    volume = abs(float(determinant)) / 6.0
    if volume == 0.0:
        raise ValueError("tetrahedron volume must be positive")

    alpha = 0.5854101966249685
    beta = 0.1381966011250105
    barycentric = np.asarray(
        [
            [alpha, beta, beta, beta],
            [beta, alpha, beta, beta],
            [beta, beta, alpha, beta],
            [beta, beta, beta, alpha],
        ],
        dtype=float,
    )
    return barycentric @ coordinates, np.full(4, volume / 4.0)


def neumaier_vector_sum(values: ArrayLike) -> NDArray[np.float64]:
    """Accumulate vectors with faithfully rounded componentwise compensation."""

    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("values must have shape (n, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("values must be finite")

    return np.asarray(
        [math.fsum(vectors[:, component]) for component in range(3)],
        dtype=float,
    )


def biot_savart_volume_h(
    *,
    receiver: ArrayLike,
    points: ArrayLike,
    current_density: ArrayLike,
    weights: ArrayLike,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Integrate a sampled volume-current density into receiver H."""

    receiver_point = np.asarray(receiver, dtype=float).reshape(3)
    integration_points = np.asarray(points, dtype=float)
    currents = np.asarray(current_density, dtype=float)
    integration_weights = np.asarray(weights, dtype=float).reshape(-1)
    if integration_points.ndim != 2 or integration_points.shape[1] != 3:
        raise ValueError("Biot integration points must have shape (n, 3)")
    if currents.shape != integration_points.shape or integration_weights.shape != (
        integration_points.shape[0],
    ):
        raise ValueError(
            "Biot samples, currents, and weights must have matching lengths"
        )
    if not (
        np.all(np.isfinite(receiver_point))
        and np.all(np.isfinite(integration_points))
        and np.all(np.isfinite(currents))
        and np.all(np.isfinite(integration_weights))
    ):
        raise ValueError("Biot receiver inputs must be finite")

    displacement = receiver_point[None, :] - integration_points
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("receiver coincides with a Biot integration point")
    contributions = (
        integration_weights[:, None]
        * np.cross(currents, displacement)
        / (4.0 * np.pi * distance[:, None] ** 3)
    )
    result = neumaier_vector_sum(contributions)
    absolute_sum = np.sum(np.abs(contributions), axis=0)
    cancellation_ratio = np.divide(
        np.abs(result),
        absolute_sum,
        out=np.ones(3, dtype=float),
        where=absolute_sum > 0.0,
    )
    return result, {
        "sample_count": int(integration_points.shape[0]),
        "absolute_contribution_sum": absolute_sum.tolist(),
        "cancellation_ratio": cancellation_ratio.tolist(),
    }


def evaluate_faraday_loop_field(
    electric_field: Any,
    mesh: Any,
    *,
    center: ArrayLike,
    radius: float,
    point_count: int,
    find_cells: Callable[[Any, NDArray[np.float64]], Sequence[int]],
) -> tuple[float, dict[str, object]]:
    """Evaluate a finite-loop Faraday receiver and fail if any point is absent."""

    rule = horizontal_loop_rule(
        center,
        radius=radius,
        point_count=point_count,
    )
    selected_cells: list[int] = []
    missing_indices: list[int] = []
    for point_index, point in enumerate(rule.points):
        candidates = np.asarray(find_cells(mesh, point), dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            missing_indices.append(point_index)
        else:
            selected_cells.append(int(np.min(candidates)))
    if missing_indices:
        raise RuntimeError(
            "Faraday loop point location failed for indices "
            + ",".join(str(index) for index in missing_indices)
        )

    cells = np.asarray(selected_cells, dtype=np.int32)
    electric_values = np.asarray(
        electric_field.eval(rule.points, cells),
        dtype=float,
    ).reshape(rule.points.shape[0], -1)
    if electric_values.shape[1] != 3:
        raise ValueError("electric field evaluation must return three components")
    value = faraday_loop_dbdt(electric_values, rule)
    return value, {
        "method": "faraday_loop",
        "radius_m": float(rule.radius),
        "point_count": int(rule.points.shape[0]),
        "located_point_count": int(rule.points.shape[0]),
        "missing_point_indices": [],
    }


def evaluate_biot_current_field(
    current_density: Any,
    *,
    receiver: ArrayLike,
    points: ArrayLike,
    cells: ArrayLike,
    weights: ArrayLike,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Evaluate a current-density function and integrate its Biot receiver H."""

    integration_points = np.asarray(points, dtype=float)
    owning_cells = np.asarray(cells, dtype=np.int32).reshape(-1)
    integration_weights = np.asarray(weights, dtype=float).reshape(-1)
    if integration_points.ndim != 2 or integration_points.shape[1] != 3:
        raise ValueError("Biot integration points must have shape (n, 3)")
    if owning_cells.shape != (integration_points.shape[0],):
        raise ValueError("Biot owning cells must match integration points")
    if integration_weights.shape != (integration_points.shape[0],):
        raise ValueError("Biot weights must match integration points")
    current_values = np.asarray(
        current_density.eval(integration_points, owning_cells),
        dtype=float,
    ).reshape(integration_points.shape[0], -1)
    if current_values.shape[1] != 3:
        raise ValueError("current density evaluation must return three components")
    value, audit = biot_savart_volume_h(
        receiver=receiver,
        points=integration_points,
        current_density=current_values,
        weights=integration_weights,
    )
    return value, {"method": "biot_tetra4", **audit}


__all__ = [
    "HorizontalLoopRule",
    "biot_savart_volume_h",
    "evaluate_biot_current_field",
    "evaluate_faraday_loop_field",
    "faraday_loop_dbdt",
    "horizontal_loop_rule",
    "neumaier_vector_sum",
    "tetra4_rule",
]
