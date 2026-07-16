"""Magnetic receiver geometry and integration operators."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Accumulate three-component vectors with Neumaier compensation."""

    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("values must have shape (n, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("values must be finite")

    total = np.zeros(3, dtype=float)
    correction = np.zeros(3, dtype=float)
    for value in vectors:
        updated = total + value
        correction += np.where(
            np.abs(total) >= np.abs(value),
            (total - updated) + value,
            (value - updated) + total,
        )
        total = updated
    return total + correction


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


__all__ = [
    "HorizontalLoopRule",
    "biot_savart_volume_h",
    "faraday_loop_dbdt",
    "horizontal_loop_rule",
    "neumaier_vector_sum",
    "tetra4_rule",
]
