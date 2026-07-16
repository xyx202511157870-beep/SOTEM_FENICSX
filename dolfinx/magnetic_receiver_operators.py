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


__all__ = [
    "HorizontalLoopRule",
    "faraday_loop_dbdt",
    "horizontal_loop_rule",
]
