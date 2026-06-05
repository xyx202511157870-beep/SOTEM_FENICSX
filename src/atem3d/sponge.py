"""Simple outer-domain sponge conductivity for finite-volume experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


_ALL_SIDES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def make_sponge_sigma(
    mesh,
    base_sigma,
    thickness_cells: int,
    strength: float,
    power: float = 2.0,
    active_sides: Sequence[str] | None = None,
) -> np.ndarray:
    """Increase conductivity smoothly in the outer ``thickness_cells`` layers.

    This is not a full CPML. It is a pragmatic direct-time-domain absorbing
    shell for FV experiments and must be validated with boundary-convergence
    sweeps.
    """

    base = np.asarray(base_sigma, dtype=float)
    if base.ndim == 0:
        base = np.full(mesh.n_cells, float(base))
    if base.shape != (mesh.n_cells,):
        raise ValueError("base_sigma must be scalar or have shape (mesh.n_cells,)")
    if thickness_cells < 0:
        raise ValueError("thickness_cells must be nonnegative")
    if strength < 0.0:
        raise ValueError("strength must be nonnegative")
    sides = _normalize_active_sides(active_sides)
    if thickness_cells == 0 or strength == 0.0 or not sides:
        return base.copy()

    centers = mesh.cell_centers
    layer_distance = np.full(mesh.n_cells, np.inf)
    side_set = set(sides)
    for axis, n_axis in enumerate(mesh.shape_cells):
        coords = np.unique(np.round(centers[:, axis], decimals=12))
        coord_to_index = {coord: i for i, coord in enumerate(coords)}
        indices = np.array([coord_to_index[round(v, 12)] for v in centers[:, axis]])
        axis_name = "xyz"[axis]
        if f"{axis_name}_min" in side_set:
            layer_distance = np.minimum(layer_distance, indices)
        if f"{axis_name}_max" in side_set:
            layer_distance = np.minimum(layer_distance, n_axis - 1 - indices)

    active = layer_distance < thickness_cells
    weight = np.zeros(mesh.n_cells, dtype=float)
    weight[active] = ((thickness_cells - layer_distance[active]) / thickness_cells) ** power
    return base + strength * weight


def _normalize_active_sides(active_sides: Sequence[str] | None) -> tuple[str, ...]:
    if active_sides is None:
        return _ALL_SIDES
    sides = tuple(str(side).strip().lower() for side in active_sides)
    unknown = sorted(set(sides) - set(_ALL_SIDES))
    if unknown:
        raise ValueError(f"unknown sponge side(s): {', '.join(unknown)}")
    return sides
