"""Synthetic complex-terrain leakage-channel example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atem3d.materials.material_map import (
    CellMaterialMap,
    apply_leakage_channel_marker,
)
from atem3d.materials.prony import DebyeTerm, PronyConductivity


@dataclass(frozen=True)
class LeakageChannelExample:
    """Small marker/material example for P8 smoke tests."""

    cell_centers: np.ndarray
    markers: np.ndarray
    material_map: CellMaterialMap
    background_material: PronyConductivity
    leakage_material: PronyConductivity
    leakage_marker: int
    diagnostics: dict[str, float | int]


def build_leakage_channel_example(
    *,
    nx: int = 9,
    ny: int = 7,
    leakage_radius: float = 0.45,
) -> LeakageChannelExample:
    """Build a small complex-terrain marker map with an irregular leakage channel."""

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.5, 1.5, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    terrain = _terrain_elevation(xx, yy)
    cell_centers = np.column_stack([xx.ravel(), yy.ravel(), (terrain - 1.0).ravel()])

    background_marker = 1
    leakage_marker = 7
    markers = np.full(cell_centers.shape[0], background_marker, dtype=int)
    channel = np.array(
        [
            [-1.6, -1.0, -0.95],
            [-0.4, -0.2, -0.85],
            [0.6, 0.1, -0.75],
            [1.7, 1.0, -0.8],
        ],
        dtype=float,
    )
    markers = apply_leakage_channel_marker(
        markers,
        cell_centers,
        channel_points=channel,
        radius=leakage_radius,
        leakage_marker=leakage_marker,
    )

    background = PronyConductivity.no_ip(0.01)
    leakage = PronyConductivity(
        sigma_inf=0.08,
        terms=[DebyeTerm(delta_sigma=0.02, tau=0.2)],
    )
    material_map = CellMaterialMap(
        markers=markers,
        materials={background_marker: background, leakage_marker: leakage},
    )
    diagnostics = {
        "cell_count": int(cell_centers.shape[0]),
        "leakage_cell_count": int(np.count_nonzero(markers == leakage_marker)),
        "terrain_elevation_min": float(np.min(terrain)),
        "terrain_elevation_max": float(np.max(terrain)),
        "leakage_marker": int(leakage_marker),
    }
    return LeakageChannelExample(
        cell_centers=cell_centers,
        markers=markers,
        material_map=material_map,
        background_material=background,
        leakage_material=leakage,
        leakage_marker=leakage_marker,
        diagnostics=diagnostics,
    )


def _terrain_elevation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.15 * np.sin(np.pi * x / 2.0) + 0.08 * np.cos(np.pi * y / 1.5)
