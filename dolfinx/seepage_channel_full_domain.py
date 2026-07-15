#!/usr/bin/env python3
"""Full-domain FEniCSx adapter for the seepage-channel benchmark.

The adapter intentionally contains no half-domain or receiver-mirroring path.
Pure geometry/material and receiver-set helpers are added here as their
test-driven implementation tasks are completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ReceiverEvaluationConfig:
    receiver: tuple[float, float, float]
    receiver_id: str
    provenance: str = "explicit_full_domain"


def receiver_configs(
    locations: Iterable[tuple[float, float, float]],
) -> tuple[ReceiverEvaluationConfig, ...]:
    configs = tuple(
        ReceiverEvaluationConfig(
            receiver=tuple(float(value) for value in location),  # type: ignore[arg-type]
            receiver_id=f"Rx{index + 1}",
        )
        for index, location in enumerate(locations)
    )
    if any(len(config.receiver) != 3 for config in configs):
        raise ValueError("each receiver location must contain x, y, z")
    if len({config.receiver for config in configs}) != len(configs):
        raise ValueError("receiver locations must be unique")
    return configs


def box_mask(
    centers: ArrayLike,
    volumes: ArrayLike,
    bounds: ArrayLike,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    cell_centers = np.asarray(centers, dtype=float)
    cell_volumes = np.asarray(volumes, dtype=float)
    box_bounds = np.asarray(bounds, dtype=float)
    if cell_centers.ndim != 2 or cell_centers.shape[1] != 3:
        raise ValueError("centers must have shape (n, 3)")
    if cell_volumes.shape != (cell_centers.shape[0],):
        raise ValueError("volumes must contain one value per center")
    if box_bounds.shape != (3, 2) or np.any(box_bounds[:, 1] <= box_bounds[:, 0]):
        raise ValueError("bounds must have shape (3, 2) with upper > lower")
    mask = np.logical_and.reduce(
        [
            (cell_centers[:, axis] >= box_bounds[axis, 0])
            & (cell_centers[:, axis] <= box_bounds[axis, 1])
            for axis in range(3)
        ]
    )
    return mask, {
        "local_cell_count": int(np.count_nonzero(mask)),
        "local_discrete_volume_m3": float(np.sum(cell_volumes[mask])),
    }


__all__ = ["ReceiverEvaluationConfig", "box_mask", "receiver_configs"]
