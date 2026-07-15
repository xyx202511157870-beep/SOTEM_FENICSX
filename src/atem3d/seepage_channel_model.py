"""Canonical physical contract for the seepage-channel benchmark.

Coordinates in this module use the project convention: ``z=0`` is the
surface, positive ``z`` is underground, and negative ``z`` is air.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray


Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(frozen=True)
class ChannelBox:
    """Axis-aligned conductive cuboid parallel to the x-directed wire."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    conductivity: float = 1.0

    def __post_init__(self) -> None:
        if any(length <= 0.0 for length in self.size):
            raise ValueError("channel dimensions must all be positive")
        if self.size[0] <= self.size[1]:
            raise ValueError("channel long axis must be parallel to the x-directed wire")
        if self.bounds[2][0] <= 0.0:
            raise ValueError("channel must be fully underground in the z-down convention")
        if self.conductivity <= 0.0:
            raise ValueError("channel conductivity must be positive")

    @property
    def bounds(self) -> Bounds3D:
        return tuple(
            (center - 0.5 * length, center + 0.5 * length)
            for center, length in zip(self.center, self.size, strict=True)
        )  # type: ignore[return-value]

    @property
    def volume_m3(self) -> float:
        return float(np.prod(self.size))

    def mask(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Return an inclusive cell/point mask in the canonical z-down frame."""

        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        inside = np.ones(coordinates.shape[0], dtype=bool)
        for axis, (lower, upper) in enumerate(self.bounds):
            inside &= (coordinates[:, axis] >= lower) & (coordinates[:, axis] <= upper)
        return inside

    def to_z_up_bounds(self) -> Bounds3D:
        """Convert bounds for SimPEG's internal positive-up coordinate frame."""

        x_bounds, y_bounds, (z_min, z_max) = self.bounds
        return x_bounds, y_bounds, (-z_max, -z_min)


@dataclass(frozen=True)
class SeepageChannelBenchmark:
    coordinate_convention: str = "z_down"
    source_endpoints: tuple[tuple[float, float, float], ...] = (
        (-50.0, 0.0, 0.1),
        (50.0, 0.0, 0.1),
    )
    source_current_a: float = 1.0
    waveform: str = "ideal_step_off"
    receiver_locations: tuple[tuple[float, float, float], ...] = tuple(
        (0.0, y, -0.1) for y in (-20.0, -10.0, 0.0, 10.0, 20.0)
    )
    components: tuple[str, ...] = ("Ex", "dBzdt", "Hz")
    times: NDArray[np.float64] = field(
        default_factory=lambda: np.logspace(-5, -2, 31)
    )
    air_conductivity: float = 1.0e-8
    background_conductivity: float = 1.0e-2
    channel: ChannelBox = field(
        default_factory=lambda: ChannelBox(
            center=(0.0, 0.0, 20.0),
            size=(60.0, 10.0, 10.0),
            conductivity=1.0,
        )
    )


MODEL = SeepageChannelBenchmark()


def benchmark_model(variant: str = "baseline_60x10x10") -> SeepageChannelBenchmark:
    """Return a named seepage-channel contract while preserving the baseline default."""

    normalized = str(variant).strip().lower()
    if normalized in {"baseline", "baseline_60x10x10", "60x10x10"}:
        return MODEL
    if normalized in {"thin", "thin_60x1x1", "60x1x1"}:
        return replace(
            MODEL,
            channel=ChannelBox(
                center=MODEL.channel.center,
                size=(60.0, 1.0, 1.0),
                conductivity=MODEL.channel.conductivity,
            ),
        )
    raise ValueError(f"unknown seepage-channel benchmark variant: {variant}")


__all__ = ["ChannelBox", "MODEL", "SeepageChannelBenchmark", "benchmark_model"]
