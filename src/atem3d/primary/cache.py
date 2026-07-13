"""Cached primary field provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PrimaryFieldProvider, as_points


@dataclass(frozen=True)
class CachedPrimaryProvider(PrimaryFieldProvider):
    """Primary provider backed by precomputed time tables."""

    times: np.ndarray
    points: np.ndarray
    receivers: np.ndarray
    Ep_on_V: np.ndarray
    receiver_E: np.ndarray
    receiver_dBdt: np.ndarray
    Ep_dc_on_V: np.ndarray
    receiver_H: np.ndarray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        points = as_points(self.points, "points")
        receivers = as_points(self.receivers, "receivers")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty 1D array")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")

        Ep_on_V = self._validate_field_table(self.Ep_on_V, times.size, points.shape[0], "Ep_on_V")
        receiver_E = self._validate_field_table(
            self.receiver_E,
            times.size,
            receivers.shape[0],
            "receiver_E",
        )
        receiver_H = (
            np.zeros_like(receiver_E)
            if self.receiver_H is None
            else self._validate_field_table(
                self.receiver_H,
                times.size,
                receivers.shape[0],
                "receiver_H",
            )
        )
        receiver_dBdt = self._validate_field_table(
            self.receiver_dBdt,
            times.size,
            receivers.shape[0],
            "receiver_dBdt",
        )
        Ep_dc_on_V = np.asarray(self.Ep_dc_on_V, dtype=float)
        if Ep_dc_on_V.shape != (points.shape[0], 3):
            raise ValueError("Ep_dc_on_V must have shape (n_points, 3)")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "Ep_on_V", Ep_on_V)
        object.__setattr__(self, "receiver_E", receiver_E)
        object.__setattr__(self, "receiver_H", receiver_H)
        object.__setattr__(self, "receiver_dBdt", receiver_dBdt)
        object.__setattr__(self, "Ep_dc_on_V", Ep_dc_on_V)

    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        self._require_same_points(V, self.points, "points")
        return self._interpolate(t, self.Ep_on_V)

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        self._require_same_points(V, self.points, "points")
        return self.Ep_dc_on_V.copy()

    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        self._require_same_points(receivers, self.receivers, "receivers")
        return self._interpolate(t, self.receiver_E)

    def get_receiver_H(self, t: float, receivers) -> np.ndarray:
        self._require_same_points(receivers, self.receivers, "receivers")
        return self._interpolate(t, self.receiver_H)

    def get_receiver_dBdt(self, t: float, receivers) -> np.ndarray:
        self._require_same_points(receivers, self.receivers, "receivers")
        return self._interpolate(t, self.receiver_dBdt)

    @staticmethod
    def _validate_field_table(values, n_times: int, n_points: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (n_times, n_points, 3):
            raise ValueError(f"{name} must have shape (n_times, n_points, 3)")
        return array

    @staticmethod
    def _require_same_points(query, cached: np.ndarray, name: str) -> None:
        points = as_points(query, name)
        if points.shape != cached.shape or not np.allclose(points, cached, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{name} must match cached {name}")

    def _interpolate(self, t: float, table: np.ndarray) -> np.ndarray:
        t = float(t)
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError("t is outside cached time range")
        exact = np.flatnonzero(np.isclose(self.times, t, rtol=0.0, atol=1.0e-15))
        if exact.size:
            return table[int(exact[0])].copy()
        upper = int(np.searchsorted(self.times, t, side="right"))
        lower = upper - 1
        weight = (t - self.times[lower]) / (self.times[upper] - self.times[lower])
        return (1.0 - weight) * table[lower] + weight * table[upper]
