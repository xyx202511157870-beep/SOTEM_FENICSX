"""Zero primary field provider."""

from __future__ import annotations

import numpy as np

from .base import PrimaryFieldProvider, as_points


class ZeroPrimaryProvider(PrimaryFieldProvider):
    """Primary provider for tests and zero-background-primary limits."""

    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        points = as_points(V, "points")
        return np.zeros((points.shape[0], 3), dtype=float)

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        points = as_points(V, "points")
        return np.zeros((points.shape[0], 3), dtype=float)

    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return np.zeros((points.shape[0], 3), dtype=float)

    def get_receiver_H(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return np.zeros((points.shape[0], 3), dtype=float)

    def get_receiver_dBdt(self, t: float, receivers) -> np.ndarray:
        points = as_points(receivers, "receivers")
        return np.zeros((points.shape[0], 3), dtype=float)
