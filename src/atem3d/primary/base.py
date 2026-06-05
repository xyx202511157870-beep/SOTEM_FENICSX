"""Primary-field provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PrimaryFieldProvider(ABC):
    """Interface for fields supplied by a 1D/background primary solver."""

    @abstractmethod
    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        """Return primary electric field samples on FEM points."""

    @abstractmethod
    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        """Return DC primary electric field samples on FEM points."""

    @abstractmethod
    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        """Return primary electric field at receiver locations."""

    @abstractmethod
    def get_receiver_dBdt(self, t: float, receivers) -> np.ndarray:
        """Return primary dB/dt at receiver locations."""


def as_points(points, name: str) -> np.ndarray:
    """Validate an array of 3D points."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return values
