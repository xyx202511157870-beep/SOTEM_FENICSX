"""Empymod-backed primary provider skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import PrimaryFieldProvider, as_points


@dataclass(frozen=True)
class EmpymodPrimaryProvider(PrimaryFieldProvider):
    """Delayed-import skeleton for an empymod primary-field provider."""

    config: dict[str, Any]

    def get_Ep_on_V(self, t: float, V) -> np.ndarray:
        as_points(V, "points")
        raise NotImplementedError("EmpymodPrimaryProvider field sampling is not implemented yet")

    def get_Ep_dc_on_V(self, V) -> np.ndarray:
        as_points(V, "points")
        raise NotImplementedError("EmpymodPrimaryProvider DC sampling is not implemented yet")

    def get_receiver_E(self, t: float, receivers) -> np.ndarray:
        as_points(receivers, "receivers")
        raise NotImplementedError("EmpymodPrimaryProvider receiver E is not implemented yet")

    def get_receiver_dBdt(self, t: float, receivers) -> np.ndarray:
        as_points(receivers, "receivers")
        raise NotImplementedError("EmpymodPrimaryProvider receiver dBdt is not implemented yet")
