"""Small analytical helpers used by examples and tests."""

from __future__ import annotations

import numpy as np
from scipy.constants import mu_0


def diffusion_time_scale(offset: float, resistivity: float) -> float:
    """Return a rough EM diffusion time scale mu0*r^2/rho."""

    if offset <= 0.0:
        raise ValueError("offset must be positive")
    if resistivity <= 0.0:
        raise ValueError("resistivity must be positive")
    return float(mu_0 * offset * offset / resistivity)


def smooth_stepoff_response(times, amplitude: float, time_scale: float, power: float = 1.5):
    """Simple monotone step-off proxy for plotting and pipeline smoke tests."""

    times = np.asarray(times, dtype=float)
    if np.any(times <= 0.0):
        raise ValueError("times must be positive")
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive")
    return float(amplitude) / (1.0 + times / float(time_scale)) ** float(power)

