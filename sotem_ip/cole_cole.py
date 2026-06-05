"""Cole-Cole material functions."""

from __future__ import annotations

import numpy as np


def cole_cole_resistivity(freq, rho0: float, m: float, tau: float, c: float):
    """Return Pelton-style Cole-Cole complex resistivity.

    Parameters use SI units: frequency in Hz, resistivity in ohm m, and tau in s.
    """

    freq = np.asarray(freq, dtype=float)
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive")
    if not 0.0 <= m < 1.0:
        raise ValueError("m must satisfy 0 <= m < 1")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if c <= 0.0:
        raise ValueError("c must be positive")
    s_tau_c = (1j * 2.0 * np.pi * freq * tau) ** c
    return rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + s_tau_c)))


def cole_cole_conductivity(freq, rho0: float, m: float, tau: float, c: float):
    """Return complex conductivity corresponding to :func:`cole_cole_resistivity`."""

    return 1.0 / cole_cole_resistivity(freq, rho0, m, tau, c)

