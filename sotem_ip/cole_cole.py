"""Cole-Cole material functions.

中文说明：参考实现使用明确的复电阻率/复电导率转换。rho0 表示直流
电阻率；与 empymod 对比时必须同时固定傅里叶号约定、频率单位和
Cole–Cole 参数定义。
"""

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

