"""Debye/Prony approximation for Cole-Cole conductivity."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import lsq_linear

from .cole_cole import cole_cole_conductivity


@dataclass(frozen=True)
class DebyeTerm:
    """One Debye memory term."""

    delta_sigma: float
    tau: float


@dataclass(frozen=True)
class DebyeFit:
    """Result of a Cole-Cole to Debye conductivity fit."""

    sigma_infinity: float
    terms: tuple[DebyeTerm, ...]
    frequencies: np.ndarray
    target: np.ndarray
    fitted: np.ndarray
    relative_l2: float


def debye_conductivity(freq, sigma_infinity: float, terms):
    """Evaluate sigma(w) = sigma_inf - sum(A_k/(1+i*w*tau_k))."""

    freq = np.asarray(freq, dtype=float)
    sigma = np.full(freq.shape, complex(sigma_infinity), dtype=complex)
    for term in terms:
        sigma -= float(term.delta_sigma) / (1.0 + 1j * 2.0 * np.pi * freq * float(term.tau))
    return sigma


def fit_cole_cole_debye(
    *,
    rho0: float,
    m: float,
    tau: float,
    c: float,
    n_terms: int = 10,
    f_min: float = 1.0e-2,
    f_max: float = 1.0e5,
    n_freq: int = 96,
    dc_weight: float = 1.0e6,
) -> DebyeFit:
    """Fit a non-negative Debye expansion to Cole-Cole conductivity.

    The fit enforces the DC conductivity constraint
    ``sum(delta_sigma) = sigma_inf - 1/rho0`` as a heavily weighted row.
    """

    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError("frequency bounds must satisfy 0 < f_min < f_max")
    if n_freq < 4:
        raise ValueError("n_freq must be at least 4")

    freqs = np.logspace(math.log10(f_min), math.log10(f_max), int(n_freq))
    target = cole_cole_conductivity(freqs, rho0=rho0, m=m, tau=tau, c=c)
    sigma_inf = 1.0 / (rho0 * (1.0 - m))
    tau_min = 1.0 / (2.0 * math.pi * freqs.max()) / 10.0
    tau_max = 1.0 / (2.0 * math.pi * freqs.min()) * 10.0
    tau_grid = np.logspace(math.log10(tau_min), math.log10(tau_max), int(n_terms))

    basis = 1.0 / (1.0 + 1j * 2.0 * np.pi * freqs[:, None] * tau_grid[None, :])
    rhs = sigma_inf - target
    dc_delta_sum = sigma_inf - 1.0 / rho0
    design = np.vstack([basis.real, basis.imag, dc_weight * np.ones((1, int(n_terms)))])
    data = np.r_[rhs.real, rhs.imag, dc_weight * dc_delta_sum]

    fit = lsq_linear(design, data, bounds=(0.0, np.inf), lsmr_tol="auto")
    terms = tuple(DebyeTerm(float(a), float(t)) for a, t in zip(fit.x, tau_grid))
    fitted = debye_conductivity(freqs, sigma_inf, terms)
    denom = np.linalg.norm(np.r_[target.real, target.imag])
    rel_l2 = float(np.linalg.norm(np.r_[fitted.real - target.real, fitted.imag - target.imag]) / denom)
    return DebyeFit(float(sigma_inf), terms, freqs, target, fitted, rel_l2)

