"""Fit frequency-dependent IP models with Debye conductivity poles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .ip import DebyeTerm
from .metrics import relative_l2


@dataclass(frozen=True)
class DebyeFitResult:
    """Result of a conductivity Debye fit."""

    sigma_infinity: float
    terms: list[DebyeTerm]
    frequencies: np.ndarray
    target_sigma: np.ndarray
    fitted_sigma: np.ndarray
    relative_l2: float


def cole_cole_conductivity(
    frequencies,
    sigma_infinity: float,
    eta: float,
    tau: float,
    c: float,
) -> np.ndarray:
    """Conductivity-form Cole-Cole model."""

    frequencies = np.asarray(frequencies, dtype=float)
    s_tau = 1j * 2.0 * np.pi * frequencies * tau
    return sigma_infinity * (1.0 - eta / (1.0 + s_tau**c))


def pelton_resistivity_to_conductivity(
    frequencies,
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
) -> np.ndarray:
    """Convert Pelton resistivity form to complex conductivity."""

    frequencies = np.asarray(frequencies, dtype=float)
    s_tau = 1j * 2.0 * np.pi * frequencies * tau
    rho = rho0 * (1.0 - chargeability * (1.0 - 1.0 / (1.0 + s_tau**c)))
    return 1.0 / rho


def fit_cole_cole_conductivity_debye(
    sigma_infinity: float,
    eta: float,
    tau: float,
    c: float,
    frequencies,
    tau_grid=None,
    n_terms: int = 10,
) -> DebyeFitResult:
    """Fit conductivity Cole-Cole with finite Debye poles."""

    frequencies = np.asarray(frequencies, dtype=float)
    target = cole_cole_conductivity(frequencies, sigma_infinity, eta, tau, c)
    if tau_grid is None:
        tau_grid = _default_tau_grid(frequencies, n_terms)
    return fit_complex_conductivity_debye(frequencies, target, sigma_infinity, tau_grid)


def fit_pelton_resistivity_debye(
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
    frequencies,
    tau_grid=None,
    n_terms: int = 10,
) -> DebyeFitResult:
    """Fit Pelton resistivity after converting it to conductivity."""

    frequencies = np.asarray(frequencies, dtype=float)
    target = pelton_resistivity_to_conductivity(frequencies, rho0, chargeability, tau, c)
    sigma_infinity = 1.0 / (float(rho0) * (1.0 - float(chargeability)))
    if tau_grid is None:
        tau_grid = _default_tau_grid(frequencies, n_terms)
    return fit_complex_conductivity_debye(frequencies, target, sigma_infinity, tau_grid)


def fit_complex_conductivity_debye(
    frequencies,
    target_sigma,
    sigma_infinity: float,
    tau_grid,
) -> DebyeFitResult:
    """Fit ``sigma = sigma_inf - sum(delta_i/(1+i*w*tau_i))``."""

    frequencies = np.asarray(frequencies, dtype=float)
    target = np.asarray(target_sigma, dtype=complex)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be positive")
    if np.any(tau_grid <= 0.0):
        raise ValueError("tau_grid must be positive")
    if target.shape != frequencies.shape:
        raise ValueError("target_sigma must have the same shape as frequencies")

    basis = 1.0 / (1.0 + 1j * 2.0 * np.pi * frequencies[:, None] * tau_grid[None, :])
    rhs = sigma_infinity - target
    design = np.vstack([basis.real, basis.imag])
    data = np.r_[rhs.real, rhs.imag]
    fit = lsq_linear(design, data, bounds=(0.0, np.inf), lsmr_tol="auto")
    delta = fit.x
    fitted = sigma_infinity - basis @ delta
    terms = [DebyeTerm(delta_sigma=np.array([value]), tau=float(tau)) for value, tau in zip(delta, tau_grid)]
    return DebyeFitResult(
        sigma_infinity=float(sigma_infinity),
        terms=terms,
        frequencies=frequencies,
        target_sigma=target,
        fitted_sigma=fitted,
        relative_l2=relative_l2(
            np.r_[fitted.real, fitted.imag],
            np.r_[target.real, target.imag],
        ),
    )


def _default_tau_grid(frequencies: np.ndarray, n_terms: int) -> np.ndarray:
    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    min_tau = 1.0 / (2.0 * np.pi * np.max(frequencies)) / 10.0
    max_tau = 1.0 / (2.0 * np.pi * np.min(frequencies)) * 10.0
    return np.logspace(np.log10(min_tau), np.log10(max_tau), n_terms)
