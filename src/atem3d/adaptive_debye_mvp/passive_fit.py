"""Passive Debye fits with a hard DC-conductivity equality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear

from atem3d.fit import DebyeFitResult
from atem3d.ip import DebyeTerm as LegacyDebyeTerm
from atem3d.materials.cole_cole import PeltonColeColeResistivity
from atem3d.materials.prony import DebyeTerm, PronyConductivity
from atem3d.metrics import relative_l2


HARD_GATE_DELTA_FLOOR: float = -1.0e-12
HARD_GATE_DC_TOLERANCE: float = 1.0e-10
_ZERO_IP_RELATIVE_BUDGET: float = 1.0e-15
_NEAR_DUPLICATE_LOG10: float = 1.0e-9
_FREE_SET_THRESHOLD: float = 1.0e-10


@dataclass(frozen=True)
class OptimizerStatus:
    """Solver diagnostics for a hard-DC Debye fit."""

    success: bool
    method: str
    status_code: int
    message: str
    n_iterations: int
    polish_iterations: int
    active_set_size: int
    cost: float
    penalty_weight: float


@dataclass(frozen=True)
class PassiveDebyeFit:
    """Result of a nonnegative Debye fit with a hard DC gate."""

    delta_sigma: np.ndarray
    tau_grid: np.ndarray
    sigma_infinity: float
    sigma0_target: float
    sigma0: float
    dc_error: float
    relative_dc_error: float
    passive: bool
    spectral_error: float
    spectral_error_linf: float
    condition_number: float
    optimizer_status: OptimizerStatus
    frequencies: np.ndarray
    target_sigma: np.ndarray
    fitted_sigma: np.ndarray
    weights: np.ndarray

    def passes_hard_gates(
        self,
        *,
        delta_floor: float = HARD_GATE_DELTA_FLOOR,
        dc_tolerance: float = HARD_GATE_DC_TOLERANCE,
    ) -> bool:
        return len(self.hard_gate_failures(delta_floor=delta_floor, dc_tolerance=dc_tolerance)) == 0

    def hard_gate_failures(
        self,
        *,
        delta_floor: float = HARD_GATE_DELTA_FLOOR,
        dc_tolerance: float = HARD_GATE_DC_TOLERANCE,
    ) -> tuple[str, ...]:
        return hard_gate_failures(self, delta_floor=delta_floor, dc_tolerance=dc_tolerance)

    def to_prony_conductivity(self) -> PronyConductivity:
        if not self.passes_hard_gates():
            raise ValueError("cannot convert a rejected hard-gate fit to PronyConductivity")
        terms = [
            DebyeTerm(delta_sigma=max(float(delta), 0.0), tau=float(tau))
            for delta, tau in zip(self.delta_sigma, self.tau_grid)
        ]
        return PronyConductivity(sigma_inf=float(self.sigma_infinity), terms=terms)

    def to_debye_fit_result(self) -> DebyeFitResult:
        terms = [
            LegacyDebyeTerm(delta_sigma=np.array([float(delta)]), tau=float(tau))
            for delta, tau in zip(self.delta_sigma, self.tau_grid)
        ]
        stacked_fit = np.r_[self.fitted_sigma.real, self.fitted_sigma.imag]
        stacked_target = np.r_[self.target_sigma.real, self.target_sigma.imag]
        return DebyeFitResult(
            sigma_infinity=float(self.sigma_infinity),
            terms=terms,
            frequencies=np.asarray(self.frequencies, dtype=float),
            target_sigma=np.asarray(self.target_sigma, dtype=complex),
            fitted_sigma=np.asarray(self.fitted_sigma, dtype=complex),
            relative_l2=relative_l2(stacked_fit, stacked_target),
        )


def validate_pelton_parameters(
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
) -> PeltonColeColeResistivity:
    """Validate Pelton parameters by constructing the existing material."""

    return PeltonColeColeResistivity(
        rho0=float(rho0),
        chargeability=float(chargeability),
        tau=float(tau),
        c=float(c),
    )


def validate_tau_grid(tau_grid) -> np.ndarray:
    """Return a strictly increasing positive pole grid."""

    tau = np.asarray(tau_grid, dtype=float)
    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau_grid must be a nonempty 1-D array")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau_grid must be finite")
    if np.any(tau <= 0.0):
        raise ValueError("tau_grid must be positive")
    if tau.size > 1:
        if not np.all(np.diff(tau) > 0.0):
            raise ValueError("tau_grid must be strictly increasing")
        if np.any(np.diff(np.log10(tau)) < _NEAR_DUPLICATE_LOG10):
            raise ValueError("tau_grid contains duplicate or indistinguishable poles")
    return tau


def passes_hard_gates(
    fit: PassiveDebyeFit,
    *,
    delta_floor: float = HARD_GATE_DELTA_FLOOR,
    dc_tolerance: float = HARD_GATE_DC_TOLERANCE,
) -> bool:
    return len(hard_gate_failures(fit, delta_floor=delta_floor, dc_tolerance=dc_tolerance)) == 0


def hard_gate_failures(
    fit: PassiveDebyeFit,
    *,
    delta_floor: float = HARD_GATE_DELTA_FLOOR,
    dc_tolerance: float = HARD_GATE_DC_TOLERANCE,
) -> tuple[str, ...]:
    failures: list[str] = []
    if float(np.min(fit.delta_sigma)) < float(delta_floor):
        failures.append("delta_sigma_negative")
    if float(fit.relative_dc_error) > float(dc_tolerance):
        failures.append("dc_error_exceeds_tolerance")
    if float(fit.sigma0) <= 0.0:
        failures.append("sigma0_not_positive")
    if not fit.optimizer_status.success:
        failures.append("optimizer_failed")
    if not np.isfinite(fit.condition_number):
        failures.append("condition_number_not_finite")
    return tuple(failures)


def fit_debye_passive_hard_dc(
    frequencies,
    target_sigma,
    sigma_infinity: float,
    tau_grid,
    sigma0_target: float,
    weights=None,
) -> PassiveDebyeFit:
    """Fit ``sigma = sigma_inf - sum(delta_k / (1 + i w tau_k))`` with hard DC."""

    frequencies = _validated_frequencies(frequencies)
    target = np.asarray(target_sigma, dtype=complex)
    if target.shape != frequencies.shape:
        raise ValueError("target_sigma must have the same shape as frequencies")
    if not np.all(np.isfinite(target.real)) or not np.all(np.isfinite(target.imag)):
        raise ValueError("target_sigma must be finite")
    if float(np.max(np.abs(target))) <= 0.0:
        raise ValueError("target_sigma must be nonzero")

    sigma_infinity = float(sigma_infinity)
    sigma0_target = float(sigma0_target)
    if not np.isfinite(sigma_infinity) or sigma_infinity <= 0.0:
        raise ValueError("sigma_infinity must be positive")
    if not np.isfinite(sigma0_target) or sigma0_target <= 0.0:
        raise ValueError("sigma0_target must be positive")
    if sigma0_target > sigma_infinity:
        raise ValueError("sigma0_target must not exceed sigma_infinity: negative IP budget cannot be passive")

    tau = validate_tau_grid(tau_grid)
    normalized_weights = _normalized_weights(weights, frequencies.size)

    omega = 2.0 * np.pi * frequencies
    basis = 1.0 / (1.0 + 1j * omega[:, None] * tau[None, :])
    scale = np.sqrt(normalized_weights)
    design = np.vstack([scale[:, None] * basis.real, scale[:, None] * basis.imag])
    rhs = sigma_infinity - target
    data = np.concatenate([scale * rhs.real, scale * rhs.imag])
    condition_number = float(np.linalg.cond(design))

    ip_budget = sigma_infinity - sigma0_target
    if ip_budget <= _ZERO_IP_RELATIVE_BUDGET * sigma_infinity:
        delta = np.zeros(tau.size, dtype=float)
        status = OptimizerStatus(
            success=True,
            method="zero_ip_budget",
            status_code=0,
            message="sigma0_target == sigma_infinity: all deltas fixed to zero",
            n_iterations=0,
            polish_iterations=0,
            active_set_size=0,
            cost=float(0.5 * np.dot(data, data)),
            penalty_weight=0.0,
        )
        return _assemble_fit(
            delta=delta,
            tau=tau,
            sigma_infinity=sigma_infinity,
            sigma0_target=sigma0_target,
            basis=basis,
            frequencies=frequencies,
            target=target,
            weights=normalized_weights,
            condition_number=condition_number,
            status=status,
        )

    scaled_design = ip_budget * design
    penalty_weight = 1.0e8 * float(np.linalg.norm(scaled_design, ord="fro") ** 2)
    augmented = np.vstack([scaled_design, np.sqrt(penalty_weight) * np.ones((1, tau.size))])
    augmented_data = np.concatenate([data, np.array([np.sqrt(penalty_weight)])])
    raw = lsq_linear(
        augmented,
        augmented_data,
        bounds=(0.0, np.inf),
        method="bvls",
        tol=1.0e-12,
        max_iter=max(10 * tau.size, 100),
    )
    fractions, polish_iterations, polish_ok = _nullspace_polish(scaled_design, data, raw.x)
    delta = ip_budget * fractions
    residual = scaled_design @ fractions - data
    success = bool(raw.success and polish_ok and np.all(np.isfinite(delta)))
    status = OptimizerStatus(
        success=success,
        method="bvls_penalized+nullspace_polish",
        status_code=int(raw.status),
        message=str(raw.message),
        n_iterations=int(raw.nit),
        polish_iterations=int(polish_iterations),
        active_set_size=int(np.count_nonzero(delta > 0.0)),
        cost=float(0.5 * np.dot(residual, residual)),
        penalty_weight=float(penalty_weight),
    )
    return _assemble_fit(
        delta=delta,
        tau=tau,
        sigma_infinity=sigma_infinity,
        sigma0_target=sigma0_target,
        basis=basis,
        frequencies=frequencies,
        target=target,
        weights=normalized_weights,
        condition_number=condition_number,
        status=status,
    )


def fit_pelton_passive_hard_dc(
    rho0: float,
    chargeability: float,
    tau: float,
    c: float,
    frequencies,
    tau_grid,
    weights=None,
) -> PassiveDebyeFit:
    """Fit a Pelton Cole-Cole spectrum with the hard-DC Debye solver."""

    material = validate_pelton_parameters(rho0, chargeability, tau, c)
    return fit_debye_passive_hard_dc(
        frequencies,
        material.complex_conductivity(frequencies),
        material.sigma_inf,
        tau_grid,
        material.sigma0,
        weights=weights,
    )


def _validated_frequencies(frequencies) -> np.ndarray:
    values = np.asarray(frequencies, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("frequencies must be a nonempty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("frequencies must be finite")
    if np.any(values <= 0.0):
        raise ValueError("frequencies must be positive")
    return values


def _normalized_weights(weights, n_frequencies: int) -> np.ndarray:
    if weights is None:
        values = np.ones(n_frequencies, dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (n_frequencies,):
            raise ValueError("weights must have the same shape as frequencies")
        if not np.all(np.isfinite(values)):
            raise ValueError("weights must be finite")
        if np.any(values < 0.0):
            raise ValueError("weights must be nonnegative")
        if float(np.sum(values)) <= 0.0:
            raise ValueError("weights must contain a positive total mass")
    return values * float(n_frequencies) / float(np.sum(values))


def _nullspace_polish(
    scaled_design: np.ndarray,
    data: np.ndarray,
    seed: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    fractions = np.asarray(seed, dtype=float)
    free = np.flatnonzero(fractions > _FREE_SET_THRESHOLD)
    if free.size == 0:
        free = np.array([int(np.argmax(fractions))], dtype=int)

    polish_iterations = 0
    for _ in range(scaled_design.shape[1] + 1):
        if free.size == 0:
            free = np.array([int(np.argmax(seed))], dtype=int)
        if free.size == 1:
            refined = np.array([1.0], dtype=float)
        else:
            constraint = np.ones((1, free.size), dtype=float)
            kernel = null_space(constraint)
            feasible = np.ones(free.size, dtype=float) / float(free.size)
            reduced = scaled_design[:, free] @ kernel
            rhs = data - scaled_design[:, free] @ feasible
            coefficients, *_ = np.linalg.lstsq(reduced, rhs, rcond=None)
            refined = feasible + kernel @ coefficients
        trial = np.zeros(scaled_design.shape[1], dtype=float)
        trial[free] = refined
        violators = free[refined < -1.0e-12]
        if violators.size == 0:
            return trial, polish_iterations, True
        free = np.setdiff1d(free, violators, assume_unique=False)
        polish_iterations += 1
    return trial, polish_iterations, False


def _assemble_fit(
    *,
    delta: np.ndarray,
    tau: np.ndarray,
    sigma_infinity: float,
    sigma0_target: float,
    basis: np.ndarray,
    frequencies: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    condition_number: float,
    status: OptimizerStatus,
) -> PassiveDebyeFit:
    fitted = sigma_infinity - basis @ delta
    residual = fitted - target
    weighted_residual = float(np.sum(weights * np.abs(residual) ** 2))
    weighted_target = float(np.sum(weights * np.abs(target) ** 2))
    spectral_error = float(np.sqrt(weighted_residual / weighted_target)) if weighted_target > 0.0 else float(np.sqrt(weighted_residual))
    target_peak = float(np.max(np.abs(target)))
    spectral_error_linf = float(np.max(np.abs(residual)) / target_peak) if target_peak > 0.0 else float(np.max(np.abs(residual)))
    sigma0 = float(sigma_infinity - float(np.sum(delta)))
    dc_error = float(sigma0 - sigma0_target)
    relative_dc_error = float(abs(dc_error) / sigma0_target)
    passive = bool(np.all(delta >= HARD_GATE_DELTA_FLOOR) and sigma0 > 0.0)
    return PassiveDebyeFit(
        delta_sigma=np.asarray(delta, dtype=float),
        tau_grid=np.asarray(tau, dtype=float),
        sigma_infinity=float(sigma_infinity),
        sigma0_target=float(sigma0_target),
        sigma0=sigma0,
        dc_error=dc_error,
        relative_dc_error=relative_dc_error,
        passive=passive,
        spectral_error=spectral_error,
        spectral_error_linf=spectral_error_linf,
        condition_number=float(condition_number),
        optimizer_status=status,
        frequencies=np.asarray(frequencies, dtype=float),
        target_sigma=np.asarray(target, dtype=complex),
        fitted_sigma=np.asarray(fitted, dtype=complex),
        weights=np.asarray(weights, dtype=float),
    )

