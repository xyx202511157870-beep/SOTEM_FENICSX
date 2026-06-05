"""Source-primary magnetic kernel diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.constants import mu_0

from .metrics import relative_l2


@dataclass(frozen=True)
class ExponentialSourcePrimaryFit:
    """Least-squares fit for ``scale * H_source * exp(-t / kernel_tau)``."""

    kernel_tau: float
    scale: float
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    components: dict[str, float]


@dataclass(frozen=True)
class ExponentialSourcePrimaryScan:
    """Fit results for several exponential kernel time constants."""

    fits: list[ExponentialSourcePrimaryFit]

    @property
    def best(self) -> ExponentialSourcePrimaryFit:
        """Return the fit with the smallest aggregate relative L2 error."""

        if not self.fits:
            raise ValueError("scan contains no fits")
        return min(self.fits, key=lambda fit: fit.relative_l2)


@dataclass(frozen=True)
class TimeDependentSourcePrimaryFit:
    """Per-time least-squares source-primary kernel fit."""

    kernel: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    components: dict[str, float]


@dataclass(frozen=True)
class SourceHistoryKernelBasisFit:
    """Least-squares fit of an empirical source-history kernel."""

    tau: float
    powers: tuple[int, ...]
    include_constant: bool
    basis_labels: list[str]
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float


@dataclass(frozen=True)
class DiscreteDebyeHistoryBasis:
    """Backward-Euler Debye relaxation basis on the simulation time grid."""

    tau: float
    times: np.ndarray
    values: np.ndarray
    basis_labels: list[str]


@dataclass(frozen=True)
class DiscreteRelaxationDifferenceBasis:
    """Zero-initial difference of two backward-Euler relaxation histories."""

    slow_tau: float
    fast_tau: float
    times: np.ndarray
    values: np.ndarray
    basis_labels: list[str]


@dataclass(frozen=True)
class DiscreteDrivenRelaxationBasis:
    """Backward-Euler fast response driven by a slow relaxation history."""

    driver_tau: float
    response_tau: float
    times: np.ndarray
    driver_values: np.ndarray
    values: np.ndarray
    basis_labels: list[str]


def fit_exponential_source_primary(
    times,
    target,
    source_amplitudes,
    *,
    kernel_tau: float,
    component_names: Sequence[str] | None = None,
) -> ExponentialSourcePrimaryFit:
    """Fit a scalar source-primary exponential basis to receiver data.

    ``target`` has shape ``(n_times, n_components)`` and is usually a physical
    IP-minus-noIP reference difference or a strict residual.  ``source_amplitudes``
    contains the static source magnetic field component for each receiver column.
    """

    times = _as_time_vector(times)
    target = _as_2d_target(target, times.size)
    source_amplitudes = np.asarray(source_amplitudes, dtype=float)
    if source_amplitudes.shape != (target.shape[1],):
        raise ValueError("source_amplitudes must have one value per target column")
    kernel_tau = float(kernel_tau)
    if kernel_tau <= 0.0:
        raise ValueError("kernel_tau must be positive")

    basis = np.exp(-times[:, None] / kernel_tau) * source_amplitudes[None, :]
    denom = float(np.dot(basis.ravel(), basis.ravel()))
    if denom == 0.0:
        raise ValueError("source-primary basis has zero norm")
    scale = float(np.dot(target.ravel(), basis.ravel()) / denom)
    fitted = scale * basis
    residual = fitted - target
    names = _component_names(component_names, target.shape[1])
    components = {
        name: relative_l2(fitted[:, index], target[:, index])
        for index, name in enumerate(names)
    }
    return ExponentialSourcePrimaryFit(
        kernel_tau=kernel_tau,
        scale=scale,
        fitted=fitted,
        residual=residual,
        relative_l2=relative_l2(fitted, target),
        components=components,
    )


def scan_exponential_source_primary(
    times,
    target,
    source_amplitudes,
    *,
    kernel_taus,
    component_names: Sequence[str] | None = None,
) -> ExponentialSourcePrimaryScan:
    """Fit the same source-primary basis for each candidate kernel time."""

    taus = np.asarray(kernel_taus, dtype=float)
    if taus.ndim != 1 or taus.size == 0:
        raise ValueError("kernel_taus must be a non-empty 1D sequence")
    return ExponentialSourcePrimaryScan(
        [
            fit_exponential_source_primary(
                times,
                target,
                source_amplitudes,
                kernel_tau=float(tau),
                component_names=component_names,
            )
            for tau in taus
        ]
    )


def fit_time_dependent_source_primary_kernel(
    times,
    target,
    source_amplitudes,
    *,
    component_names: Sequence[str] | None = None,
) -> TimeDependentSourcePrimaryFit:
    """Fit an independent common source-primary amplitude at each time.

    The fitted model is ``target[t, :] ~= kernel[t] * source_amplitudes``.  It
    is a non-parametric diagnostic for extracting the empirical source-history
    kernel before assuming an exponential or derived convolution form.
    """

    times = _as_time_vector(times)
    target = _as_2d_target(target, times.size)
    source_amplitudes = np.asarray(source_amplitudes, dtype=float)
    if source_amplitudes.shape != (target.shape[1],):
        raise ValueError("source_amplitudes must have one value per target column")
    denom = float(np.dot(source_amplitudes, source_amplitudes))
    if denom == 0.0:
        raise ValueError("source_amplitudes has zero norm")

    kernel = (target @ source_amplitudes) / denom
    fitted = kernel[:, None] * source_amplitudes[None, :]
    residual = fitted - target
    names = _component_names(component_names, target.shape[1])
    components = {
        name: relative_l2(fitted[:, index], target[:, index])
        for index, name in enumerate(names)
    }
    return TimeDependentSourcePrimaryFit(
        kernel=np.asarray(kernel, dtype=float),
        fitted=fitted,
        residual=residual,
        relative_l2=relative_l2(fitted, target),
        components=components,
    )


def discrete_debye_history_basis(
    time_steps,
    *,
    tau: float,
    max_order: int = 1,
) -> DiscreteDebyeHistoryBasis:
    """Return discrete Debye relaxation/cascade bases at all time nodes.

    Order zero is the homogeneous backward-Euler relaxation of an initial
    Debye memory.  Higher orders are same-pole cascades driven by the already
    updated lower order, which is the discrete counterpart of
    ``(t/tau)^p exp(-t/tau)`` factors on nonuniform time grids.
    """

    steps = _as_time_steps(time_steps)
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")

    times = np.r_[0.0, np.cumsum(steps)]
    values = np.zeros((times.size, max_order + 1), dtype=float)
    values[0, 0] = 1.0
    for step_index, dt in enumerate(steps):
        alpha = tau / (tau + float(dt))
        beta = float(dt) / (tau + float(dt))
        values[step_index + 1, 0] = alpha * values[step_index, 0]
        for order in range(1, max_order + 1):
            values[step_index + 1, order] = (
                alpha * values[step_index, order]
                + beta * values[step_index + 1, order - 1]
            )

    labels = ["BE relaxation"] + [
        f"BE cascade {order}" for order in range(1, max_order + 1)
    ]
    return DiscreteDebyeHistoryBasis(
        tau=tau,
        times=times,
        values=values,
        basis_labels=labels,
    )


def discrete_relaxation_difference_basis(
    time_steps,
    *,
    slow_tau: float,
    fast_tau: float,
) -> DiscreteRelaxationDifferenceBasis:
    """Return ``g_slow - g_fast`` on the backward-Euler time grid.

    Both ``g`` histories are order-zero Debye relaxations.  With
    ``fast_tau < slow_tau`` the column starts at zero, rises as the fast pole
    decays away, then decays on the slow pole.  This is a diagnostic basis for
    source-history/MMR recovery studies, not a production IP law by itself.
    """

    slow_tau = float(slow_tau)
    fast_tau = float(fast_tau)
    if slow_tau <= 0.0:
        raise ValueError("slow_tau must be positive")
    if fast_tau <= 0.0:
        raise ValueError("fast_tau must be positive")
    if fast_tau >= slow_tau:
        raise ValueError("fast_tau must be smaller than slow_tau")

    slow = discrete_debye_history_basis(time_steps, tau=slow_tau, max_order=0)
    fast = discrete_debye_history_basis(time_steps, tau=fast_tau, max_order=0)
    return DiscreteRelaxationDifferenceBasis(
        slow_tau=slow_tau,
        fast_tau=fast_tau,
        times=slow.times,
        values=np.asarray(slow.values[:, 0] - fast.values[:, 0], dtype=float),
        basis_labels=["BE relaxation difference slow-fast"],
    )


def discrete_driven_relaxation_basis(
    time_steps,
    *,
    driver_tau: float,
    response_tau: float,
) -> DiscreteDrivenRelaxationBasis:
    """Return a zero-initial fast state driven by slow Debye relaxation.

    The discrete equation is

    ``z^{n+1} = a_f z^n + b_f g_slow^{n+1}``,

    where ``a_f = response_tau/(response_tau + dt_n)`` and
    ``b_f = dt_n/(response_tau + dt_n)``.  On uniform time steps this is
    proportional to ``g_slow - g_fast``; writing it as a driven state makes the
    source-recovery convolution interpretation explicit.
    """

    steps = _as_time_steps(time_steps)
    driver_tau = float(driver_tau)
    response_tau = float(response_tau)
    if driver_tau <= 0.0:
        raise ValueError("driver_tau must be positive")
    if response_tau <= 0.0:
        raise ValueError("response_tau must be positive")
    if response_tau >= driver_tau:
        raise ValueError("response_tau must be smaller than driver_tau")

    driver = discrete_debye_history_basis(
        steps,
        tau=driver_tau,
        max_order=0,
    )
    values = np.zeros(driver.times.size, dtype=float)
    for step_index, dt in enumerate(steps):
        alpha = response_tau / (response_tau + float(dt))
        beta = float(dt) / (response_tau + float(dt))
        values[step_index + 1] = (
            alpha * values[step_index]
            + beta * driver.values[step_index + 1, 0]
        )

    return DiscreteDrivenRelaxationBasis(
        driver_tau=driver_tau,
        response_tau=response_tau,
        times=driver.times,
        driver_values=np.asarray(driver.values[:, 0], dtype=float),
        values=values,
        basis_labels=["BE driven relaxation response"],
    )


def fit_source_history_kernel_discrete_debye_basis(
    time_steps,
    sample_times,
    kernel,
    *,
    tau: float,
    max_order: int = 1,
    include_constant: bool = False,
    time_atol: float = 1.0e-12,
) -> SourceHistoryKernelBasisFit:
    """Fit ``K(t)`` to discrete backward-Euler Debye history bases."""

    sample_times = _as_time_vector(sample_times)
    kernel = _as_kernel_vector(kernel, sample_times.size)
    basis = discrete_debye_history_basis(time_steps, tau=tau, max_order=max_order)
    indices = _time_node_indices(basis.times, sample_times, atol=time_atol)
    design = basis.values[indices]
    labels = list(basis.basis_labels)
    if include_constant:
        design = np.column_stack([np.ones(sample_times.size, dtype=float), design])
        labels = ["constant"] + labels

    coefficients, _, _, _ = np.linalg.lstsq(design, kernel, rcond=None)
    fitted = design @ coefficients
    residual = fitted - kernel
    return SourceHistoryKernelBasisFit(
        tau=float(tau),
        powers=tuple(range(int(max_order) + 1)),
        include_constant=bool(include_constant),
        basis_labels=labels,
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, kernel),
    )


def fit_source_history_kernel_basis(
    times,
    kernel,
    *,
    tau: float,
    powers: Sequence[int] = (0, 1),
    include_constant: bool = False,
) -> SourceHistoryKernelBasisFit:
    """Fit ``K(t)`` to a Debye-time exponential polynomial basis.

    The fitted basis is optionally a constant plus
    ``(t/tau)**p * exp(-t/tau)`` for each integer power in ``powers``.
    This is a diagnostic reduction of the non-parametric empirical kernel; it
    is not itself the production source-history formula.
    """

    times = _as_time_vector(times)
    kernel = _as_kernel_vector(kernel, times.size)
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    power_values = _basis_powers(powers)
    if not include_constant and not power_values:
        raise ValueError("at least one basis function is required")

    u = times / tau
    columns = []
    labels: list[str] = []
    if include_constant:
        columns.append(np.ones_like(times, dtype=float))
        labels.append("constant")
    exp_u = np.exp(-u)
    for power in power_values:
        if power == 0:
            columns.append(exp_u)
            labels.append("exp(-t/tau)")
        else:
            columns.append((u**power) * exp_u)
            labels.append(f"(t/tau)^{power} exp(-t/tau)")

    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, kernel, rcond=None)
    fitted = design @ coefficients
    residual = fitted - kernel
    return SourceHistoryKernelBasisFit(
        tau=tau,
        powers=tuple(power_values),
        include_constant=bool(include_constant),
        basis_labels=labels,
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, kernel),
    )


def normalized_source_primary_scale(
    scale: float,
    *,
    delta_sigma: float,
    source_length: float,
    mu: float = mu_0,
) -> float:
    """Return ``scale / (mu * delta_sigma * source_length**2)``."""

    scale = float(scale)
    delta_sigma = float(delta_sigma)
    source_length = float(source_length)
    mu = float(mu)
    if delta_sigma <= 0.0:
        raise ValueError("delta_sigma must be positive")
    if source_length <= 0.0:
        raise ValueError("source_length must be positive")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    return scale / (mu * delta_sigma * source_length**2)


def _as_time_vector(times) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty 1D sequence")
    if np.any(values < 0.0):
        raise ValueError("times must be nonnegative")
    return values


def _as_time_steps(time_steps) -> np.ndarray:
    values = np.asarray(time_steps, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("time_steps must be a non-empty 1D sequence")
    if np.any(values <= 0.0):
        raise ValueError("time_steps must be positive")
    return values


def _as_2d_target(target, n_times: int) -> np.ndarray:
    values = np.asarray(target, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("target must have shape (n_times, n_components)")
    if values.shape[0] != n_times:
        raise ValueError("target row count must match times")
    return values


def _time_node_indices(times: np.ndarray, sample_times: np.ndarray, *, atol: float) -> np.ndarray:
    if atol < 0.0:
        raise ValueError("time_atol must be nonnegative")
    indices = []
    for sample_time in sample_times:
        matches = np.flatnonzero(np.isclose(times, sample_time, rtol=1.0e-9, atol=atol))
        if matches.size == 0:
            raise ValueError(f"sample time {sample_time:g} is not on the simulation time grid")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def _as_kernel_vector(kernel, n_times: int) -> np.ndarray:
    values = np.asarray(kernel, dtype=float)
    if values.ndim != 1:
        raise ValueError("kernel must be a 1D sequence")
    if values.shape != (n_times,):
        raise ValueError("kernel length must match times")
    return values


def _basis_powers(powers: Sequence[int]) -> list[int]:
    values = []
    for power in powers:
        value = int(power)
        if value != power or value < 0:
            raise ValueError("powers must contain nonnegative integers")
        values.append(value)
    return values


def _component_names(names: Sequence[str] | None, n_components: int) -> list[str]:
    if names is None:
        return [f"component_{index}" for index in range(n_components)]
    values = [str(name) for name in names]
    if len(values) != n_components:
        raise ValueError("component_names length must match the number of columns")
    return values
