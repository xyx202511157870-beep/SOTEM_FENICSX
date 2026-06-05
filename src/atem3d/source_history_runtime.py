"""Runtime source-history magnetic correction hooks.

The objects in this module intentionally do not fit coefficients.  They only
evaluate coefficients supplied by a derivation or diagnostic report against the
same discrete BE history and FV/MMR receiver matrices used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import mu_0

from .local_coupling import source_edge_moment_basis, source_face_moment_basis
from .magnetic_recovery import (
    cell_current_biot_matrix,
    edge_basis_biot_matrix,
    edge_current_biot_matrix,
    face_basis_biot_matrix,
    face_current_biot_matrix,
)
from .source_history_operator import project_vector_to_spatial_basis
from .source_primary import (
    discrete_debye_history_basis,
    discrete_driven_relaxation_basis,
)


def _coefficient_tuple(
    value: Sequence[float],
    expected: int,
    message: str,
) -> tuple[float, ...]:
    coefficients = tuple(float(item) for item in value)
    if len(coefficients) != expected:
        raise ValueError(message)
    return coefficients


@dataclass(frozen=True)
class SourceHistoryCorrection:
    """Prescribed source-history coefficients for magnetic receiver recovery."""

    tau: float
    coefficients: Sequence[float] | None = None
    normalized_coefficients: Sequence[float] | None = None
    max_order: int = 1
    source_moment_degrees: Sequence[int] = (0, 2)
    receiver_matrix: str = "auto"
    kind: str = "prescribed_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "prescribed_source_moments":
            raise ValueError("source-history correction kind must be 'prescribed_source_moments'")
        tau = float(self.tau)
        if tau <= 0.0:
            raise ValueError("source-history correction tau must be positive")
        max_order = int(self.max_order)
        if max_order < 0:
            raise ValueError("source-history correction max_order must be nonnegative")
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")
        expected = (max_order + 1) * len(degrees)
        has_coefficients = self.coefficients is not None
        has_normalized = self.normalized_coefficients is not None
        if has_coefficients == has_normalized:
            raise ValueError(
                "prescribed source-history correction requires exactly one of "
                "coefficients or normalized_coefficients"
            )
        coefficients = (
            _coefficient_tuple(
                self.coefficients,
                expected,
                "source-history coefficient count must equal "
                "(max_order + 1) * len(source_moment_degrees)",
            )
            if has_coefficients
            else None
        )
        normalized_coefficients = (
            _coefficient_tuple(
                self.normalized_coefficients,
                expected,
                "source-history normalized coefficient count must equal "
                "(max_order + 1) * len(source_moment_degrees)",
            )
            if has_normalized
            else None
        )
        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "source-history receiver_matrix must be 'auto', 'current_biot', "
                "'edge_current', 'edge_basis', 'face_current', or 'face_basis'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "max_order", max_order)
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "normalized_coefficients", normalized_coefficients)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class InitialPolarizationSourceHistoryCorrection:
    """Project the initial Debye polarization current onto source moments.

    This is a non-fitted diagnostic candidate for the missing source-primary
    magnetic history.  For each Debye term it builds the initial polarization
    current ``-delta_sigma * y0``, projects it onto the selected source-moment
    vectors, and advances that source-like contribution with the same BE
    relaxation factor as the Debye memory.
    """

    source_moment_degrees: Sequence[int] = (0, 2)
    receiver_matrix: str = "auto"
    projection: str = "receiver_l2"
    kind: str = "initial_polarization_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "initial_polarization_source_moments":
            raise ValueError(
                "initial-polarization correction kind must be "
                "'initial_polarization_source_moments'"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")
        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "source-history receiver_matrix must be 'auto', 'current_biot', "
                "'edge_current', 'edge_basis', 'face_current', or 'face_basis'"
            )
        projection = str(self.projection).strip().lower()
        if projection not in {"receiver_l2", "dof_l2"}:
            raise ValueError(
                "initial-polarization source-history projection must be "
                "'receiver_l2' or 'dof_l2'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class ChargeConservingInitialPolarizationSourceHistoryCorrection:
    """Project initial polarization current through a charge-conserving FV solve.

    The raw initial Debye current ``-delta_sigma * y0`` is first projected with
    the discrete continuity equation ``D(j_p - M_rho^-1 G phi) = 0``.  The
    resulting solenoidal face current is then mapped onto source moments and
    advanced with the Debye relaxation.  This is a non-fitted diagnostic
    candidate for H/J face-current recovery.
    """

    source_moment_degrees: Sequence[int] = (0, 2)
    receiver_matrix: str = "auto"
    projection: str = "receiver_l2"
    kind: str = "charge_conserving_initial_polarization_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "charge_conserving_initial_polarization_source_moments":
            raise ValueError(
                "charge-conserving correction kind must be "
                "'charge_conserving_initial_polarization_source_moments'"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")
        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "charge-conserving source-history receiver_matrix must be "
                "'auto', 'current_biot', 'face_current', or 'face_basis'"
            )
        projection = str(self.projection).strip().lower()
        if projection not in {"receiver_l2", "dof_l2"}:
            raise ValueError(
                "charge-conserving source-history projection must be "
                "'receiver_l2' or 'dof_l2'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class DrivenRecoverySourceHistoryCorrection:
    """Prescribed source moments with a driven MMR recovery history.

    This diagnostic term evaluates one zero-initial recovery state driven by a
    Debye source-memory relaxation.  The coefficients are supplied externally;
    this class only provides the runtime path for a derived local operator.
    """

    driver_tau: float
    response_tau: float | Sequence[float]
    coefficients: Sequence[float] | None = None
    normalized_coefficients: Sequence[float] | None = None
    source_moment_degrees: Sequence[int] = (0, 2)
    receiver_matrix: str = "auto"
    kind: str = "driven_recovery_source_moments"
    source_edge_atol: float = 0.0

    @property
    def response_taus(self) -> tuple[float, ...]:
        value = self.response_tau
        if isinstance(value, tuple):
            return value
        return (float(value),)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "driven_recovery_source_moments":
            raise ValueError(
                "driven-recovery correction kind must be "
                "'driven_recovery_source_moments'"
        )
        driver_tau = float(self.driver_tau)
        response_taus = _as_tau_sequence(
            self.response_tau,
            "driven-recovery response_tau",
        )
        if driver_tau <= 0.0:
            raise ValueError("driven-recovery driver_tau must be positive")
        if any(response_tau >= driver_tau for response_tau in response_taus):
            raise ValueError(
                "driven-recovery response_tau must be smaller than driver_tau"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")
        expected = len(response_taus) * len(degrees)
        has_coefficients = self.coefficients is not None
        has_normalized = self.normalized_coefficients is not None
        if has_coefficients == has_normalized:
            raise ValueError(
                "driven-recovery source-history correction requires exactly one of "
                "coefficients or normalized_coefficients"
            )
        coefficients = (
            _coefficient_tuple(
                self.coefficients,
                expected,
                "driven-recovery coefficient count must equal "
                "len(response_taus) * len(source_moment_degrees)",
            )
            if has_coefficients
            else None
        )
        normalized_coefficients = (
            _coefficient_tuple(
                self.normalized_coefficients,
                expected,
                "driven-recovery normalized coefficient count must equal "
                "len(response_taus) * len(source_moment_degrees)",
            )
            if has_normalized
            else None
        )
        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "source-history receiver_matrix must be 'auto', 'current_biot', "
                "'edge_current', 'edge_basis', 'face_current', or 'face_basis'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "driver_tau", driver_tau)
        object.__setattr__(
            self,
            "response_tau",
            response_taus[0] if len(response_taus) == 1 else response_taus,
        )
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "normalized_coefficients", normalized_coefficients)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class SourcePrimaryDelta6SourceHistoryCorrection:
    """Delta6 source-primary diagnostic expressed as face source moments.

    This is the source-history equivalent of the H/J
    ``magnetic_recovery_source_primary_delta6_basis='face_current'`` diagnostic.
    It keeps the same non-fitted test kernel, but routes the spatial part
    through the FV/MMR source-moment operator.
    """

    source_moment_degrees: Sequence[int] = (0,)
    receiver_matrix: str = "auto"
    kind: str = "source_primary_delta6_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "source_primary_delta6_source_moments":
            raise ValueError(
                "delta6 source-history correction kind must be "
                "'source_primary_delta6_source_moments'"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if 0 not in degrees:
            raise ValueError("delta6 source-history correction requires degree 0")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")
        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "delta6 source-history receiver_matrix must be 'auto', "
                "'current_biot', 'face_current', or 'face_basis'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class TimeSeriesSourceHistoryCorrection:
    """Explicit time-node source-moment coefficients for diagnostics."""

    times: Sequence[float]
    coefficients: Sequence[Sequence[float]]
    source_moment_degrees: Sequence[int] = (0, 2)
    receiver_matrix: str = "auto"
    kind: str = "time_series_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "time_series_source_moments":
            raise ValueError(
                "time-series source-history correction kind must be "
                "'time_series_source_moments'"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")

        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("time-series source-history times must be a nonempty vector")
        if not np.all(np.isfinite(times)):
            raise ValueError("time-series source-history times must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time-series source-history times must be strictly increasing")

        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim == 1 and len(degrees) == 1:
            coefficients = coefficients.reshape(-1, 1)
        if coefficients.shape != (times.size, len(degrees)):
            raise ValueError(
                "time-series source-history coefficients must have shape "
                "(len(times), len(source_moment_degrees))"
            )

        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "source-history receiver_matrix must be 'auto', 'current_biot', "
                "'edge_current', 'edge_basis', 'face_current', or 'face_basis'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "times", tuple(float(value) for value in times))
        object.__setattr__(
            self,
            "coefficients",
            tuple(tuple(float(value) for value in row) for row in coefficients),
        )
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


@dataclass(frozen=True)
class SourceDiffusionKernelSourceHistoryCorrection:
    """Exponential source-diffusion source-moment diagnostic.

    The time constant is computed from the source midpoint conductivity as
    ``tau = tau_multiplier * mu * sigma * L**2``.  This is a diagnostic replay
    path for source-neighborhood audits; amplitudes must come from a derivation
    or an explicitly documented diagnostic fit.
    """

    amplitude: float = 0.0
    normalized_amplitude: float | None = None
    tau_multiplier: float = 1.0
    amplitude_time: float = 0.0
    basis_kind: str = "continuous"
    source_moment_degrees: Sequence[int] = (0,)
    coefficients: Sequence[float] | None = None
    receiver_matrix: str = "auto"
    kind: str = "source_diffusion_kernel_source_moments"
    source_edge_atol: float = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "source_diffusion_kernel_source_moments":
            raise ValueError(
                "source-diffusion correction kind must be "
                "'source_diffusion_kernel_source_moments'"
            )
        amplitude = float(self.amplitude)
        if not np.isfinite(amplitude):
            raise ValueError("source-diffusion amplitude must be finite")
        normalized_amplitude = (
            None
            if self.normalized_amplitude is None
            else float(self.normalized_amplitude)
        )
        if normalized_amplitude is not None and not np.isfinite(normalized_amplitude):
            raise ValueError("source-diffusion normalized_amplitude must be finite")
        if normalized_amplitude is not None and self.coefficients is not None:
            raise ValueError(
                "source-diffusion normalized_amplitude cannot be combined with "
                "absolute coefficients"
            )
        tau_multiplier = float(self.tau_multiplier)
        if tau_multiplier <= 0.0:
            raise ValueError("source-diffusion tau_multiplier must be positive")
        amplitude_time = float(self.amplitude_time)
        if not np.isfinite(amplitude_time):
            raise ValueError("source-diffusion amplitude_time must be finite")
        basis_kind = str(self.basis_kind).strip().lower()
        if basis_kind not in {"continuous", "be_decay"}:
            raise ValueError(
                "source-diffusion basis_kind must be 'continuous' or 'be_decay'"
            )
        degrees = tuple(int(degree) for degree in self.source_moment_degrees)
        if not degrees:
            raise ValueError("source_moment_degrees must contain at least one degree")
        if any(degree < 0 for degree in degrees):
            raise ValueError("source_moment_degrees must be nonnegative")

        if normalized_amplitude is not None:
            if len(degrees) > 1 and 0 not in degrees:
                raise ValueError(
                    "source-diffusion normalized_amplitude requires degree 0 "
                    "when multiple source moments are requested"
                )
            coefficients = None
        elif self.coefficients is None:
            coefficients = np.zeros(len(degrees), dtype=float)
            if len(degrees) == 1:
                coefficients[0] = amplitude
            elif 0 in degrees:
                coefficients[degrees.index(0)] = amplitude
            else:
                raise ValueError(
                    "source-diffusion scalar amplitude requires degree 0 when "
                    "multiple source moments are requested"
                )
        else:
            coefficients = np.asarray(self.coefficients, dtype=float)
            if coefficients.ndim != 1 or coefficients.size != len(degrees):
                raise ValueError(
                    "source-diffusion coefficients must have shape "
                    "(len(source_moment_degrees),)"
                )
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("source-diffusion coefficients must be finite")

        receiver_matrix = str(self.receiver_matrix).strip().lower()
        if receiver_matrix not in {
            "auto",
            "current_biot",
            "edge_current",
            "edge_basis",
            "face_current",
            "face_basis",
        }:
            raise ValueError(
                "source-history receiver_matrix must be 'auto', 'current_biot', "
                "'edge_current', 'edge_basis', 'face_current', or 'face_basis'"
            )
        source_edge_atol = float(self.source_edge_atol)
        if source_edge_atol < 0.0:
            raise ValueError("source_edge_atol must be nonnegative")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "normalized_amplitude", normalized_amplitude)
        object.__setattr__(self, "tau_multiplier", tau_multiplier)
        object.__setattr__(self, "amplitude_time", amplitude_time)
        object.__setattr__(self, "basis_kind", basis_kind)
        object.__setattr__(self, "source_moment_degrees", degrees)
        object.__setattr__(
            self,
            "coefficients",
            (
                None
                if coefficients is None
                else tuple(float(value) for value in coefficients)
            ),
        )
        object.__setattr__(self, "receiver_matrix", receiver_matrix)
        object.__setattr__(self, "source_edge_atol", source_edge_atol)


SourceHistoryCorrectionTerm = (
    SourceHistoryCorrection
    | InitialPolarizationSourceHistoryCorrection
    | ChargeConservingInitialPolarizationSourceHistoryCorrection
    | DrivenRecoverySourceHistoryCorrection
    | SourcePrimaryDelta6SourceHistoryCorrection
    | TimeSeriesSourceHistoryCorrection
    | SourceDiffusionKernelSourceHistoryCorrection
)


def source_history_correction_terms(value) -> tuple[SourceHistoryCorrectionTerm, ...]:
    """Normalize an optional source-history correction value to a tuple."""

    if value is None:
        return ()
    if isinstance(
        value,
        (
            SourceHistoryCorrection,
            InitialPolarizationSourceHistoryCorrection,
            ChargeConservingInitialPolarizationSourceHistoryCorrection,
            DrivenRecoverySourceHistoryCorrection,
            SourcePrimaryDelta6SourceHistoryCorrection,
            TimeSeriesSourceHistoryCorrection,
            SourceDiffusionKernelSourceHistoryCorrection,
        ),
    ):
        return (value,)
    terms = tuple(value)
    if not terms:
        raise ValueError("source-history correction terms must not be empty")
    if not all(
        isinstance(
            term,
            (
                SourceHistoryCorrection,
                InitialPolarizationSourceHistoryCorrection,
                ChargeConservingInitialPolarizationSourceHistoryCorrection,
                DrivenRecoverySourceHistoryCorrection,
                SourcePrimaryDelta6SourceHistoryCorrection,
                TimeSeriesSourceHistoryCorrection,
                SourceDiffusionKernelSourceHistoryCorrection,
            ),
        )
        for term in terms
    ):
        raise ValueError("source-history correction terms must be source-history corrections")
    return terms


def source_history_correction_requires_ip(correction: SourceHistoryCorrectionTerm) -> bool:
    """Return whether a correction should vanish for purely Ohmic models."""

    return not isinstance(correction, SourceDiffusionKernelSourceHistoryCorrection)


def source_history_ip_model_has_contrast(ip_model) -> bool:
    """Return whether an IP model contains nonzero Debye polarization contrast."""

    for term in getattr(ip_model, "terms", ()):
        delta = np.asarray(term.delta_sigma, dtype=float)
        if np.any(delta > 0.0):
            return True
    return False


def source_history_correction_field(
    mesh,
    sources,
    time_steps,
    time: float,
    locations,
    *,
    correction: SourceHistoryCorrectionTerm,
    magnetic_receiver_mode: str,
    subdivisions: int = 1,
    field_location: str = "edge",
    ip_model=None,
    initial_ip_model=None,
    initial_memories=None,
    mu: float = mu_0,
    cache: dict | None = None,
) -> np.ndarray:
    """Evaluate the prescribed source-history correction at receiver locations."""

    locations = _as_locations(locations)
    field_location = _normalize_field_location(field_location)
    if isinstance(correction, InitialPolarizationSourceHistoryCorrection):
        return _initial_polarization_source_history_field(
            mesh,
            sources,
            time_steps,
            time,
            locations,
            correction=correction,
            magnetic_receiver_mode=magnetic_receiver_mode,
            subdivisions=subdivisions,
            field_location=field_location,
            ip_model=ip_model,
            initial_memories=initial_memories,
            cache=cache,
        )
    if isinstance(correction, ChargeConservingInitialPolarizationSourceHistoryCorrection):
        return _charge_conserving_initial_polarization_source_history_field(
            mesh,
            sources,
            time_steps,
            time,
            locations,
            correction=correction,
            magnetic_receiver_mode=magnetic_receiver_mode,
            subdivisions=subdivisions,
            field_location=field_location,
            ip_model=ip_model,
            initial_memories=initial_memories,
            cache=cache,
        )
    if isinstance(correction, SourcePrimaryDelta6SourceHistoryCorrection):
        return _source_primary_delta6_source_history_field(
            mesh,
            sources,
            time,
            locations,
            correction=correction,
            magnetic_receiver_mode=magnetic_receiver_mode,
            subdivisions=subdivisions,
            field_location=field_location,
            ip_model=ip_model,
            mu=mu,
            cache=cache,
        )

    if len(sources) != 1:
        raise ValueError("source-history source_moments correction requires exactly one source")
    source = sources[0]
    cached = _cached_static_response(
        mesh,
        source,
        locations,
        correction=correction,
        magnetic_receiver_mode=magnetic_receiver_mode,
        subdivisions=subdivisions,
        field_location=field_location,
        cache=cache,
    )
    if cached is None:
        return np.zeros((locations.shape[0], 3), dtype=float)
    _, _, static_response = cached

    if isinstance(correction, TimeSeriesSourceHistoryCorrection):
        time_index = _time_node_index(np.asarray(correction.times, dtype=float), float(time))
        coefficients = np.asarray(correction.coefficients, dtype=float)
        return np.einsum(
            "s,slc->lc",
            coefficients[time_index],
            static_response,
        )

    if isinstance(correction, SourceDiffusionKernelSourceHistoryCorrection):
        tau0 = _source_diffusion_time(
            mesh,
            source,
            initial_ip_model if initial_ip_model is not None else ip_model,
            mu,
        )
        tau = correction.tau_multiplier * tau0
        kernel = _source_diffusion_kernel_value(
            correction,
            time_steps,
            time=float(time),
            tau=tau,
        )
        coefficients = kernel * _source_diffusion_coefficients(correction, tau0)
        return np.einsum("s,slc->lc", coefficients, static_response)

    if isinstance(correction, DrivenRecoverySourceHistoryCorrection):
        response_taus = correction.response_taus
        coefficients = _source_history_runtime_coefficients(
            correction,
            mesh,
            ip_model,
            source,
            mu,
        ).reshape(
            len(response_taus),
            len(correction.source_moment_degrees),
        )
        field = np.zeros((locations.shape[0], 3), dtype=float)
        for mode_index, response_tau in enumerate(response_taus):
            history = discrete_driven_relaxation_basis(
                time_steps,
                driver_tau=correction.driver_tau,
                response_tau=response_tau,
            )
            time_index = _time_node_index(history.times, float(time))
            field += history.values[time_index] * np.einsum(
                "s,slc->lc",
                coefficients[mode_index],
                static_response,
            )
        return field

    history = discrete_debye_history_basis(
        time_steps,
        tau=correction.tau,
        max_order=correction.max_order,
    )
    time_index = _time_node_index(history.times, float(time))
    coefficients = _source_history_runtime_coefficients(
        correction,
        mesh,
        ip_model,
        source,
        mu,
    ).reshape(
        correction.max_order + 1,
        len(correction.source_moment_degrees),
    )
    return np.einsum(
        "ps,p,slc->lc",
        coefficients,
        history.values[time_index],
        static_response,
    )


def _initial_polarization_source_history_field(
    mesh,
    sources,
    time_steps,
    time: float,
    locations: np.ndarray,
    *,
    correction: InitialPolarizationSourceHistoryCorrection,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
    ip_model,
    initial_memories,
    cache: dict | None,
) -> np.ndarray:
    field = np.zeros((locations.shape[0], 3), dtype=float)
    if ip_model is None:
        raise ValueError("initial-polarization source history requires ip_model")
    if initial_memories is None:
        raise ValueError("initial-polarization source history requires initial_memories")
    if not getattr(ip_model, "terms", []):
        return field
    initial_memories = [np.asarray(memory, dtype=float) for memory in initial_memories]
    if len(initial_memories) != len(ip_model.terms):
        raise ValueError("one initial memory vector is required for each Debye term")
    if len(sources) != 1:
        raise ValueError("source-history source_moments correction requires exactly one source")

    source = sources[0]
    cached = _cached_static_response(
        mesh,
        source,
        locations,
        correction=correction,
        magnetic_receiver_mode=magnetic_receiver_mode,
        subdivisions=subdivisions,
        field_location=field_location,
        cache=cache,
    )
    if cached is None:
        return field
    moments, receiver_matrix, static_response = cached

    for term, initial_memory in zip(ip_model.terms, initial_memories):
        polarization = _initial_polarization_current_vector(
            mesh,
            term.delta_sigma,
            initial_memory,
            field_location,
        )
        coefficients = project_vector_to_spatial_basis(
            receiver_matrix,
            moments.basis_vectors,
            -polarization,
            static_response=static_response,
            projection=correction.projection,
        )
        history = discrete_debye_history_basis(
            time_steps,
            tau=float(term.tau),
            max_order=0,
        )
        time_index = _time_node_index(history.times, float(time))
        field += history.values[time_index, 0] * np.einsum(
            "s,slc->lc",
            coefficients,
            static_response,
        )
    return field


def _source_primary_delta6_source_history_field(
    mesh,
    sources,
    time: float,
    locations: np.ndarray,
    *,
    correction: SourcePrimaryDelta6SourceHistoryCorrection,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
    ip_model,
    mu: float,
    cache: dict | None,
) -> np.ndarray:
    field = np.zeros((locations.shape[0], 3), dtype=float)
    if field_location != "face":
        raise ValueError("delta6 source-history correction requires field_location='face'")
    if ip_model is None:
        raise ValueError("delta6 source-history correction requires ip_model")
    if time <= 0.0 or not getattr(ip_model, "terms", []):
        return field
    if len(sources) != 1:
        raise ValueError("source-history source_moments correction requires exactly one source")

    source = sources[0]
    cached = _cached_static_response(
        mesh,
        source,
        locations,
        correction=correction,
        magnetic_receiver_mode=magnetic_receiver_mode,
        subdivisions=subdivisions,
        field_location=field_location,
        cache=cache,
    )
    if cached is None:
        return field
    _, _, static_response = cached

    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    if length == 0.0:
        return field
    kernel = 0.0
    for term in ip_model.terms:
        delta_sigma = _source_delta_sigma(mesh, term.delta_sigma, source)
        if delta_sigma > 0.0:
            kernel += (
                -6.0
                * float(mu)
                * delta_sigma
                * length**2
                * np.exp(-float(time) / (2.0 * float(term.tau)))
            )
    if kernel == 0.0:
        return field

    coefficients = np.zeros(len(correction.source_moment_degrees), dtype=float)
    coefficients[correction.source_moment_degrees.index(0)] = kernel
    return np.einsum("s,slc->lc", coefficients, static_response)


def _charge_conserving_initial_polarization_source_history_field(
    mesh,
    sources,
    time_steps,
    time: float,
    locations: np.ndarray,
    *,
    correction: ChargeConservingInitialPolarizationSourceHistoryCorrection,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
    ip_model,
    initial_memories,
    cache: dict | None,
) -> np.ndarray:
    field = np.zeros((locations.shape[0], 3), dtype=float)
    if field_location != "face":
        raise ValueError(
            "charge-conserving initial-polarization source history requires "
            "field_location='face'"
        )
    if ip_model is None:
        raise ValueError("charge-conserving source history requires ip_model")
    if initial_memories is None:
        raise ValueError("charge-conserving source history requires initial_memories")
    if not getattr(ip_model, "terms", []):
        return field
    initial_memories = [np.asarray(memory, dtype=float) for memory in initial_memories]
    if len(initial_memories) != len(ip_model.terms):
        raise ValueError("one initial memory vector is required for each Debye term")
    if len(sources) != 1:
        raise ValueError("source-history source_moments correction requires exactly one source")

    source = sources[0]
    cached = _cached_static_response(
        mesh,
        source,
        locations,
        correction=correction,
        magnetic_receiver_mode=magnetic_receiver_mode,
        subdivisions=subdivisions,
        field_location=field_location,
        cache=cache,
    )
    if cached is None:
        return field
    moments, receiver_matrix, static_response = cached

    coefficient_vectors = _charge_conserving_coefficient_vectors(
        mesh,
        source,
        locations,
        correction=correction,
        magnetic_receiver_mode=magnetic_receiver_mode,
        subdivisions=subdivisions,
        field_location=field_location,
        ip_model=ip_model,
        initial_memories=initial_memories,
        moments=moments,
        receiver_matrix=receiver_matrix,
        static_response=static_response,
        cache=cache,
    )

    for term, coefficients in zip(ip_model.terms, coefficient_vectors):
        history = discrete_debye_history_basis(
            time_steps,
            tau=float(term.tau),
            max_order=0,
        )
        time_index = _time_node_index(history.times, float(time))
        field += history.values[time_index, 0] * np.einsum(
            "s,slc->lc",
            coefficients,
            static_response,
        )
    return field


def _charge_conserving_coefficient_vectors(
    mesh,
    source,
    locations: np.ndarray,
    *,
    correction: ChargeConservingInitialPolarizationSourceHistoryCorrection,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
    ip_model,
    initial_memories: list[np.ndarray],
    moments,
    receiver_matrix: np.ndarray,
    static_response: np.ndarray,
    cache: dict | None,
) -> tuple[np.ndarray, ...]:
    key = None
    if cache is not None:
        location_values = np.ascontiguousarray(locations, dtype=float)
        key = (
            "charge_conserving_initial_polarization_coefficients",
            id(mesh),
            id(source),
            id(correction),
            id(ip_model),
            tuple(id(memory) for memory in initial_memories),
            field_location,
            str(magnetic_receiver_mode).strip().lower(),
            int(subdivisions),
            location_values.shape,
            location_values.tobytes(),
        )
        if key in cache:
            return cache[key]

    coefficients = []
    for term, initial_memory in zip(ip_model.terms, initial_memories):
        polarization = _initial_polarization_current_vector(
            mesh,
            term.delta_sigma,
            initial_memory,
            field_location,
        )
        projected_current = charge_conserving_face_current(
            mesh,
            ip_model.sigma_infinity,
            -polarization,
        )
        coefficients.append(
            project_vector_to_spatial_basis(
                receiver_matrix,
                moments.basis_vectors,
                projected_current,
                static_response=static_response,
                projection=correction.projection,
            )
        )
    value = tuple(np.asarray(item, dtype=float) for item in coefficients)
    if key is not None:
        cache[key] = value
    return value


def charge_conserving_face_current(
    mesh,
    conductivity,
    current: np.ndarray,
    *,
    divergence_atol: float = 1.0e-12,
) -> np.ndarray:
    """Return the sigma-weighted solenoidal projection of a face current."""

    current = np.asarray(current, dtype=float)
    if current.shape != (mesh.n_faces,):
        raise ValueError("current must have shape (mesh.n_faces,)")
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    rhs = np.asarray(divergence @ current, dtype=float)
    if np.linalg.norm(rhs) <= float(divergence_atol):
        return current.copy()

    face_rho = _face_resistivity_inner_product(mesh, conductivity)
    face_rho_inverse = _inverse_diagonal_matrix(face_rho, "face resistivity mass")
    gradient = divergence.T.tocsr()
    matrix = (divergence @ face_rho_inverse @ gradient).tocsr()
    phi = spla.lsmr(matrix, rhs, atol=1.0e-12, btol=1.0e-12)[0]
    gradient_current = np.asarray(face_rho_inverse @ (gradient @ phi), dtype=float)
    return np.asarray(current - gradient_current, dtype=float)


def _cached_static_response(
    mesh,
    source,
    locations: np.ndarray,
    *,
    correction: SourceHistoryCorrectionTerm,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
    cache: dict | None,
):
    key = None
    if cache is not None:
        location_values = np.ascontiguousarray(locations, dtype=float)
        key = (
            id(mesh),
            id(source),
            id(correction),
            field_location,
            str(magnetic_receiver_mode).strip().lower(),
            int(subdivisions),
            location_values.shape,
            location_values.tobytes(),
        )
        if key in cache:
            return cache[key]

    source_vector = _source_vector(mesh, source, field_location)
    if np.linalg.norm(source_vector) == 0.0:
        value = None
    else:
        moments = _source_moment_basis(
            mesh,
            source,
            source_vector,
            correction,
            field_location,
        )
        receiver_matrix = _receiver_matrix(
            mesh,
            locations,
            correction=correction,
            magnetic_receiver_mode=magnetic_receiver_mode,
            subdivisions=subdivisions,
            field_location=field_location,
        )
        static_response = np.einsum(
            "lci,si->slc",
            receiver_matrix,
            moments.basis_vectors,
        )
        value = (moments, receiver_matrix, static_response)

    if key is not None:
        cache[key] = value
    return value


def _initial_polarization_current_vector(
    mesh,
    delta_sigma,
    initial_memory: np.ndarray,
    field_location: str,
) -> np.ndarray:
    initial_memory = np.asarray(initial_memory, dtype=float)
    if field_location == "edge":
        if initial_memory.shape != (mesh.n_edges,):
            raise ValueError("edge initial memory must have length mesh.n_edges")
        delta = np.asarray(delta_sigma, dtype=float)
        if delta.shape == (mesh.n_edges,):
            return delta * initial_memory
        return np.asarray(
            mesh.get_edge_inner_product(
                _cell_property(mesh, delta, field_location="edge")
            ).tocsr()
            @ initial_memory,
            dtype=float,
        )

    if initial_memory.shape != (mesh.n_faces,):
        raise ValueError("face initial memory must have length mesh.n_faces")
    delta = np.asarray(delta_sigma, dtype=float)
    if delta.shape == (mesh.n_faces,):
        return delta * initial_memory
    unit_face = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    delta_face = mesh.get_face_inner_product(
        _cell_property(mesh, delta, field_location="face")
    ).tocsr()
    return np.asarray(delta_face.diagonal() / unit_face.diagonal() * initial_memory)


def _face_resistivity_inner_product(mesh, conductivity) -> sp.csr_matrix:
    conductivity = np.asarray(conductivity, dtype=float)
    if conductivity.size == 1:
        sigma = np.full(mesh.n_cells, float(conductivity.reshape(-1)[0]))
        return mesh.get_face_inner_product(1.0 / sigma).tocsr()
    if conductivity.shape == (mesh.n_cells,):
        if np.any(conductivity <= 0.0):
            raise ValueError("cell conductivity must be positive")
        return mesh.get_face_inner_product(1.0 / conductivity).tocsr()
    if conductivity.shape == (mesh.n_faces,):
        if np.any(conductivity <= 0.0):
            raise ValueError("face conductivity must be positive")
        unit_face = _unit_face_inner_product(mesh)
        return (unit_face @ sp.diags(1.0 / conductivity, format="csr")).tocsr()
    raise ValueError("conductivity must be scalar, cell-centered, or face-centered")


def _as_tau_sequence(value, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        taus = (float(values),)
    else:
        taus = tuple(float(item) for item in values.reshape(-1))
    if not taus:
        raise ValueError(f"{name} must contain at least one value")
    if any(tau <= 0.0 for tau in taus):
        raise ValueError(f"{name} must be positive")
    return taus


def _unit_face_inner_product(mesh) -> sp.csr_matrix:
    matrix = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError("face source-history prototype requires diagonal face mass")
    return matrix


def _inverse_diagonal_matrix(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError(f"{name} has zero diagonal entries")
    return sp.diags(1.0 / diagonal, format="csr")


def _cell_property(mesh, values, *, field_location: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.full(mesh.n_cells, float(values.reshape(-1)[0]))
    if values.shape == (mesh.n_cells,):
        return values
    if field_location == "face" and values.shape == (mesh.n_faces,):
        return values
    if field_location == "edge" and values.shape == (mesh.n_edges,):
        return values
    raise ValueError(
        "delta_sigma must be scalar, cell-centered, or match the source-history dofs"
    )


def _source_delta_sigma(mesh, delta_sigma, source) -> float:
    delta = np.asarray(delta_sigma, dtype=float)
    if delta.size == 1:
        return float(delta.reshape(-1)[0])
    if delta.shape == (mesh.n_cells,):
        return float(delta[_source_midpoint_cell_index(mesh, source)])
    if delta.shape == (mesh.n_faces,):
        source_vector = np.abs(source.initial_face_vector(mesh))
        weight = float(np.sum(source_vector))
        if weight == 0.0:
            return 0.0
        return float(np.dot(delta, source_vector) / weight)
    raise ValueError("delta_sigma must be scalar, cell-centered, or face-centered")


def _source_history_runtime_coefficients(
    correction: SourceHistoryCorrection | DrivenRecoverySourceHistoryCorrection,
    mesh,
    ip_model,
    source,
    mu: float,
) -> np.ndarray:
    if correction.coefficients is not None:
        return np.asarray(correction.coefficients, dtype=float)
    scale = _source_history_normalization_factor(mesh, ip_model, source, mu)
    return scale * np.asarray(correction.normalized_coefficients, dtype=float)


def _source_history_normalization_factor(mesh, ip_model, source, mu: float) -> float:
    if ip_model is None:
        raise ValueError("normalized source-history coefficients require ip_model")
    mu = float(mu)
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    source_length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    if source_length <= 0.0:
        raise ValueError("source length must be positive")
    delta_sigma = 0.0
    for term in getattr(ip_model, "terms", ()):
        delta_sigma += _source_delta_sigma(mesh, term.delta_sigma, source)
    return mu * delta_sigma * source_length**2


def _source_diffusion_time(mesh, source, ip_model, mu: float) -> float:
    if ip_model is None:
        raise ValueError("source-diffusion correction requires ip_model")
    length = float(np.linalg.norm(source.locations[-1] - source.locations[0]))
    if length <= 0.0:
        raise ValueError("source length must be positive")
    conductivity = _source_midpoint_conductivity(mesh, ip_model, source)
    return float(mu) * conductivity * length**2


def _source_diffusion_coefficients(
    correction: SourceDiffusionKernelSourceHistoryCorrection,
    source_diffusion_time: float,
) -> np.ndarray:
    if correction.normalized_amplitude is None:
        return np.asarray(correction.coefficients, dtype=float)
    coefficients = np.zeros(len(correction.source_moment_degrees), dtype=float)
    value = correction.normalized_amplitude * float(source_diffusion_time)
    if len(correction.source_moment_degrees) == 1:
        coefficients[0] = value
    else:
        coefficients[correction.source_moment_degrees.index(0)] = value
    return coefficients


def _source_diffusion_kernel_value(
    correction: SourceDiffusionKernelSourceHistoryCorrection,
    time_steps,
    *,
    time: float,
    tau: float,
) -> float:
    if correction.basis_kind == "continuous":
        return float(np.exp(-(float(time) - correction.amplitude_time) / tau))
    if correction.basis_kind == "be_decay":
        history = discrete_debye_history_basis(
            time_steps,
            tau=float(tau),
            max_order=0,
        )
        time_index = _time_node_index(history.times, float(time))
        amplitude_index = _time_node_index(
            history.times,
            float(correction.amplitude_time),
        )
        denominator = float(history.values[amplitude_index, 0])
        if denominator == 0.0:
            raise ValueError("source-diffusion BE amplitude_time has zero kernel")
        return float(history.values[time_index, 0] / denominator)
    raise ValueError(f"unknown source-diffusion basis_kind {correction.basis_kind!r}")


def _source_midpoint_conductivity(mesh, ip_model, source) -> float:
    if hasattr(ip_model, "low_frequency_sigma"):
        conductivity = np.asarray(ip_model.low_frequency_sigma(), dtype=float)
    else:
        conductivity = np.asarray(ip_model.sigma_infinity, dtype=float)
    if conductivity.size == 1:
        sigma = float(conductivity.reshape(-1)[0])
    elif conductivity.shape == (mesh.n_cells,):
        sigma = float(conductivity[_source_midpoint_cell_index(mesh, source)])
    else:
        raise ValueError(
            "source-diffusion conductivity must be scalar or cell-centered"
        )
    if sigma <= 0.0:
        raise ValueError("source midpoint conductivity must be positive")
    return sigma


def _source_midpoint_cell_index(mesh, source) -> int:
    midpoint = np.mean(source.locations, axis=0).reshape(1, 3)
    if hasattr(mesh, "closest_points_index"):
        index = mesh.closest_points_index(midpoint, "CC")
        return int(np.asarray(index).reshape(-1)[0])
    centers = np.asarray(mesh.cell_centers, dtype=float)
    distances = np.sum((centers - midpoint[0]) ** 2, axis=1)
    return int(np.argmin(distances))


def _receiver_matrix(
    mesh,
    locations: np.ndarray,
    *,
    correction: SourceHistoryCorrectionTerm,
    magnetic_receiver_mode: str,
    subdivisions: int,
    field_location: str,
) -> np.ndarray:
    kind = correction.receiver_matrix
    if kind == "auto":
        kind = _receiver_matrix_from_mode(magnetic_receiver_mode, field_location)
    elif kind == "current_biot" and field_location == "face":
        kind = "face_current"
    if kind == "current_biot":
        return cell_current_biot_matrix(mesh, locations, subdivisions=subdivisions)
    if kind == "edge_current":
        _require_field_location(field_location, "edge", kind)
        return edge_current_biot_matrix(mesh, locations)
    if kind == "edge_basis":
        _require_field_location(field_location, "edge", kind)
        return edge_basis_biot_matrix(mesh, locations, subdivisions=subdivisions)
    if kind == "face_current":
        _require_field_location(field_location, "face", kind)
        return face_current_biot_matrix(mesh, locations, subdivisions=subdivisions)
    if kind == "face_basis":
        _require_field_location(field_location, "face", kind)
        return face_basis_biot_matrix(mesh, locations, subdivisions=subdivisions)
    raise ValueError(f"unsupported source-history receiver matrix: {kind}")


def _receiver_matrix_from_mode(mode: str, field_location: str) -> str:
    normalized = str(mode).strip().lower()
    if field_location == "edge":
        if normalized == "current_biot":
            return "current_biot"
        if normalized == "edge_current_biot":
            return "edge_current"
        if normalized in {"edge_basis_biot", "edge_basis_cell_biot"}:
            return "edge_basis"
    if field_location == "face":
        if normalized == "current_biot":
            return "face_current"
        if normalized in {"face_basis_biot", "face_basis_cell_biot"}:
            return "face_basis"
    raise ValueError(
        "source-history receiver_matrix='auto' requires a Biot magnetic receiver mode"
    )


def _time_node_index(times: np.ndarray, time: float) -> int:
    matches = np.flatnonzero(np.isclose(times, float(time), rtol=1.0e-9, atol=1.0e-12))
    if matches.size == 0:
        raise ValueError(f"time {time:g} is not on the source-history time grid")
    return int(matches[0])


def _as_locations(locations) -> np.ndarray:
    values = np.asarray(locations, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    return values


def _normalize_field_location(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"edge", "face"}:
        raise ValueError("source-history field_location must be 'edge' or 'face'")
    return normalized


def _source_vector(mesh, source, field_location: str) -> np.ndarray:
    if field_location == "edge":
        return source.initial_edge_vector(mesh)
    return -source.initial_face_vector(mesh)


def _source_moment_basis(mesh, source, source_vector, correction, field_location: str):
    if field_location == "edge":
        return source_edge_moment_basis(
            mesh,
            source_vector,
            start=source.locations[0],
            end=source.locations[-1],
            degrees=correction.source_moment_degrees,
            source_edge_atol=correction.source_edge_atol,
        )
    return source_face_moment_basis(
        mesh,
        source_vector,
        start=source.locations[0],
        end=source.locations[-1],
        degrees=correction.source_moment_degrees,
        source_face_atol=correction.source_edge_atol,
    )


def _require_field_location(actual: str, expected: str, receiver_matrix: str) -> None:
    if actual != expected:
        raise ValueError(
            f"source-history receiver_matrix='{receiver_matrix}' requires "
            f"field_location='{expected}'"
        )
