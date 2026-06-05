"""Source consistency diagnostics for grounded-wire TDEM validation."""

from __future__ import annotations

import numpy as np


def diagnose_source_consistency(
    *,
    gradient_transpose,
    source_vector,
    endpoint_source,
    divergence_operator,
    conductive_current,
    initial_electric_field,
    curl_operator,
    time_intervals,
    interval_average_didt,
    current_initial: float,
    current_final: float,
) -> dict[str, float]:
    """Return task-book source consistency residuals.

    The operators may be dense arrays or sparse matrices.  The residuals are
    L2 norms of the weak algebraic checks used by the validation report:
    ``G^T s_Gamma``, ``D J_dc``, ``C E0``, and the integrated source waveform.
    """

    source_vector = _as_vector(source_vector, "source_vector")
    endpoint_source = _as_vector(endpoint_source, "endpoint_source")
    conductive_current = _as_vector(conductive_current, "conductive_current")
    initial_electric_field = _as_vector(initial_electric_field, "initial_electric_field")
    time_intervals = _as_vector(time_intervals, "time_intervals")
    interval_average_didt = _as_vector(interval_average_didt, "interval_average_didt")
    if time_intervals.shape != interval_average_didt.shape:
        raise ValueError("time_intervals and interval_average_didt must have the same shape")
    if np.any(time_intervals <= 0.0):
        raise ValueError("time_intervals must be positive")

    endpoint_balance = _matvec(gradient_transpose, source_vector) - endpoint_source
    dc_conservation = _matvec(divergence_operator, conductive_current)
    initial_curl = _matvec(curl_operator, initial_electric_field)
    integrated_didt = float(np.dot(time_intervals, interval_average_didt))
    current_delta = float(current_final) - float(current_initial)

    return {
        "source_endpoint_balance_residual": _l2(endpoint_balance),
        "dc_current_conservation_residual": _l2(dc_conservation),
        "initial_curl_residual": _l2(initial_curl),
        "waveform_integral_residual": float(abs(integrated_didt - current_delta)),
    }


def _matvec(operator, vector: np.ndarray) -> np.ndarray:
    values = operator @ vector
    return _as_vector(values, "operator result")


def _as_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _l2(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).reshape(-1)))
