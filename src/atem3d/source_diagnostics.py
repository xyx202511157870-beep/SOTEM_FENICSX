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


def diagnose_edge_source_orientation(
    *,
    source_start,
    source_end,
    edge_source_vector,
    edge_block_sizes,
    current: float = 1.0,
    reversed_cosine_threshold: float = -0.99,
) -> dict[str, object]:
    """Audit whether an E-form edge source vector follows ``start -> end``.

    For a Nedelec edge line source, the unit-current edge vector represents
    integrated path lengths on x/y/z oriented edge blocks.  Summing each block
    and dividing by the source current should recover the physical
    displacement ``source_end - source_start`` for an aligned discretization.
    """

    start = _as_point(source_start, "source_start")
    end = _as_point(source_end, "source_end")
    displacement = end - start
    source_length = float(np.linalg.norm(displacement))
    if source_length <= 0.0:
        raise ValueError("source_start and source_end must be distinct")
    current = float(current)
    if current == 0.0:
        raise ValueError("current must be nonzero")
    if reversed_cosine_threshold < -1.0 or reversed_cosine_threshold > 1.0:
        raise ValueError("reversed_cosine_threshold must be between -1 and 1")

    vector = _as_vector(edge_source_vector, "edge_source_vector")
    block_sizes = _edge_block_sizes(edge_block_sizes)
    if vector.size != sum(block_sizes):
        raise ValueError("edge_source_vector length must equal sum(edge_block_sizes)")

    nx, ny, nz = block_sizes
    offsets = np.cumsum([0, nx, ny, nz])
    integrated = np.array(
        [
            np.sum(vector[offsets[0] : offsets[1]]),
            np.sum(vector[offsets[1] : offsets[2]]),
            np.sum(vector[offsets[2] : offsets[3]]),
        ],
        dtype=float,
    ) / current
    axis = displacement / source_length
    signed_parallel = float(np.dot(integrated, axis))
    transverse = integrated - signed_parallel * axis
    integrated_norm = float(np.linalg.norm(integrated))
    if integrated_norm == 0.0:
        orientation_cosine = 0.0
    else:
        orientation_cosine = float(np.dot(integrated, displacement) / (integrated_norm * source_length))

    return {
        "source_start": _float_list(start),
        "source_end": _float_list(end),
        "source_length_m": source_length,
        "current_a": current,
        "edge_block_sizes": [int(value) for value in block_sizes],
        "expected_displacement_m": _float_list(displacement),
        "integrated_displacement_m": _float_list(integrated),
        "signed_parallel_projection_m": signed_parallel,
        "transverse_residual_m": float(np.linalg.norm(transverse)),
        "orientation_cosine": orientation_cosine,
        "relative_parallel_length_error": float(abs(signed_parallel - source_length) / source_length),
        "reversed_orientation": bool(orientation_cosine <= float(reversed_cosine_threshold)),
    }


def _matvec(operator, vector: np.ndarray) -> np.ndarray:
    values = operator @ vector
    return _as_vector(values, "operator result")


def _as_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_point(values, name: str) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must be a 3D coordinate")
    if np.any(~np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values")
    return point


def _edge_block_sizes(values) -> tuple[int, int, int]:
    sizes = tuple(int(value) for value in values)
    if len(sizes) != 3:
        raise ValueError("edge_block_sizes must contain x/y/z block sizes")
    if any(value < 0 for value in sizes):
        raise ValueError("edge_block_sizes must be nonnegative")
    return sizes


def _float_list(values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _l2(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float).reshape(-1)))
