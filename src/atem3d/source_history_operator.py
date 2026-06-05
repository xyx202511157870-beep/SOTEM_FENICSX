"""Source-history receiver basis operators for small-mesh derivations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import relative_l2
from .source_primary import discrete_debye_history_basis


@dataclass(frozen=True)
class SourceHistoryReceiverBasis:
    """Receiver responses for Debye source-history basis vectors."""

    times: np.ndarray
    basis_labels: list[str]
    responses: np.ndarray


@dataclass(frozen=True)
class SourceHistoryCoefficientFit:
    """Least-squares coefficients for source-history receiver basis responses."""

    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    rank: int
    singular_values: np.ndarray
    design_shape: tuple[int, int]
    column_norms: np.ndarray
    condition_number: float
    column_normalized_condition_number: float


@dataclass(frozen=True)
class StaticSpatialCoefficientFit:
    """Per-time coefficients from projecting target data onto static sources."""

    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    rank: int
    singular_values: np.ndarray
    design_shape: tuple[int, int]
    column_norms: np.ndarray
    condition_number: float
    column_normalized_condition_number: float


@dataclass(frozen=True)
class SpatialCoefficientTraceBasisFit:
    """Least-squares fit of static spatial coefficient traces to history bases."""

    tau: float
    basis_labels: list[str]
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_l2: float
    per_trace_relative_l2: np.ndarray
    rank: int
    singular_values: np.ndarray
    design_shape: tuple[int, int]
    column_norms: np.ndarray
    condition_number: float
    column_normalized_condition_number: float


def source_history_receiver_basis(
    time_steps,
    *,
    tau: float,
    source_vector,
    receiver_matrix,
    max_order: int = 1,
) -> SourceHistoryReceiverBasis:
    """Return ``g_p(t) * B * s0`` receiver responses.

    ``receiver_matrix`` must have shape ``(n_locations, 3, n_edges)`` and
    ``source_vector`` must have one value per edge.  The result has shape
    ``(n_times, n_basis, n_locations, 3)``.
    """

    source_vector = np.asarray(source_vector, dtype=float)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    source_vectors = np.repeat(source_vector[None, :], max_order + 1, axis=0)
    return source_history_receiver_basis_from_vectors(
        time_steps,
        tau=tau,
        source_vectors=source_vectors,
        receiver_matrix=receiver_matrix,
    )


def source_history_receiver_basis_from_vectors(
    time_steps,
    *,
    tau: float,
    source_vectors,
    receiver_matrix,
) -> SourceHistoryReceiverBasis:
    """Return ``g_p(t) * B * s_p`` responses with one vector per basis order."""

    source_vectors = np.asarray(source_vectors, dtype=float)
    receiver_matrix = np.asarray(receiver_matrix, dtype=float)
    if source_vectors.ndim != 2:
        raise ValueError("source_vectors must have shape (n_basis, n_edges)")
    if source_vectors.shape[0] == 0:
        raise ValueError("source_vectors must contain at least one basis vector")
    if receiver_matrix.ndim != 3 or receiver_matrix.shape[1] != 3:
        raise ValueError("receiver_matrix must have shape (n_locations, 3, n_edges)")
    if source_vectors.shape[1] != receiver_matrix.shape[2]:
        raise ValueError("source_vectors edge dimension must match receiver_matrix")

    max_order = source_vectors.shape[0] - 1
    history = discrete_debye_history_basis(
        time_steps,
        tau=tau,
        max_order=max_order,
    )
    static_response = np.einsum("lce,pe->plc", receiver_matrix, source_vectors)
    responses = history.values[:, :, None, None] * static_response[None, :, :, :]
    return SourceHistoryReceiverBasis(
        times=history.times,
        basis_labels=list(history.basis_labels),
        responses=np.asarray(responses, dtype=float),
    )


def source_history_receiver_basis_from_spatial_vectors(
    time_steps,
    *,
    tau: float,
    spatial_vectors,
    receiver_matrix,
    max_order: int = 1,
    spatial_labels: list[str] | None = None,
) -> SourceHistoryReceiverBasis:
    """Return all ``g_p(t) * B * v_s`` time-history/spatial-basis responses.

    ``spatial_vectors`` has shape ``(n_spatial, n_edges)``.  The returned basis
    dimension is ordered by time-history order first and spatial vector second:
    ``g_0*v_0, g_0*v_1, ..., g_1*v_0, ...``.
    """

    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    receiver_matrix = np.asarray(receiver_matrix, dtype=float)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if spatial_vectors.ndim != 2:
        raise ValueError("spatial_vectors must have shape (n_spatial, n_edges)")
    if spatial_vectors.shape[0] == 0:
        raise ValueError("spatial_vectors must contain at least one spatial vector")
    if receiver_matrix.ndim != 3 or receiver_matrix.shape[1] != 3:
        raise ValueError("receiver_matrix must have shape (n_locations, 3, n_edges)")
    if spatial_vectors.shape[1] != receiver_matrix.shape[2]:
        raise ValueError("spatial_vectors edge dimension must match receiver_matrix")
    if spatial_labels is None:
        spatial_labels = [f"spatial:{index}" for index in range(spatial_vectors.shape[0])]
    if len(spatial_labels) != spatial_vectors.shape[0]:
        raise ValueError("spatial_labels must have one label per spatial vector")

    history = discrete_debye_history_basis(
        time_steps,
        tau=tau,
        max_order=max_order,
    )
    static_response = np.einsum("lce,se->slc", receiver_matrix, spatial_vectors)
    response_grid = history.values[:, :, None, None, None] * static_response[
        None,
        None,
        :,
        :,
        :,
    ]
    responses = response_grid.reshape(
        history.times.size,
        history.values.shape[1] * spatial_vectors.shape[0],
        receiver_matrix.shape[0],
        3,
    )
    labels = [
        f"{history_label} * {spatial_label}"
        for history_label in history.basis_labels
        for spatial_label in spatial_labels
    ]
    return SourceHistoryReceiverBasis(
        times=history.times,
        basis_labels=labels,
        responses=np.asarray(responses, dtype=float),
    )


def source_history_receiver_basis_from_static_response(
    time_steps,
    *,
    tau: float,
    static_response,
    max_order: int = 1,
    spatial_labels: list[str] | None = None,
) -> SourceHistoryReceiverBasis:
    """Return history responses from precomputed ``B_R v_s`` responses.

    ``static_response`` has shape ``(n_spatial, n_locations, 3)``.  This is the
    matrix-free equivalent of
    :func:`source_history_receiver_basis_from_spatial_vectors` for cases where
    only a few source/spatial vectors are needed and constructing the full
    receiver matrix would be wasteful.
    """

    static_response = np.asarray(static_response, dtype=float)
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if static_response.ndim != 3 or static_response.shape[2] != 3:
        raise ValueError("static_response must have shape (n_spatial, n_locations, 3)")
    if static_response.shape[0] == 0:
        raise ValueError("static_response must contain at least one spatial vector")
    if spatial_labels is None:
        spatial_labels = [f"spatial:{index}" for index in range(static_response.shape[0])]
    if len(spatial_labels) != static_response.shape[0]:
        raise ValueError("spatial_labels must have one label per spatial vector")

    history = discrete_debye_history_basis(
        time_steps,
        tau=tau,
        max_order=max_order,
    )
    response_grid = history.values[:, :, None, None, None] * static_response[
        None,
        None,
        :,
        :,
        :,
    ]
    responses = response_grid.reshape(
        history.times.size,
        history.values.shape[1] * static_response.shape[0],
        static_response.shape[1],
        3,
    )
    labels = [
        f"{history_label} * {spatial_label}"
        for history_label in history.basis_labels
        for spatial_label in spatial_labels
    ]
    return SourceHistoryReceiverBasis(
        times=history.times,
        basis_labels=labels,
        responses=np.asarray(responses, dtype=float),
    )


def fit_source_history_coefficients(
    basis_responses,
    target,
) -> SourceHistoryCoefficientFit:
    """Fit common coefficients for source-history receiver basis responses."""

    basis_responses = np.asarray(basis_responses, dtype=float)
    target = np.asarray(target, dtype=float)
    if basis_responses.ndim != 4:
        raise ValueError(
            "basis_responses must have shape (n_times, n_basis, n_locations, 3)"
        )
    if basis_responses.shape[0] != target.shape[0] or basis_responses.shape[2:] != target.shape[1:]:
        raise ValueError("target must have shape (n_times, n_locations, 3)")

    n_basis = basis_responses.shape[1]
    design = np.moveaxis(basis_responses, 1, -1).reshape(-1, n_basis)
    rhs = target.reshape(-1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, rhs, rcond=None)
    fitted = np.einsum("p,tplc->tlc", coefficients, basis_responses)
    residual = fitted - target
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    return SourceHistoryCoefficientFit(
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, target),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def fit_source_history_coefficients_for_components(
    basis_responses,
    target_columns,
    *,
    receiver_indices,
    component_indices,
) -> SourceHistoryCoefficientFit:
    """Fit coefficients using selected receiver/component scalar columns."""

    basis_responses = np.asarray(basis_responses, dtype=float)
    target_columns = np.asarray(target_columns, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if basis_responses.ndim != 4:
        raise ValueError(
            "basis_responses must have shape (n_times, n_basis, n_locations, 3)"
        )
    if receiver_indices.ndim != 1 or component_indices.ndim != 1:
        raise ValueError("receiver_indices and component_indices must be 1D")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= basis_responses.shape[2]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")
    expected_shape = (basis_responses.shape[0], receiver_indices.size)
    if target_columns.shape != expected_shape:
        raise ValueError("target_columns must have shape (n_times, n_columns)")

    selected = basis_responses[:, :, receiver_indices, component_indices]
    design = np.moveaxis(selected, 1, -1).reshape(-1, basis_responses.shape[1])
    rhs = target_columns.reshape(-1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, rhs, rcond=None)
    fitted = (design @ coefficients).reshape(target_columns.shape)
    residual = fitted - target_columns
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    return SourceHistoryCoefficientFit(
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, target_columns),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def fit_static_spatial_coefficients_for_components(
    receiver_matrix,
    spatial_vectors,
    target_columns,
    *,
    receiver_indices,
    component_indices,
) -> StaticSpatialCoefficientFit:
    """Project each target time sample onto static receiver responses.

    This separates the spatial question, ``target(t) ~= B_R v_s a_s(t)``,
    from any Debye time-history fit of the coefficient traces ``a_s(t)``.
    """

    receiver_matrix = np.asarray(receiver_matrix, dtype=float)
    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    target_columns = np.asarray(target_columns, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if receiver_matrix.ndim != 3 or receiver_matrix.shape[1] != 3:
        raise ValueError("receiver_matrix must have shape (n_locations, 3, n_dofs)")
    if spatial_vectors.ndim != 2:
        raise ValueError("spatial_vectors must have shape (n_spatial, n_dofs)")
    if spatial_vectors.shape[1] != receiver_matrix.shape[2]:
        raise ValueError("spatial_vectors dof dimension must match receiver_matrix")
    if receiver_indices.ndim != 1 or component_indices.ndim != 1:
        raise ValueError("receiver_indices and component_indices must be 1D")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= receiver_matrix.shape[0]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")
    expected_shape = (target_columns.shape[0], receiver_indices.size)
    if target_columns.ndim != 2 or target_columns.shape != expected_shape:
        raise ValueError("target_columns must have shape (n_times, n_columns)")

    static_response = np.einsum("lce,se->slc", receiver_matrix, spatial_vectors)
    design = static_response[:, receiver_indices, component_indices].T
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        target_columns.T,
        rcond=None,
    )
    coefficients = coefficients.T
    fitted = coefficients @ design.T
    residual = fitted - target_columns
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    return StaticSpatialCoefficientFit(
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, target_columns),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def fit_static_spatial_coefficients_from_static_response(
    static_response,
    target_columns,
    *,
    receiver_indices,
    component_indices,
) -> StaticSpatialCoefficientFit:
    """Project target samples onto precomputed static spatial responses."""

    static_response = np.asarray(static_response, dtype=float)
    target_columns = np.asarray(target_columns, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if static_response.ndim != 3 or static_response.shape[2] != 3:
        raise ValueError("static_response must have shape (n_spatial, n_locations, 3)")
    if receiver_indices.ndim != 1 or component_indices.ndim != 1:
        raise ValueError("receiver_indices and component_indices must be 1D")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= static_response.shape[1]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")
    expected_shape = (target_columns.shape[0], receiver_indices.size)
    if target_columns.ndim != 2 or target_columns.shape != expected_shape:
        raise ValueError("target_columns must have shape (n_times, n_columns)")

    design = static_response[:, receiver_indices, component_indices].T
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        target_columns.T,
        rcond=None,
    )
    coefficients = coefficients.T
    fitted = coefficients @ design.T
    residual = fitted - target_columns
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    return StaticSpatialCoefficientFit(
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, target_columns),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def project_vector_to_spatial_basis(
    receiver_matrix,
    spatial_vectors,
    target_vector,
    *,
    projection: str = "receiver_l2",
    static_response=None,
) -> np.ndarray:
    """Project one FV vector onto a source/spatial basis.

    ``projection='dof_l2'`` solves in the vector DOF space.  The default
    ``receiver_l2`` solves after applying the selected receiver/MMR matrix, so
    coefficients are tied to the same magnetic recovery operator used later in
    prescribed source-history audits.
    """

    receiver_matrix = np.asarray(receiver_matrix, dtype=float)
    spatial_vectors = np.asarray(spatial_vectors, dtype=float)
    target_vector = np.asarray(target_vector, dtype=float)
    projection = str(projection).strip().lower()
    if projection not in {"receiver_l2", "dof_l2"}:
        raise ValueError("projection must be 'receiver_l2' or 'dof_l2'")
    if spatial_vectors.ndim != 2:
        raise ValueError("spatial_vectors must have shape (n_spatial, n_dofs)")
    if target_vector.shape != (spatial_vectors.shape[1],):
        raise ValueError("target_vector must have shape (n_dofs,)")

    if projection == "dof_l2":
        design = spatial_vectors.T
        rhs = target_vector
    else:
        if receiver_matrix.ndim != 3 or receiver_matrix.shape[1] != 3:
            raise ValueError("receiver_matrix must have shape (n_locations, 3, n_dofs)")
        if receiver_matrix.shape[2] != spatial_vectors.shape[1]:
            raise ValueError("receiver_matrix dof dimension must match spatial vectors")
        if static_response is None:
            static_response = np.einsum("lci,si->slc", receiver_matrix, spatial_vectors)
        static_response = np.asarray(static_response, dtype=float)
        if static_response.shape != (
            spatial_vectors.shape[0],
            receiver_matrix.shape[0],
            3,
        ):
            raise ValueError(
                "static_response must have shape (n_spatial, n_locations, 3)"
            )
        target_response = np.einsum("lci,i->lc", receiver_matrix, target_vector)
        design = np.moveaxis(static_response, 0, -1).reshape(-1, spatial_vectors.shape[0])
        rhs = target_response.reshape(-1)

    if design.size == 0 or not np.any(design) or not np.any(rhs):
        return np.zeros(spatial_vectors.shape[0], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    return np.asarray(coefficients, dtype=float)


def fit_spatial_coefficient_traces_to_history_basis(
    time_steps,
    sample_times,
    coefficient_matrix,
    *,
    tau: float,
    max_order: int = 1,
    time_atol: float = 1.0e-12,
) -> SpatialCoefficientTraceBasisFit:
    """Fit per-time spatial coefficients to discrete Debye history columns."""

    sample_times = np.asarray(sample_times, dtype=float)
    coefficient_matrix = np.asarray(coefficient_matrix, dtype=float)
    if sample_times.ndim != 1:
        raise ValueError("sample_times must be a 1D array")
    if coefficient_matrix.ndim != 2:
        raise ValueError("coefficient_matrix must have shape (n_times, n_traces)")
    if coefficient_matrix.shape[0] != sample_times.size:
        raise ValueError("coefficient_matrix first dimension must match sample_times")

    history = discrete_debye_history_basis(
        time_steps,
        tau=float(tau),
        max_order=int(max_order),
    )
    indices = _time_node_indices(history.times, sample_times, atol=float(time_atol))
    design = history.values[indices]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        coefficient_matrix,
        rcond=None,
    )
    fitted = design @ coefficients
    residual = fitted - coefficient_matrix
    diagnostics = _least_squares_design_diagnostics(design, singular_values, int(rank))
    per_trace_relative_l2 = np.array(
        [
            relative_l2(fitted[:, index], coefficient_matrix[:, index])
            for index in range(coefficient_matrix.shape[1])
        ],
        dtype=float,
    )
    return SpatialCoefficientTraceBasisFit(
        tau=float(tau),
        basis_labels=list(history.basis_labels),
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, coefficient_matrix),
        per_trace_relative_l2=per_trace_relative_l2,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def evaluate_spatial_coefficient_traces_with_history_basis(
    time_steps,
    sample_times,
    coefficient_matrix,
    *,
    tau: float,
    coefficients,
    max_order: int | None = None,
    time_atol: float = 1.0e-12,
) -> SpatialCoefficientTraceBasisFit:
    """Evaluate prescribed BE history coefficients against spatial traces."""

    sample_times = np.asarray(sample_times, dtype=float)
    coefficient_matrix = np.asarray(coefficient_matrix, dtype=float)
    prescribed = np.asarray(coefficients, dtype=float)
    if sample_times.ndim != 1:
        raise ValueError("sample_times must be a 1D array")
    if coefficient_matrix.ndim != 2:
        raise ValueError("coefficient_matrix must have shape (n_times, n_traces)")
    if coefficient_matrix.shape[0] != sample_times.size:
        raise ValueError("coefficient_matrix first dimension must match sample_times")
    if prescribed.ndim > 2:
        raise ValueError("coefficients must be a flat vector or a 2D table")

    n_traces = coefficient_matrix.shape[1]
    if prescribed.ndim == 1:
        if prescribed.size == 0 or prescribed.size % n_traces != 0:
            raise ValueError(
                "flat coefficients must contain (max_order + 1) * n_traces values"
            )
        inferred_order = prescribed.size // n_traces - 1
        prescribed = prescribed.reshape(inferred_order + 1, n_traces)
    else:
        if prescribed.shape[1] != n_traces:
            raise ValueError("coefficient table second dimension must match n_traces")
        inferred_order = prescribed.shape[0] - 1
    if inferred_order < 0:
        raise ValueError("coefficients must contain at least one history row")
    if max_order is None:
        max_order = inferred_order
    max_order = int(max_order)
    if max_order != inferred_order:
        raise ValueError("max_order must match the prescribed coefficient rows")

    history = discrete_debye_history_basis(
        time_steps,
        tau=float(tau),
        max_order=max_order,
    )
    indices = _time_node_indices(history.times, sample_times, atol=float(time_atol))
    design = history.values[indices]
    fitted = design @ prescribed
    residual = fitted - coefficient_matrix
    singular_values = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    per_trace_relative_l2 = np.array(
        [
            relative_l2(fitted[:, index], coefficient_matrix[:, index])
            for index in range(n_traces)
        ],
        dtype=float,
    )
    return SpatialCoefficientTraceBasisFit(
        tau=float(tau),
        basis_labels=list(history.basis_labels),
        coefficients=np.asarray(prescribed, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, coefficient_matrix),
        per_trace_relative_l2=per_trace_relative_l2,
        rank=rank,
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def evaluate_source_history_coefficients_for_components(
    basis_responses,
    target_columns,
    *,
    coefficients,
    receiver_indices,
    component_indices,
) -> SourceHistoryCoefficientFit:
    """Evaluate prescribed coefficients on selected receiver/component columns."""

    basis_responses = np.asarray(basis_responses, dtype=float)
    target_columns = np.asarray(target_columns, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    receiver_indices = np.asarray(receiver_indices, dtype=int)
    component_indices = np.asarray(component_indices, dtype=int)
    if basis_responses.ndim != 4:
        raise ValueError(
            "basis_responses must have shape (n_times, n_basis, n_locations, 3)"
        )
    if coefficients.shape != (basis_responses.shape[1],):
        raise ValueError("coefficients must have one value per basis response")
    if receiver_indices.ndim != 1 or component_indices.ndim != 1:
        raise ValueError("receiver_indices and component_indices must be 1D")
    if receiver_indices.shape != component_indices.shape:
        raise ValueError("receiver_indices and component_indices must have the same length")
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= basis_responses.shape[2]):
        raise ValueError("receiver_indices contains an out-of-range index")
    if np.any(component_indices < 0) or np.any(component_indices >= 3):
        raise ValueError("component_indices must contain values 0, 1, or 2")
    expected_shape = (basis_responses.shape[0], receiver_indices.size)
    if target_columns.shape != expected_shape:
        raise ValueError("target_columns must have shape (n_times, n_columns)")

    selected = basis_responses[:, :, receiver_indices, component_indices]
    design = np.moveaxis(selected, 1, -1).reshape(-1, basis_responses.shape[1])
    rhs_shape = target_columns.shape
    fitted = (design @ coefficients).reshape(rhs_shape)
    residual = fitted - target_columns
    singular_values = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    diagnostics = _least_squares_design_diagnostics(design, singular_values, rank)
    return SourceHistoryCoefficientFit(
        coefficients=np.asarray(coefficients, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        relative_l2=relative_l2(fitted, target_columns),
        rank=rank,
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def _least_squares_design_diagnostics(
    design: np.ndarray,
    singular_values: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray | tuple[int, int] | float]:
    design = np.asarray(design, dtype=float)
    singular_values = np.asarray(singular_values, dtype=float)
    column_norms = np.linalg.norm(design, axis=0)
    normalized_design = design.copy()
    nonzero_columns = column_norms > 0.0
    normalized_design[:, nonzero_columns] /= column_norms[nonzero_columns]
    if normalized_design.size:
        normalized_singular_values = np.linalg.svd(
            normalized_design,
            full_matrices=False,
            compute_uv=False,
        )
        normalized_rank = int(np.linalg.matrix_rank(normalized_design))
    else:
        normalized_singular_values = np.asarray([], dtype=float)
        normalized_rank = 0
    return {
        "shape": (int(design.shape[0]), int(design.shape[1])),
        "column_norms": np.asarray(column_norms, dtype=float),
        "condition_number": _condition_number(
            singular_values,
            rank=int(rank),
            n_columns=design.shape[1],
        ),
        "column_normalized_condition_number": _condition_number(
            normalized_singular_values,
            rank=normalized_rank,
            n_columns=design.shape[1],
        ),
    }


def _condition_number(
    singular_values: np.ndarray,
    *,
    rank: int,
    n_columns: int,
) -> float:
    singular_values = np.asarray(singular_values, dtype=float)
    if n_columns <= 0 or singular_values.size == 0:
        return float("nan")
    if rank < n_columns:
        return float("inf")
    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def _time_node_indices(times: np.ndarray, sample_times: np.ndarray, *, atol: float) -> np.ndarray:
    indices: list[int] = []
    for time in sample_times:
        matches = np.flatnonzero(np.isclose(times, float(time), rtol=1.0e-9, atol=atol))
        if matches.size == 0:
            raise ValueError(f"sample time {time:g} is not on the source-history time grid")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)
