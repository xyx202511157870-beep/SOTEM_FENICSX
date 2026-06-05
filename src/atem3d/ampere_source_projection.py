"""Project discrete Ampere residuals onto the grounded-source edge vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class AmpereSourceProjection:
    """Ampere residual source-vector projection over selected time nodes."""

    times: np.ndarray
    coefficients: np.ndarray
    residual_norms: np.ndarray
    relative_residual_norms: np.ndarray
    source_norm: float


def ampere_source_projection(
    sim,
    times,
    electric_fields,
    magnetic_fluxes,
    *,
    include_t0: bool = False,
) -> AmpereSourceProjection:
    """Project ``C.T M_mu^-1 b - j_c - s_e`` onto the initial source vector.

    This diagnostic uses only the discrete finite-volume fields and source
    vector.  It does not use empymod references or fitted receiver residuals.
    """

    times = np.asarray(times, dtype=float)
    electric_fields = np.asarray(electric_fields, dtype=float)
    magnetic_fluxes = np.asarray(magnetic_fluxes, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D sequence")
    if electric_fields.shape != (times.size, sim.mesh.n_edges):
        raise ValueError("electric_fields shape does not match times and mesh edges")
    if magnetic_fluxes.shape != (times.size, sim.mesh.n_faces):
        raise ValueError("magnetic_fluxes shape does not match times and mesh faces")
    if times.size > sim.time_steps.size + 1:
        raise ValueError("more field rows were supplied than configured time nodes")

    source_vector = np.zeros(sim.mesh.n_edges, dtype=float)
    for source in sim.sources:
        source_vector += source.initial_edge_vector(sim.mesh)
    source_norm_sq = float(np.dot(source_vector, source_vector))
    if source_norm_sq == 0.0:
        raise ValueError("initial grounded-source vector has zero norm")

    selected_times: list[float] = []
    coefficients: list[float] = []
    residual_norms: list[float] = []
    relative_norms: list[float] = []

    memories = sim.ip_model.initial_memory(sim.mesh.n_edges, electric_fields[0])
    if include_t0:
        _append_projection(
            sim,
            float(times[0]),
            electric_fields[0],
            magnetic_fluxes[0],
            memories,
            source_vector,
            source_norm_sq,
            use_left_limit_source=True,
            selected_times=selected_times,
            coefficients=coefficients,
            residual_norms=residual_norms,
            relative_norms=relative_norms,
        )

    for step_index in range(times.size - 1):
        dt = float(sim.time_steps[step_index])
        memories = sim.ip_model.update_memory(
            memories,
            electric_fields[step_index + 1],
            dt,
        )
        _append_projection(
            sim,
            float(times[step_index + 1]),
            electric_fields[step_index + 1],
            magnetic_fluxes[step_index + 1],
            memories,
            source_vector,
            source_norm_sq,
            use_left_limit_source=False,
            selected_times=selected_times,
            coefficients=coefficients,
            residual_norms=residual_norms,
            relative_norms=relative_norms,
        )

    return AmpereSourceProjection(
        times=np.asarray(selected_times, dtype=float),
        coefficients=np.asarray(coefficients, dtype=float),
        residual_norms=np.asarray(residual_norms, dtype=float),
        relative_residual_norms=np.asarray(relative_norms, dtype=float),
        source_norm=float(np.sqrt(source_norm_sq)),
    )


def hj_ampere_source_projection(
    sim,
    times,
    electric_fields,
    magnetic_fields,
    *,
    include_t0: bool = False,
) -> AmpereSourceProjection:
    """Project H/J Ampere residuals onto the signed grounded-source face vector.

    The H/J convention used by :mod:`atem3d.hj` is
    ``j = C h - s_face`` with ``s_face = -source.face_vector``.  This
    diagnostic therefore evaluates ``C h - q(E, y) - s_face`` directly on
    face unknowns.
    """

    times = np.asarray(times, dtype=float)
    electric_fields = np.asarray(electric_fields, dtype=float)
    magnetic_fields = np.asarray(magnetic_fields, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D sequence")
    if electric_fields.shape != (times.size, sim.mesh.n_faces):
        raise ValueError("electric_fields shape does not match times and mesh faces")
    if magnetic_fields.shape != (times.size, sim.mesh.n_edges):
        raise ValueError("magnetic_fields shape does not match times and mesh edges")
    if times.size > sim.time_steps.size + 1:
        raise ValueError("more field rows were supplied than configured time nodes")

    source_vector = _hj_initial_source_vector(sim)
    source_norm_sq = float(np.dot(source_vector, source_vector))
    if source_norm_sq == 0.0:
        raise ValueError("initial grounded-source face vector has zero norm")

    selected_times: list[float] = []
    coefficients: list[float] = []
    residual_norms: list[float] = []
    relative_norms: list[float] = []

    memories = sim.ip_model.initial_memory(sim.mesh.n_faces, electric_fields[0])
    if include_t0:
        _append_hj_projection(
            sim,
            float(times[0]),
            electric_fields[0],
            magnetic_fields[0],
            memories,
            source_vector,
            source_norm_sq,
            use_initial_source=True,
            selected_times=selected_times,
            coefficients=coefficients,
            residual_norms=residual_norms,
            relative_norms=relative_norms,
        )

    for step_index in range(times.size - 1):
        dt = float(sim.time_steps[step_index])
        memories = sim.ip_model.update_memory(
            memories,
            electric_fields[step_index + 1],
            dt,
        )
        _append_hj_projection(
            sim,
            float(times[step_index + 1]),
            electric_fields[step_index + 1],
            magnetic_fields[step_index + 1],
            memories,
            source_vector,
            source_norm_sq,
            use_initial_source=False,
            selected_times=selected_times,
            coefficients=coefficients,
            residual_norms=residual_norms,
            relative_norms=relative_norms,
        )

    return AmpereSourceProjection(
        times=np.asarray(selected_times, dtype=float),
        coefficients=np.asarray(coefficients, dtype=float),
        residual_norms=np.asarray(residual_norms, dtype=float),
        relative_residual_norms=np.asarray(relative_norms, dtype=float),
        source_norm=float(np.sqrt(source_norm_sq)),
    )


def _append_projection(
    sim,
    time: float,
    electric_field: np.ndarray,
    magnetic_flux: np.ndarray,
    memories: list[np.ndarray],
    source_vector: np.ndarray,
    source_norm_sq: float,
    *,
    use_left_limit_source: bool,
    selected_times: list[float],
    coefficients: list[float],
    residual_norms: list[float],
    relative_norms: list[float],
) -> None:
    residual, reference_norm = _ampere_residual(
        sim,
        time,
        electric_field,
        magnetic_flux,
        memories,
        use_left_limit_source=use_left_limit_source,
    )
    selected_times.append(time)
    coefficients.append(float(np.dot(residual, source_vector) / source_norm_sq))
    norm = float(np.linalg.norm(residual))
    residual_norms.append(norm)
    if reference_norm == 0.0:
        relative_norms.append(norm)
    else:
        relative_norms.append(float(norm / reference_norm))


def _ampere_residual(
    sim,
    time: float,
    electric_field: np.ndarray,
    magnetic_flux: np.ndarray,
    memories: list[np.ndarray],
    *,
    use_left_limit_source: bool,
) -> tuple[np.ndarray, float]:
    current = sim.mesh.get_edge_inner_product(sim.ip_model.sigma_infinity).tocsr() @ electric_field
    for term, memory in zip(sim.ip_model.terms, memories):
        current -= sim.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ memory
    source = (
        sim.previous_electric_source_term(time)
        if use_left_limit_source
        else sim.electric_source_term(time)
    )
    ampere = sim.mesh.edge_curl.T @ sim.face_mu_inverse_matrix @ magnetic_flux
    residual = np.asarray(ampere - current - source, dtype=float)
    reference_norm = float(np.linalg.norm(current + source))
    return residual, reference_norm


def _append_hj_projection(
    sim,
    time: float,
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    memories: list[np.ndarray],
    source_vector: np.ndarray,
    source_norm_sq: float,
    *,
    use_initial_source: bool,
    selected_times: list[float],
    coefficients: list[float],
    residual_norms: list[float],
    relative_norms: list[float],
) -> None:
    residual, reference_norm = _hj_ampere_residual(
        sim,
        time,
        electric_field,
        magnetic_field,
        memories,
        use_initial_source=use_initial_source,
    )
    selected_times.append(time)
    coefficients.append(float(np.dot(residual, source_vector) / source_norm_sq))
    norm = float(np.linalg.norm(residual))
    residual_norms.append(norm)
    if reference_norm == 0.0:
        relative_norms.append(norm)
    else:
        relative_norms.append(float(norm / reference_norm))


def _hj_ampere_residual(
    sim,
    time: float,
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    memories: list[np.ndarray],
    *,
    use_initial_source: bool,
) -> tuple[np.ndarray, float]:
    current = _hj_conductive_current_density(
        sim.mesh,
        sim.ip_model,
        electric_field,
        memories,
    )
    source = (
        _hj_initial_source_vector(sim)
        if use_initial_source
        else sim.electric_source_term(time)
    )
    ampere = sim.mesh.edge_curl @ magnetic_field
    residual = np.asarray(ampere - current - source, dtype=float)
    reference_norm = float(np.linalg.norm(current + source))
    return residual, reference_norm


def _hj_conductive_current_density(
    mesh,
    ip_model,
    electric_field: np.ndarray,
    memories: list[np.ndarray],
) -> np.ndarray:
    electric_field = np.asarray(electric_field, dtype=float)
    if electric_field.shape != (mesh.n_faces,):
        raise ValueError("electric_field must have shape (mesh.n_faces,)")
    if len(memories) != len(ip_model.terms):
        raise ValueError("one memory vector is required for each Debye term")

    unit_face = _hj_unit_face_inner_product(mesh)
    unit_diagonal = np.asarray(unit_face.diagonal(), dtype=float)
    current = (
        _hj_face_inner_product_from_coefficients(mesh, ip_model.sigma_infinity)
        @ electric_field
    ) / unit_diagonal
    for term, memory in zip(ip_model.terms, memories):
        memory = np.asarray(memory, dtype=float)
        if memory.shape != (mesh.n_faces,):
            raise ValueError("each H/J memory vector must have shape (mesh.n_faces,)")
        current -= (
            _hj_face_inner_product_from_coefficients(mesh, term.delta_sigma) @ memory
        ) / unit_diagonal
    return np.asarray(current, dtype=float)


def _hj_initial_source_vector(sim) -> np.ndarray:
    source = np.zeros(sim.mesh.n_faces, dtype=float)
    for item in sim.sources:
        source -= item.initial_face_vector(sim.mesh)
    return source


def _hj_face_inner_product_from_coefficients(mesh, coefficients: np.ndarray) -> sp.csr_matrix:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape == (mesh.n_cells,):
        return mesh.get_face_inner_product(coefficients).tocsr()
    if coefficients.shape == (mesh.n_faces,):
        return (_hj_unit_face_inner_product(mesh) @ sp.diags(coefficients, format="csr")).tocsr()
    if coefficients.size == 1:
        return mesh.get_face_inner_product(
            np.full(mesh.n_cells, float(coefficients.reshape(-1)[0]))
        ).tocsr()
    raise ValueError("face coefficients must be scalar, cell-centered, or face-centered")


def _hj_unit_face_inner_product(mesh) -> sp.csr_matrix:
    matrix = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError("H/J Ampere projection requires diagonal face mass")
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError("unit face mass contains zero diagonal entries")
    return matrix
