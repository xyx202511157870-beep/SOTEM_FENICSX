"""Magnetic-diffusion recovery spectra for local H/J MMR diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from discretize import TensorMesh
from scipy.constants import mu_0

from .magnetic_recovery import face_current_biot_matrix
from .metrics import relative_l2
from .source_history_runtime import charge_conserving_face_current


@dataclass(frozen=True)
class MagneticDiffusionSpectrum:
    """Positive generalized eigenmodes of ``K h = lambda M h``."""

    eigenvalues: np.ndarray
    time_constants: np.ndarray
    eigenvectors: np.ndarray
    eigenvalue_floor: float = 0.0
    discarded_count: int = 0
    raw_eigenvalue_min: float = 0.0
    raw_eigenvalue_max: float = 0.0
    last_discarded_eigenvalue: float | None = None
    first_kept_eigenvalue: float | None = None


@dataclass(frozen=True)
class LocalTensorMeshSupport:
    """Rectangular TensorMesh sub-support extracted from global cells."""

    mesh: TensorMesh
    global_cell_indices: np.ndarray
    global_face_indices: np.ndarray
    global_edge_indices: np.ndarray
    ijk_min: tuple[int, int, int]
    ijk_max: tuple[int, int, int]


@dataclass(frozen=True)
class LocalMagneticDiffusionSpectrum:
    """Local support and its positive magnetic-diffusion modes."""

    support: LocalTensorMeshSupport
    spectrum: MagneticDiffusionSpectrum


@dataclass(frozen=True)
class MagneticDiffusionModalCoupling:
    """Face-source forcing and receiver projection for local diffusion modes."""

    eigenvalues: np.ndarray
    time_constants: np.ndarray
    eigenvectors: np.ndarray
    modal_forcing: np.ndarray
    modal_receiver_response: np.ndarray
    source_receiver_response: np.ndarray
    receiver_mode: str


@dataclass(frozen=True)
class ModalSourceMomentProjection:
    """Least-squares projection of modal receiver responses onto source moments."""

    coefficients: np.ndarray
    fitted_response: np.ndarray
    residual_response: np.ndarray
    relative_l2: np.ndarray
    aggregate_relative_l2: float
    rank: int
    singular_values: np.ndarray
    design_shape: tuple[int, int]
    column_norms: np.ndarray
    condition_number: float
    column_normalized_condition_number: float


@dataclass(frozen=True)
class MagneticDiffusionDrivenResponse:
    """Backward-Euler nonhomogeneous local magnetic-diffusion response."""

    times: np.ndarray
    driver_values: np.ndarray
    source_projection: str
    initial_state_kind: str
    forcing_kind: str
    receiver_response: np.ndarray
    source_moment_projection: ModalSourceMomentProjection


def magnetic_diffusion_matrices(
    mesh,
    conductivity,
    *,
    mu: float = mu_0,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return H/J magnetic diffusion stiffness and mass matrices.

    The semi-discrete no-source H/J magnetic equation has the form
    ``M_mu dh/dt + C.T M_rho C h = 0``.  Its positive generalized eigenvalues
    define non-fitted magnetic recovery rates for local MMR/source-history
    diagnostics.
    """

    sigma = _cell_conductivity(mesh, conductivity)
    rho = 1.0 / sigma
    face_rho = mesh.get_face_inner_product(rho).tocsr()
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, float(mu))).tocsr()
    curl = mesh.edge_curl.tocsr()
    stiffness = (curl.T @ face_rho @ curl).tocsr()
    return stiffness, edge_mu


def magnetic_diffusion_mmr_initial_state(
    mesh,
    face_currents,
    *,
    conductivity=None,
    source_projection: str = "raw",
    mu: float = mu_0,
) -> np.ndarray:
    """Return MMR edge ``h`` states whose curl recovers face currents.

    ``face_currents`` is shaped ``(n_sources, mesh.n_faces)`` or
    ``(mesh.n_faces,)``.  The returned array is shaped
    ``(mesh.n_edges, n_sources)`` so it can be passed directly as
    ``initial_state`` to :func:`magnetic_diffusion_driven_response`.
    """

    currents = _face_source_vectors(mesh, face_currents)
    projection = _source_projection(source_projection)
    if projection == "charge_conserving":
        if conductivity is None:
            raise ValueError("conductivity is required for charge_conserving projection")
        currents = _project_source_vectors(
            mesh,
            conductivity,
            currents,
            source_projection=projection,
        )
    elif conductivity is not None:
        _cell_conductivity(mesh, conductivity)

    if not np.any(currents):
        return np.zeros((mesh.n_edges, currents.shape[0]), dtype=float)

    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, float(mu))).tocsr()
    edge_mu_inverse = _inverse_diagonal_matrix(edge_mu, "edge permeability mass")
    curl = mesh.edge_curl.tocsr()
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    stabilization = (
        divergence.T
        @ sp.diags(
            np.full(mesh.n_cells, 1.0 / float(mu)) / mesh.cell_volumes,
            format="csr",
        )
        @ divergence
    )
    matrix = (curl @ edge_mu_inverse @ curl.T - stabilization).tocsc()
    solver = spla.factorized(matrix)
    potentials = np.column_stack(
        [solver(currents[index]) for index in range(currents.shape[0])]
    )
    states = edge_mu_inverse @ (curl.T @ potentials)
    return np.asarray(states, dtype=float)


def tensor_mesh_cell_submesh(mesh, cell_indices, *, padding: int = 0) -> LocalTensorMeshSupport:
    """Return the smallest padded rectangular TensorMesh containing cells."""

    shape = _mesh_shape(mesh)
    cells = _cell_indices(cell_indices, upper=mesh.n_cells)
    padding = int(padding)
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    ijk = np.column_stack(np.unravel_index(cells, shape, order="F"))
    ijk_min = tuple(
        max(0, int(np.min(ijk[:, axis])) - padding) for axis in range(3)
    )
    ijk_max = tuple(
        min(shape[axis] - 1, int(np.max(ijk[:, axis])) + padding)
        for axis in range(3)
    )

    widths = [
        np.asarray(mesh.h[axis], dtype=float)[ijk_min[axis] : ijk_max[axis] + 1]
        for axis in range(3)
    ]
    nodes = _mesh_nodes(mesh)
    origin = tuple(float(nodes[axis][ijk_min[axis]]) for axis in range(3))
    local_mesh = TensorMesh(widths, origin=origin)

    local_shape = tuple(ijk_max[axis] - ijk_min[axis] + 1 for axis in range(3))
    local_ijk = np.indices(local_shape)
    global_ijk = tuple(
        (local_ijk[axis] + ijk_min[axis]).ravel(order="F") for axis in range(3)
    )
    global_cell_indices = np.ravel_multi_index(global_ijk, shape, order="F").astype(int)
    global_face_indices = _local_to_global_face_indices(mesh, ijk_min, ijk_max)
    global_edge_indices = _local_to_global_edge_indices(mesh, ijk_min, ijk_max)

    return LocalTensorMeshSupport(
        mesh=local_mesh,
        global_cell_indices=global_cell_indices,
        global_face_indices=global_face_indices,
        global_edge_indices=global_edge_indices,
        ijk_min=ijk_min,
        ijk_max=ijk_max,
    )


def local_magnetic_diffusion_positive_spectrum(
    mesh,
    conductivity,
    cell_indices,
    *,
    padding: int = 0,
    mu: float = mu_0,
    max_modes: int = 12,
    zero_tol: float = 1.0e-10,
    dense_dof_limit: int = 2000,
) -> LocalMagneticDiffusionSpectrum:
    """Compute the positive H/J magnetic spectrum on a local cell support."""

    support = tensor_mesh_cell_submesh(mesh, cell_indices, padding=padding)
    sigma = _cell_conductivity(mesh, conductivity)[support.global_cell_indices]
    spectrum = magnetic_diffusion_positive_spectrum(
        support.mesh,
        sigma,
        mu=mu,
        max_modes=max_modes,
        zero_tol=zero_tol,
        dense_dof_limit=dense_dof_limit,
    )
    return LocalMagneticDiffusionSpectrum(support=support, spectrum=spectrum)


def magnetic_diffusion_positive_spectrum(
    mesh,
    conductivity,
    *,
    mu: float = mu_0,
    max_modes: int = 12,
    zero_tol: float = 1.0e-10,
    dense_dof_limit: int = 2000,
) -> MagneticDiffusionSpectrum:
    """Compute sorted positive H/J magnetic diffusion eigenmodes.

    This dense helper is intended for small local-support operators.  Full-domain
    production spectra should use a targeted sparse eigensolver once the local
    recovery support is fixed.
    """

    max_modes = int(max_modes)
    if max_modes <= 0:
        raise ValueError("max_modes must be positive")
    stiffness, mass = magnetic_diffusion_matrices(mesh, conductivity, mu=mu)
    if stiffness.shape[0] > int(dense_dof_limit):
        raise ValueError("dense magnetic diffusion spectrum requires a smaller local mesh")

    raw_eigenvalues, raw_eigenvectors = la.eigh(
        stiffness.toarray(),
        mass.toarray(),
        check_finite=False,
    )
    positive_floor = max(
        float(zero_tol),
        float(zero_tol) * np.max(np.abs(raw_eigenvalues)),
    )
    keep = np.flatnonzero(raw_eigenvalues > positive_floor)
    eigenvalues = np.asarray(raw_eigenvalues[keep][:max_modes], dtype=float)
    eigenvectors = np.asarray(raw_eigenvectors[:, keep][:, :max_modes], dtype=float)
    discarded = raw_eigenvalues[raw_eigenvalues <= positive_floor]
    return MagneticDiffusionSpectrum(
        eigenvalues=eigenvalues,
        time_constants=1.0 / eigenvalues,
        eigenvectors=eigenvectors,
        eigenvalue_floor=float(positive_floor),
        discarded_count=int(discarded.size),
        raw_eigenvalue_min=float(np.min(raw_eigenvalues)),
        raw_eigenvalue_max=float(np.max(raw_eigenvalues)),
        last_discarded_eigenvalue=(
            float(discarded[-1]) if discarded.size else None
        ),
        first_kept_eigenvalue=float(eigenvalues[0]) if eigenvalues.size else None,
    )


def magnetic_diffusion_modal_coupling(
    mesh,
    conductivity,
    source_vectors,
    receiver_locations,
    *,
    mu: float = mu_0,
    max_modes: int = 12,
    receiver_mode: str = "face_current_biot",
    subdivisions: int = 1,
    zero_tol: float = 1.0e-10,
    dense_dof_limit: int = 2000,
) -> MagneticDiffusionModalCoupling:
    """Project face-source vectors through local magnetic diffusion modes.

    For ``M dh/dt + K h = f(t)`` with ``K h_k = lambda_k M h_k`` and
    ``h_k.T M h_l = delta_kl``, a face-source vector ``s`` drives mode ``k`` by
    ``h_k.T C.T M_rho s``.  The returned ``source_receiver_response`` stores the
    steady modal amplitude factor ``forcing / lambda`` times the requested
    receiver projection of each mode.
    """

    source_vectors = _face_source_vectors(mesh, source_vectors)
    locations = _locations(receiver_locations)
    spectrum = magnetic_diffusion_positive_spectrum(
        mesh,
        conductivity,
        mu=mu,
        max_modes=max_modes,
        zero_tol=zero_tol,
        dense_dof_limit=dense_dof_limit,
    )
    sigma = _cell_conductivity(mesh, conductivity)
    face_rho = mesh.get_face_inner_product(1.0 / sigma).tocsr()
    curl = mesh.edge_curl.tocsr()
    forcing_vectors = curl.T @ (face_rho @ source_vectors.T)
    modal_forcing = np.asarray(spectrum.eigenvectors.T @ forcing_vectors, dtype=float)
    modal_receiver_response = _modal_receiver_response(
        mesh,
        locations,
        spectrum.eigenvectors,
        receiver_mode=receiver_mode,
        subdivisions=subdivisions,
    )
    source_receiver_response = (
        modal_forcing[:, :, None, None]
        / spectrum.eigenvalues[:, None, None, None]
        * modal_receiver_response[:, None, :, :]
    )
    return MagneticDiffusionModalCoupling(
        eigenvalues=spectrum.eigenvalues,
        time_constants=spectrum.time_constants,
        eigenvectors=spectrum.eigenvectors,
        modal_forcing=modal_forcing,
        modal_receiver_response=np.asarray(modal_receiver_response, dtype=float),
        source_receiver_response=np.asarray(source_receiver_response, dtype=float),
        receiver_mode=str(receiver_mode).strip().lower(),
    )


def magnetic_diffusion_driven_response(
    mesh,
    conductivity,
    source_vectors,
    receiver_locations,
    *,
    time_steps,
    driver_values,
    mu: float = mu_0,
    receiver_mode: str = "face_current_biot",
    subdivisions: int = 1,
    source_projection: str = "raw",
    initial_state=None,
    initial_state_kind: str | None = None,
    forcing_vectors=None,
    forcing_kind: str = "source_edge_rhs",
) -> MagneticDiffusionDrivenResponse:
    """Solve a local driven H/J magnetic-diffusion response on a BE grid.

    The local diagnostic equation is

    ``M_mu (h^{n+1}-h^n)/dt + C.T M_rho C h^{n+1}
      = C.T M_rho s_f g^{n+1}``,

    with zero initial magnetic recovery state.  The receiver response is then
    projected back onto the same face source-moment response basis used by the
    runtime source-history hook.  This is a non-fitted local operator audit; it
    does not use empymod residuals.
    """

    mode = str(receiver_mode).strip().lower()
    if mode != "face_current_biot":
        raise ValueError("driven response currently requires receiver_mode='face_current_biot'")
    source_vectors = _face_source_vectors(mesh, source_vectors)
    source_projection = _source_projection(source_projection)
    driving_source_vectors = _project_source_vectors(
        mesh,
        conductivity,
        source_vectors,
        source_projection=source_projection,
    )
    locations = _locations(receiver_locations)
    steps = _time_steps(time_steps)
    driver = _driver_values(driver_values, expected=steps.size + 1)
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    stiffness, mass = magnetic_diffusion_matrices(mesh, conductivity, mu=mu)
    sigma = _cell_conductivity(mesh, conductivity)
    face_rho = mesh.get_face_inner_product(1.0 / sigma).tocsr()
    curl = mesh.edge_curl.tocsr()
    default_forcing = curl.T @ (face_rho @ driving_source_vectors.T)
    if forcing_vectors is None:
        forcing = np.asarray(default_forcing, dtype=float)
        forcing_kind = "source_edge_rhs"
    else:
        forcing = _edge_forcing_vectors(
            mesh,
            forcing_vectors,
            n_sources=source_vectors.shape[0],
        )
        forcing_kind = str(forcing_kind)
    receiver_matrix = face_current_biot_matrix(
        mesh,
        locations,
        subdivisions=subdivisions,
    )

    state = _initial_edge_state(
        mesh,
        initial_state,
        n_sources=source_vectors.shape[0],
    )
    if initial_state is None:
        if initial_state_kind not in (None, "zero"):
            raise ValueError("initial_state_kind must be 'zero' when initial_state is None")
        initial_state_kind = "zero"
    elif initial_state_kind is None:
        initial_state_kind = "provided"
    else:
        initial_state_kind = str(initial_state_kind)
    response = np.zeros(
        (steps.size + 1, source_vectors.shape[0], locations.shape[0], 3),
        dtype=float,
    )
    response[0] = np.einsum(
        "lcf,fs->slc",
        receiver_matrix,
        curl @ state,
    )
    solvers: dict[float, object] = {}
    for step_index, dt in enumerate(steps):
        dt = float(dt)
        solver = solvers.get(dt)
        if solver is None:
            system = (mass + dt * stiffness).tocsc()
            solver = spla.factorized(system)
            solvers[dt] = solver
        rhs = mass @ state + dt * np.asarray(forcing, dtype=float) * driver[step_index + 1]
        state = np.asarray(solver(rhs), dtype=float)
        face_currents = curl @ state
        response[step_index + 1] = np.einsum(
            "lcf,fs->slc",
            receiver_matrix,
            face_currents,
        )

    static_response = np.einsum("lcf,sf->slc", receiver_matrix, source_vectors)
    projection = project_modal_response_to_source_moments(
        static_response,
        response,
    )
    return MagneticDiffusionDrivenResponse(
        times=np.r_[0.0, np.cumsum(steps)],
        driver_values=driver,
        source_projection=source_projection,
        initial_state_kind=initial_state_kind,
        forcing_kind=forcing_kind,
        receiver_response=response,
        source_moment_projection=projection,
    )


def project_modal_response_to_source_moments(
    static_response,
    modal_source_receiver_response,
) -> ModalSourceMomentProjection:
    """Project modal receiver responses back onto source-moment responses.

    ``static_response`` has shape ``(n_source_moments, n_locations, n_components)``.
    ``modal_source_receiver_response`` has shape
    ``(n_modes, n_source_drives, n_locations, n_components)``.  The returned
    coefficients are shaped ``(n_modes, n_source_drives, n_source_moments)``.
    They are a non-fitted FV/MMR bridge: each modal source-drive receiver
    response is represented in the same source-moment basis used by the runtime
    source-history hook.
    """

    static_response = np.asarray(static_response, dtype=float)
    modal_response = np.asarray(modal_source_receiver_response, dtype=float)
    if static_response.ndim != 3:
        raise ValueError(
            "static_response must have shape "
            "(n_source_moments, n_locations, n_components)"
        )
    if modal_response.ndim != 4:
        raise ValueError(
            "modal_source_receiver_response must have shape "
            "(n_modes, n_source_drives, n_locations, n_components)"
        )
    if static_response.shape[0] == 0:
        raise ValueError("static_response must contain at least one source moment")
    if modal_response.shape[0] == 0 or modal_response.shape[1] == 0:
        raise ValueError("modal_source_receiver_response must contain modes and drives")
    if modal_response.shape[2:] != static_response.shape[1:]:
        raise ValueError(
            "modal_source_receiver_response location/component shape must match "
            "static_response"
        )

    n_modes, n_drives = modal_response.shape[:2]
    n_sources = static_response.shape[0]
    design = static_response.reshape(n_sources, -1).T
    targets = modal_response.reshape(n_modes * n_drives, -1).T
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        targets,
        rcond=None,
    )
    fitted_targets = design @ coefficients
    fitted_response = fitted_targets.T.reshape(modal_response.shape)
    coefficient_table = coefficients.T.reshape(n_modes, n_drives, n_sources)
    relative = np.asarray(
        [
            [
                relative_l2(
                    fitted_response[mode_index, drive_index],
                    modal_response[mode_index, drive_index],
                )
                for drive_index in range(n_drives)
            ]
            for mode_index in range(n_modes)
        ],
        dtype=float,
    )
    diagnostics = _projection_design_diagnostics(
        design,
        np.asarray(singular_values, dtype=float),
    )
    return ModalSourceMomentProjection(
        coefficients=np.asarray(coefficient_table, dtype=float),
        fitted_response=np.asarray(fitted_response, dtype=float),
        residual_response=np.asarray(fitted_response - modal_response, dtype=float),
        relative_l2=relative,
        aggregate_relative_l2=relative_l2(fitted_response, modal_response),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        design_shape=diagnostics["shape"],
        column_norms=diagnostics["column_norms"],
        condition_number=diagnostics["condition_number"],
        column_normalized_condition_number=diagnostics[
            "column_normalized_condition_number"
        ],
    )


def magnetic_diffusion_time_constants(
    mesh,
    conductivity,
    *,
    mu: float = mu_0,
    max_modes: int = 12,
    zero_tol: float = 1.0e-10,
    dense_dof_limit: int = 2000,
) -> np.ndarray:
    """Return sorted positive magnetic diffusion time constants."""

    return magnetic_diffusion_positive_spectrum(
        mesh,
        conductivity,
        mu=mu,
        max_modes=max_modes,
        zero_tol=zero_tol,
        dense_dof_limit=dense_dof_limit,
    ).time_constants


def _cell_conductivity(mesh, conductivity) -> np.ndarray:
    sigma = np.asarray(conductivity, dtype=float)
    if sigma.size == 1:
        sigma = np.full(mesh.n_cells, float(sigma.reshape(-1)[0]))
    if sigma.shape != (mesh.n_cells,):
        raise ValueError("conductivity must be scalar or cell-centered")
    if np.any(sigma <= 0.0):
        raise ValueError("conductivity must be positive")
    return sigma


def _face_source_vectors(mesh, source_vectors) -> np.ndarray:
    vectors = np.asarray(source_vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[1] != mesh.n_faces:
        raise ValueError("source_vectors must have shape (n_sources, mesh.n_faces)")
    if vectors.shape[0] == 0:
        raise ValueError("source_vectors must contain at least one vector")
    return vectors


def _locations(receiver_locations) -> np.ndarray:
    locations = np.asarray(receiver_locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (n_locations, 3)")
    return locations


def _modal_receiver_response(
    mesh,
    locations: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    receiver_mode: str,
    subdivisions: int,
) -> np.ndarray:
    mode = str(receiver_mode).strip().lower()
    if mode == "stored_h":
        receiver_matrix = np.zeros((locations.shape[0], 3, mesh.n_edges), dtype=float)
        for axis, component in enumerate(("Ex", "Ey", "Ez")):
            receiver_matrix[:, axis, :] = mesh.get_interpolation_matrix(
                locations,
                component,
            ).toarray()
        return np.einsum("lce,ek->klc", receiver_matrix, eigenvectors)
    if mode == "face_current_biot":
        face_currents = mesh.edge_curl.tocsr() @ eigenvectors
        receiver_matrix = face_current_biot_matrix(
            mesh,
            locations,
            subdivisions=subdivisions,
        )
        return np.einsum("lcf,fk->klc", receiver_matrix, face_currents)
    raise ValueError("receiver_mode must be 'stored_h' or 'face_current_biot'")


def _mesh_shape(mesh) -> tuple[int, int, int]:
    if not hasattr(mesh, "h") or len(mesh.h) != 3:
        raise ValueError("magnetic diffusion spectrum requires a 3D TensorMesh")
    return tuple(int(len(np.asarray(widths))) for widths in mesh.h)


def _mesh_nodes(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(hasattr(mesh, name) for name in ("nodes_x", "nodes_y", "nodes_z")):
        return (
            np.asarray(mesh.nodes_x, dtype=float),
            np.asarray(mesh.nodes_y, dtype=float),
            np.asarray(mesh.nodes_z, dtype=float),
        )
    origin = np.asarray(mesh.origin, dtype=float)
    return tuple(
        np.r_[origin[axis], origin[axis] + np.cumsum(np.asarray(mesh.h[axis], dtype=float))]
        for axis in range(3)
    )


def _local_to_global_face_indices(
    mesh,
    ijk_min: tuple[int, int, int],
    ijk_max: tuple[int, int, int],
) -> np.ndarray:
    x_faces = _structured_dof_indices(
        mesh.vnFx,
        (
            (ijk_min[0], ijk_max[0] + 1),
            (ijk_min[1], ijk_max[1]),
            (ijk_min[2], ijk_max[2]),
        ),
        offset=0,
    )
    y_faces = _structured_dof_indices(
        mesh.vnFy,
        (
            (ijk_min[0], ijk_max[0]),
            (ijk_min[1], ijk_max[1] + 1),
            (ijk_min[2], ijk_max[2]),
        ),
        offset=mesh.nFx,
    )
    z_faces = _structured_dof_indices(
        mesh.vnFz,
        (
            (ijk_min[0], ijk_max[0]),
            (ijk_min[1], ijk_max[1]),
            (ijk_min[2], ijk_max[2] + 1),
        ),
        offset=mesh.nFx + mesh.nFy,
    )
    return np.r_[x_faces, y_faces, z_faces].astype(int)


def _local_to_global_edge_indices(
    mesh,
    ijk_min: tuple[int, int, int],
    ijk_max: tuple[int, int, int],
) -> np.ndarray:
    x_edges = _structured_dof_indices(
        mesh.vnEx,
        (
            (ijk_min[0], ijk_max[0]),
            (ijk_min[1], ijk_max[1] + 1),
            (ijk_min[2], ijk_max[2] + 1),
        ),
        offset=0,
    )
    y_edges = _structured_dof_indices(
        mesh.vnEy,
        (
            (ijk_min[0], ijk_max[0] + 1),
            (ijk_min[1], ijk_max[1]),
            (ijk_min[2], ijk_max[2] + 1),
        ),
        offset=mesh.nEx,
    )
    z_edges = _structured_dof_indices(
        mesh.vnEz,
        (
            (ijk_min[0], ijk_max[0] + 1),
            (ijk_min[1], ijk_max[1] + 1),
            (ijk_min[2], ijk_max[2]),
        ),
        offset=mesh.nEx + mesh.nEy,
    )
    return np.r_[x_edges, y_edges, z_edges].astype(int)


def _structured_dof_indices(
    shape: tuple[int, int, int],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    offset: int,
) -> np.ndarray:
    ranges = [np.arange(start, stop + 1, dtype=int) for start, stop in bounds]
    grids = np.meshgrid(*ranges, indexing="ij")
    indices = np.ravel_multi_index(
        tuple(grid.ravel(order="F") for grid in grids),
        tuple(int(value) for value in shape),
        order="F",
    )
    return np.asarray(indices + int(offset), dtype=int)


def _cell_indices(cell_indices, *, upper: int) -> np.ndarray:
    cells = np.asarray(cell_indices, dtype=int).reshape(-1)
    if cells.size == 0:
        raise ValueError("cell_indices must contain at least one cell")
    if np.any(cells < 0) or np.any(cells >= int(upper)):
        raise ValueError("cell_indices contains out-of-range entries")
    return np.unique(cells)


def _time_steps(time_steps) -> np.ndarray:
    steps = np.asarray(time_steps, dtype=float).reshape(-1)
    if steps.size == 0:
        raise ValueError("time_steps must contain at least one step")
    if np.any(steps <= 0.0):
        raise ValueError("time_steps must be positive")
    return steps


def _driver_values(driver_values, *, expected: int) -> np.ndarray:
    values = np.asarray(driver_values, dtype=float).reshape(-1)
    if values.size != int(expected):
        raise ValueError("driver_values length must equal len(time_steps) + 1")
    return values


def _initial_edge_state(mesh, initial_state, *, n_sources: int) -> np.ndarray:
    if initial_state is None:
        return np.zeros((mesh.n_edges, int(n_sources)), dtype=float)
    values = np.asarray(initial_state, dtype=float)
    if values.ndim == 1:
        if int(n_sources) != 1:
            raise ValueError(
                "1D initial_state is only valid when there is exactly one source vector"
            )
        values = values.reshape(mesh.n_edges, 1)
    if values.shape != (mesh.n_edges, int(n_sources)):
        raise ValueError("initial_state must have shape (mesh.n_edges, n_sources)")
    return values.copy()


def _edge_forcing_vectors(mesh, forcing_vectors, *, n_sources: int) -> np.ndarray:
    values = np.asarray(forcing_vectors, dtype=float)
    if values.ndim == 1:
        if int(n_sources) != 1:
            raise ValueError(
                "1D forcing_vectors is only valid when there is exactly one source vector"
            )
        values = values.reshape(mesh.n_edges, 1)
    if values.shape != (mesh.n_edges, int(n_sources)):
        raise ValueError("forcing_vectors must have shape (mesh.n_edges, n_sources)")
    return values.copy()


def _source_projection(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"raw", "charge_conserving"}:
        raise ValueError("source_projection must be 'raw' or 'charge_conserving'")
    return normalized


def _project_source_vectors(
    mesh,
    conductivity,
    source_vectors: np.ndarray,
    *,
    source_projection: str,
) -> np.ndarray:
    if source_projection == "raw":
        return np.asarray(source_vectors, dtype=float)
    return np.asarray(
        [
            charge_conserving_face_current(mesh, conductivity, vector)
            for vector in np.asarray(source_vectors, dtype=float)
        ],
        dtype=float,
    )


def _inverse_diagonal_matrix(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError(f"{name} must be diagonal")
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError(f"{name} contains zero diagonal entries")
    return sp.diags(1.0 / diagonal, format="csr")


def _projection_design_diagnostics(
    design: np.ndarray,
    singular_values: np.ndarray,
) -> dict[str, np.ndarray | float | tuple[int, int]]:
    design = np.asarray(design, dtype=float)
    singular_values = np.asarray(singular_values, dtype=float)
    column_norms = np.linalg.norm(design, axis=0)
    if singular_values.size == 0 or singular_values[-1] == 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    normalized = design.copy()
    nonzero = column_norms > 0.0
    normalized[:, nonzero] /= column_norms[nonzero]
    if np.any(~nonzero):
        normalized[:, ~nonzero] = 0.0
    if normalized.size == 0:
        normalized_condition = float("inf")
    else:
        normalized_singular = np.linalg.svd(
            normalized,
            compute_uv=False,
        )
        if normalized_singular.size == 0 or normalized_singular[-1] == 0.0:
            normalized_condition = float("inf")
        else:
            normalized_condition = float(
                normalized_singular[0] / normalized_singular[-1]
            )
    return {
        "shape": tuple(int(value) for value in design.shape),
        "column_norms": np.asarray(column_norms, dtype=float),
        "condition_number": condition_number,
        "column_normalized_condition_number": normalized_condition,
    }
