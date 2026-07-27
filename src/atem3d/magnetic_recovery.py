"""Magnetic receiver recovery from cell-centered conduction currents.

中文说明：本模块用离散传导电流恢复磁接收响应。Biot–Savart 叉乘顺序、
接收点到积分点的方向和 H/B 单位必须统一；接收点靠近电流单元时还要通过
高阶求积、局部加密或近奇异积分检查结果稳定性。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def biot_savart_h_from_cell_currents(
    mesh,
    current_density,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Recover magnetic field from cell-centered volume currents.

    This is a magnetoquasistatic receiver recovery:

    ``H(r) = integral J(r') x (r-r') / (4 pi |r-r'|^3) dV``.

    The integral is evaluated with one midpoint sample per mesh cell.  It is a
    receiver recovery operator, not a replacement for the finite-volume time
    stepping equations.
    """

    current_density = np.asarray(current_density, dtype=float)
    if current_density.ndim == 1:
        if current_density.size != 3 * mesh.n_cells:
            raise ValueError("flat current_density must have length 3 * mesh.n_cells")
        current_density = current_density.reshape((mesh.n_cells, 3), order="F")
    if current_density.shape != (mesh.n_cells, 3):
        raise ValueError("current_density must have shape (mesh.n_cells, 3)")

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    fields = np.zeros((locations.shape[0], 3), dtype=float)
    centers, volumes, cell_indices = _cell_midpoint_quadrature(mesh, subdivisions)
    quadrature_current = current_density[cell_indices]
    for index, location in enumerate(locations):
        displacement = location[None, :] - centers
        radius = np.linalg.norm(displacement, axis=1)
        mask = radius > 0.0
        weights = np.zeros(centers.shape[0], dtype=float)
        weights[mask] = volumes[mask] / (4.0 * np.pi * radius[mask] ** 3)
        fields[index] = np.sum(
            weights[:, None] * np.cross(quadrature_current, displacement),
            axis=0,
        )
    return fields


def cell_current_density_biot_matrix(
    mesh,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Return the Biot matrix for flat cell-centered current density.

    The returned array has shape ``(n_locations, 3, 3 * mesh.n_cells)`` and maps
    ``current_density.reshape(-1, order="F")`` to receiver ``H``.
    """

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    centers, volumes, cell_indices = _cell_midpoint_quadrature(mesh, subdivisions)
    matrix = np.zeros((locations.shape[0], 3, 3 * mesh.n_cells), dtype=float)
    y_columns = cell_indices + mesh.n_cells
    z_columns = cell_indices + 2 * mesh.n_cells
    for index, location in enumerate(locations):
        displacement = location[None, :] - centers
        radius = np.linalg.norm(displacement, axis=1)
        weights = np.zeros(centers.shape[0], dtype=float)
        mask = radius > 0.0
        weights[mask] = volumes[mask] / (4.0 * np.pi * radius[mask] ** 3)
        dx = displacement[:, 0]
        dy = displacement[:, 1]
        dz = displacement[:, 2]

        np.add.at(matrix[index, 0], y_columns, weights * dz)
        np.add.at(matrix[index, 0], z_columns, -weights * dy)
        np.add.at(matrix[index, 1], cell_indices, -weights * dz)
        np.add.at(matrix[index, 1], z_columns, weights * dx)
        np.add.at(matrix[index, 2], cell_indices, weights * dy)
        np.add.at(matrix[index, 2], y_columns, -weights * dx)
    return matrix


def cell_current_biot_matrix(
    mesh,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Return the runtime ``current_biot`` matrix for edge current moments.

    This linearizes the same recovery used by ``magnetic_receiver_mode:
    current_biot``: edge current moments are divided by the unit edge mass,
    averaged to cell-centered vector currents, and integrated with the cell
    current Biot operator.
    """

    cell_matrix = cell_current_density_biot_matrix(
        mesh,
        locations,
        subdivisions=subdivisions,
    )
    inverse_edge_mass = 1.0 / _unit_edge_mass_diagonal(mesh)
    average_edge_to_cell = mesh.average_edge_to_cell_vector.tocsr()
    projection = average_edge_to_cell @ sp.diags(inverse_edge_mass, format="csr")
    matrix_2d = (projection.T @ cell_matrix.reshape(-1, 3 * mesh.n_cells).T).T
    return np.asarray(matrix_2d, dtype=float).reshape(
        cell_matrix.shape[0],
        3,
        mesh.n_edges,
    )


def face_current_biot_matrix(
    mesh,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Return the H/J ``current_biot`` matrix for face current moments.

    This linearizes ``HJMagneticSimulation._current_biot_h``: face current
    moments are averaged to cell-centered vector currents and integrated with
    the cell-current Biot operator.
    """

    cell_matrix = cell_current_density_biot_matrix(
        mesh,
        locations,
        subdivisions=subdivisions,
    )
    average_face_to_cell = mesh.average_face_to_cell_vector.tocsr()
    matrix_2d = (average_face_to_cell.T @ cell_matrix.reshape(-1, 3 * mesh.n_cells).T).T
    return np.asarray(matrix_2d, dtype=float).reshape(
        cell_matrix.shape[0],
        3,
        mesh.n_faces,
    )


def biot_savart_h_from_edge_current_moments(
    mesh,
    edge_current_moments,
    locations,
) -> np.ndarray:
    """Recover magnetic field from finite-volume edge current moments.

    ``edge_current_moments`` is the dual edge current vector, e.g.
    ``M_sigma e``.  Each entry is treated as the integrated vector-current
    moment associated with its edge basis function and evaluated at the edge
    location.  This preserves the FV constitutive current moment without first
    averaging it to cell centers.
    """

    edge_current_moments = np.asarray(edge_current_moments, dtype=float)
    if edge_current_moments.shape != (mesh.n_edges,):
        raise ValueError("edge_current_moments must have length mesh.n_edges")

    matrix = edge_current_biot_matrix(mesh, locations)
    return np.einsum("lce,e->lc", matrix, edge_current_moments)


def edge_current_biot_matrix(mesh, locations) -> np.ndarray:
    """Return the linear Biot matrix for FV edge current moments.

    The returned array has shape ``(n_locations, 3, mesh.n_edges)`` and maps an
    edge-current moment vector ``q`` to receiver magnetic field by
    ``H[l, c] = sum_e matrix[l, c, e] * q[e]``.
    """

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    edge_locations = np.vstack([mesh.edges_x, mesh.edges_y, mesh.edges_z])
    edge_axes = np.zeros((mesh.n_edges, 3), dtype=float)
    edge_axes[: mesh.n_edges_x, 0] = 1.0
    y_start = mesh.n_edges_x
    y_end = y_start + mesh.n_edges_y
    edge_axes[y_start:y_end, 1] = 1.0
    edge_axes[y_end:, 2] = 1.0

    matrix = np.zeros((locations.shape[0], 3, mesh.n_edges), dtype=float)
    for index, location in enumerate(locations):
        displacement = location[None, :] - edge_locations
        radius = np.linalg.norm(displacement, axis=1)
        mask = radius > 0.0
        weights = np.zeros(mesh.n_edges, dtype=float)
        weights[mask] = 1.0 / (4.0 * np.pi * radius[mask] ** 3)
        matrix[index] = (weights[:, None] * np.cross(edge_axes, displacement)).T
    return matrix


def biot_savart_h_from_edge_basis_currents(
    mesh,
    edge_current_field,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Recover magnetic field from edge-basis reconstructed currents.

    The input is an edge-vector current field, not a dual current moment.  Each
    cell is reconstructed with the lowest-order edge basis: the x component is
    bilinear in local y/z, the y component in x/z, and the z component in x/y.
    The resulting cell-local vector current is integrated with midpoint
    quadrature.
    """

    edge_current_field = np.asarray(edge_current_field, dtype=float)
    if edge_current_field.shape != (mesh.n_edges,):
        raise ValueError("edge_current_field must have length mesh.n_edges")

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    if not hasattr(mesh, "h") or len(mesh.h) != 3:
        raise ValueError("edge-basis recovery requires a 3D TensorMesh-like mesh")

    nx, ny, nz = (len(np.asarray(widths)) for widths in mesh.h)
    x_edges = edge_current_field[: mesh.n_edges_x].reshape(
        (nx, ny + 1, nz + 1),
        order="F",
    )
    y_start = mesh.n_edges_x
    y_end = y_start + mesh.n_edges_y
    y_edges = edge_current_field[y_start:y_end].reshape(
        (nx + 1, ny, nz + 1),
        order="F",
    )
    z_edges = edge_current_field[y_end:].reshape(
        (nx + 1, ny + 1, nz),
        order="F",
    )

    centers = np.asarray(mesh.cell_centers, dtype=float)
    volumes = np.asarray(mesh.cell_volumes, dtype=float) / float(subdivisions**3)
    widths = np.meshgrid(
        *[np.asarray(axis_widths, dtype=float) for axis_widths in mesh.h],
        indexing="ij",
    )
    cell_widths = np.column_stack([axis.ravel(order="F") for axis in widths])
    offsets_1d = (np.arange(subdivisions, dtype=float) + 0.5) / subdivisions
    fields = np.zeros((locations.shape[0], 3), dtype=float)

    for xi in offsets_1d:
        for eta in offsets_1d:
            for zeta in offsets_1d:
                current = _edge_basis_current_at_local_points(
                    x_edges,
                    y_edges,
                    z_edges,
                    xi,
                    eta,
                    zeta,
                )
                points = centers + (np.array([xi, eta, zeta]) - 0.5) * cell_widths
                _accumulate_biot_savart(fields, locations, points, current, volumes)
    return fields


def biot_savart_h_from_edge_basis_cell_ip_currents(
    mesh,
    electric_field,
    sigma_infinity,
    terms,
    memories,
    locations,
    subdivisions: int = 1,
    polarization_scale: float | str | np.ndarray = 1.0,
    initial_polarization_scale: float = 0.0,
    initial_memories=None,
) -> np.ndarray:
    """Recover ``H`` from cell-local edge-basis Debye constitutive currents.

    Unlike ``biot_savart_h_from_edge_basis_currents``, this reconstructs the
    electric field and each Debye memory inside every cell first, then applies
    that cell's own material parameters:

    ``J = sigma_inf E - sum(scale * delta_sigma_i y_i)``.

    This avoids smearing layer/IP contrasts onto shared edges before the
    receiver-side Biot integration.
    """

    electric_field = _validate_edge_vector(mesh, electric_field, "electric_field")
    terms = list(terms or [])
    memories = [np.asarray(memory, dtype=float) for memory in memories or []]
    if len(memories) != len(terms):
        raise ValueError("one memory vector is required for each Debye term")
    for memory in memories:
        _validate_edge_vector(mesh, memory, "memory")

    sigma = _cell_property(sigma_infinity, mesh.n_cells, "sigma_infinity")
    deltas = [
        _cell_property(term.delta_sigma, mesh.n_cells, "delta_sigma")
        for term in terms
    ]
    polarization_scale = _normalize_cell_ip_polarization_scale(
        polarization_scale,
        sigma,
        deltas,
    )
    initial_polarization_scale = float(initial_polarization_scale)
    if not np.isfinite(initial_polarization_scale):
        raise ValueError("initial_polarization_scale must be finite")
    if initial_polarization_scale != 0.0:
        if initial_memories is None:
            raise ValueError(
                "initial_memories are required when initial_polarization_scale is nonzero"
            )
        initial_memories = [
            _validate_edge_vector(mesh, memory, "initial_memory")
            for memory in initial_memories
        ]
        if len(initial_memories) != len(terms):
            raise ValueError("one initial memory vector is required for each Debye term")
    else:
        initial_memories = []

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    if not hasattr(mesh, "h") or len(mesh.h) != 3:
        raise ValueError("edge-basis recovery requires a 3D TensorMesh-like mesh")

    electric_components = _edge_component_arrays(mesh, electric_field)
    memory_components = [_edge_component_arrays(mesh, memory) for memory in memories]
    initial_components = [
        _edge_component_arrays(mesh, memory) for memory in initial_memories
    ]

    centers = np.asarray(mesh.cell_centers, dtype=float)
    volumes = np.asarray(mesh.cell_volumes, dtype=float) / float(subdivisions**3)
    widths = np.meshgrid(
        *[np.asarray(axis_widths, dtype=float) for axis_widths in mesh.h],
        indexing="ij",
    )
    cell_widths = np.column_stack([axis.ravel(order="F") for axis in widths])
    offsets_1d = (np.arange(subdivisions, dtype=float) + 0.5) / subdivisions
    fields = np.zeros((locations.shape[0], 3), dtype=float)

    for xi in offsets_1d:
        for eta in offsets_1d:
            for zeta in offsets_1d:
                electric = _edge_basis_current_at_local_points(
                    *electric_components,
                    xi,
                    eta,
                    zeta,
                )
                current = sigma[:, None] * electric
                for delta, components in zip(deltas, memory_components):
                    memory = _edge_basis_current_at_local_points(
                        *components,
                        xi,
                        eta,
                        zeta,
                    )
                    current -= _scaled_cell_memory_current(
                        delta,
                        memory,
                        polarization_scale,
                    )
                if initial_polarization_scale != 0.0:
                    for delta, components in zip(deltas, initial_components):
                        initial = _edge_basis_current_at_local_points(
                            *components,
                            xi,
                            eta,
                            zeta,
                        )
                        current += initial_polarization_scale * delta[:, None] * initial

                points = centers + (np.array([xi, eta, zeta]) - 0.5) * cell_widths
                _accumulate_biot_savart(fields, locations, points, current, volumes)
    return fields


def biot_savart_h_from_face_basis_cell_ip_currents(
    mesh,
    electric_field,
    sigma_infinity,
    terms,
    memories,
    locations,
    subdivisions: int = 1,
    polarization_scale: float | str | np.ndarray = 1.0,
    initial_polarization_scale: float = 0.0,
    initial_memories=None,
) -> np.ndarray:
    """Recover ``H`` from cell-local face-basis Debye constitutive currents.

    This is the H/J counterpart of
    ``biot_savart_h_from_edge_basis_cell_ip_currents``.  H/J electric fields and
    Debye memories live on faces, so each cell reconstructs the normal vector
    field from its two x, two y, and two z faces before applying that cell's
    material parameters.
    """

    electric_field = _validate_face_vector(mesh, electric_field, "electric_field")
    terms = list(terms or [])
    memories = [np.asarray(memory, dtype=float) for memory in memories or []]
    if len(memories) != len(terms):
        raise ValueError("one memory vector is required for each Debye term")
    for memory in memories:
        _validate_face_vector(mesh, memory, "memory")

    sigma = _cell_property(sigma_infinity, mesh.n_cells, "sigma_infinity")
    deltas = [
        _cell_property(term.delta_sigma, mesh.n_cells, "delta_sigma")
        for term in terms
    ]
    polarization_scale = _normalize_cell_ip_polarization_scale(
        polarization_scale,
        sigma,
        deltas,
    )
    initial_polarization_scale = float(initial_polarization_scale)
    if not np.isfinite(initial_polarization_scale):
        raise ValueError("initial_polarization_scale must be finite")
    if initial_polarization_scale != 0.0:
        if initial_memories is None:
            raise ValueError(
                "initial_memories are required when initial_polarization_scale is nonzero"
            )
        initial_memories = [
            _validate_face_vector(mesh, memory, "initial_memory")
            for memory in initial_memories
        ]
        if len(initial_memories) != len(terms):
            raise ValueError("one initial memory vector is required for each Debye term")
    else:
        initial_memories = []

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")

    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    if not hasattr(mesh, "h") or len(mesh.h) != 3:
        raise ValueError("face-basis recovery requires a 3D TensorMesh-like mesh")

    electric_components = _face_component_arrays(mesh, electric_field)
    memory_components = [_face_component_arrays(mesh, memory) for memory in memories]
    initial_components = [
        _face_component_arrays(mesh, memory) for memory in initial_memories
    ]

    centers = np.asarray(mesh.cell_centers, dtype=float)
    volumes = np.asarray(mesh.cell_volumes, dtype=float) / float(subdivisions**3)
    widths = np.meshgrid(
        *[np.asarray(axis_widths, dtype=float) for axis_widths in mesh.h],
        indexing="ij",
    )
    cell_widths = np.column_stack([axis.ravel(order="F") for axis in widths])
    offsets_1d = (np.arange(subdivisions, dtype=float) + 0.5) / subdivisions
    fields = np.zeros((locations.shape[0], 3), dtype=float)

    for xi in offsets_1d:
        for eta in offsets_1d:
            for zeta in offsets_1d:
                electric = _face_basis_current_at_local_points(
                    *electric_components,
                    xi,
                    eta,
                    zeta,
                )
                current = sigma[:, None] * electric
                for delta, components in zip(deltas, memory_components):
                    memory = _face_basis_current_at_local_points(
                        *components,
                        xi,
                        eta,
                        zeta,
                    )
                    current -= _scaled_cell_memory_current(
                        delta,
                        memory,
                        polarization_scale,
                    )
                if initial_polarization_scale != 0.0:
                    for delta, components in zip(deltas, initial_components):
                        initial = _face_basis_current_at_local_points(
                            *components,
                            xi,
                            eta,
                            zeta,
                        )
                        current += initial_polarization_scale * delta[:, None] * initial

                points = centers + (np.array([xi, eta, zeta]) - 0.5) * cell_widths
                _accumulate_biot_savart(fields, locations, points, current, volumes)
    return fields


def biot_savart_h_from_face_basis_currents(
    mesh,
    face_current_field,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Recover ``H`` from an H/J face-basis current field."""

    return biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        face_current_field,
        np.ones(mesh.n_cells, dtype=float),
        [],
        [],
        locations,
        subdivisions=subdivisions,
    )


def edge_basis_biot_matrix(
    mesh,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Return a diagnostic explicit matrix for edge-basis Biot recovery.

    This constructs one column per edge by applying the existing edge-basis
    recovery to unit edge fields.  It is intended for small-mesh derivations and
    tests where an explicit ``H = B q`` operator is more useful than speed.
    """

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    matrix = np.zeros((locations.shape[0], 3, mesh.n_edges), dtype=float)
    unit = np.zeros(mesh.n_edges, dtype=float)
    for edge_index in range(mesh.n_edges):
        unit[edge_index] = 1.0
        matrix[:, :, edge_index] = biot_savart_h_from_edge_basis_currents(
            mesh,
            unit,
            locations,
            subdivisions=subdivisions,
        )
        unit[edge_index] = 0.0
    return matrix


def face_basis_biot_matrix(
    mesh,
    locations,
    subdivisions: int = 1,
) -> np.ndarray:
    """Return a diagnostic explicit matrix for H/J face-basis Biot recovery."""

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    matrix = np.zeros((locations.shape[0], 3, mesh.n_faces), dtype=float)
    unit = np.zeros(mesh.n_faces, dtype=float)
    for face_index in range(mesh.n_faces):
        unit[face_index] = 1.0
        matrix[:, :, face_index] = biot_savart_h_from_face_basis_currents(
            mesh,
            unit,
            locations,
            subdivisions=subdivisions,
        )
        unit[face_index] = 0.0
    return matrix


def edge_basis_cell_ip_biot_matrices(
    mesh,
    sigma_infinity,
    terms,
    locations,
    subdivisions: int = 1,
    polarization_scale: float | str | np.ndarray = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return explicit small-mesh matrices for cell-local IP Biot recovery.

    The first matrix maps the electric edge field to the ohmic contribution.
    Each matrix in the returned list maps one Debye memory edge field to its
    signed polarization-current contribution, including ``polarization_scale``.
    """

    locations = np.asarray(locations, dtype=float)
    if locations.ndim == 1:
        locations = locations.reshape(1, 3)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    subdivisions = int(subdivisions)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    terms = list(terms or [])
    zero_edge = np.zeros(mesh.n_edges, dtype=float)
    zero_memories = [zero_edge.copy() for _ in terms]
    ohmic = np.zeros((locations.shape[0], 3, mesh.n_edges), dtype=float)
    unit = np.zeros(mesh.n_edges, dtype=float)
    for edge_index in range(mesh.n_edges):
        unit[edge_index] = 1.0
        ohmic[:, :, edge_index] = biot_savart_h_from_edge_basis_cell_ip_currents(
            mesh,
            unit,
            sigma_infinity,
            terms,
            zero_memories,
            locations,
            subdivisions=subdivisions,
            polarization_scale=polarization_scale,
        )
        unit[edge_index] = 0.0

    memory_matrices: list[np.ndarray] = []
    for term_index in range(len(terms)):
        matrix = np.zeros((locations.shape[0], 3, mesh.n_edges), dtype=float)
        memories = [zero_edge.copy() for _ in terms]
        for edge_index in range(mesh.n_edges):
            memories[term_index][edge_index] = 1.0
            matrix[:, :, edge_index] = biot_savart_h_from_edge_basis_cell_ip_currents(
                mesh,
                zero_edge,
                sigma_infinity,
                terms,
                memories,
                locations,
                subdivisions=subdivisions,
                polarization_scale=polarization_scale,
            )
            memories[term_index][edge_index] = 0.0
        memory_matrices.append(matrix)

    return ohmic, memory_matrices


def _edge_basis_current_at_local_points(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    jx = (
        (1.0 - eta) * (1.0 - zeta) * x_edges[:, :-1, :-1]
        + eta * (1.0 - zeta) * x_edges[:, 1:, :-1]
        + (1.0 - eta) * zeta * x_edges[:, :-1, 1:]
        + eta * zeta * x_edges[:, 1:, 1:]
    )
    jy = (
        (1.0 - xi) * (1.0 - zeta) * y_edges[:-1, :, :-1]
        + xi * (1.0 - zeta) * y_edges[1:, :, :-1]
        + (1.0 - xi) * zeta * y_edges[:-1, :, 1:]
        + xi * zeta * y_edges[1:, :, 1:]
    )
    jz = (
        (1.0 - xi) * (1.0 - eta) * z_edges[:-1, :-1, :]
        + xi * (1.0 - eta) * z_edges[1:, :-1, :]
        + (1.0 - xi) * eta * z_edges[:-1, 1:, :]
        + xi * eta * z_edges[1:, 1:, :]
    )
    return np.column_stack(
        [
            jx.ravel(order="F"),
            jy.ravel(order="F"),
            jz.ravel(order="F"),
        ]
    )


def _face_basis_current_at_local_points(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    z_faces: np.ndarray,
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    jx = (1.0 - xi) * x_faces[:-1, :, :] + xi * x_faces[1:, :, :]
    jy = (1.0 - eta) * y_faces[:, :-1, :] + eta * y_faces[:, 1:, :]
    jz = (1.0 - zeta) * z_faces[:, :, :-1] + zeta * z_faces[:, :, 1:]
    return np.column_stack(
        [
            jx.ravel(order="F"),
            jy.ravel(order="F"),
            jz.ravel(order="F"),
        ]
    )


def _validate_edge_vector(mesh, values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (mesh.n_edges,):
        raise ValueError(f"{name} must have length mesh.n_edges")
    return values


def _validate_face_vector(mesh, values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (mesh.n_faces,):
        raise ValueError(f"{name} must have length mesh.n_faces")
    return values


def _unit_edge_mass_diagonal(mesh) -> np.ndarray:
    matrix = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError("current_biot matrix recovery requires diagonal edge mass")
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError("unit edge mass contains zero diagonal entries")
    return diagonal


def _edge_component_arrays(mesh, edge_field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = (len(np.asarray(widths)) for widths in mesh.h)
    x_edges = edge_field[: mesh.n_edges_x].reshape(
        (nx, ny + 1, nz + 1),
        order="F",
    )
    y_start = mesh.n_edges_x
    y_end = y_start + mesh.n_edges_y
    y_edges = edge_field[y_start:y_end].reshape(
        (nx + 1, ny, nz + 1),
        order="F",
    )
    z_edges = edge_field[y_end:].reshape(
        (nx + 1, ny + 1, nz),
        order="F",
    )
    return x_edges, y_edges, z_edges


def _face_component_arrays(mesh, face_field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = (len(np.asarray(widths)) for widths in mesh.h)
    x_faces = face_field[: mesh.n_faces_x].reshape(
        (nx + 1, ny, nz),
        order="F",
    )
    y_start = mesh.n_faces_x
    y_end = y_start + mesh.n_faces_y
    y_faces = face_field[y_start:y_end].reshape(
        (nx, ny + 1, nz),
        order="F",
    )
    z_faces = face_field[y_end:].reshape(
        (nx, ny, nz + 1),
        order="F",
    )
    return x_faces, y_faces, z_faces


def _cell_property(values, n_cells: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(n_cells, float(values))
    if values.shape == (1,):
        return np.full(n_cells, float(values[0]))
    if values.shape == (n_cells,):
        return values.copy()
    raise ValueError(f"{name} must be scalar or have length mesh.n_cells")


def _normalize_cell_ip_polarization_scale(
    scale: float | str | np.ndarray,
    sigma: np.ndarray,
    deltas: list[np.ndarray],
) -> float | np.ndarray:
    if isinstance(scale, str):
        normalized = scale.strip().lower()
        if normalized != "low_frequency_ratio":
            raise ValueError(
                "polarization_scale must be nonnegative or 'low_frequency_ratio'"
            )
        low_frequency_sigma = sigma.copy()
        for delta in deltas:
            low_frequency_sigma -= delta
        if np.any(low_frequency_sigma <= 0.0):
            raise ValueError("low-frequency conductivity must remain positive")
        return sigma / low_frequency_sigma

    array = np.asarray(scale, dtype=float)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError("component polarization_scale must have length 3")
        if np.any(array < 0.0):
            raise ValueError("polarization_scale must be nonnegative")
        return array.copy()
    scalar = float(array)
    if scalar < 0.0:
        raise ValueError("polarization_scale must be nonnegative")
    return scalar


def _scaled_cell_memory_current(
    delta: np.ndarray,
    memory: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    if isinstance(scale, np.ndarray) and scale.shape == (3,):
        return delta[:, None] * memory * scale[None, :]
    if isinstance(scale, np.ndarray):
        return (scale * delta)[:, None] * memory
    return float(scale) * delta[:, None] * memory


def _accumulate_biot_savart(
    fields: np.ndarray,
    locations: np.ndarray,
    points: np.ndarray,
    current_density: np.ndarray,
    volumes: np.ndarray,
) -> None:
    for index, location in enumerate(locations):
        displacement = location[None, :] - points
        radius = np.linalg.norm(displacement, axis=1)
        mask = radius > 0.0
        weights = np.zeros(points.shape[0], dtype=float)
        weights[mask] = volumes[mask] / (4.0 * np.pi * radius[mask] ** 3)
        fields[index] += np.sum(
            weights[:, None] * np.cross(current_density, displacement),
            axis=0,
        )


def _cell_midpoint_quadrature(mesh, subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if subdivisions == 1:
        return (
            np.asarray(mesh.cell_centers, dtype=float),
            np.asarray(mesh.cell_volumes, dtype=float),
            np.arange(mesh.n_cells, dtype=int),
        )

    if not hasattr(mesh, "h"):
        raise ValueError("subcell quadrature requires a TensorMesh-like mesh with h widths")

    widths = np.meshgrid(*[np.asarray(axis_widths, dtype=float) for axis_widths in mesh.h], indexing="ij")
    cell_widths = np.column_stack([axis.ravel(order="F") for axis in widths])
    offsets_1d = (np.arange(subdivisions, dtype=float) + 0.5) / subdivisions - 0.5
    offsets = np.array(np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij"))
    offsets = offsets.reshape(3, -1).T

    centers = np.asarray(mesh.cell_centers, dtype=float)
    quadrature_points = (
        centers[:, None, :] + offsets[None, :, :] * cell_widths[:, None, :]
    ).reshape(-1, 3)
    quadrature_volumes = np.repeat(
        np.asarray(mesh.cell_volumes, dtype=float) / float(subdivisions**3),
        subdivisions**3,
    )
    cell_indices = np.repeat(np.arange(mesh.n_cells, dtype=int), subdivisions**3)
    return quadrature_points, quadrature_volumes, cell_indices
