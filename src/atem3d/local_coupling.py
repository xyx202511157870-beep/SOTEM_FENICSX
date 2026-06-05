"""Local FV edge supports for source/MMR coupling diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalEdgeBasis:
    """Canonical edge-current basis over source/receiver local support."""

    source_edge_indices: np.ndarray
    source_cell_indices: np.ndarray
    receiver_cell_indices: list[np.ndarray]
    support_cell_indices: np.ndarray
    support_edge_indices: np.ndarray
    basis_vectors: np.ndarray
    basis_labels: list[str]


@dataclass(frozen=True)
class EdgeBasis:
    """Canonical edge-current vectors for an explicit edge index set."""

    edge_indices: np.ndarray
    basis_vectors: np.ndarray
    basis_labels: list[str]


@dataclass(frozen=True)
class FaceBasis:
    """Canonical face-current vectors for an explicit face index set."""

    face_indices: np.ndarray
    basis_vectors: np.ndarray
    basis_labels: list[str]


@dataclass(frozen=True)
class LocalCellSupport:
    """Source/receiver cell support for edge- or face-based local diagnostics."""

    field_location: str
    source_dof_indices: np.ndarray
    source_cell_indices: np.ndarray
    receiver_cell_indices: list[np.ndarray]
    support_cell_indices: np.ndarray


def canonical_edge_basis(mesh, edge_indices, *, label_prefix: str = "edge") -> EdgeBasis:
    """Return one unit edge-current vector per selected global edge index."""

    indices = _unique_indices(edge_indices, upper=mesh.n_edges, name="edge_indices")
    basis_vectors = np.zeros((indices.size, mesh.n_edges), dtype=float)
    if indices.size:
        basis_vectors[np.arange(indices.size), indices] = 1.0
    labels = [f"{label_prefix}:{edge_index}" for edge_index in indices]
    return EdgeBasis(
        edge_indices=indices,
        basis_vectors=basis_vectors,
        basis_labels=labels,
    )


def source_edge_moment_basis(
    mesh,
    source_vector,
    *,
    start,
    end,
    max_degree: int = 2,
    degrees=None,
    source_edge_atol: float = 0.0,
) -> EdgeBasis:
    """Return low-order longitudinal moment vectors on active source edges.

    The normalized source coordinate is ``xi = 2 * t - 1`` with ``t`` the
    projection of each active edge location onto the straight source segment.
    Degree zero is the projected source vector itself; higher degrees multiply
    that vector by ``xi**degree`` on the same active source-edge support.
    """

    source = np.asarray(source_vector, dtype=float)
    if source.shape != (mesh.n_edges,):
        raise ValueError("source_vector must have length mesh.n_edges")
    max_degree = int(max_degree)
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    if degrees is None:
        degree_values = list(range(max_degree + 1))
    else:
        degree_values = [int(degree) for degree in degrees]
        if not degree_values:
            raise ValueError("degrees must contain at least one value")
        if any(degree < 0 for degree in degree_values):
            raise ValueError("degrees must be nonnegative")
    source_edge_atol = float(source_edge_atol)
    if source_edge_atol < 0.0:
        raise ValueError("source_edge_atol must be nonnegative")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must be 3D coordinates")
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        raise ValueError("source segment must have distinct endpoints")

    active_edges = np.flatnonzero(np.abs(source) > source_edge_atol).astype(int)
    edge_locations = _edge_locations(mesh)[active_edges]
    t = ((edge_locations - start[None, :]) @ direction) / length_squared
    xi = 2.0 * t - 1.0

    basis_vectors = np.zeros((len(degree_values), mesh.n_edges), dtype=float)
    for row, degree in enumerate(degree_values):
        basis_vectors[row, active_edges] = source[active_edges] * xi**degree
    return EdgeBasis(
        edge_indices=active_edges,
        basis_vectors=basis_vectors,
        basis_labels=[f"source_moment:{degree}" for degree in degree_values],
    )


def source_face_moment_basis(
    mesh,
    source_vector,
    *,
    start,
    end,
    max_degree: int = 2,
    degrees=None,
    source_face_atol: float = 0.0,
) -> FaceBasis:
    """Return low-order longitudinal moment vectors on active source faces."""

    source = np.asarray(source_vector, dtype=float)
    if source.shape != (mesh.n_faces,):
        raise ValueError("source_vector must have length mesh.n_faces")
    max_degree = int(max_degree)
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    if degrees is None:
        degree_values = list(range(max_degree + 1))
    else:
        degree_values = [int(degree) for degree in degrees]
        if not degree_values:
            raise ValueError("degrees must contain at least one value")
        if any(degree < 0 for degree in degree_values):
            raise ValueError("degrees must be nonnegative")
    source_face_atol = float(source_face_atol)
    if source_face_atol < 0.0:
        raise ValueError("source_face_atol must be nonnegative")
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must be 3D coordinates")
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        raise ValueError("source segment must have distinct endpoints")

    active_faces = np.flatnonzero(np.abs(source) > source_face_atol).astype(int)
    face_locations = _face_locations(mesh)[active_faces]
    t = ((face_locations - start[None, :]) @ direction) / length_squared
    xi = 2.0 * t - 1.0

    basis_vectors = np.zeros((len(degree_values), mesh.n_faces), dtype=float)
    for row, degree in enumerate(degree_values):
        basis_vectors[row, active_faces] = source[active_faces] * xi**degree
    return FaceBasis(
        face_indices=active_faces,
        basis_vectors=basis_vectors,
        basis_labels=[f"source_face_moment:{degree}" for degree in degree_values],
    )


def edge_indices_for_cells(mesh, cell_indices) -> np.ndarray:
    """Return all global edge indices touching the requested TensorMesh cells."""

    nx, ny, nz = _mesh_shape(mesh)
    cells = _unique_indices(cell_indices, upper=mesh.n_cells, name="cell_indices")
    if cells.size == 0:
        return np.array([], dtype=int)

    shape_x = (nx, ny + 1, nz + 1)
    shape_y = (nx + 1, ny, nz + 1)
    shape_z = (nx + 1, ny + 1, nz)
    y_offset = int(mesh.n_edges_x)
    z_offset = int(mesh.n_edges_x + mesh.n_edges_y)

    edges: list[int] = []
    for cell in cells:
        i, j, k = np.unravel_index(int(cell), (nx, ny, nz), order="F")
        edges.extend(
            int(np.ravel_multi_index(index, shape_x, order="F"))
            for index in (
                (i, j, k),
                (i, j + 1, k),
                (i, j, k + 1),
                (i, j + 1, k + 1),
            )
        )
        edges.extend(
            y_offset + int(np.ravel_multi_index(index, shape_y, order="F"))
            for index in (
                (i, j, k),
                (i + 1, j, k),
                (i, j, k + 1),
                (i + 1, j, k + 1),
            )
        )
        edges.extend(
            z_offset + int(np.ravel_multi_index(index, shape_z, order="F"))
            for index in (
                (i, j, k),
                (i + 1, j, k),
                (i, j + 1, k),
                (i + 1, j + 1, k),
            )
        )
    return np.unique(np.asarray(edges, dtype=int))


def adjacent_cells_for_edges(mesh, edge_indices) -> np.ndarray:
    """Return all TensorMesh cells adjacent to the requested global edges."""

    nx, ny, nz = _mesh_shape(mesh)
    edges = _unique_indices(edge_indices, upper=mesh.n_edges, name="edge_indices")
    if edges.size == 0:
        return np.array([], dtype=int)

    cells: list[int] = []
    for edge in edges:
        edge = int(edge)
        if edge < mesh.n_edges_x:
            i, j, k = np.unravel_index(edge, (nx, ny + 1, nz + 1), order="F")
            candidates = ((i, j + dj, k + dk) for dj in (-1, 0) for dk in (-1, 0))
        elif edge < mesh.n_edges_x + mesh.n_edges_y:
            local = edge - mesh.n_edges_x
            i, j, k = np.unravel_index(local, (nx + 1, ny, nz + 1), order="F")
            candidates = ((i + di, j, k + dk) for di in (-1, 0) for dk in (-1, 0))
        else:
            local = edge - mesh.n_edges_x - mesh.n_edges_y
            i, j, k = np.unravel_index(local, (nx + 1, ny + 1, nz), order="F")
            candidates = ((i + di, j + dj, k) for di in (-1, 0) for dj in (-1, 0))

        for candidate in candidates:
            ci, cj, ck = candidate
            if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
                cells.append(int(np.ravel_multi_index(candidate, (nx, ny, nz), order="F")))
    return np.unique(np.asarray(cells, dtype=int))


def adjacent_cells_for_faces(mesh, face_indices) -> np.ndarray:
    """Return all TensorMesh cells adjacent to the requested global faces."""

    nx, ny, nz = _mesh_shape(mesh)
    faces = _unique_indices(face_indices, upper=mesh.n_faces, name="face_indices")
    if faces.size == 0:
        return np.array([], dtype=int)

    x_count = int(mesh.n_faces_x)
    y_count = int(mesh.n_faces_y)
    cells: list[int] = []
    for face in faces:
        face = int(face)
        if face < x_count:
            i, j, k = np.unravel_index(face, (nx + 1, ny, nz), order="F")
            candidates = ((i - 1, j, k), (i, j, k))
        elif face < x_count + y_count:
            local = face - x_count
            i, j, k = np.unravel_index(local, (nx, ny + 1, nz), order="F")
            candidates = ((i, j - 1, k), (i, j, k))
        else:
            local = face - x_count - y_count
            i, j, k = np.unravel_index(local, (nx, ny, nz + 1), order="F")
            candidates = ((i, j, k - 1), (i, j, k))

        for candidate in candidates:
            ci, cj, ck = candidate
            if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
                cells.append(int(np.ravel_multi_index(candidate, (nx, ny, nz), order="F")))
    return np.unique(np.asarray(cells, dtype=int))


def expand_cell_indices(mesh, cell_indices, radius: int = 0) -> np.ndarray:
    """Expand cells by an integer TensorMesh index radius."""

    nx, ny, nz = _mesh_shape(mesh)
    cells = _unique_indices(cell_indices, upper=mesh.n_cells, name="cell_indices")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if cells.size == 0 or radius == 0:
        return cells

    expanded: list[int] = []
    for cell in cells:
        i, j, k = np.unravel_index(int(cell), (nx, ny, nz), order="F")
        for ii in range(max(0, i - radius), min(nx, i + radius + 1)):
            for jj in range(max(0, j - radius), min(ny, j + radius + 1)):
                for kk in range(max(0, k - radius), min(nz, k + radius + 1)):
                    expanded.append(
                        int(np.ravel_multi_index((ii, jj, kk), (nx, ny, nz), order="F"))
                    )
    return np.unique(np.asarray(expanded, dtype=int))


def nearest_cell_indices(mesh, locations) -> np.ndarray:
    """Return the closest cell-center index for each receiver location."""

    points = _as_locations(locations)
    if points.shape[0] == 0:
        return np.array([], dtype=int)
    centers = np.asarray(mesh.cell_centers, dtype=float)
    distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.asarray(np.argmin(distances, axis=1), dtype=int)


def local_edge_basis(
    mesh,
    source_vector,
    receiver_locations,
    *,
    source_cell_radius: int = 0,
    receiver_cell_radius: int = 0,
    source_edge_atol: float = 0.0,
) -> LocalEdgeBasis:
    """Build canonical edge-current basis vectors on local source/receiver support.

    The support is the union of cells adjacent to nonzero source edges and cells
    nearest to receiver locations, optionally expanded by integer cell radii.
    Each returned basis vector is a unit edge-current vector on one support edge.
    """

    source = np.asarray(source_vector, dtype=float)
    if source.shape != (mesh.n_edges,):
        raise ValueError("source_vector must have length mesh.n_edges")
    source_edge_atol = float(source_edge_atol)
    if source_edge_atol < 0.0:
        raise ValueError("source_edge_atol must be nonnegative")

    source_edges = np.flatnonzero(np.abs(source) > source_edge_atol).astype(int)
    source_cells = expand_cell_indices(
        mesh,
        adjacent_cells_for_edges(mesh, source_edges),
        radius=source_cell_radius,
    )

    receiver_cells: list[np.ndarray] = []
    for cell in nearest_cell_indices(mesh, receiver_locations):
        receiver_cells.append(
            expand_cell_indices(mesh, [int(cell)], radius=receiver_cell_radius)
        )

    support_parts = [source_cells, *receiver_cells]
    if support_parts:
        support_cells = np.unique(
            np.concatenate([part for part in support_parts if part.size])
            if any(part.size for part in support_parts)
            else np.array([], dtype=int)
        )
    else:
        support_cells = np.array([], dtype=int)
    support_edges = edge_indices_for_cells(mesh, support_cells)

    edge_basis = canonical_edge_basis(mesh, support_edges)
    return LocalEdgeBasis(
        source_edge_indices=source_edges,
        source_cell_indices=source_cells,
        receiver_cell_indices=receiver_cells,
        support_cell_indices=support_cells,
        support_edge_indices=support_edges,
        basis_vectors=edge_basis.basis_vectors,
        basis_labels=edge_basis.basis_labels,
    )


def local_cell_support(
    mesh,
    source_vector,
    receiver_locations,
    *,
    field_location: str,
    source_cell_radius: int = 0,
    receiver_cell_radius: int = 0,
    source_atol: float = 0.0,
) -> LocalCellSupport:
    """Build source/receiver cell support for local recovery diagnostics."""

    field_location = _normalize_field_location(field_location)
    source = np.asarray(source_vector, dtype=float)
    expected = mesh.n_edges if field_location == "edge" else mesh.n_faces
    if source.shape != (expected,):
        raise ValueError(f"{field_location} source_vector has wrong length")
    source_atol = float(source_atol)
    if source_atol < 0.0:
        raise ValueError("source_atol must be nonnegative")

    source_dofs = np.flatnonzero(np.abs(source) > source_atol).astype(int)
    if field_location == "edge":
        adjacent = adjacent_cells_for_edges(mesh, source_dofs)
    else:
        adjacent = adjacent_cells_for_faces(mesh, source_dofs)
    source_cells = expand_cell_indices(
        mesh,
        adjacent,
        radius=source_cell_radius,
    )

    receiver_cells: list[np.ndarray] = []
    for cell in nearest_cell_indices(mesh, receiver_locations):
        receiver_cells.append(
            expand_cell_indices(mesh, [int(cell)], radius=receiver_cell_radius)
        )

    return LocalCellSupport(
        field_location=field_location,
        source_dof_indices=source_dofs,
        source_cell_indices=source_cells,
        receiver_cell_indices=receiver_cells,
        support_cell_indices=_combine_cell_supports([source_cells, *receiver_cells]),
    )


def _mesh_shape(mesh) -> tuple[int, int, int]:
    if not hasattr(mesh, "h") or len(mesh.h) != 3:
        raise ValueError("local coupling requires a 3D TensorMesh-like mesh")
    return tuple(int(len(np.asarray(widths))) for widths in mesh.h)


def _edge_locations(mesh) -> np.ndarray:
    if hasattr(mesh, "edges"):
        return np.asarray(mesh.edges, dtype=float)
    required = ("edges_x", "edges_y", "edges_z")
    if not all(hasattr(mesh, name) for name in required):
        raise ValueError("edge locations require a TensorMesh-like mesh")
    return np.vstack([mesh.edges_x, mesh.edges_y, mesh.edges_z])


def _face_locations(mesh) -> np.ndarray:
    if hasattr(mesh, "faces"):
        return np.asarray(mesh.faces, dtype=float)
    required = ("faces_x", "faces_y", "faces_z")
    if not all(hasattr(mesh, name) for name in required):
        raise ValueError("face locations require a TensorMesh-like mesh")
    return np.vstack([mesh.faces_x, mesh.faces_y, mesh.faces_z])


def _unique_indices(indices, *, upper: int, name: str) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if values.size and (np.any(values < 0) or np.any(values >= int(upper))):
        raise ValueError(f"{name} contains an out-of-range index")
    return np.unique(values)


def _combine_cell_supports(parts: list[np.ndarray]) -> np.ndarray:
    if not parts or not any(part.size for part in parts):
        return np.array([], dtype=int)
    return np.unique(np.concatenate([part for part in parts if part.size]))


def _normalize_field_location(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"edge", "face"}:
        raise ValueError("field_location must be 'edge' or 'face'")
    return normalized


def _as_locations(locations) -> np.ndarray:
    values = np.asarray(locations, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    return values
