"""CPML profile and curl-splitting utilities for structured FV experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CPMLConfig:
    """Parameters for convolutional PML stretching profiles."""

    thickness_cells: int
    sigma_max: float
    alpha_max: float = 0.0
    kappa_max: float = 1.0
    power: float = 2.0

    def __post_init__(self) -> None:
        if self.thickness_cells < 0:
            raise ValueError("thickness_cells must be nonnegative")
        if self.sigma_max < 0.0:
            raise ValueError("sigma_max must be nonnegative")
        if self.alpha_max < 0.0:
            raise ValueError("alpha_max must be nonnegative")
        if self.kappa_max < 1.0:
            raise ValueError("kappa_max must be at least 1")
        if self.power <= 0.0:
            raise ValueError("power must be positive")


@dataclass(frozen=True)
class CPMLProfiles:
    """Direction-wise CPML coefficients on mesh faces and edges."""

    face_b: np.ndarray
    face_c: np.ndarray
    face_kappa: np.ndarray
    edge_b: np.ndarray
    edge_c: np.ndarray
    edge_kappa: np.ndarray


@dataclass(frozen=True)
class CPMLState:
    """Auxiliary memory arrays for stretched curl terms."""

    face_curl_memory: np.ndarray
    edge_curl_memory: np.ndarray

    @classmethod
    def zeros(cls, mesh) -> "CPMLState":
        return cls(
            face_curl_memory=np.zeros((3, mesh.n_faces), dtype=float),
            edge_curl_memory=np.zeros((3, mesh.n_edges), dtype=float),
        )


@dataclass(frozen=True)
class CurlSplit:
    """Directional pieces of ``mesh.edge_curl``."""

    cx: sp.csr_matrix
    cy: sp.csr_matrix
    cz: sp.csr_matrix


def build_cpml_profiles(mesh, config: CPMLConfig, dt: float) -> CPMLProfiles:
    """Build recursive-convolution CPML coefficients for faces and edges."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    face_locations = np.vstack((mesh.faces_x, mesh.faces_y, mesh.faces_z))
    edge_locations = np.vstack((mesh.edges_x, mesh.edges_y, mesh.edges_z))

    face_b, face_c, face_kappa = _coefficient_arrays(mesh, face_locations, config, dt)
    edge_b, edge_c, edge_kappa = _coefficient_arrays(mesh, edge_locations, config, dt)
    return CPMLProfiles(
        face_b=face_b,
        face_c=face_c,
        face_kappa=face_kappa,
        edge_b=edge_b,
        edge_c=edge_c,
        edge_kappa=edge_kappa,
    )


def split_edge_curl(mesh) -> CurlSplit:
    """Split ``edge_curl`` into x-, y-, and z-derivative sparse matrices."""

    curl = mesh.edge_curl.tocsr()
    parts = [sp.lil_matrix(curl.shape, dtype=curl.dtype) for _ in range(3)]
    face_offsets = np.r_[0, np.cumsum(mesh.n_faces_per_direction)]
    edge_offsets = np.r_[0, np.cumsum(mesh.n_edges_per_direction)]

    for face_axis in range(3):
        row_slice = slice(face_offsets[face_axis], face_offsets[face_axis + 1])
        for edge_axis in range(3):
            if face_axis == edge_axis:
                continue
            col_slice = slice(edge_offsets[edge_axis], edge_offsets[edge_axis + 1])
            derivative_axis = _remaining_axis(face_axis, edge_axis)
            block = curl[row_slice, col_slice]
            if block.nnz:
                parts[derivative_axis][row_slice, col_slice] = block

    return CurlSplit(*(part.tocsr() for part in parts))


def stretched_edge_curl(
    mesh,
    profiles: CPMLProfiles,
    state: CPMLState,
    edge_field: np.ndarray,
    split: CurlSplit | None = None,
) -> tuple[np.ndarray, CPMLState]:
    """Apply the CPML-stretched edge curl to an edge field."""

    edge_field = np.asarray(edge_field, dtype=float)
    if edge_field.shape != (mesh.n_edges,):
        raise ValueError("edge_field must have shape (mesh.n_edges,)")
    if state.face_curl_memory.shape != (3, mesh.n_faces):
        raise ValueError("face_curl_memory has incompatible shape")

    if split is None:
        split = split_edge_curl(mesh)
    pieces = np.vstack(
        [
            split.cx @ edge_field,
            split.cy @ edge_field,
            split.cz @ edge_field,
        ]
    )
    new_face_memory = profiles.face_b * state.face_curl_memory + profiles.face_c * pieces
    stretched = pieces / profiles.face_kappa + new_face_memory
    return np.sum(stretched, axis=0), CPMLState(
        face_curl_memory=new_face_memory,
        edge_curl_memory=state.edge_curl_memory.copy(),
    )


def stretched_face_curl_transpose(
    mesh,
    profiles: CPMLProfiles,
    state: CPMLState,
    face_field: np.ndarray,
    split: CurlSplit | None = None,
) -> tuple[np.ndarray, CPMLState]:
    """Apply the CPML-stretched adjoint curl to a face field."""

    face_field = np.asarray(face_field, dtype=float)
    if face_field.shape != (mesh.n_faces,):
        raise ValueError("face_field must have shape (mesh.n_faces,)")
    if state.edge_curl_memory.shape != (3, mesh.n_edges):
        raise ValueError("edge_curl_memory has incompatible shape")

    if split is None:
        split = split_edge_curl(mesh)
    pieces = np.vstack(
        [
            split.cx.T @ face_field,
            split.cy.T @ face_field,
            split.cz.T @ face_field,
        ]
    )
    new_edge_memory = profiles.edge_b * state.edge_curl_memory + profiles.edge_c * pieces
    stretched = pieces / profiles.edge_kappa + new_edge_memory
    return np.sum(stretched, axis=0), CPMLState(
        face_curl_memory=state.face_curl_memory.copy(),
        edge_curl_memory=new_edge_memory,
    )


def effective_stretched_edge_curl_matrix(
    mesh,
    profiles: CPMLProfiles,
    split: CurlSplit | None = None,
) -> sp.csr_matrix:
    """Return the implicit linear part of the stretched edge curl."""

    if split is None:
        split = split_edge_curl(mesh)
    parts = (split.cx, split.cy, split.cz)
    matrix = sp.csr_matrix(mesh.edge_curl.shape, dtype=float)
    for axis, part in enumerate(parts):
        weights = profiles.face_kappa[axis] ** -1 + profiles.face_c[axis]
        matrix = matrix + sp.diags(weights, format="csr") @ part
    return matrix.tocsr()


def effective_stretched_face_curl_transpose_matrix(
    mesh,
    profiles: CPMLProfiles,
    face_matrix: sp.spmatrix | None = None,
    split: CurlSplit | None = None,
) -> sp.csr_matrix:
    """Return the implicit linear part of the stretched adjoint curl.

    If ``face_matrix`` is provided, it is applied to the face field before the
    split adjoint curl.  The EB solver uses this for ``M_mu^-1 b``.
    """

    if face_matrix is None:
        face_matrix = sp.eye(mesh.n_faces, format="csr")
    if split is None:
        split = split_edge_curl(mesh)
    parts = (split.cx, split.cy, split.cz)
    matrix = sp.csr_matrix((mesh.n_edges, mesh.n_faces), dtype=float)
    for axis, part in enumerate(parts):
        weights = profiles.edge_kappa[axis] ** -1 + profiles.edge_c[axis]
        matrix = matrix + sp.diags(weights, format="csr") @ part.T @ face_matrix
    return matrix.tocsr()


def _coefficient_arrays(mesh, locations: np.ndarray, config: CPMLConfig, dt: float):
    b = np.ones((3, locations.shape[0]), dtype=float)
    c = np.zeros_like(b)
    kappa = np.ones_like(b)
    for axis in range(3):
        sigma, alpha, kappa_axis = _axis_stretch_profile(mesh, locations[:, axis], axis, config)
        decay = sigma / kappa_axis + alpha
        b_axis = np.exp(-decay * dt)
        c_axis = np.zeros_like(b_axis)
        mask = sigma > 0.0
        denominator = sigma[mask] * kappa_axis[mask] + kappa_axis[mask] ** 2 * alpha[mask]
        c_axis[mask] = sigma[mask] * (b_axis[mask] - 1.0) / denominator
        b[axis] = b_axis
        c[axis] = c_axis
        kappa[axis] = kappa_axis
    return b, c, kappa


def _axis_stretch_profile(mesh, coordinates: np.ndarray, axis: int, config: CPMLConfig):
    sigma = np.zeros_like(coordinates, dtype=float)
    alpha = np.zeros_like(coordinates, dtype=float)
    kappa = np.ones_like(coordinates, dtype=float)
    if config.thickness_cells == 0 or config.sigma_max == 0.0:
        return sigma, alpha, kappa

    nodes = np.asarray((mesh.nodes_x, mesh.nodes_y, mesh.nodes_z)[axis], dtype=float)
    n_cells = nodes.size - 1
    thickness = min(config.thickness_cells, n_cells)
    if thickness == 0:
        return sigma, alpha, kappa

    left_width = float(nodes[thickness] - nodes[0])
    right_width = float(nodes[-1] - nodes[-thickness - 1])
    left_depth = np.zeros_like(coordinates, dtype=float)
    right_depth = np.zeros_like(coordinates, dtype=float)
    if left_width > 0.0:
        left_depth = np.clip((nodes[0] + left_width - coordinates) / left_width, 0.0, 1.0)
    if right_width > 0.0:
        right_depth = np.clip((coordinates - (nodes[-1] - right_width)) / right_width, 0.0, 1.0)
    depth = np.maximum(left_depth, right_depth)
    active = depth > 0.0
    taper = depth[active] ** config.power
    sigma[active] = config.sigma_max * taper
    alpha[active] = config.alpha_max * (1.0 - depth[active])
    kappa[active] = 1.0 + (config.kappa_max - 1.0) * taper
    return sigma, alpha, kappa


def _remaining_axis(first: int, second: int) -> int:
    return int(({0, 1, 2} - {first, second}).pop())
