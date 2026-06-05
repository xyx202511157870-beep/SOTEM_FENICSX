import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh

from atem3d.cpml import (
    CPMLConfig,
    CPMLState,
    build_cpml_profiles,
    effective_stretched_edge_curl_matrix,
    effective_stretched_face_curl_transpose_matrix,
    split_edge_curl,
    stretched_edge_curl,
    stretched_face_curl_transpose,
)


def test_cpml_profiles_are_face_and_edge_sized_and_leave_core_unstretched():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=1, sigma_max=2.0, alpha_max=0.1, kappa_max=1.0),
        dt=0.01,
    )

    assert profiles.face_b.shape == (3, mesh.n_faces)
    assert profiles.face_c.shape == (3, mesh.n_faces)
    assert profiles.edge_b.shape == (3, mesh.n_edges)
    assert profiles.edge_c.shape == (3, mesh.n_edges)

    center_face = np.argmin(np.linalg.norm(mesh.faces, axis=1))
    np.testing.assert_allclose(profiles.face_b[:, center_face], 1.0)
    np.testing.assert_allclose(profiles.face_c[:, center_face], 0.0)
    assert np.any(profiles.face_b < 1.0)
    assert np.any(profiles.face_c != 0.0)


def test_split_edge_curl_sums_to_mesh_edge_curl():
    mesh = TensorMesh([np.ones(3), np.ones(4), np.ones(2)], origin="CCC")

    split = split_edge_curl(mesh)

    assert split.cx.shape == mesh.edge_curl.shape
    assert split.cy.shape == mesh.edge_curl.shape
    assert split.cz.shape == mesh.edge_curl.shape
    np.testing.assert_allclose(
        (split.cx + split.cy + split.cz).toarray(),
        mesh.edge_curl.toarray(),
    )


def test_cpml_state_allocates_directional_curl_memories():
    mesh = TensorMesh([np.ones(2), np.ones(3), np.ones(4)], origin="CCC")

    state = CPMLState.zeros(mesh)

    assert state.face_curl_memory.shape == (3, mesh.n_faces)
    assert state.edge_curl_memory.shape == (3, mesh.n_edges)
    np.testing.assert_allclose(state.face_curl_memory, 0.0)
    np.testing.assert_allclose(state.edge_curl_memory, 0.0)


def test_stretched_edge_curl_reduces_to_plain_curl_without_pml():
    mesh = TensorMesh([np.ones(3), np.ones(4), np.ones(2)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=0, sigma_max=0.0),
        dt=0.01,
    )
    state = CPMLState.zeros(mesh)
    e = np.arange(mesh.n_edges, dtype=float)

    curl_e, updated = stretched_edge_curl(mesh, profiles, state, e)

    np.testing.assert_allclose(curl_e, mesh.edge_curl @ e)
    np.testing.assert_allclose(updated.face_curl_memory, 0.0)


def test_stretched_face_curl_transpose_reduces_to_plain_adjoint_without_pml():
    mesh = TensorMesh([np.ones(3), np.ones(4), np.ones(2)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=0, sigma_max=0.0),
        dt=0.01,
    )
    state = CPMLState.zeros(mesh)
    face_field = np.arange(mesh.n_faces, dtype=float)

    curl_t_h, updated = stretched_face_curl_transpose(mesh, profiles, state, face_field)

    np.testing.assert_allclose(curl_t_h, mesh.edge_curl.T @ face_field)
    np.testing.assert_allclose(updated.edge_curl_memory, 0.0)


def test_effective_stretched_curl_matrices_reduce_to_plain_operators_without_pml():
    mesh = TensorMesh([np.ones(3), np.ones(4), np.ones(2)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=0, sigma_max=0.0),
        dt=0.01,
    )

    faraday = effective_stretched_edge_curl_matrix(mesh, profiles)
    ampere = effective_stretched_face_curl_transpose_matrix(
        mesh,
        profiles,
        sp.eye(mesh.n_faces, format="csr"),
    )

    np.testing.assert_allclose(faraday.toarray(), mesh.edge_curl.toarray())
    np.testing.assert_allclose(ampere.toarray(), mesh.edge_curl.T.toarray())


def test_stretched_edge_curl_updates_face_memories_in_pml():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=1, sigma_max=2.0),
        dt=0.01,
    )
    state = CPMLState.zeros(mesh)
    e = np.linspace(0.0, 1.0, mesh.n_edges)

    curl_e, updated = stretched_edge_curl(mesh, profiles, state, e)

    assert curl_e.shape == (mesh.n_faces,)
    assert np.linalg.norm(updated.face_curl_memory) > 0.0


def test_stretched_face_curl_transpose_updates_edge_memories_in_pml():
    mesh = TensorMesh([np.ones(5), np.ones(5), np.ones(5)], origin="CCC")
    profiles = build_cpml_profiles(
        mesh,
        CPMLConfig(thickness_cells=1, sigma_max=2.0),
        dt=0.01,
    )
    state = CPMLState.zeros(mesh)
    face_field = np.linspace(0.0, 1.0, mesh.n_faces)

    curl_t_h, updated = stretched_face_curl_transpose(mesh, profiles, state, face_field)

    assert curl_t_h.shape == (mesh.n_edges,)
    assert np.linalg.norm(updated.edge_curl_memory) > 0.0
