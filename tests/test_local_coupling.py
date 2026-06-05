import numpy as np
from discretize import TensorMesh

from atem3d.local_coupling import (
    adjacent_cells_for_edges,
    adjacent_cells_for_faces,
    canonical_edge_basis,
    edge_indices_for_cells,
    local_cell_support,
    local_edge_basis,
    source_edge_moment_basis,
)


def test_edge_indices_for_cells_returns_all_edges_of_single_cell():
    mesh = TensorMesh([[2.0], [3.0], [4.0]], origin=(0.0, 0.0, 0.0))

    indices = edge_indices_for_cells(mesh, [0])

    np.testing.assert_array_equal(indices, np.arange(mesh.n_edges))


def test_adjacent_cells_for_edges_finds_cells_touching_shared_x_edge():
    mesh = TensorMesh([[1.0, 1.0], [1.0, 1.0], [1.0]], origin=(0.0, 0.0, 0.0))
    shared_x_edge = 2

    cells = adjacent_cells_for_edges(mesh, [shared_x_edge])

    np.testing.assert_array_equal(cells, np.array([0, 2]))


def test_adjacent_cells_for_faces_finds_cells_touching_shared_x_face():
    mesh = TensorMesh([[1.0, 1.0], [1.0], [1.0]], origin=(0.0, 0.0, 0.0))
    shared_x_face = 1

    cells = adjacent_cells_for_faces(mesh, [shared_x_face])

    np.testing.assert_array_equal(cells, np.array([0, 1]))


def test_local_cell_support_builds_face_source_and_receiver_support():
    mesh = TensorMesh([[1.0, 1.0], [1.0, 1.0], [1.0]], origin=(0.0, 0.0, 0.0))
    source_vector = np.zeros(mesh.n_faces)
    source_vector[1] = 2.0
    receiver_locations = np.array([[1.5, 1.5, 0.5]])

    support = local_cell_support(
        mesh,
        source_vector,
        receiver_locations,
        field_location="face",
        source_cell_radius=0,
        receiver_cell_radius=0,
    )

    np.testing.assert_array_equal(support.source_dof_indices, np.array([1]))
    np.testing.assert_array_equal(support.source_cell_indices, np.array([0, 1]))
    np.testing.assert_array_equal(support.receiver_cell_indices[0], np.array([3]))
    np.testing.assert_array_equal(support.support_cell_indices, np.array([0, 1, 3]))
    assert support.field_location == "face"


def test_local_edge_basis_builds_canonical_support_from_source_and_receiver_cells():
    mesh = TensorMesh([[1.0, 1.0], [1.0, 1.0], [1.0]], origin=(0.0, 0.0, 0.0))
    source_vector = np.zeros(mesh.n_edges)
    source_vector[2] = 2.0
    receiver_locations = np.array([[1.5, 1.5, 0.5]])

    basis = local_edge_basis(
        mesh,
        source_vector,
        receiver_locations,
        source_cell_radius=0,
        receiver_cell_radius=0,
    )

    expected_cells = np.array([0, 2, 3])
    expected_edges = edge_indices_for_cells(mesh, expected_cells)
    np.testing.assert_array_equal(basis.source_edge_indices, np.array([2]))
    np.testing.assert_array_equal(basis.source_cell_indices, np.array([0, 2]))
    np.testing.assert_array_equal(basis.receiver_cell_indices[0], np.array([3]))
    np.testing.assert_array_equal(basis.support_cell_indices, expected_cells)
    np.testing.assert_array_equal(basis.support_edge_indices, expected_edges)
    assert basis.basis_vectors.shape == (expected_edges.size, mesh.n_edges)
    for row, edge_index in zip(basis.basis_vectors, expected_edges):
        expected = np.zeros(mesh.n_edges)
        expected[edge_index] = 1.0
        np.testing.assert_array_equal(row, expected)
    assert basis.basis_labels == [f"edge:{edge_index}" for edge_index in expected_edges]


def test_canonical_edge_basis_builds_one_vector_per_selected_edge():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))

    basis = canonical_edge_basis(mesh, [3, 0], label_prefix="source")

    np.testing.assert_array_equal(basis.edge_indices, np.array([0, 3]))
    assert basis.basis_vectors.shape == (2, mesh.n_edges)
    np.testing.assert_array_equal(basis.basis_vectors[0], np.eye(mesh.n_edges)[0])
    np.testing.assert_array_equal(basis.basis_vectors[1], np.eye(mesh.n_edges)[3])
    assert basis.basis_labels == ["source:0", "source:3"]


def test_source_edge_moment_basis_weights_active_edges_by_wire_coordinate():
    mesh = TensorMesh([[1.0, 1.0], [1.0], [1.0]], origin=(-1.0, -0.5, -0.5))
    source_vector = np.zeros(mesh.n_edges)
    left_edge = int(np.argmin(np.sum((mesh.edges - np.array([-0.5, -0.5, -0.5])) ** 2, axis=1)))
    right_edge = int(np.argmin(np.sum((mesh.edges - np.array([0.5, -0.5, -0.5])) ** 2, axis=1)))
    source_vector[left_edge] = 2.0
    source_vector[right_edge] = 4.0

    basis = source_edge_moment_basis(
        mesh,
        source_vector,
        start=(-1.0, -0.5, -0.5),
        end=(1.0, -0.5, -0.5),
        max_degree=2,
    )

    expected0 = np.zeros(mesh.n_edges)
    expected0[left_edge] = 2.0
    expected0[right_edge] = 4.0
    expected1 = np.zeros(mesh.n_edges)
    expected1[left_edge] = -1.0
    expected1[right_edge] = 2.0
    expected2 = np.zeros(mesh.n_edges)
    expected2[left_edge] = 0.5
    expected2[right_edge] = 1.0
    np.testing.assert_array_equal(basis.edge_indices, np.array([left_edge, right_edge]))
    np.testing.assert_allclose(basis.basis_vectors, np.vstack([expected0, expected1, expected2]))
    assert basis.basis_labels == [
        "source_moment:0",
        "source_moment:1",
        "source_moment:2",
    ]


def test_source_edge_moment_basis_accepts_explicit_degrees():
    mesh = TensorMesh([[1.0, 1.0], [1.0], [1.0]], origin=(-1.0, -0.5, -0.5))
    source_vector = np.zeros(mesh.n_edges)
    active_edge = int(np.argmin(np.sum((mesh.edges - np.array([0.5, -0.5, -0.5])) ** 2, axis=1)))
    source_vector[active_edge] = 4.0

    basis = source_edge_moment_basis(
        mesh,
        source_vector,
        start=(-1.0, -0.5, -0.5),
        end=(1.0, -0.5, -0.5),
        degrees=[0, 2],
    )

    expected0 = np.zeros(mesh.n_edges)
    expected0[active_edge] = 4.0
    expected2 = np.zeros(mesh.n_edges)
    expected2[active_edge] = 1.0
    np.testing.assert_allclose(basis.basis_vectors, np.vstack([expected0, expected2]))
    assert basis.basis_labels == ["source_moment:0", "source_moment:2"]
