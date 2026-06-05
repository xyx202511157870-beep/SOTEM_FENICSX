import numpy as np
from discretize import TensorMesh

from atem3d.magnetic_recovery import (
    biot_savart_h_from_cell_currents,
    biot_savart_h_from_edge_basis_cell_ip_currents,
    biot_savart_h_from_edge_basis_currents,
    biot_savart_h_from_edge_current_moments,
    biot_savart_h_from_face_basis_currents,
    biot_savart_h_from_face_basis_cell_ip_currents,
    cell_current_biot_matrix,
    cell_current_density_biot_matrix,
    edge_basis_biot_matrix,
    edge_basis_cell_ip_biot_matrices,
    edge_current_biot_matrix,
    face_basis_biot_matrix,
    face_current_biot_matrix,
)
from atem3d.ip import DebyeTerm


def test_biot_savart_cell_current_matches_point_volume_limit():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    current_density = np.array([[2.0, 0.0, 0.0]])
    location = np.array([[0.0, 2.0, 0.0]])

    recovered = biot_savart_h_from_cell_currents(mesh, current_density, location)

    radius = 2.0
    expected_hz = 2.0 * mesh.cell_volumes[0] * radius / (4.0 * np.pi * radius**3)
    np.testing.assert_allclose(recovered[0], [0.0, 0.0, expected_hz])


def test_biot_savart_cell_current_supports_subcell_midpoint_quadrature():
    mesh = TensorMesh([[2.0], [2.0], [2.0]], origin=(-1.0, -1.0, -1.0))
    current_density = np.array([[1.0, 0.0, 0.0]])
    location = np.array([[0.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        location,
        subdivisions=2,
    )

    offsets = np.array([-0.5, 0.5])
    expected_hz = 0.0
    for x in offsets:
        for y in offsets:
            for z in offsets:
                displacement = location[0] - np.array([x, y, z])
                radius = np.linalg.norm(displacement)
                expected_hz += displacement[1] / (4.0 * np.pi * radius**3)

    np.testing.assert_allclose(recovered[0], [0.0, 0.0, expected_hz])


def test_biot_savart_edge_current_moments_match_single_x_moment():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    moments = np.zeros(mesh.n_edges)
    moments[0] = 2.0
    location = np.array([[0.0, 2.0, 0.0]])

    recovered = biot_savart_h_from_edge_current_moments(mesh, moments, location)

    displacement = location[0] - mesh.edges_x[0]
    radius = np.linalg.norm(displacement)
    expected = np.cross([2.0, 0.0, 0.0], displacement) / (4.0 * np.pi * radius**3)
    np.testing.assert_allclose(recovered[0], expected, atol=1.0e-15)


def test_biot_savart_edge_current_moments_preserve_uniform_cell_current_moment():
    mesh = TensorMesh([[2.0], [2.0], [2.0]], origin=(-1.0, -1.0, -1.0))
    unit_mass = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).diagonal()
    moments = np.zeros(mesh.n_edges)
    moments[: mesh.n_edges_x] = unit_mass[: mesh.n_edges_x]
    location = np.array([[0.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_edge_current_moments(mesh, moments, location)

    expected = np.zeros(3)
    for edge_location, moment in zip(mesh.edges_x, unit_mass[: mesh.n_edges_x]):
        displacement = location[0] - edge_location
        radius = np.linalg.norm(displacement)
        expected += np.cross([moment, 0.0, 0.0], displacement) / (
            4.0 * np.pi * radius**3
        )
    np.testing.assert_allclose(recovered[0], expected, atol=1.0e-15)


def test_edge_current_biot_matrix_matches_direct_edge_current_recovery():
    mesh = TensorMesh([[1.0, 2.0], [1.0], [1.0]], origin=(-1.0, -0.5, -0.5))
    moments = np.linspace(-0.5, 0.75, mesh.n_edges)
    locations = np.array([[0.0, 2.0, 0.0], [1.5, 2.5, 0.5]])

    matrix = edge_current_biot_matrix(mesh, locations)
    recovered = np.einsum("lce,e->lc", matrix, moments)
    direct = biot_savart_h_from_edge_current_moments(mesh, moments, locations)

    assert matrix.shape == (locations.shape[0], 3, mesh.n_edges)
    np.testing.assert_allclose(recovered, direct)


def test_cell_current_density_biot_matrix_matches_cell_current_recovery():
    mesh = TensorMesh([[1.0, 2.0], [1.0], [1.5]], origin=(-1.0, -0.5, -0.75))
    current_density = np.linspace(-0.75, 0.5, 3 * mesh.n_cells).reshape(
        (mesh.n_cells, 3),
        order="F",
    )
    locations = np.array([[0.0, 2.0, 0.0], [1.25, 2.5, 0.25]])

    matrix = cell_current_density_biot_matrix(mesh, locations, subdivisions=2)
    recovered = np.einsum(
        "lcp,p->lc",
        matrix,
        current_density.reshape(-1, order="F"),
    )
    direct = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        locations,
        subdivisions=2,
    )

    assert matrix.shape == (locations.shape[0], 3, 3 * mesh.n_cells)
    np.testing.assert_allclose(recovered, direct)


def test_cell_current_biot_matrix_matches_runtime_current_biot_projection():
    mesh = TensorMesh([[1.0, 2.0], [1.0], [1.5]], origin=(-1.0, -0.5, -0.75))
    edge_current_moments = np.linspace(-0.5, 0.75, mesh.n_edges)
    locations = np.array([[0.0, 2.0, 0.0], [1.25, 2.5, 0.25]])

    matrix = cell_current_biot_matrix(mesh, locations, subdivisions=2)
    recovered = np.einsum("lce,e->lc", matrix, edge_current_moments)

    unit_mass = mesh.get_edge_inner_product(np.ones(mesh.n_cells)).diagonal()
    edge_current_field = edge_current_moments / unit_mass
    current_density = mesh.average_edge_to_cell_vector @ edge_current_field
    current_density = np.asarray(current_density).reshape((mesh.n_cells, 3), order="F")
    direct = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        locations,
        subdivisions=2,
    )

    assert matrix.shape == (locations.shape[0], 3, mesh.n_edges)
    np.testing.assert_allclose(recovered, direct)


def test_face_current_biot_matrix_matches_hj_runtime_current_biot_projection():
    mesh = TensorMesh([[1.0, 2.0], [1.0], [1.5]], origin=(-1.0, -0.5, -0.75))
    face_current = np.linspace(-0.5, 0.75, mesh.n_faces)
    locations = np.array([[0.0, 2.0, 0.0], [1.25, 2.5, 0.25]])

    matrix = face_current_biot_matrix(mesh, locations, subdivisions=2)
    recovered = np.einsum("lcf,f->lc", matrix, face_current)

    current_density = mesh.average_face_to_cell_vector @ face_current
    current_density = np.asarray(current_density).reshape((mesh.n_cells, 3), order="F")
    direct = biot_savart_h_from_cell_currents(
        mesh,
        current_density,
        locations,
        subdivisions=2,
    )

    assert matrix.shape == (locations.shape[0], 3, mesh.n_faces)
    np.testing.assert_allclose(recovered, direct, atol=1.0e-15)


def test_biot_savart_edge_basis_currents_match_constant_cell_current_quadrature():
    mesh = TensorMesh([[2.0], [2.0], [2.0]], origin=(-1.0, -1.0, -1.0))
    edge_current = np.zeros(mesh.n_edges)
    edge_current[: mesh.n_edges_x] = 1.0
    location = np.array([[0.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_edge_basis_currents(
        mesh,
        edge_current,
        location,
        subdivisions=2,
    )
    expected = biot_savart_h_from_cell_currents(
        mesh,
        np.array([[1.0, 0.0, 0.0]]),
        location,
        subdivisions=2,
    )

    np.testing.assert_allclose(recovered, expected)


def test_biot_savart_edge_basis_currents_use_bilinear_component_shape_functions():
    mesh = TensorMesh([[2.0], [2.0], [2.0]], origin=(0.0, 0.0, 0.0))
    edge_current = np.zeros(mesh.n_edges)
    edge_current[: mesh.n_edges_x] = [1.0, 3.0, 5.0, 7.0]
    location = np.array([[1.0, 4.0, 1.0]])

    recovered = biot_savart_h_from_edge_basis_currents(
        mesh,
        edge_current,
        location,
        subdivisions=2,
    )

    expected = np.zeros(3)
    for xi in (0.25, 0.75):
        for eta in (0.25, 0.75):
            for zeta in (0.25, 0.75):
                point = np.array([2.0 * xi, 2.0 * eta, 2.0 * zeta])
                jx = (
                    (1.0 - eta) * (1.0 - zeta) * 1.0
                    + eta * (1.0 - zeta) * 3.0
                    + (1.0 - eta) * zeta * 5.0
                    + eta * zeta * 7.0
                )
                displacement = location[0] - point
                radius = np.linalg.norm(displacement)
                expected += (mesh.cell_volumes[0] / 8.0) * np.cross(
                    [jx, 0.0, 0.0],
                    displacement,
                ) / (4.0 * np.pi * radius**3)

    np.testing.assert_allclose(recovered[0], expected)


def test_edge_basis_biot_matrix_matches_direct_edge_basis_recovery():
    mesh = TensorMesh([[2.0], [1.0], [1.0]], origin=(0.0, -0.5, -0.5))
    edge_current = np.linspace(-1.0, 1.0, mesh.n_edges)
    locations = np.array([[1.0, 3.0, 0.0], [0.5, 2.0, 0.5]])

    matrix = edge_basis_biot_matrix(mesh, locations, subdivisions=2)
    recovered = np.einsum("lce,e->lc", matrix, edge_current)
    direct = biot_savart_h_from_edge_basis_currents(
        mesh,
        edge_current,
        locations,
        subdivisions=2,
    )

    assert matrix.shape == (locations.shape[0], 3, mesh.n_edges)
    np.testing.assert_allclose(recovered, direct)


def test_face_basis_biot_matrix_matches_direct_face_basis_recovery():
    mesh = TensorMesh([[2.0], [1.0], [1.0]], origin=(0.0, -0.5, -0.5))
    face_current = np.linspace(-1.0, 1.0, mesh.n_faces)
    locations = np.array([[1.0, 3.0, 0.0], [0.5, 2.0, 0.5]])

    matrix = face_basis_biot_matrix(mesh, locations, subdivisions=2)
    recovered = np.einsum("lcf,f->lc", matrix, face_current)
    direct = biot_savart_h_from_face_basis_currents(
        mesh,
        face_current,
        locations,
        subdivisions=2,
    )

    assert matrix.shape == (locations.shape[0], 3, mesh.n_faces)
    np.testing.assert_allclose(recovered, direct, atol=1.0e-15)


def test_biot_savart_edge_basis_cell_ip_currents_use_cell_local_conductivity():
    mesh = TensorMesh([[1.0, 1.0], [1.0], [1.0]], origin=(0.0, -0.5, -0.5))
    electric = np.zeros(mesh.n_edges)
    electric[: mesh.n_edges_x] = 1.0
    sigma = np.array([1.0, 3.0])
    location = np.array([[1.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_edge_basis_cell_ip_currents(
        mesh,
        electric,
        sigma,
        [],
        [],
        location,
        subdivisions=1,
    )
    expected = biot_savart_h_from_cell_currents(
        mesh,
        np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        location,
    )

    np.testing.assert_allclose(recovered, expected)


def test_biot_savart_face_basis_cell_ip_currents_match_constant_cell_current():
    mesh = TensorMesh([[2.0], [2.0], [2.0]], origin=(-1.0, -1.0, -1.0))
    electric = np.zeros(mesh.n_faces)
    electric[: mesh.n_faces_x] = 1.0
    location = np.array([[0.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        electric,
        np.array([2.0]),
        [],
        [],
        location,
        subdivisions=2,
    )
    expected = biot_savart_h_from_cell_currents(
        mesh,
        np.array([[2.0, 0.0, 0.0]]),
        location,
        subdivisions=2,
    )

    np.testing.assert_allclose(recovered, expected)


def test_biot_savart_face_basis_cell_ip_currents_use_cell_local_ip_parameters():
    mesh = TensorMesh([[1.0, 1.0], [1.0], [1.0]], origin=(0.0, -0.5, -0.5))
    electric = np.zeros(mesh.n_faces)
    memory = np.zeros(mesh.n_faces)
    electric[: mesh.n_faces_x] = 1.0
    memory[: mesh.n_faces_x] = 0.5
    sigma = np.array([1.0, 3.0])
    delta = np.array([0.2, 0.4])
    location = np.array([[1.0, 3.0, 0.0]])

    recovered = biot_savart_h_from_face_basis_cell_ip_currents(
        mesh,
        electric,
        sigma,
        [DebyeTerm(delta, tau=0.1)],
        [memory],
        location,
        subdivisions=1,
    )
    expected = biot_savart_h_from_cell_currents(
        mesh,
        np.array([[0.9, 0.0, 0.0], [2.8, 0.0, 0.0]]),
        location,
    )

    np.testing.assert_allclose(recovered, expected)


def test_biot_savart_edge_basis_cell_ip_currents_apply_component_memory_scale():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    electric = np.zeros(mesh.n_edges)
    memory = np.zeros(mesh.n_edges)
    memory[: mesh.n_edges_x] = 2.0
    y_start = mesh.n_edges_x
    y_end = y_start + mesh.n_edges_y
    memory[y_start:y_end] = 3.0
    location = np.array([[0.0, 2.0, 0.0]])

    recovered = biot_savart_h_from_edge_basis_cell_ip_currents(
        mesh,
        electric,
        np.array([4.0]),
        [DebyeTerm(np.array([1.0]), tau=0.1)],
        [memory],
        location,
        polarization_scale=np.array([0.5, 2.0, 1.0]),
    )
    expected = biot_savart_h_from_cell_currents(
        mesh,
        np.array([[-1.0, -6.0, 0.0]]),
        location,
    )

    np.testing.assert_allclose(recovered, expected)


def test_edge_basis_cell_ip_biot_matrices_match_direct_recovery():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    electric = np.linspace(-0.5, 0.5, mesh.n_edges)
    memory = np.linspace(0.25, -0.25, mesh.n_edges)
    sigma = np.array([2.0])
    terms = [DebyeTerm(np.array([0.75]), tau=0.1)]
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])

    ohmic_matrix, memory_matrices = edge_basis_cell_ip_biot_matrices(
        mesh,
        sigma,
        terms,
        locations,
        subdivisions=2,
        polarization_scale=1.5,
    )
    recovered = np.einsum("lce,e->lc", ohmic_matrix, electric)
    recovered += np.einsum("lce,e->lc", memory_matrices[0], memory)
    direct = biot_savart_h_from_edge_basis_cell_ip_currents(
        mesh,
        electric,
        sigma,
        terms,
        [memory],
        locations,
        subdivisions=2,
        polarization_scale=1.5,
    )

    assert ohmic_matrix.shape == (locations.shape[0], 3, mesh.n_edges)
    assert len(memory_matrices) == 1
    assert memory_matrices[0].shape == ohmic_matrix.shape
    np.testing.assert_allclose(recovered, direct)
