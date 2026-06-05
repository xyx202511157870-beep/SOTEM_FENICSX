import numpy as np
from discretize import TensorMesh

from atem3d.recovery_spectrum import (
    MagneticDiffusionSpectrum,
    magnetic_diffusion_driven_response,
    magnetic_diffusion_mmr_initial_state,
    local_magnetic_diffusion_positive_spectrum,
    magnetic_diffusion_modal_coupling,
    magnetic_diffusion_matrices,
    magnetic_diffusion_positive_spectrum,
    magnetic_diffusion_time_constants,
    project_modal_response_to_source_moments,
    tensor_mesh_cell_submesh,
)
from atem3d.magnetic_recovery import face_current_biot_matrix


def test_magnetic_diffusion_spectrum_returns_positive_sorted_modes():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")

    spectrum = magnetic_diffusion_positive_spectrum(
        mesh,
        conductivity=np.full(mesh.n_cells, 0.2),
        max_modes=6,
    )

    assert spectrum.eigenvalues.shape == (6,)
    assert np.all(spectrum.eigenvalues > 0.0)
    assert spectrum.discarded_count > 0
    assert spectrum.eigenvalue_floor > 0.0
    assert spectrum.raw_eigenvalue_min <= spectrum.raw_eigenvalue_max
    assert spectrum.last_discarded_eigenvalue <= spectrum.eigenvalue_floor
    assert spectrum.first_kept_eigenvalue == spectrum.eigenvalues[0]
    np.testing.assert_allclose(np.diff(spectrum.eigenvalues) >= 0.0, True)
    np.testing.assert_allclose(spectrum.time_constants, 1.0 / spectrum.eigenvalues)


def test_magnetic_diffusion_time_constants_shorten_when_resistivity_increases():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")

    conductive = magnetic_diffusion_time_constants(
        mesh,
        conductivity=np.full(mesh.n_cells, 1.0),
        max_modes=4,
    )
    resistive = magnetic_diffusion_time_constants(
        mesh,
        conductivity=np.full(mesh.n_cells, 0.1),
        max_modes=4,
    )

    assert np.all(resistive < conductive)


def test_magnetic_diffusion_spectrum_helpers_are_public_api():
    from atem3d import (
        LocalMagneticDiffusionSpectrum as ExportedLocalSpectrum,
        LocalTensorMeshSupport as ExportedSupport,
        MagneticDiffusionModalCoupling as ExportedModalCoupling,
        MagneticDiffusionSpectrum as ExportedSpectrum,
        ModalSourceMomentProjection as ExportedModalProjectionResult,
        magnetic_diffusion_driven_response as exported_driven_response,
        magnetic_diffusion_mmr_initial_state as exported_mmr_initial_state,
        local_magnetic_diffusion_positive_spectrum as exported_local_spectrum,
        magnetic_diffusion_modal_coupling as exported_modal_coupling,
        magnetic_diffusion_matrices as exported_matrices,
        magnetic_diffusion_positive_spectrum as exported_spectrum,
        magnetic_diffusion_time_constants as exported_time_constants,
        project_modal_response_to_source_moments as exported_modal_projection,
        tensor_mesh_cell_submesh as exported_submesh,
    )

    assert ExportedLocalSpectrum is not None
    assert ExportedSupport is not None
    assert ExportedModalCoupling is not None
    assert ExportedModalProjectionResult is not None
    assert ExportedSpectrum is MagneticDiffusionSpectrum
    assert exported_driven_response is magnetic_diffusion_driven_response
    assert exported_mmr_initial_state is magnetic_diffusion_mmr_initial_state
    assert exported_local_spectrum is local_magnetic_diffusion_positive_spectrum
    assert exported_modal_coupling is magnetic_diffusion_modal_coupling
    assert exported_matrices is magnetic_diffusion_matrices
    assert exported_modal_projection is project_modal_response_to_source_moments
    assert exported_spectrum is magnetic_diffusion_positive_spectrum
    assert exported_time_constants is magnetic_diffusion_time_constants
    assert exported_submesh is tensor_mesh_cell_submesh


def test_project_modal_response_to_source_moments_recovers_known_coefficients():
    static_response = np.array(
        [
            [[1.0, 0.5, 0.0]],
            [[0.0, 2.0, -1.0]],
        ]
    )
    coefficients = np.array(
        [
            [[2.0, -1.0], [0.5, 3.0]],
            [[-4.0, 1.5], [1.0, 0.0]],
        ]
    )
    modal_response = np.einsum("kjs,slc->kjlc", coefficients, static_response)

    projection = project_modal_response_to_source_moments(
        static_response,
        modal_response,
    )

    np.testing.assert_allclose(projection.coefficients, coefficients, atol=1.0e-14)
    np.testing.assert_allclose(
        projection.fitted_response,
        modal_response,
        atol=1.0e-14,
    )
    assert projection.aggregate_relative_l2 < 1.0e-14
    np.testing.assert_allclose(projection.relative_l2, 0.0, atol=1.0e-14)
    assert projection.rank == 2
    assert projection.design_shape == (3, 2)


def test_magnetic_diffusion_driven_response_projects_time_traces():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    conductivity = np.full(mesh.n_cells, 0.2)
    source_vectors = np.zeros((2, mesh.n_faces))
    source_vectors[0, 0] = 1.0
    source_vectors[1, -1] = -0.5
    locations = np.array([[0.0, 0.0, 0.0]])
    time_steps = np.array([0.1, 0.2])
    driver_values = np.array([0.0, 1.0, 0.5])

    response = magnetic_diffusion_driven_response(
        mesh,
        conductivity,
        source_vectors,
        locations,
        time_steps=time_steps,
        driver_values=driver_values,
        receiver_mode="face_current_biot",
    )

    assert response.times.shape == (3,)
    assert response.driver_values.shape == (3,)
    assert response.receiver_response.shape == (3, 2, 1, 3)
    assert response.source_moment_projection.coefficients.shape == (3, 2, 2)
    np.testing.assert_allclose(response.receiver_response[0], 0.0)
    assert response.source_moment_projection.aggregate_relative_l2 >= 0.0

    projected = magnetic_diffusion_driven_response(
        mesh,
        conductivity,
        source_vectors,
        locations,
        time_steps=time_steps,
        driver_values=driver_values,
        receiver_mode="face_current_biot",
        source_projection="charge_conserving",
    )

    assert projected.source_projection == "charge_conserving"
    np.testing.assert_allclose(
        projected.receiver_response,
        response.receiver_response,
        atol=1.0e-14,
    )


def test_magnetic_diffusion_driven_response_accepts_initial_state():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    conductivity = np.full(mesh.n_cells, 0.2)
    source_vectors = np.zeros((2, mesh.n_faces))
    source_vectors[0, 0] = 1.0
    source_vectors[1, -1] = -0.5
    locations = np.array([[0.0, 0.0, 0.0]])
    time_steps = np.array([0.1])
    driver_values = np.array([0.0, 0.0])
    initial_state = np.column_stack(
        [
            np.linspace(-0.2, 0.3, mesh.n_edges),
            np.linspace(0.1, -0.4, mesh.n_edges),
        ]
    )

    response = magnetic_diffusion_driven_response(
        mesh,
        conductivity,
        source_vectors,
        locations,
        time_steps=time_steps,
        driver_values=driver_values,
        receiver_mode="face_current_biot",
        initial_state=initial_state,
    )

    receiver_matrix = face_current_biot_matrix(mesh, locations)
    expected_initial = np.einsum(
        "lcf,fs->slc",
        receiver_matrix,
        mesh.edge_curl @ initial_state,
    )
    np.testing.assert_allclose(response.receiver_response[0], expected_initial)
    assert response.initial_state_kind == "provided"
    assert response.source_moment_projection.coefficients.shape == (2, 2, 2)


def test_magnetic_diffusion_driven_response_accepts_edge_forcing_override():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    conductivity = np.full(mesh.n_cells, 0.2)
    source_vectors = np.zeros((2, mesh.n_faces))
    source_vectors[0, 0] = 1.0
    source_vectors[1, -1] = -0.5
    locations = np.array([[0.0, 0.0, 0.0]])
    dt = 0.1
    forcing_vectors = np.column_stack(
        [
            np.linspace(-0.3, 0.2, mesh.n_edges),
            np.linspace(0.4, -0.1, mesh.n_edges),
        ]
    )

    response = magnetic_diffusion_driven_response(
        mesh,
        conductivity,
        source_vectors,
        locations,
        time_steps=[dt],
        driver_values=[0.0, 1.0],
        receiver_mode="face_current_biot",
        forcing_vectors=forcing_vectors,
        forcing_kind="custom_edge_rhs",
    )

    stiffness, mass = magnetic_diffusion_matrices(mesh, conductivity)
    expected_state = np.linalg.solve(
        (mass + dt * stiffness).toarray(),
        dt * forcing_vectors,
    )
    receiver_matrix = face_current_biot_matrix(mesh, locations)
    expected_response = np.einsum(
        "lcf,fs->slc",
        receiver_matrix,
        mesh.edge_curl @ expected_state,
    )

    assert response.forcing_kind == "custom_edge_rhs"
    np.testing.assert_allclose(response.receiver_response[1], expected_response)


def test_magnetic_diffusion_mmr_initial_state_recovers_divergence_free_current():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    h_state = np.linspace(-0.2, 0.3, mesh.n_edges)
    face_current = mesh.edge_curl @ h_state

    recovered = magnetic_diffusion_mmr_initial_state(
        mesh,
        face_current,
        mu=1.0,
    )

    assert recovered.shape == (mesh.n_edges, 1)
    np.testing.assert_allclose(
        mesh.edge_curl @ recovered[:, 0],
        face_current,
        atol=1.0e-10,
    )


def test_tensor_mesh_cell_submesh_extracts_rectangular_padded_support():
    mesh = TensorMesh(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0]],
        origin=(-10.0, 2.0, -3.0),
    )
    selected = np.ravel_multi_index((2, 1, 1), (4, 3, 2), order="F")

    support = tensor_mesh_cell_submesh(mesh, [selected], padding=1)

    np.testing.assert_allclose(support.mesh.h[0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(support.mesh.h[1], [5.0, 6.0, 7.0])
    np.testing.assert_allclose(support.mesh.h[2], [8.0, 9.0])
    np.testing.assert_allclose(support.mesh.origin, [-9.0, 2.0, -3.0])
    assert support.ijk_min == (1, 0, 0)
    assert support.ijk_max == (3, 2, 1)
    assert support.global_cell_indices.shape == (support.mesh.n_cells,)
    assert selected in set(support.global_cell_indices.tolist())


def test_tensor_mesh_cell_submesh_maps_local_faces_and_edges_to_global_dofs():
    mesh = TensorMesh(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0]],
        origin=(-10.0, 2.0, -3.0),
    )
    selected = np.ravel_multi_index((2, 1, 1), (4, 3, 2), order="F")

    support = tensor_mesh_cell_submesh(mesh, [selected], padding=1)

    global_faces = np.vstack([mesh.faces_x, mesh.faces_y, mesh.faces_z])
    local_faces = np.vstack(
        [support.mesh.faces_x, support.mesh.faces_y, support.mesh.faces_z]
    )
    global_edges = np.vstack([mesh.edges_x, mesh.edges_y, mesh.edges_z])
    local_edges = np.vstack(
        [support.mesh.edges_x, support.mesh.edges_y, support.mesh.edges_z]
    )

    assert support.global_face_indices.shape == (support.mesh.n_faces,)
    assert support.global_edge_indices.shape == (support.mesh.n_edges,)
    np.testing.assert_allclose(
        global_faces[support.global_face_indices],
        local_faces,
    )
    np.testing.assert_allclose(
        global_edges[support.global_edge_indices],
        local_edges,
    )


def test_local_magnetic_diffusion_spectrum_uses_local_cell_conductivity():
    mesh = TensorMesh([np.ones(3), np.ones(3), np.ones(2)], origin="CCC")
    selected = [
        np.ravel_multi_index((1, 1, 0), (3, 3, 2), order="F"),
        np.ravel_multi_index((1, 1, 1), (3, 3, 2), order="F"),
    ]
    conductivity = np.full(mesh.n_cells, 0.2)
    conductivity[selected] = 0.8

    local = local_magnetic_diffusion_positive_spectrum(
        mesh,
        conductivity,
        selected,
        padding=0,
        max_modes=3,
    )
    direct = magnetic_diffusion_positive_spectrum(
        local.support.mesh,
        conductivity[local.support.global_cell_indices],
        max_modes=3,
    )

    np.testing.assert_allclose(local.spectrum.eigenvalues, direct.eigenvalues)
    np.testing.assert_array_equal(local.support.global_cell_indices, np.asarray(selected))


def test_magnetic_diffusion_modal_coupling_projects_face_source_to_modes():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    conductivity = np.full(mesh.n_cells, 0.2)
    source_vectors = np.zeros((2, mesh.n_faces))
    source_vectors[0, 0] = 1.0
    source_vectors[1, -1] = -0.5
    locations = np.array([[0.0, 0.0, 0.0]])

    coupling = magnetic_diffusion_modal_coupling(
        mesh,
        conductivity,
        source_vectors,
        locations,
        max_modes=4,
        receiver_mode="stored_h",
    )
    stiffness, mass = magnetic_diffusion_matrices(mesh, conductivity)
    expected_forcing = coupling.eigenvectors.T @ (
        mesh.edge_curl.T
        @ mesh.get_face_inner_product(1.0 / conductivity)
        @ source_vectors.T
    )

    assert coupling.modal_forcing.shape == (4, 2)
    assert coupling.modal_receiver_response.shape == (4, 1, 3)
    assert coupling.source_receiver_response.shape == (4, 2, 1, 3)
    np.testing.assert_allclose(coupling.modal_forcing, expected_forcing)
    np.testing.assert_allclose(
        coupling.source_receiver_response,
        (
            coupling.modal_forcing[:, :, None, None]
            / coupling.eigenvalues[:, None, None, None]
        )
        * coupling.modal_receiver_response[:, None, :, :],
    )
    np.testing.assert_allclose(
        coupling.eigenvectors.T @ (mass @ coupling.eigenvectors),
        np.eye(4),
        atol=1.0e-10,
    )
