import numpy as np
from discretize import TensorMesh

from atem3d.magnetic_recovery import edge_current_biot_matrix
from atem3d.source_history_operator import (
    evaluate_spatial_coefficient_traces_with_history_basis,
    evaluate_source_history_coefficients_for_components,
    fit_spatial_coefficient_traces_to_history_basis,
    fit_static_spatial_coefficients_from_static_response,
    fit_static_spatial_coefficients_for_components,
    fit_source_history_coefficients,
    fit_source_history_coefficients_for_components,
    project_vector_to_spatial_basis,
    source_history_receiver_basis,
    source_history_receiver_basis_from_vectors,
    source_history_receiver_basis_from_spatial_vectors,
    source_history_receiver_basis_from_static_response,
)
from atem3d.source_primary import discrete_debye_history_basis


def test_fit_static_spatial_coefficients_for_components_recovers_time_series():
    receiver_matrix = np.zeros((2, 3, 2))
    receiver_matrix[0, 2, 0] = 1.0
    receiver_matrix[1, 0, 1] = 2.0
    spatial_vectors = np.eye(2)
    receiver_indices = np.array([0, 1])
    component_indices = np.array([2, 0])
    coefficients = np.array(
        [
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.75, 2.0],
        ]
    )
    static_design = np.array([[1.0, 0.0], [0.0, 2.0]])
    target = coefficients @ static_design.T

    fit = fit_static_spatial_coefficients_for_components(
        receiver_matrix,
        spatial_vectors,
        target,
        receiver_indices=receiver_indices,
        component_indices=component_indices,
    )

    np.testing.assert_allclose(fit.coefficients, coefficients)
    np.testing.assert_allclose(fit.fitted, target)
    assert fit.relative_l2 < 1.0e-14
    assert fit.design_shape == (2, 2)
    assert fit.rank == 2
    np.testing.assert_allclose(fit.column_norms, [1.0, 2.0])
    assert np.isfinite(fit.condition_number)


def test_fit_spatial_coefficient_traces_to_history_basis_recovers_known_traces():
    time_steps = np.array([0.1, 0.1, 0.1])
    history = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    coefficients = np.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
        ]
    )
    traces = history.values @ coefficients

    fit = fit_spatial_coefficient_traces_to_history_basis(
        time_steps,
        history.times,
        traces,
        tau=0.2,
        max_order=1,
    )

    np.testing.assert_allclose(fit.coefficients, coefficients)
    np.testing.assert_allclose(fit.fitted, traces, atol=1.0e-14)
    assert fit.relative_l2 < 1.0e-14
    np.testing.assert_allclose(fit.per_trace_relative_l2, np.zeros(2), atol=1.0e-14)
    assert fit.design_shape == (history.times.size, 2)
    assert fit.rank == 2
    assert fit.basis_labels == ["BE relaxation", "BE cascade 1"]


def test_evaluate_spatial_coefficient_traces_with_history_basis_uses_prescribed_table():
    time_steps = np.array([0.1, 0.1, 0.1])
    history = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    coefficients = np.array(
        [
            [2.0, -1.0],
            [0.5, 3.0],
        ]
    )
    traces = history.values @ coefficients

    evaluation = evaluate_spatial_coefficient_traces_with_history_basis(
        time_steps,
        history.times,
        traces,
        tau=0.2,
        coefficients=coefficients.reshape(-1),
    )

    np.testing.assert_allclose(evaluation.coefficients, coefficients)
    np.testing.assert_allclose(evaluation.fitted, traces, atol=1.0e-14)
    assert evaluation.relative_l2 < 1.0e-14
    np.testing.assert_allclose(
        evaluation.per_trace_relative_l2,
        np.zeros(2),
        atol=1.0e-14,
    )
    assert evaluation.design_shape == (history.times.size, 2)
    assert evaluation.rank == 2


def test_project_vector_to_spatial_basis_dof_l2_recovers_coefficients():
    spatial_vectors = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, -0.5],
        ]
    )
    expected = np.array([0.25, -0.75])
    target = expected @ spatial_vectors
    receiver_matrix = np.zeros((1, 3, spatial_vectors.shape[1]))

    coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        spatial_vectors,
        target,
        projection="dof_l2",
    )

    np.testing.assert_allclose(coefficients, expected)


def test_project_vector_to_spatial_basis_receiver_l2_uses_receiver_matrix():
    receiver_matrix = np.zeros((2, 3, 3))
    receiver_matrix[0, 2, 0] = 1.0
    receiver_matrix[1, 0, 1] = 2.0
    spatial_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 1.0],
        ]
    )
    target_vector = np.array([0.2, -0.3, 7.0])
    static_response = np.einsum("lci,si->slc", receiver_matrix, spatial_vectors)
    design = np.moveaxis(static_response, 0, -1).reshape(-1, spatial_vectors.shape[0])
    target_response = np.einsum("lci,i->lc", receiver_matrix, target_vector)
    expected, *_ = np.linalg.lstsq(design, target_response.reshape(-1), rcond=None)

    coefficients = project_vector_to_spatial_basis(
        receiver_matrix,
        spatial_vectors,
        target_vector,
        projection="receiver_l2",
    )

    np.testing.assert_allclose(coefficients, expected)


def test_source_history_receiver_basis_scales_static_matrix_response():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    source_vector = np.linspace(-0.25, 0.5, mesh.n_edges)
    time_steps = np.array([0.1, 0.1, 0.1])

    basis = source_history_receiver_basis(
        time_steps,
        tau=0.2,
        source_vector=source_vector,
        receiver_matrix=receiver_matrix,
        max_order=1,
    )

    history = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    static_response = np.einsum("lce,e->lc", receiver_matrix, source_vector)
    expected = history.values[:, :, None, None] * static_response[None, None, :, :]

    np.testing.assert_allclose(basis.times, history.times)
    assert basis.basis_labels == ["BE relaxation", "BE cascade 1"]
    assert basis.responses.shape == (4, 2, locations.shape[0], 3)
    np.testing.assert_allclose(basis.responses, expected)


def test_source_history_receiver_basis_from_vectors_uses_one_vector_per_order():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    source_vectors = np.vstack(
        [
            np.linspace(-0.25, 0.5, mesh.n_edges),
            np.linspace(0.75, -0.5, mesh.n_edges),
        ]
    )
    time_steps = np.array([0.1, 0.1, 0.1])

    basis = source_history_receiver_basis_from_vectors(
        time_steps,
        tau=0.2,
        source_vectors=source_vectors,
        receiver_matrix=receiver_matrix,
    )

    history = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    static_response = np.einsum("lce,pe->plc", receiver_matrix, source_vectors)
    expected = history.values[:, :, None, None] * static_response[None, :, :, :]

    np.testing.assert_allclose(basis.times, history.times)
    assert basis.basis_labels == ["BE relaxation", "BE cascade 1"]
    assert basis.responses.shape == (4, 2, locations.shape[0], 3)
    np.testing.assert_allclose(basis.responses, expected)


def test_source_history_receiver_basis_from_spatial_vectors_uses_all_time_space_pairs():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    spatial_vectors = np.zeros((2, mesh.n_edges))
    spatial_vectors[0, 0] = 1.0
    spatial_vectors[1, mesh.n_edges_x] = -2.0
    time_steps = np.array([0.1, 0.1])

    basis = source_history_receiver_basis_from_spatial_vectors(
        time_steps,
        tau=0.2,
        spatial_vectors=spatial_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=["source edge", "receiver edge"],
    )

    history = discrete_debye_history_basis(time_steps, tau=0.2, max_order=1)
    static_response = np.einsum("lce,se->slc", receiver_matrix, spatial_vectors)
    expected = np.zeros((history.times.size, 4, locations.shape[0], 3))
    expected[:, 0, :, :] = history.values[:, 0, None, None] * static_response[0]
    expected[:, 1, :, :] = history.values[:, 0, None, None] * static_response[1]
    expected[:, 2, :, :] = history.values[:, 1, None, None] * static_response[0]
    expected[:, 3, :, :] = history.values[:, 1, None, None] * static_response[1]

    np.testing.assert_allclose(basis.times, history.times)
    assert basis.basis_labels == [
        "BE relaxation * source edge",
        "BE relaxation * receiver edge",
        "BE cascade 1 * source edge",
        "BE cascade 1 * receiver edge",
    ]
    assert basis.responses.shape == expected.shape
    np.testing.assert_allclose(basis.responses, expected)


def test_source_history_receiver_basis_from_static_response_matches_matrix_path():
    receiver_matrix = np.zeros((2, 3, 3))
    receiver_matrix[0, 2, 0] = 1.0
    receiver_matrix[1, 0, 1] = 2.0
    spatial_vectors = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, -1.0, 2.0],
        ]
    )
    static_response = np.einsum("lci,si->slc", receiver_matrix, spatial_vectors)
    time_steps = np.array([0.1, 0.1])

    from_matrix = source_history_receiver_basis_from_spatial_vectors(
        time_steps,
        tau=0.2,
        spatial_vectors=spatial_vectors,
        receiver_matrix=receiver_matrix,
        max_order=1,
        spatial_labels=["m0", "m2"],
    )
    from_static = source_history_receiver_basis_from_static_response(
        time_steps,
        tau=0.2,
        static_response=static_response,
        max_order=1,
        spatial_labels=["m0", "m2"],
    )

    np.testing.assert_allclose(from_static.times, from_matrix.times)
    assert from_static.basis_labels == from_matrix.basis_labels
    np.testing.assert_allclose(from_static.responses, from_matrix.responses)


def test_fit_static_spatial_coefficients_from_static_response_matches_matrix_path():
    receiver_matrix = np.zeros((2, 3, 3))
    receiver_matrix[0, 2, 0] = 1.0
    receiver_matrix[1, 0, 1] = 2.0
    spatial_vectors = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, -1.0, 2.0],
        ]
    )
    static_response = np.einsum("lci,si->slc", receiver_matrix, spatial_vectors)
    receiver_indices = np.array([0, 1])
    component_indices = np.array([2, 0])
    coefficients = np.array([[1.5, -0.25], [0.2, 2.0]])
    static_design = static_response[:, receiver_indices, component_indices].T
    target = coefficients @ static_design.T

    from_matrix = fit_static_spatial_coefficients_for_components(
        receiver_matrix,
        spatial_vectors,
        target,
        receiver_indices=receiver_indices,
        component_indices=component_indices,
    )
    from_static = fit_static_spatial_coefficients_from_static_response(
        static_response,
        target,
        receiver_indices=receiver_indices,
        component_indices=component_indices,
    )

    np.testing.assert_allclose(from_static.coefficients, from_matrix.coefficients)
    np.testing.assert_allclose(from_static.fitted, from_matrix.fitted)
    assert from_static.relative_l2 == from_matrix.relative_l2
    assert from_static.design_shape == from_matrix.design_shape


def test_fit_source_history_coefficients_recovers_known_coefficients():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    source_vector = np.linspace(-0.25, 0.5, mesh.n_edges)
    basis = source_history_receiver_basis(
        np.array([0.1, 0.1, 0.1]),
        tau=0.2,
        source_vector=source_vector,
        receiver_matrix=receiver_matrix,
        max_order=1,
    )
    coefficients = np.array([-6.0, 2.5])
    target = np.einsum("p,tplc->tlc", coefficients, basis.responses)

    fit = fit_source_history_coefficients(basis.responses, target)

    np.testing.assert_allclose(fit.coefficients, coefficients)
    np.testing.assert_allclose(fit.fitted, target)
    assert fit.relative_l2 < 1.0e-14
    assert fit.design_shape == (basis.responses.shape[0] * locations.shape[0] * 3, 2)
    assert np.all(fit.column_norms > 0.0)
    assert np.isfinite(fit.condition_number)
    assert np.isfinite(fit.column_normalized_condition_number)


def test_fit_source_history_coefficients_for_components_recovers_known_coefficients():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    source_vector = np.linspace(-0.25, 0.5, mesh.n_edges)
    basis = source_history_receiver_basis(
        np.array([0.1, 0.1, 0.1]),
        tau=0.2,
        source_vector=source_vector,
        receiver_matrix=receiver_matrix,
        max_order=1,
    )
    receiver_indices = np.array([0, 1])
    component_indices = np.array([2, 0])
    coefficients = np.array([-3.0, 0.75])
    basis_columns = basis.responses[:, :, receiver_indices, component_indices]
    target_columns = np.einsum("p,tnp->tn", coefficients, np.moveaxis(basis_columns, 1, -1))

    fit = fit_source_history_coefficients_for_components(
        basis.responses,
        target_columns,
        receiver_indices=receiver_indices,
        component_indices=component_indices,
    )

    np.testing.assert_allclose(fit.coefficients, coefficients)
    np.testing.assert_allclose(fit.fitted, target_columns)
    assert fit.relative_l2 < 1.0e-14
    assert fit.design_shape == (basis.responses.shape[0] * receiver_indices.size, 2)
    assert np.all(fit.column_norms > 0.0)
    assert np.isfinite(fit.condition_number)
    assert np.isfinite(fit.column_normalized_condition_number)


def test_evaluate_source_history_coefficients_for_components_uses_prescribed_values():
    mesh = TensorMesh([[1.0], [1.0], [1.0]], origin=(-0.5, -0.5, -0.5))
    locations = np.array([[0.0, 2.0, 0.0], [0.5, 2.5, 0.5]])
    receiver_matrix = edge_current_biot_matrix(mesh, locations)
    source_vector = np.linspace(-0.25, 0.5, mesh.n_edges)
    basis = source_history_receiver_basis(
        np.array([0.1, 0.1, 0.1]),
        tau=0.2,
        source_vector=source_vector,
        receiver_matrix=receiver_matrix,
        max_order=1,
    )
    receiver_indices = np.array([0, 1])
    component_indices = np.array([2, 0])
    coefficients = np.array([-3.0, 0.75])
    basis_columns = basis.responses[:, :, receiver_indices, component_indices]
    target_columns = np.einsum("p,tnp->tn", coefficients, np.moveaxis(basis_columns, 1, -1))

    evaluation = evaluate_source_history_coefficients_for_components(
        basis.responses,
        target_columns,
        coefficients=coefficients,
        receiver_indices=receiver_indices,
        component_indices=component_indices,
    )

    np.testing.assert_allclose(evaluation.coefficients, coefficients)
    np.testing.assert_allclose(evaluation.fitted, target_columns)
    assert evaluation.relative_l2 < 1.0e-14
    assert evaluation.design_shape == (basis.responses.shape[0] * receiver_indices.size, 2)
