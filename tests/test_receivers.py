import numpy as np
from discretize import TensorMesh

from atem3d.receivers import PointReceiver


def test_point_receiver_samples_db_dt_component_from_two_b_states():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="dBzdt")
    e = np.zeros(mesh.n_edges)
    b_old = np.zeros(mesh.n_faces)
    b_new = np.ones(mesh.n_faces)

    value = receiver.sample_time_derivative(mesh, e, b_new, b_old, dt=2.0)

    assert value == 0.5


def test_point_receiver_marks_db_dt_as_two_state_observation():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="dBzdt")

    assert receiver.requires_previous_magnetic_state is True
    assert receiver.sample(
        mesh,
        np.zeros(mesh.n_edges),
        np.zeros(mesh.n_faces),
    ) == 0.0


def test_point_receiver_samples_hj_electric_faces_and_magnetic_edges():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    location = np.array([[0.0, 0.0, 0.0]])
    e_faces = np.linspace(-1.0, 1.0, mesh.n_faces)
    h_edges = np.linspace(0.5, 2.5, mesh.n_edges)

    ex_receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="Ex")
    hz_receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="Hz")
    bz_receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="Bz")

    expected_ex = mesh.get_interpolation_matrix(location, "Fx") @ e_faces
    expected_hz = mesh.get_interpolation_matrix(location, "Ez") @ h_edges

    np.testing.assert_allclose(
        ex_receiver.sample_hj(mesh, e_faces, h_edges),
        expected_ex[0],
    )
    np.testing.assert_allclose(
        hz_receiver.sample_hj(mesh, e_faces, h_edges),
        expected_hz[0],
    )
    np.testing.assert_allclose(
        bz_receiver.sample_hj(mesh, e_faces, h_edges, mu=4.0),
        4.0 * expected_hz[0],
    )


def test_point_receiver_samples_hj_db_dt_from_edge_h_states():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="dBzdt")
    e_faces = np.zeros(mesh.n_faces)
    h_old = np.zeros(mesh.n_edges)
    h_new = np.ones(mesh.n_edges)

    value = receiver.sample_hj_time_derivative(
        mesh,
        e_faces,
        h_new,
        h_old,
        dt=2.0,
        mu=4.0,
    )

    assert value == 2.0


def test_point_receiver_hj_initial_db_dt_convention_is_explicit_zero():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    receiver = PointReceiver(location=(0.0, 0.0, 0.0), component="dBzdt")

    assert receiver.requires_previous_magnetic_state is True
    assert receiver.sample_hj(
        mesh,
        np.zeros(mesh.n_faces),
        np.zeros(mesh.n_edges),
    ) == 0.0
