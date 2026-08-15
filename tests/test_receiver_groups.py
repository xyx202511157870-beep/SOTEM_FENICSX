import numpy as np
from discretize import TensorMesh

from atem3d.receiver_groups import (
    H3_COMPONENTS,
    build_h3_receiver_array,
    build_h3_receivers,
    reshape_h3_data,
)


def _constant_face_vector(mesh, values):
    hx, hy, hz = (float(value) for value in values)
    return np.r_[
        np.full(mesh.n_faces_x, hx),
        np.full(mesh.n_faces_y, hy),
        np.full(mesh.n_faces_z, hz),
    ]


def _constant_edge_vector(mesh, values):
    hx, hy, hz = (float(value) for value in values)
    return np.r_[
        np.full(mesh.n_edges_x, hx),
        np.full(mesh.n_edges_y, hy),
        np.full(mesh.n_edges_z, hz),
    ]


def test_build_h3_point_receivers_has_fixed_component_order():
    receivers = build_h3_receivers(location=(0.0, 0.0, 0.0))

    assert tuple(receiver.component for receiver in receivers) == H3_COMPONENTS
    assert len(receivers) == 3


def test_build_h3_disk_receivers_uses_three_orthogonal_normals():
    receivers = build_h3_receivers(
        location=(0.0, 0.0, 0.0),
        receiver_type="disk_average",
        radius=1.0,
    )

    normals = np.array([receiver.normal for receiver in receivers])
    np.testing.assert_allclose(normals, np.eye(3))
    assert all(receiver.sample_count == 36 for receiver in receivers)


def test_h3_receivers_return_h_from_eb_and_hj_formulations():
    mesh = TensorMesh([np.ones(2), np.ones(2), np.ones(2)], origin="CCC")
    receivers = build_h3_receivers(location=(0.0, 0.0, 0.0))
    mu = 4.0

    e_edges = np.zeros(mesh.n_edges)
    b_faces = _constant_face_vector(mesh, mu * np.array([1.0, 2.0, 3.0]))
    eb_values = np.array(
        [receiver.sample(mesh, e_edges, b_faces, mu=mu) for receiver in receivers]
    )

    e_faces = np.zeros(mesh.n_faces)
    h_edges = _constant_edge_vector(mesh, [1.0, 2.0, 3.0])
    hj_values = np.array(
        [receiver.sample_hj(mesh, e_faces, h_edges, mu=mu) for receiver in receivers]
    )

    np.testing.assert_allclose(eb_values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(hj_values, [1.0, 2.0, 3.0])


def test_h3_receiver_array_is_location_major_and_reshape_is_consistent():
    receivers = build_h3_receiver_array(
        locations=[(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
    )

    assert [receiver.component for receiver in receivers] == [
        "Hx",
        "Hy",
        "Hz",
        "Hx",
        "Hy",
        "Hz",
    ]

    data = np.arange(12.0).reshape(2, 6)
    reshaped = reshape_h3_data(data, n_locations=2)

    assert reshaped.shape == (2, 2, 3)
    np.testing.assert_allclose(reshaped[0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(reshaped[0, 1], [3.0, 4.0, 5.0])
