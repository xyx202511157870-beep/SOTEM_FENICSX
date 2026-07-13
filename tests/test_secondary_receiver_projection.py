import numpy as np

from atem3d.solvers.receiver_projection import SecondaryReceiverProjection
from atem3d.solvers.tdem_secondary import SecondaryState


def test_secondary_receiver_projection_maps_electric_and_dbdt_components():
    state = SecondaryState(
        Es=np.array([[1.0, 2.0, 3.0]]),
        deltaJ=np.zeros((1, 3)),
        chi=[],
    )
    receiver_locations = np.array(
        [
            [0.0, -300.0, -0.1],
            [10.0, -300.0, -0.1],
        ]
    )
    seen = {}

    def electric_sampler(sample_state, Ep_new, time_value, dt, receivers):
        seen["electric"] = (sample_state, Ep_new.copy(), time_value, dt, receivers.copy())
        return np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    def dbdt_sampler(sample_state, Ep_new, time_value, dt, receivers):
        seen["dbdt"] = (sample_state, Ep_new.copy(), time_value, dt, receivers.copy())
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    projector = SecondaryReceiverProjection(
        receiver_locations=receiver_locations,
        electric_sampler=electric_sampler,
        dbdt_sampler=dbdt_sampler,
    )

    values = projector(
        state,
        np.array([[0.5, 0.0, 0.0]]),
        1.0e-5,
        1.0e-6,
        ("Ex", "Ey", "dBzdt"),
    )

    np.testing.assert_allclose(values, [[10.0, 20.0, 3.0], [40.0, 50.0, 6.0]])
    assert seen["electric"][0] is state
    assert seen["dbdt"][0] is state
    np.testing.assert_allclose(seen["electric"][4], receiver_locations)


def test_secondary_receiver_projection_maps_h_components_with_magnetic_sampler():
    state = SecondaryState(
        Es=np.array([[1.0, 2.0, 3.0]]),
        deltaJ=np.zeros((1, 3)),
        chi=[],
    )
    receiver_locations = np.array([[0.0, -300.0, -0.1]])
    seen = {}

    def electric_sampler(sample_state, Ep_new, time_value, dt, receivers):
        return np.array([[10.0, 20.0, 30.0]])

    def dbdt_sampler(sample_state, Ep_new, time_value, dt, receivers):
        return np.array([[1.0, 2.0, 3.0]])

    def magnetic_sampler(sample_state, Ep_new, time_value, dt, receivers):
        seen["magnetic"] = (sample_state, Ep_new.copy(), time_value, dt, receivers.copy())
        return np.array([[4.0, 5.0, 6.0]])

    projector = SecondaryReceiverProjection(
        receiver_locations=receiver_locations,
        electric_sampler=electric_sampler,
        dbdt_sampler=dbdt_sampler,
        magnetic_sampler=magnetic_sampler,
    )

    values = projector(
        state,
        np.array([[0.5, 0.0, 0.0]]),
        1.0e-5,
        1.0e-6,
        ("Ex", "Hz", "dBzdt"),
    )

    np.testing.assert_allclose(values, [[10.0, 6.0, 3.0]])
    assert seen["magnetic"][0] is state
    np.testing.assert_allclose(seen["magnetic"][4], receiver_locations)
