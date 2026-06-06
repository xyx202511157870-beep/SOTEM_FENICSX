import numpy as np

from atem3d.materials.prony import PronyConductivity
from atem3d.primary import CachedPrimaryProvider
from atem3d.solvers.primary_secondary_forward import PrimarySecondaryForwardOperator


def test_primary_secondary_forward_zero_contrast_returns_primary_receiver_response():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5, 2.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.array(
            [
                [[10.0, 1.0, 0.0]],
                [[5.0, 0.5, 0.0]],
            ]
        ),
        receiver_dBdt=np.array(
            [
                [[0.0, 0.0, -3.0]],
                [[0.0, 0.0, -1.5]],
            ]
        ),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.01, terms=[]),
        sigma_background=0.01,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(
        predicted,
        [
            [10.0, 1.0, -3.0],
            [5.0, 0.5, -1.5],
        ],
    )


def test_primary_secondary_forward_adds_2d_secondary_receiver_projection():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, -300.0, -0.1]])
    times = np.array([1.0e-5])
    provider = CachedPrimaryProvider(
        times=times,
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]]]),
        Ep_dc_on_V=np.array([[2.0, 0.0, 0.0]]),
    )
    seen = {}

    def dc_solver(contrast_current):
        seen["dc_contrast_current"] = contrast_current.copy()
        return None, np.array([[0.25, 0.0, 0.0]])

    def step_solver(rhs, sigma_eff, dt):
        seen["step_rhs"] = rhs.copy()
        seen["step_sigma_eff"] = sigma_eff
        seen["step_dt"] = dt
        return np.array([[0.5, 0.0, 0.0]])

    def receiver_projector(state, Ep_new, time_value, dt, components):
        seen["receiver_state_Es"] = state.Es.copy()
        seen["receiver_Ep_new"] = Ep_new.copy()
        seen["receiver_time"] = time_value
        seen["receiver_dt"] = dt
        seen["receiver_components"] = tuple(components)
        return np.array([[0.2, -0.1, 0.5]])

    operator = PrimarySecondaryForwardOperator(
        primary=provider,
        fem_points=points,
        receiver_locations=receivers,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity(sigma_inf=0.02, terms=[]),
        sigma_background=0.01,
        secondary_field_solver=dc_solver,
        secondary_step_solver=step_solver,
        secondary_receiver_projector=receiver_projector,
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.2, 0.9, -2.5]])
    np.testing.assert_allclose(seen["dc_contrast_current"], [[0.02, 0.0, 0.0]])
    np.testing.assert_allclose(seen["receiver_state_Es"], [[0.5, 0.0, 0.0]])
    np.testing.assert_allclose(seen["receiver_Ep_new"], [[1.0, 0.0, 0.0]])
    assert seen["receiver_time"] == 1.0e-5
    assert seen["receiver_dt"] == 1.0e-5
    assert seen["receiver_components"] == ("Ex", "Ey", "dBzdt")
