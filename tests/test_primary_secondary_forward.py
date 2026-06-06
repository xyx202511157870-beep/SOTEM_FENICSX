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
