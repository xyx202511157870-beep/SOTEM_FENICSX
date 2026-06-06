import numpy as np

from atem3d.solvers.primary_secondary_forward import _flatten_components


def test_dbdt_component_columns_are_flattened_from_receiver_dbdt_vectors():
    receiver_e = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
    receiver_dbdt = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    row = _flatten_components(
        receiver_e[0],
        receiver_dbdt[0],
        ("dBxdt", "dBydt", "dBzdt"),
    )

    np.testing.assert_allclose(row, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_mixed_electric_and_dbdt_components_preserve_receiver_order():
    receiver_e = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    receiver_dbdt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    row = _flatten_components(receiver_e, receiver_dbdt, ("Ex", "Ey", "dBzdt"))

    np.testing.assert_allclose(row, [10.0, 20.0, 3.0, 40.0, 50.0, 6.0])
