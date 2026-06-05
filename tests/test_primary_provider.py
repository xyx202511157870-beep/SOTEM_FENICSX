import builtins

import numpy as np
import pytest

from atem3d.primary import (
    CachedPrimaryProvider,
    EmpymodPrimaryProvider,
    PrimaryFieldProvider,
    ZeroPrimaryProvider,
)


def test_zero_primary_provider_returns_zero_arrays_with_expected_shapes():
    provider = ZeroPrimaryProvider()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    receivers = np.array([[5.0, 6.0, 7.0]])

    np.testing.assert_allclose(provider.get_Ep_on_V(0.1, points), np.zeros((2, 3)))
    np.testing.assert_allclose(provider.get_Ep_dc_on_V(points), np.zeros((2, 3)))
    np.testing.assert_allclose(provider.get_receiver_E(0.1, receivers), np.zeros((1, 3)))
    np.testing.assert_allclose(provider.get_receiver_dBdt(0.1, receivers), np.zeros((1, 3)))


def test_cached_primary_provider_returns_exact_cached_values():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    receivers = np.array([[0.5, 0.0, 0.0]])
    provider = CachedPrimaryProvider(
        times=np.array([0.1, 1.0]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            ]
        ),
        receiver_E=np.array([[[7.0, 8.0, 9.0]], [[70.0, 80.0, 90.0]]]),
        receiver_dBdt=np.array([[[0.7, 0.8, 0.9]], [[7.0, 8.0, 9.0]]]),
        Ep_dc_on_V=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )

    np.testing.assert_allclose(provider.get_Ep_on_V(1.0, points), provider.Ep_on_V[1])
    np.testing.assert_allclose(provider.get_Ep_dc_on_V(points), provider.Ep_dc_on_V)
    np.testing.assert_allclose(provider.get_receiver_E(0.1, receivers), provider.receiver_E[0])
    np.testing.assert_allclose(
        provider.get_receiver_dBdt(1.0, receivers),
        provider.receiver_dBdt[1],
    )


def test_cached_primary_provider_interpolates_in_time():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[0.0, 0.0, 0.0]])
    provider = CachedPrimaryProvider(
        times=np.array([0.0, 2.0]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[0.0, 0.0, 0.0]], [[2.0, 4.0, 6.0]]]),
        receiver_E=np.array([[[10.0, 0.0, 0.0]], [[14.0, 0.0, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 1.0, 0.0]], [[0.0, 5.0, 0.0]]]),
        Ep_dc_on_V=np.array([[1.0, 2.0, 3.0]]),
    )

    np.testing.assert_allclose(provider.get_Ep_on_V(0.5, points), [[0.5, 1.0, 1.5]])
    np.testing.assert_allclose(provider.get_receiver_E(0.5, receivers), [[11.0, 0.0, 0.0]])
    np.testing.assert_allclose(provider.get_receiver_dBdt(0.5, receivers), [[0.0, 2.0, 0.0]])


def test_cached_primary_provider_rejects_wrong_query_locations():
    provider = CachedPrimaryProvider(
        times=np.array([0.0]),
        points=np.array([[0.0, 0.0, 0.0]]),
        receivers=np.array([[1.0, 0.0, 0.0]]),
        Ep_on_V=np.zeros((1, 1, 3)),
        receiver_E=np.zeros((1, 1, 3)),
        receiver_dBdt=np.zeros((1, 1, 3)),
        Ep_dc_on_V=np.zeros((1, 3)),
    )

    with pytest.raises(ValueError, match="points"):
        provider.get_Ep_on_V(0.0, np.array([[9.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="receivers"):
        provider.get_receiver_E(0.0, np.array([[9.0, 0.0, 0.0]]))


def test_empymod_primary_provider_delays_import_until_evaluation():
    provider = EmpymodPrimaryProvider(config={"src": "placeholder"})
    assert isinstance(provider, PrimaryFieldProvider)

    with pytest.raises(NotImplementedError, match="EmpymodPrimaryProvider"):
        provider.get_receiver_E(0.1, np.array([[0.0, 0.0, 0.0]]))


def test_empymod_primary_provider_skeleton_does_not_require_empymod(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "empymod":
            raise ModuleNotFoundError("No module named 'empymod'", name="empymod")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = EmpymodPrimaryProvider(config={})

    with pytest.raises(NotImplementedError, match="EmpymodPrimaryProvider"):
        provider.get_receiver_dBdt(0.1, np.array([[0.0, 0.0, 0.0]]))


def test_empymod_primary_provider_get_receiver_E_uses_reference_runner():
    seen = {}

    def fake_runner(survey, **kwargs):
        seen["components"] = survey.components
        seen["receiver_components"] = survey.receiver_components
        seen["times"] = survey.times.copy()
        seen["kwargs"] = kwargs
        return np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        reference_runner=fake_runner,
        empymod_kwargs={"srcpts": 7},
    )
    receivers = np.array([[0.0, 10.0, -0.5], [1.0, 11.0, -0.5]])

    values = provider.get_receiver_E(0.25, receivers)

    np.testing.assert_allclose(values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert seen["components"] == ["Ex", "Ey", "Ez"]
    assert seen["receiver_components"][0] == ((0.0, 10.0, -0.5), "Ex")
    np.testing.assert_allclose(seen["times"], [0.25])
    assert seen["kwargs"] == {"srcpts": 7}


def test_empymod_primary_provider_get_receiver_dBdt_uses_reference_runner():
    def fake_runner(survey, **kwargs):
        assert survey.components == ["dBxdt", "dBydt", "dBzdt"]
        return np.array([[7.0, 8.0, 9.0]])

    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        reference_runner=fake_runner,
    )

    values = provider.get_receiver_dBdt(1.0e-3, np.array([[0.0, 10.0, -0.5]]))

    np.testing.assert_allclose(values, [[7.0, 8.0, 9.0]])


def _empymod_provider_config():
    return {
        "source_start": [-25.0, 0.0, -0.5],
        "source_end": [25.0, 0.0, -0.5],
        "depths": [0.0],
        "resistivities": [1.0e8, 100.0],
        "strength": 10.0,
        "signal": -1,
        "coordinate_system": "z_up",
    }
