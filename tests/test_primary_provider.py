import builtins

import numpy as np
import pytest

from atem3d.primary import (
    CachedPrimaryProvider,
    EmpymodPrimaryProvider,
    PrimaryFEMInterpolator,
    PrimaryFieldProvider,
    TabulatedVectorField,
    ZeroPrimaryProvider,
    analytic_halfspace_dc_runner,
    analytic_halfspace_grounded_wire_dc_electric_field,
    empymod_quasistatic_dc_runner,
    make_tabulated_vector_assembler,
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


def test_empymod_primary_provider_skeleton_does_not_require_empymod(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "empymod":
            raise ModuleNotFoundError("No module named 'empymod'", name="empymod")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = EmpymodPrimaryProvider(config=_empymod_provider_config())

    with pytest.raises(ModuleNotFoundError, match="empymod"):
        provider.get_receiver_dBdt(0.1, np.array([[0.0, 0.0, 0.0]]))


def test_empymod_primary_provider_uses_default_run_empymod_reference(monkeypatch):
    from atem3d import empymod_compare

    seen = {}

    def fake_reference(survey, **kwargs):
        seen["components"] = survey.components
        seen["times"] = survey.times.copy()
        seen["kwargs"] = kwargs
        return np.array([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(empymod_compare, "run_empymod_reference", fake_reference)
    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        empymod_kwargs={"srcpts": 11},
    )

    values = provider.get_receiver_E(0.5, np.array([[0.0, 10.0, -0.5]]))

    np.testing.assert_allclose(values, [[1.0, 2.0, 3.0]])
    assert seen["components"] == ["Ex", "Ey", "Ez"]
    np.testing.assert_allclose(seen["times"], [0.5])
    assert seen["kwargs"] == {"srcpts": 11}


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


def test_empymod_primary_provider_get_receiver_H_uses_reference_runner():
    def fake_runner(survey, **kwargs):
        assert survey.components == ["Hx", "Hy", "Hz"]
        return np.array([[4.0, 5.0, 6.0]])

    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        reference_runner=fake_runner,
    )

    values = provider.get_receiver_H(1.0e-3, np.array([[0.0, 10.0, -0.5]]))

    np.testing.assert_allclose(values, [[4.0, 5.0, 6.0]])


def test_empymod_primary_provider_get_Ep_on_V_uses_reference_runner():
    seen = {}

    def fake_runner(survey, **kwargs):
        seen["components"] = survey.components
        seen["receiver_locations"] = survey.receiver_locations
        seen["receiver_components"] = survey.receiver_components
        seen["times"] = survey.times.copy()
        seen["kwargs"] = kwargs
        return np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])

    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        reference_runner=fake_runner,
        empymod_kwargs={"srcpts": 9},
    )
    points = np.array([[0.0, 10.0, -0.5], [1.0, 11.0, -0.5]])

    values = provider.get_Ep_on_V(0.25, points)

    np.testing.assert_allclose(values, [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    assert seen["components"] == ["Ex", "Ey", "Ez"]
    assert seen["receiver_locations"] == [(0.0, 10.0, -0.5), (1.0, 11.0, -0.5)]
    assert seen["receiver_components"][0] == ((0.0, 10.0, -0.5), "Ex")
    np.testing.assert_allclose(seen["times"], [0.25])
    assert seen["kwargs"] == {"srcpts": 9}


def test_empymod_primary_provider_get_Ep_dc_on_V_uses_dc_runner():
    seen = {}

    def fake_dc_runner(points, **kwargs):
        seen["points"] = points.copy()
        seen["kwargs"] = kwargs
        return np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])

    provider = EmpymodPrimaryProvider(
        config=_empymod_provider_config(),
        dc_runner=fake_dc_runner,
        dc_kwargs={"method": "analytic_halfspace"},
    )
    points = np.array([[0.0, 10.0, -0.5], [1.0, 11.0, -0.5]])

    values = provider.get_Ep_dc_on_V(points)

    np.testing.assert_allclose(values, [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(seen["points"], points)
    assert seen["kwargs"]["config"] == _empymod_provider_config()
    assert seen["kwargs"]["method"] == "analytic_halfspace"


def test_empymod_primary_provider_get_Ep_dc_on_V_uses_default_halfspace_runner():
    config = _empymod_provider_config()
    provider = EmpymodPrimaryProvider(config=config)
    points = np.array([[0.0, 10.0, -0.5], [1.0, 11.0, -0.5]])

    values = provider.get_Ep_dc_on_V(points)

    np.testing.assert_allclose(values, analytic_halfspace_dc_runner(points, config=config))


def test_empymod_quasistatic_dc_runner_uses_low_frequency_empymod_reference(monkeypatch):
    from atem3d import empymod_compare

    seen = {}

    def fake_reference(survey, **kwargs):
        seen["survey"] = survey
        seen["kwargs"] = kwargs
        return np.array([[0.1, 0.2, 0.3, 1.0, 2.0, 3.0]])

    monkeypatch.setattr(empymod_compare, "run_empymod_reference", fake_reference)
    config = _empymod_provider_config()
    points = np.array([[0.0, 10.0, -0.5], [1.0, 11.0, -0.5]])

    values = empymod_quasistatic_dc_runner(
        points,
        config=config,
        frequency=1.0e-9,
        empymod_kwargs={"srcpts": 11},
    )

    np.testing.assert_allclose(values, [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    assert seen["survey"].components == ["Ex", "Ey", "Ez"]
    assert seen["survey"].signal is None
    np.testing.assert_allclose(seen["survey"].times, [1.0e-9])
    assert seen["kwargs"] == {"srcpts": 11}


def test_analytic_halfspace_grounded_wire_dc_field_matches_endpoint_formula():
    points = np.array([[0.0, 1.0, 0.0]])

    values = analytic_halfspace_grounded_wire_dc_electric_field(
        points,
        source_start=(-1.0, 0.0, 0.0),
        source_end=(1.0, 0.0, 0.0),
        current=2.0,
        resistivity=100.0,
    )

    np.testing.assert_allclose(values, [[-100.0 / (np.pi * np.sqrt(2.0)), 0.0, 0.0]])


def test_analytic_halfspace_dc_runner_reads_empymod_provider_config():
    config = _empymod_provider_config()
    points = np.array([[0.0, 10.0, -0.5]])

    runner_values = analytic_halfspace_dc_runner(points, config=config)
    direct_values = analytic_halfspace_grounded_wire_dc_electric_field(
        points,
        source_start=config["source_start"],
        source_end=config["source_end"],
        current=config["strength"],
        resistivity=config["resistivities"][-1],
    )

    np.testing.assert_allclose(runner_values, direct_values)


def test_primary_fem_interpolator_calls_provider_and_injected_assembler():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    seen = {}

    class RecordingProvider(ZeroPrimaryProvider):
        def get_Ep_on_V(self, t, V):
            seen["transient"] = (float(t), np.asarray(V).copy())
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def get_Ep_dc_on_V(self, V):
            seen["dc"] = np.asarray(V).copy()
            return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def assembler(query_points, values):
        seen["assembled"] = (query_points.copy(), values.copy())
        return values.reshape(-1)

    interpolator = PrimaryFEMInterpolator(
        provider=RecordingProvider(),
        points=points,
        assembler=assembler,
    )

    np.testing.assert_allclose(interpolator.interpolate_Ep(0.25), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(interpolator.interpolate_Ep_dc(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert seen["transient"][0] == 0.25
    np.testing.assert_allclose(seen["transient"][1], points)
    np.testing.assert_allclose(seen["dc"], points)
    np.testing.assert_allclose(seen["assembled"][0], points)


def test_primary_fem_interpolator_zero_provider_returns_point_samples_without_assembler():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    interpolator = PrimaryFEMInterpolator(provider=ZeroPrimaryProvider(), points=points)

    np.testing.assert_allclose(interpolator.interpolate_Ep(1.0e-3), np.zeros((2, 3)))
    np.testing.assert_allclose(interpolator.interpolate_Ep_dc(), np.zeros((2, 3)))
    np.testing.assert_allclose(interpolator.sample_Ep_times([0.0, 1.0]), np.zeros((2, 2, 3)))


def test_primary_fem_interpolator_preserves_cached_provider_time_samples():
    points = np.array([[0.0, 0.0, 0.0]])
    receivers = np.array([[10.0, 0.0, 0.0]])
    provider = CachedPrimaryProvider(
        times=np.array([0.0, 2.0]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array([[[0.0, 0.0, 0.0]], [[2.0, 4.0, 6.0]]]),
        receiver_E=np.zeros((2, 1, 3)),
        receiver_dBdt=np.zeros((2, 1, 3)),
        Ep_dc_on_V=np.array([[10.0, 20.0, 30.0]]),
    )
    interpolator = PrimaryFEMInterpolator(provider=provider, points=points)

    np.testing.assert_allclose(interpolator.sample_Ep(1.0), [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(interpolator.sample_Ep_dc(), [[10.0, 20.0, 30.0]])
    np.testing.assert_allclose(
        interpolator.sample_Ep_times([0.0, 1.0, 2.0]),
        [[[0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0]], [[2.0, 4.0, 6.0]]],
    )


def test_tabulated_vector_field_returns_dolfinx_style_component_matrix():
    field = TabulatedVectorField(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        values=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
    )
    x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    np.testing.assert_allclose(field(x), [[40.0, 10.0], [50.0, 20.0], [60.0, 30.0]])


def test_tabulated_vector_field_rejects_unknown_interpolation_point():
    field = TabulatedVectorField(
        points=np.array([[0.0, 0.0, 0.0]]),
        values=np.array([[1.0, 2.0, 3.0]]),
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="not in tabulated primary field points"):
        field(np.array([[1.0], [0.0], [0.0]]))


def test_primary_fem_interpolator_can_return_tabulated_dolfinx_callable():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    receivers = np.array([[0.0, 0.0, 0.0]])
    provider = CachedPrimaryProvider(
        times=np.array([0.0, 2.0]),
        points=points,
        receivers=receivers,
        Ep_on_V=np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            ]
        ),
        receiver_E=np.zeros((2, 1, 3)),
        receiver_dBdt=np.zeros((2, 1, 3)),
        Ep_dc_on_V=np.array([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]),
    )
    interpolator = PrimaryFEMInterpolator(
        provider=provider,
        points=points,
        assembler=make_tabulated_vector_assembler(),
    )

    transient = interpolator.interpolate_Ep(1.0)
    dc = interpolator.interpolate_Ep_dc()
    x = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    np.testing.assert_allclose(transient(x), [[11.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(dc(x), [[200.0, 100.0], [0.0, 0.0], [0.0, 0.0]])


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
