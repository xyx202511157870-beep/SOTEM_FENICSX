from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_analytic_dc_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analytic_halfspace_dc_field_matches_grounded_wire_formula():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-50.0, 0.0, -0.1),
        source_end=(50.0, 0.0, -0.1),
        receiver=(500.0, 50.0, -0.1),
        rho_earth=100.0,
        source_current=1.0,
    )

    value = sp._analytic_halfspace_dc_electric_field(np.asarray([config.receiver]), config)[0]

    r_start = np.asarray(config.receiver) - np.asarray(config.source_start)
    r_end = np.asarray(config.receiver) - np.asarray(config.source_end)
    expected = config.rho_earth * config.source_current / (2.0 * np.pi) * (
        r_end / np.linalg.norm(r_end) ** 3 - r_start / np.linalg.norm(r_start) ** 3
    )
    assert expected[0] > 0.0
    assert expected[1] > 0.0
    np.testing.assert_allclose(value, expected)


def test_analytic_halfspace_dc_field_uses_uniform_layer_resistivity():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-50.0, 0.0, -0.1),
        source_end=(50.0, 0.0, -0.1),
        receiver=(500.0, 50.0, -0.1),
        rho_earth=100.0,
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 200.0, 200.0),
        source_current=1.0,
    )

    value = sp._analytic_halfspace_dc_electric_field(np.asarray([config.receiver]), config)[0]

    r_start = np.asarray(config.receiver) - np.asarray(config.source_start)
    r_end = np.asarray(config.receiver) - np.asarray(config.source_end)
    expected = 200.0 * config.source_current / (2.0 * np.pi) * (
        r_end / np.linalg.norm(r_end) ** 3 - r_start / np.linalg.norm(r_start) ** 3
    )
    np.testing.assert_allclose(value, expected)


def test_initial_dc_mode_defaults_to_fem():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().initial_dc_mode == "fem"


def test_interpolate_vector_callable_to_nedelec_function_uses_dolfinx_function(monkeypatch):
    sp = _load_pipeline_module()
    calls = {}

    class FakeX:
        def __init__(self):
            self.scatter_count = 0

        def scatter_forward(self):
            self.scatter_count += 1

    class FakeFunction:
        def __init__(self, function_space, name):
            calls["space"] = function_space
            calls["name"] = name
            self.x = FakeX()

        def interpolate(self, field_callable):
            x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
            calls["interpolated"] = field_callable(x)

    fake_fem = types.SimpleNamespace(Function=FakeFunction)
    monkeypatch.setitem(sys.modules, "dolfinx", types.SimpleNamespace(fem=fake_fem))
    V = object()
    spaces = {"V": V}

    function = sp._interpolate_vector_callable_to_nedelec_function(
        spaces,
        name="E_primary_test",
        field_callable=lambda x: x + 10.0,
    )

    assert calls["space"] is V
    assert calls["name"] == "E_primary_test"
    np.testing.assert_allclose(calls["interpolated"], [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
    assert function.x.scatter_count == 1


def test_ramp_average_is_applied_to_dbdt_reference_component():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0, 4.0])
    values = np.asarray([10.0, 8.0, 4.0])

    averaged = sp._apply_reference_ramp_average("dBzdt", times, values, ramp_time=1.0)

    assert not np.allclose(averaged, values)


def test_ramp_average_from_dense_samples_without_sparse_extrapolation():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0])
    dense_times = np.asarray([0.25, 0.5, 1.0, 2.0])
    dense_values = dense_times.copy()

    averaged = sp._ramp_average_from_dense(times, dense_times, dense_values, ramp_time=1.0)

    np.testing.assert_allclose(averaged, np.asarray([0.625, 1.5]))


def test_ramp_average_from_dense_supports_after_ramp_window():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0])
    dense_times = np.asarray([1.0, 2.0, 3.0])
    dense_values = dense_times.copy()

    averaged = sp._ramp_average_from_dense(
        times, dense_times, dense_values, ramp_time=1.0, window="after_ramp"
    )

    np.testing.assert_allclose(averaged, np.asarray([1.5, 2.5]))


def test_empymod_call_kwargs_include_configured_source_points_and_qwe_transforms():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(empymod_srcpts=33, empymod_ht="qwe", empymod_ft="qwe")

    kwargs = sp._empymod_call_kwargs(config)

    assert kwargs["srcpts"] == 33
    assert kwargs["ht"] == "qwe"
    assert kwargs["ft"] == "qwe"
    assert kwargs["htarg"]["rtol"] < 1.0e-10
    assert kwargs["ftarg"]["rtol"] <= 1.0e-10


def test_empymod_call_kwargs_omit_default_dlf_transform_arguments():
    sp = _load_pipeline_module()

    kwargs = sp._empymod_call_kwargs(sp.PipelineConfig())

    assert kwargs == {"srcpts": 5}


def test_horizontal_electric_vector_error_uses_vector_norm_denominator():
    sp = _load_pipeline_module()
    fem = np.asarray([[3.0, 4.0, 1.0], [6.0, 8.0, 2.0]])
    ref = np.asarray([[3.0, 4.0, 1.0], [3.0, 4.0, 2.0]])

    values = sp.compute_horizontal_electric_error(fem, ref)

    assert values["mean"] == np.mean([0.0, 1.0])
    assert values["max"] == 1.0
    np.testing.assert_allclose(values["relative"], np.asarray([0.0, 1.0]))


def test_compute_windowed_error_metrics_filters_times_before_error_min_time():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0])
    fem = np.asarray([[10.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    ref = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])

    metrics = sp.compute_windowed_error_metrics(times, fem, ref, ["Ex", "Ey", "dBzdt"], error_min_time=2.0)

    assert metrics["time_count"] == 1
    assert metrics["errors"]["Ex"]["mean"] == 0.0
    assert metrics["horizontal_electric"]["mean"] == 0.0


def test_find_physical_error_passing_window_returns_earliest_strict_start():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0, 3.0])
    fem = np.asarray(
        [
            [2.0, 0.0, 2.0],
            [1.02, 0.0, 1.01],
            [0.98, 0.0, 0.99],
        ]
    )
    ref = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    window = sp.find_physical_error_passing_window(times, fem, ref, ["Ex", "Ey", "dBzdt"], tolerance=0.05)

    assert window["time_min"] == pytest.approx(2.0)
    assert window["time_count"] == 2
    assert window["maxima"]["Ex"] == pytest.approx(0.02)
    assert window["maxima"]["dBzdt"] == pytest.approx(0.01)
    assert window["maxima"]["Eh_vector"] == pytest.approx(0.02)


def test_check_physical_error_window_reports_pass_and_maxima():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0, 3.0])
    fem = np.asarray([[2.0, 0.0, 2.0], [1.02, 0.0, 1.01], [0.98, 0.0, 0.99]])
    ref = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    result = sp.check_physical_error_window(times, fem, ref, ["Ex", "Ey", "dBzdt"], error_min_time=2.0, tolerance=0.05)

    assert result["passed"] is True
    assert result["time_count"] == 2
    assert result["maxima"]["Ex"] == pytest.approx(0.02)
    assert result["maxima"]["dBzdt"] == pytest.approx(0.01)
    assert result["maxima"]["Eh_vector"] == pytest.approx(0.02)


def test_check_physical_error_window_reports_failure():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 2.0])
    fem = np.asarray([[1.0, 0.0, 1.0], [1.2, 0.0, 1.0]])
    ref = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    result = sp.check_physical_error_window(times, fem, ref, ["Ex", "Ey", "dBzdt"], error_min_time=2.0, tolerance=0.05)

    assert result["passed"] is False
    assert result["maxima"]["Ex"] == pytest.approx(0.2)


def test_report_distinguishes_physical_pass_from_weak_component_relative_error(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, error_tolerance=0.05)
    times = np.asarray([1.0, 2.0])
    fem = np.asarray([[100.0, 1.0e-7, 1.0], [100.0, 1.0e-7, 1.0]])
    ref = np.asarray([[100.0, 1.0e-9, 1.0], [100.0, 1.0e-9, 1.0]])
    components = ["Ex", "Ey", "dBzdt"]
    result = {
        "times": times,
        "data": fem,
        "components": components,
        "solver_log": [{"step": 0, "time": 1.0, "dt": 1.0, "its": 1, "residual": 0.0, "reason": 2}],
    }
    reference = {"times": times, "data": ref}
    errors = sp.compute_error(fem, ref, components)

    sp.write_report(
        config,
        env={},
        fem_result=result,
        ref_result=reference,
        errors=errors,
        source_info={"mode": "manual_line"},
    )

    report = config.output_report().read_text(encoding="utf-8")
    assert "strict physical-error passing window" in report
    assert "[optimization note] The configured run exceeds the physical gate" not in report
    assert "Component relative error exceeds tolerance for weak components: Ey." in report
    assert "weak horizontal-component absolute gate also passes" in report


def test_plot_verification_supports_hz_component(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0e-5, 1.0e-4])
    components = ["Ex", "Ey", "Hz", "dBzdt"]
    fem = np.asarray(
        [
            [1.0, 0.1, 0.01, 0.001],
            [0.8, 0.08, 0.008, 0.0008],
        ]
    )
    ref = fem * 1.1
    errors = sp.compute_error(fem, ref, components)

    sp.plot_verification(times, fem, ref, errors, components, config)

    assert config.output_png().exists()


def test_compute_error_uses_pointwise_relative_error_not_floor_denominator():
    sp = _load_pipeline_module()
    fem = np.asarray([[10.0], [1.1e-12]])
    ref = np.asarray([[10.0], [1.0e-12]])

    errors = sp.compute_error(fem, ref, ["Ex"])

    assert errors["Ex"]["floor"] == pytest.approx(1.0e-5)
    assert errors["Ex"]["relative"][1] == pytest.approx(0.1)


def test_compute_error_uses_component_minimum_floor_for_near_zero_electric_field():
    sp = _load_pipeline_module()
    fem = np.asarray([[1.0e-6], [2.0e-6]])
    ref = np.asarray([[0.0], [1.0e-21]])

    errors = sp.compute_error(fem, ref, ["Ey"])

    assert errors["Ey"]["floor"] == pytest.approx(1.0e-14)
    assert errors["Ey"]["relative"][0] == np.inf
    assert errors["Ey"]["relative"][1] == pytest.approx(2.0e15)


def test_receiver_sampling_points_point_receiver_uses_center_only():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(1.0, 2.0, -0.1), receiver_type="point")

    points = sp._receiver_sampling_points(config)

    np.testing.assert_allclose(points, [[1.0, 2.0, -0.1]])


def test_receiver_sampling_points_volume_average_uses_center_and_axis_offsets():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(1.0, 2.0, -0.1),
        receiver_type="volume_average",
        receiver_average_radius=2.0,
    )

    points = sp._receiver_sampling_points(config)

    expected = np.asarray(
        [
            [1.0, 2.0, -0.1],
            [3.0, 2.0, -0.1],
            [-1.0, 2.0, -0.1],
            [1.0, 4.0, -0.1],
            [1.0, 0.0, -0.1],
            [1.0, 2.0, 1.9],
            [1.0, 2.0, -2.1],
        ]
    )
    np.testing.assert_allclose(points, expected)


def test_receiver_sampling_points_disk_average_uses_horizontal_offsets():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(1.0, 2.0, -0.1),
        receiver_type="disk_average",
        receiver_average_radius=2.0,
    )

    points = sp._receiver_sampling_points(config)

    expected = np.asarray(
        [
            [1.0, 2.0, -0.1],
            [3.0, 2.0, -0.1],
            [-1.0, 2.0, -0.1],
            [1.0, 4.0, -0.1],
            [1.0, 0.0, -0.1],
        ]
    )
    np.testing.assert_allclose(points, expected)


def test_receiver_sample_aggregation_collapses_cell_candidates_then_averages_points():
    sp = _load_pipeline_module()
    sample_values = [
        np.asarray([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0], [9.0, 90.0, 900.0]]),
        np.asarray([[10.0, 20.0, 30.0]]),
    ]

    median = sp._aggregate_receiver_sample_values(sample_values, "median")
    mean = sp._aggregate_receiver_sample_values(sample_values, "mean")
    first = sp._aggregate_receiver_sample_values(sample_values, "first_cell")

    np.testing.assert_allclose(median, [6.5, 25.0, 165.0])
    np.testing.assert_allclose(mean, [43.0 / 6.0, 95.0 / 3.0, 695.0 / 3.0])
    np.testing.assert_allclose(first, [5.5, 15.0, 65.0])


def test_debye_backward_euler_coefficients_are_dimensionless():
    sp = _load_pipeline_module()
    term = sp.DebyeTerm(delta_sigma=2.0, tau=4.0)

    alpha, beta = sp._debye_backward_euler_coefficients(term, dt=1.0)

    assert alpha == pytest.approx(4.0 / 5.0)
    assert beta == pytest.approx(1.0 / 5.0)
    assert alpha + beta == pytest.approx(1.0)


def test_debye_exponential_coefficients_use_exact_constant_field_update():
    sp = _load_pipeline_module()
    term = sp.DebyeTerm(delta_sigma=2.0, tau=4.0)

    alpha, beta = sp._debye_time_coefficients(term, dt=1.0, scheme="exponential")

    assert alpha == pytest.approx(np.exp(-0.25))
    assert beta == pytest.approx(1.0 - np.exp(-0.25))
    assert alpha + beta == pytest.approx(1.0)


def test_debye_time_coefficients_default_to_backward_euler():
    sp = _load_pipeline_module()
    term = sp.DebyeTerm(delta_sigma=2.0, tau=4.0)

    alpha, beta = sp._debye_time_coefficients(term, dt=1.0, scheme="backward_euler")

    assert alpha == pytest.approx(4.0 / 5.0)
    assert beta == pytest.approx(1.0 / 5.0)
