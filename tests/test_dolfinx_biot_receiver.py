from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_biot_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_biot_savart_line_h_matches_finite_wire_midpoint_formula():
    sp = _load_pipeline_module()

    value = sp._biot_savart_line_h(
        np.asarray([[0.0, 10.0, 0.0]]),
        np.asarray([-25.0, 0.0, 0.0]),
        np.asarray([25.0, 0.0, 0.0]),
        current=1.0,
        n_quad=801,
    )[0]

    half_length = 25.0
    offset = 10.0
    expected_hz = half_length / (2.0 * np.pi * offset * np.sqrt(half_length**2 + offset**2))
    np.testing.assert_allclose(value, [0.0, 0.0, expected_hz], rtol=2.0e-5, atol=1.0e-12)


def test_biot_savart_line_h_at_receiver_respects_disk_average_sampling():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-10.0, 0.0, -0.1),
        source_end=(10.0, 0.0, -0.1),
        receiver=(0.0, 5.0, -0.1),
        receiver_type="disk_average",
        receiver_average_radius=1.0,
    )

    value = sp._biot_savart_line_h_at_receiver(config, current=2.0, n_quad=401)
    expected = np.mean(
        sp._biot_savart_line_h(
            sp._receiver_sampling_points(config),
            np.asarray(config.source_start, dtype=float),
            np.asarray(config.source_end, dtype=float),
            current=2.0,
            n_quad=401,
        ),
        axis=0,
    )

    np.testing.assert_allclose(value, expected, rtol=1.0e-12, atol=1.0e-14)


def test_debye_cell_current_density_uses_memory_current():
    sp = _load_pipeline_module()

    e_vals = np.asarray([[10.0, 1.0, 0.0], [2.0, 20.0, 0.0]])
    sigma_inf = np.asarray([0.02, 0.03])
    delta_values = [
        np.asarray([0.005, 0.001]),
        np.asarray([0.002, 0.004]),
    ]
    memory_values = [
        np.asarray([[3.0, 4.0, 0.0], [5.0, 6.0, 0.0]]),
        np.asarray([[7.0, 8.0, 0.0], [9.0, 10.0, 0.0]]),
    ]

    currents = sp._cell_current_density_from_debye_values(e_vals, sigma_inf, delta_values, memory_values)

    expected = sigma_inf[:, None] * e_vals
    for delta, memory in zip(delta_values, memory_values):
        expected -= delta[:, None] * memory
    np.testing.assert_allclose(currents, expected)


def test_debye_memories_are_initialised_to_dc_field():
    sp = _load_pipeline_module()

    class X:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float)

        def scatter_forward(self):
            return None

    class FunctionLike:
        def __init__(self, values):
            self.x = X(values)

    e_old = FunctionLike([1.0, 2.0, 3.0])
    memories = [FunctionLike([0.0, 0.0, 0.0]), FunctionLike([9.0, 9.0, 9.0])]

    sp._initialise_debye_memories_to_field(memories, e_old)

    for memory in memories:
        np.testing.assert_allclose(memory.x.array, e_old.x.array)


def test_biot_receiver_preserves_instantaneous_dbdt():
    sp = _load_pipeline_module()

    receiver = {"Ex": 1.0, "Ey": 2.0, "dBzdt": 3.0}

    sp._assign_biot_receiver_hz(receiver, np.asarray([0.0, 0.0, -4.0]))

    assert receiver["Hz"] == -4.0
    assert receiver["dBzdt"] == 3.0


def test_faraday_receiver_hz_update_uses_backward_euler_dbdt():
    sp = _load_pipeline_module()

    updated = sp._advance_faraday_receiver_hz(
        previous_hz=2.0,
        dbzdt_new=4.0e-6,
        dt=0.25,
        mu=2.0e-6,
    )

    assert updated == 2.5


def test_faraday_integrated_receiver_mode_outputs_hz_component():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(magnetic_receiver_mode="faraday_integrated")

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["magnetic_receiver_mode"] == "faraday_integrated"
    assert sp._forward_components(config) == ["Ex", "Ey", "Hz", "dBzdt"]


def test_cole_cole_debye_fit_preserves_dc_conductivity():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=10,
        cole_f_min=0.001,
        cole_f_max=10000.0,
        cole_n_freq=81,
    )

    fit = sp.fit_cole_cole_to_debye(config)

    sigma_dc = fit.sigma_infinity - sum(term.delta_sigma for term in fit.terms)
    np.testing.assert_allclose(sigma_dc, 1.0 / config.cole_rho0, rtol=1.0e-8, atol=1.0e-12)


def test_song_prony_fit_passes_one_percent_material_gate():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=0.01,
    )

    fit = sp.fit_cole_cole_to_debye(config)

    assert fit.relative_l2 <= 0.01
    assert fit.sigma_infinity == pytest.approx(1.0 / 70.0)
    assert sum(term.delta_sigma for term in fit.terms) == pytest.approx(1.0 / 70.0 - 0.01)


def test_cole_cole_debye_fit_rejects_material_error_above_tolerance():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=1.0e-12,
    )

    with pytest.raises(ValueError) as exc_info:
        sp.fit_cole_cole_to_debye(config)

    assert str(exc_info.value).startswith("Cole-Cole Debye fit relative L2 ")
    assert " exceeds tolerance " in str(exc_info.value)


def test_cole_cole_debye_fit_accepts_material_error_equal_to_tolerance():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=1.0,
    )
    measured = sp.fit_cole_cole_to_debye(config).relative_l2
    config.cole_fit_tolerance = measured

    fit = sp.fit_cole_cole_to_debye(config)

    assert fit.relative_l2 == measured


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.01])
def test_cole_cole_debye_fit_rejects_invalid_tolerance_before_scipy(monkeypatch, value):
    sp = _load_pipeline_module()
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "lsq_linear",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SciPy fit must not run")),
    )
    config = sp.PipelineConfig(cole_fit_tolerance=value)

    with pytest.raises(ValueError, match="^cole_fit_tolerance must be finite and positive$"):
        sp.fit_cole_cole_to_debye(config)


def test_exact_empymod_material_uses_cole_cole_not_debye_fit(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(300.0,),
        layer_resistivities=(100.0, 100.0),
        cole_layer_top=0.0,
        cole_layer_bottom=300.0,
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
    )
    monkeypatch.setattr(
        sp,
        "fit_cole_cole_to_debye",
        lambda _config: (_ for _ in ()).throw(AssertionError("exact reference must not use the Debye fit")),
    )

    depth, material = sp._exact_cole_cole_empymod_material(config)
    frequencies = np.array([1.0e-3, 1.0, 1.0e4])
    eta_h, eta_v = material["func_eta"](None, {"freq": frequencies})
    expected = sp.cole_cole_complex_conductivity(frequencies, 100.0, 0.3, 1.0, 0.3)

    assert depth == pytest.approx([0.0, 300.0])
    np.testing.assert_allclose(eta_h[:, 1], expected)
    np.testing.assert_allclose(eta_v[:, 1], expected)
    np.testing.assert_allclose(eta_h[:, 0], 1.0 / config.rho_air)
    np.testing.assert_allclose(eta_h[:, 2], 0.01)
    assert eta_h is not eta_v
    assert not np.shares_memory(eta_h, eta_v)


def test_exact_empymod_material_changes_only_overlapping_earth_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(300.0, 600.0),
        layer_resistivities=(100.0, 200.0, 300.0),
        cole_layer_top=300.0,
        cole_layer_bottom=600.0,
        cole_rho0=200.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
    )

    _depth, material = sp._exact_cole_cole_empymod_material(config)
    frequencies = np.array([0.1, 10.0])
    eta_h, eta_v = material["func_eta"](None, {"freq": frequencies})
    expected = sp.cole_cole_complex_conductivity(frequencies, 200.0, 0.3, 1.0, 0.3)

    np.testing.assert_allclose(eta_h[:, 0], 1.0 / config.rho_air)
    np.testing.assert_allclose(eta_h[:, 1], 1.0 / 100.0)
    np.testing.assert_allclose(eta_h[:, 2], expected)
    np.testing.assert_allclose(eta_h[:, 3], 1.0 / 300.0)
    np.testing.assert_allclose(eta_v, eta_h)
    eta_h[0, 2] = -1.0
    assert eta_v[0, 2] != -1.0


def test_exact_empymod_material_splits_internal_cole_cole_window():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(300.0,),
        layer_resistivities=(100.0, 100.0),
        cole_layer_top=100.0,
        cole_layer_bottom=200.0,
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
    )

    depth, material = sp._exact_cole_cole_empymod_material(config)
    frequencies = np.array([0.1, 10.0])
    eta_h, eta_v = material["func_eta"](None, {"freq": frequencies})
    expected = sp.cole_cole_complex_conductivity(frequencies, 100.0, 0.3, 1.0, 0.3)

    assert depth == pytest.approx([0.0, 100.0, 200.0, 300.0])
    assert material["res"] == pytest.approx([config.rho_air, 100.0, 100.0, 100.0, 100.0])
    np.testing.assert_allclose(eta_h[:, 0], 1.0 / config.rho_air)
    np.testing.assert_allclose(eta_h[:, 1], 0.01)
    np.testing.assert_allclose(eta_h[:, 2], expected)
    np.testing.assert_allclose(eta_h[:, 3], 0.01)
    np.testing.assert_allclose(eta_h[:, 4], 0.01)
    np.testing.assert_allclose(eta_v, eta_h)


def test_debye_empymod_material_splits_internal_cole_cole_window():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(300.0,),
        layer_resistivities=(100.0, 100.0),
        cole_layer_top=100.0,
        cole_layer_bottom=200.0,
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=0.01,
    )

    depth, material = sp._debye_cole_cole_empymod_material(config)
    frequencies = np.array([0.1, 10.0])
    eta_h, eta_v = material["func_eta"](None, {"freq": frequencies})

    assert depth == pytest.approx([0.0, 100.0, 200.0, 300.0])
    assert material["res"] == pytest.approx([config.rho_air, 100.0, 70.0, 100.0, 100.0])
    np.testing.assert_allclose(eta_h[:, 0], 1.0 / config.rho_air)
    np.testing.assert_allclose(eta_h[:, 1], 0.01)
    assert not np.allclose(eta_h[:, 2], 0.01)
    np.testing.assert_allclose(eta_h[:, 3], 0.01)
    np.testing.assert_allclose(eta_h[:, 4], 0.01)
    np.testing.assert_allclose(eta_v, eta_h)


def test_empymod_cole_layer_split_preserves_existing_boundaries_and_infinite_bottom():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_layer_top=300.0, cole_layer_bottom=float("inf"))

    depth, res = sp._split_empymod_cole_cole_layers(
        [0.0, 300.0],
        [config.rho_air, 100.0, 200.0],
        config,
    )

    assert depth == pytest.approx([0.0, 300.0])
    assert res == pytest.approx([config.rho_air, 100.0, 200.0])


def test_exact_empymod_material_rejects_interval_without_earth_overlap():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_layer_top=-300.0, cole_layer_bottom=-100.0)

    with pytest.raises(RuntimeError, match="^no empymod layer overlaps the Cole-Cole interval$"):
        sp._exact_cole_cole_empymod_material(config)


def test_exact_empymod_material_rejects_interval_below_finite_fem_earth():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        earth_depth=1000.0,
        cole_layer_top=1100.0,
        cole_layer_bottom=1200.0,
    )

    with pytest.raises(RuntimeError, match="^no empymod layer overlaps the Cole-Cole interval$"):
        sp._exact_cole_cole_empymod_material(config)


def test_empymod_reference_has_distinct_exact_and_debye_modes(monkeypatch):
    sp = _load_pipeline_module()
    seen_materials = []

    def bipole(**kwargs):
        seen_materials.append(kwargs["res"])
        return np.zeros_like(np.asarray(kwargs["freqtime"], dtype=float))

    monkeypatch.setitem(sys.modules, "empymod", SimpleNamespace(bipole=bipole))
    config = sp.PipelineConfig(
        layer_depths=(300.0,),
        layer_resistivities=(100.0, 100.0),
        cole_layer_top=0.0,
        cole_layer_bottom=300.0,
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=0.01,
        ramp_off_time=0.0,
    )

    exact = sp.get_empymod_reference(np.array([1.0e-4, 1.0e-3]), config, mode="cole-cole-exact")
    exact_materials = list(seen_materials)
    seen_materials.clear()
    debye = sp.get_empymod_reference(np.array([1.0e-4, 1.0e-3]), config, mode="cole-cole-debye")

    assert exact["data"].shape == debye["data"].shape == (2, 4)
    assert exact["components"] == debye["components"] == ["Ex", "Ey", "Hz", "dBzdt"]
    assert exact["reference_mode"] == "cole-cole-exact"
    assert debye["reference_mode"] == "cole-cole-debye"
    assert exact_materials and seen_materials
    assert all(material["res"][1] == pytest.approx(100.0) for material in exact_materials)
    assert all(material["res"][1] == pytest.approx(70.0) for material in seen_materials)


def test_exact_empymod_reference_never_calls_debye_fit(monkeypatch):
    sp = _load_pipeline_module()

    def bipole(**kwargs):
        return np.zeros_like(np.asarray(kwargs["freqtime"], dtype=float))

    monkeypatch.setitem(sys.modules, "empymod", SimpleNamespace(bipole=bipole))
    monkeypatch.setattr(
        sp,
        "fit_cole_cole_to_debye",
        lambda _config: (_ for _ in ()).throw(AssertionError("exact reference called Debye fit")),
    )
    config = sp.PipelineConfig(ramp_off_time=0.0)

    result = sp.get_empymod_reference(np.array([1.0e-4, 1.0e-3]), config, mode="cole-cole-exact")

    assert result["data"].shape == (2, 4)
    assert result["components"] == ["Ex", "Ey", "Hz", "dBzdt"]


def test_debye_empymod_reference_obeys_material_fit_gate(monkeypatch):
    sp = _load_pipeline_module()
    monkeypatch.setitem(
        sys.modules,
        "empymod",
        SimpleNamespace(bipole=lambda **kwargs: np.zeros_like(np.asarray(kwargs["freqtime"], dtype=float))),
    )
    config = sp.PipelineConfig(
        cole_rho0=100.0,
        cole_m=0.3,
        cole_tau=1.0,
        cole_c=0.3,
        cole_n_terms=16,
        cole_f_min=1.0e-3,
        cole_f_max=1.0e4,
        cole_n_freq=81,
        cole_fit_tolerance=1.0e-12,
        ramp_off_time=0.0,
    )

    with pytest.raises(ValueError, match="^Cole-Cole Debye fit relative L2 "):
        sp.get_empymod_reference(np.array([1.0e-4, 1.0e-3]), config, mode="cole-cole-debye")


@pytest.mark.parametrize("mode", ["cole-cole", "unknown"])
def test_empymod_reference_rejects_ambiguous_or_unknown_modes(monkeypatch, mode):
    sp = _load_pipeline_module()
    monkeypatch.setitem(sys.modules, "empymod", SimpleNamespace(bipole=lambda **_kwargs: np.zeros(1)))

    with pytest.raises(ValueError, match="reference mode|migrat"):
        sp.get_empymod_reference(np.array([1.0e-4]), sp.PipelineConfig(ramp_off_time=0.0), mode=mode)
