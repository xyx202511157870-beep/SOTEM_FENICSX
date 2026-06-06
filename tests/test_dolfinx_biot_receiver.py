from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
