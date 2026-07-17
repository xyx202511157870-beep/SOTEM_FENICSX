from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_selected_magnetic_outputs_do_not_modify_electric_component():
    sp = _load_pipeline_module()
    methods = {
        "Ex": 3.0,
        "dBzdt_curl": 1.0,
        "dBzdt_biot_rate": 2.0,
        "dBzdt_faraday_loop": 4.0,
        "Hz_biot_center": 5.0,
        "Hz_biot_tetra4": 6.0,
    }

    selected = sp._select_magnetic_receiver_outputs(
        methods,
        magnetic_dbdt_mode="faraday_loop",
        biot_current_integration="tetra4",
    )

    assert selected == {"Ex": 3.0, "dBzdt": 4.0, "Hz": 6.0}


def test_selected_magnetic_outputs_name_missing_diagnostic():
    import pytest

    sp = _load_pipeline_module()
    with pytest.raises(KeyError, match="dBzdt_faraday_loop"):
        sp._select_magnetic_receiver_outputs(
            {"Ex": 3.0, "Hz_biot_center": 5.0},
            magnetic_dbdt_mode="faraday_loop",
            biot_current_integration="cell_center",
        )


def test_evaluate_magnetic_receiver_methods_records_named_methods(monkeypatch):
    sp = _load_pipeline_module()
    calls = []
    config = sp.PipelineConfig(
        receiver=(1.0, 2.0, 3.0),
        faraday_loop_radius=1.5,
        faraday_loop_quadrature_points=16,
    )

    monkeypatch.setattr(
        sp,
        "evaluate_receivers",
        lambda *_args: {"Ex": 7.0, "dBzdt": 11.0},
    )
    monkeypatch.setattr(
        sp,
        "evaluate_faraday_loop_field",
        lambda *_args, **kwargs: (
            13.0,
            {"method": "faraday_loop", "radius_m": kwargs["radius"]},
        ),
        raising=False,
    )

    def fake_biot(*_args, integration, return_audit=False, **_kwargs):
        calls.append((integration, return_audit))
        value = np.asarray([0.0, 0.0, 17.0 if integration == "cell_center" else 19.0])
        return (value, {"method": "biot_tetra4"}) if return_audit else value

    monkeypatch.setattr(sp, "_biot_savart_total_h_at_receiver", fake_biot)
    methods = sp.evaluate_magnetic_receiver_methods(
        object(),
        object(),
        object(),
        {},
        config,
        source_current=2.0,
        config=config,
        h_new=np.asarray([0.0, 0.0, 5.0]),
        h_old=np.asarray([0.0, 0.0, 3.0]),
        dt=0.5,
    )

    assert methods["dBzdt_curl"] == 11.0
    assert methods["dBzdt_faraday_loop"] == 13.0
    assert methods["Hz_biot_center"] == 17.0
    assert methods["Hz_biot_tetra4"] == 19.0
    assert methods["dBzdt_biot_rate"] == pytest.approx(1.2566370614359173e-6 * 4.0)
    assert methods["faraday_audit"]["radius_m"] == 1.5
    assert methods["biot_tetra4_audit"]["method"] == "biot_tetra4"
    assert calls == [("cell_center", False), ("tetra4", True)]


def test_evaluate_magnetic_receiver_methods_reuses_selected_biot_field(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unrequested magnetic diagnostic was recomputed")

    monkeypatch.setattr(sp, "evaluate_receivers", unexpected)
    monkeypatch.setattr(sp, "evaluate_faraday_loop_field", unexpected)
    monkeypatch.setattr(sp, "_biot_savart_total_h_at_receiver", unexpected)

    methods = sp.evaluate_magnetic_receiver_methods(
        object(),
        object(),
        object(),
        {},
        config,
        source_current=0.0,
        config=config,
        h_new=np.asarray([0.0, 0.0, 5.0]),
        h_old=np.asarray([0.0, 0.0, 3.0]),
        dt=0.5,
        diagnostic_methods=(),
        selected_integration="tetra4",
    )

    assert methods == {
        "Hz_biot_tetra4": 5.0,
        "dBzdt_biot_rate": pytest.approx(1.2566370614359173e-6 * 4.0),
    }


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


def test_primary_secondary_biot_rate_keeps_core_components_to_supported_fields():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        source_term_mode="primary_secondary",
        magnetic_receiver_mode="biot_ohmic",
        magnetic_dbdt_mode="biot_rate",
    )

    assert sp._forward_components(config) == ["Ex", "Ey", "dBzdt"]


def test_primary_secondary_ampere_rate_keeps_core_components_to_supported_fields():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        source_term_mode="primary_secondary",
        magnetic_dbdt_mode="ampere_rate",
    )

    assert sp._forward_components(config) == ["Ex", "Ey", "dBzdt"]


def test_biot_savart_function_current_h_at_receiver_integrates_cell_current():
    import pytest

    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")

    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        receiver=(0.5, 1.5, 0.5),
        receiver_type="point",
    )
    spaces = sp.build_function_spaces(msh, config)
    current = fem.Function(spaces["V"], name="test_current_density")
    current.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])]))
    current.x.scatter_forward()

    h = sp._biot_savart_function_current_h_at_receiver(current, msh, config)

    assert h.shape == (3,)
    assert np.all(np.isfinite(h))
    assert h[2] > 0.0


def test_biot_savart_cell_center_path_matches_function_current_path():
    import pytest

    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")

    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    config = sp.PipelineConfig(
        receiver=(0.5, 1.5, 0.5),
        receiver_type="point",
    )
    spaces = sp.build_function_spaces(msh, config)
    current = fem.Function(spaces["V"], name="test_current_density")
    current.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])]))
    current.x.scatter_forward()

    full = sp._biot_savart_function_current_h_at_receiver(current, msh, config)
    fast = sp._biot_savart_function_current_h_at_receiver(
        current,
        msh,
        config,
        integration="cell_center",
    )

    assert fast.shape == (3,)
    assert np.all(np.isfinite(fast))
    np.testing.assert_allclose(fast[2], full[2], rtol=5.0e-3, atol=1.0e-12)


def test_biot_total_tetra4_path_returns_finite_audit():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")

    from dolfinx import fem, mesh
    from mpi4py import MPI
    from types import SimpleNamespace

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(receiver=(0.5, 1.5, 0.5), receiver_type="point")
    spaces = sp.build_function_spaces(msh, config)
    electric = fem.Function(spaces["V"], name="constant_electric")
    electric.interpolate(
        lambda x: np.vstack(
            [np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])]
        )
    )
    electric.x.scatter_forward()
    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    sigma = SimpleNamespace(x=SimpleNamespace(array=np.ones(n_cells)))

    value, audit = sp._biot_savart_total_h_at_receiver(
        electric,
        msh,
        {"sigma": sigma},
        config,
        0.0,
        integration="tetra4",
        return_audit=True,
    )

    assert value.shape == (3,)
    assert np.all(np.isfinite(value))
    assert value[2] > 0.0
    assert audit["method"] == "biot_tetra4"
    assert audit["samples"][0]["sample_count"] == 4 * n_cells


def test_ampere_h_recovery_zero_current_returns_zero_receiver_h():
    import pytest

    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")

    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(receiver=(0.25, 0.25, 0.25), outer_boundary_mode="natural")
    spaces = sp.build_function_spaces(msh, config)
    zero_current = fem.Function(spaces["V"], name="zero_current")
    zero_current.x.array[:] = 0.0
    zero_current.x.scatter_forward()

    solver = sp._make_ampere_h_recovery_solver(msh, spaces, config)
    h_new = solver(zero_current)
    h_old = solver(zero_current)
    receiver_h = sp._evaluate_ampere_h_receiver(h_new, h_old, 1.0, msh, config)

    np.testing.assert_allclose(receiver_h["H"], np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(receiver_h["dBdt"], np.zeros(3), atol=1.0e-12)


def test_tetrahedron_biot_quadrature_weights_sum_to_volume_and_points_are_internal():
    sp = _load_pipeline_module()
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    volume = 1.0 / 6.0

    points, weights = sp._tetrahedron_quadrature_points_weights(coords, volume)

    assert points.shape == (4, 3)
    assert weights.shape == (4,)
    np.testing.assert_allclose(np.sum(weights), volume)
    assert np.all(points >= 0.0)
    assert np.all(np.sum(points, axis=1) < 1.0)


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
