import builtins
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

import atem3d.sotem_simpeg_adapter as adapter
from atem3d.config import build_simulation
from atem3d.sotem_benchmark import load_benchmark_case
from atem3d.sotem_simpeg_adapter import (
    build_benchmark_config,
    build_internal_time_steps,
    paired_model_dicts,
    run_simpeg_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def song_case():
    return load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")


@pytest.fixture
def lei_case():
    return load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")


def test_internal_steps_hit_every_output_and_apply_substeps():
    outputs = np.array([1.0e-5, 1.0e-4, 1.0e-3])

    steps, indices = build_internal_time_steps(outputs, substeps=4)

    np.testing.assert_allclose(np.cumsum(steps)[indices], outputs)
    assert len(steps) == 12


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [[1.0e-5]],
        [0.0, 1.0e-4],
        [-1.0e-5, 1.0e-4],
        [1.0e-5, 1.0e-5],
        [1.0e-4, 1.0e-5],
        [1.0e-5, np.nan],
        [1.0e-5, np.inf],
    ],
)
def test_internal_steps_reject_invalid_output_grids(outputs):
    with pytest.raises(ValueError, match="strictly increasing"):
        build_internal_time_steps(outputs, substeps=1)


@pytest.mark.parametrize("substeps", [True, False, 0, -1, 1.0, "1"])
def test_internal_steps_require_an_exact_positive_integer(substeps):
    with pytest.raises(ValueError, match="excluding bool"):
        build_internal_time_steps([1.0e-5], substeps=substeps)


def test_internal_step_results_are_independent_read_only_arrays():
    outputs = np.array([1.0e-5, 1.0e-4])
    steps, indices = build_internal_time_steps(outputs, substeps=2)
    outputs[0] = 9.0

    np.testing.assert_allclose(np.cumsum(steps)[indices], [1.0e-5, 1.0e-4])
    with pytest.raises(ValueError):
        steps[0] = 1.0
    with pytest.raises(ValueError):
        indices[0] = 1


def test_song_noip_and_ip_share_mesh_and_dc_conductivity(song_case):
    noip, ip = paired_model_dicts(
        song_case,
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )

    assert noip["mesh"] == ip["mesh"]
    assert noip["model"]["layers"][1]["sigma_infinity"] == pytest.approx(0.01)
    material = ip["adapter_metadata"]["material_fit"]
    assert material["rho0_ohm_m"] == pytest.approx(100.0)
    assert material["m"] == pytest.approx(0.3)
    assert material["tau_s"] == pytest.approx(1.0)
    assert material["c"] == pytest.approx(0.3)
    assert ip["model"]["layers"][1]["sigma_infinity"] == pytest.approx(1.0 / 70.0)
    assert sum(
        term["delta_sigma"] for term in ip["model"]["layers"][1]["debye_terms"]
    ) == pytest.approx(1.0 / 70.0 - 0.01, abs=1.0e-14)


def test_song_pair_is_structurally_equal_except_model_and_material_metadata(song_case):
    noip, ip = paired_model_dicts(
        song_case,
        spatial_level="S1",
        boundary_level="B1",
        substeps=3,
    )

    for field in ("mesh", "source", "receivers", "time_steps", "boundary"):
        assert noip[field] == ip[field]
        assert noip[field] is not ip[field]
    assert noip["adapter_metadata"]["output_indices"] == ip["adapter_metadata"]["output_indices"]
    assert noip["adapter_metadata"]["time_hash"] == ip["adapter_metadata"]["time_hash"]

    ip["mesh"]["hx"][0] = -1.0
    ip["source"]["start"][0] = 123.0
    ip["receivers"][0]["location"][0] = 123.0
    ip["time_steps"][0] = 123.0
    assert noip["mesh"]["hx"][0] > 0.0
    assert noip["source"]["start"] == [-500.0, 0.0, -0.1]
    assert noip["receivers"][0]["location"] == [0.0, -500.0, -0.1]
    assert noip["time_steps"][0] != 123.0


def test_public_z_down_geometry_is_mapped_once_to_internal_z_up(song_case):
    config = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )

    assert config["coordinate_system"] == "z_up"
    assert config["model"]["coordinate_system"] == "z_up"
    assert config["initial_magnetic_field"] == "ampere"
    assert config["solver"] == {
        "type": "petsc_ams",
        "tolerance": 1.0e-8,
        "internal_tolerance": 1.0e-11,
        "maxiter": 2000,
        "preconditioner": "hypre_ams",
        "ksp_type": "gmres",
        "residual_replacement_steps": 2,
    }
    assert config["initialization_solver"] == {
        "type": "petsc_hypre",
        "tolerance": 1.0e-8,
        "internal_tolerance": 1.0e-11,
        "maxiter": 2000,
        "residual_replacement_steps": 2,
        "dc_ksp_type": "cg",
        "dc_preconditioner": "hypre_boomeramg",
        "magnetic_ksp_type": "gmres",
        "magnetic_preconditioner": "hypre_ams",
    }
    assert config["adapter_metadata"]["transient_solver"] == "petsc_gmres_hypre_ams"
    assert config["adapter_metadata"]["initial_magnetic_field"] == "ampere"
    assert config["adapter_metadata"]["initialization_solver"] == {
        "dc_electric": "petsc_cg_hypre_boomeramg",
        "ampere_magnetic": "petsc_gmres_hypre_ams",
    }
    assert config["source"] == {
        "start": [song_case.source_start_down[0], song_case.source_start_down[1], -0.1],
        "end": [song_case.source_end_down[0], song_case.source_end_down[1], -0.1],
        "current": song_case.current_a,
        "waveform": {"type": "step_off", "off_time": 0.0},
    }
    assert [receiver["component"] for receiver in config["receivers"]] == [
        "Ex",
        "Ey",
        "Hz",
        "dBzdt",
    ]
    assert all(
        receiver["location"]
        == [song_case.receiver_down[0], song_case.receiver_down[1], -0.1]
        for receiver in config["receivers"]
    )
    layers = config["model"]["layers"]
    assert layers[0]["top"] == float("inf") and layers[0]["bottom"] == 0.0
    assert layers[1]["top"] == 0.0 and layers[1]["bottom"] == -300.0
    assert layers[2]["top"] == -300.0 and layers[2]["bottom"] == float("-inf")
    transform = config["adapter_metadata"]["coordinate_transform"]
    assert transform["public_coordinates"] == "z_down"
    assert transform["internal_coordinates"] == "z_up"
    assert transform["position_mapping"] == "(x, y, z_up) = (x, y, -z_down)"
    assert transform["output_component_signs"] == {
        "Ex": 1.0,
        "Ey": 1.0,
        "Hz": 1.0,
        "dBzdt": 1.0,
    }
    assert transform["Hz_vector_type"] == "axial"
    assert config["boundary"] == {"kind": "none", "thickness_cells": 0}


def _nodes(mesh, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    # Discretize constructs TensorMesh nodes by cumulatively summing the
    # origin together with widths, so this must mirror its floating arithmetic.
    return np.cumsum(
        np.r_[mesh["origin"][axis_index], np.asarray(mesh[f"h{axis}"])]
    )


def _assert_strong_weak_parity(
    actual,
    reference,
    *,
    strong_floor_fraction,
    limit,
):
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert actual.shape == reference.shape
    reference_peak = float(np.max(np.abs(reference)))
    assert reference_peak > 0.0
    strong = np.abs(reference) >= strong_floor_fraction * reference_peak
    weak = ~strong
    strong_error = float(
        np.max(np.abs(actual[strong] - reference[strong]) / np.abs(reference[strong]))
    )
    weak_error = (
        float(np.max(np.abs(actual[weak] - reference[weak]))) / reference_peak
        if np.any(weak)
        else 0.0
    )
    assert strong_error <= limit, (
        f"strong relative error {strong_error:.6e} exceeds {limit:.6e}"
    )
    assert weak_error <= limit, (
        f"weak peak-normalized error {weak_error:.6e} exceeds {limit:.6e}"
    )
    return {
        "strong_relative_error": strong_error,
        "weak_peak_normalized_error": weak_error,
    }


@pytest.mark.parametrize("spatial_level", ["S0", "S1", "S2"])
@pytest.mark.parametrize("boundary_level", ["B0", "B1", "B2"])
def test_mesh_has_positive_widths_exact_interfaces_and_contains_anchors(
    song_case,
    spatial_level,
    boundary_level,
):
    config = build_benchmark_config(
        song_case,
        variant="noip",
        spatial_level=spatial_level,
        boundary_level=boundary_level,
        substeps=1,
    )
    mesh = config["mesh"]

    for axis in "xyz":
        assert np.all(np.asarray(mesh[f"h{axis}"]) > 0.0)
        nodes = _nodes(mesh, axis)
        assert np.all(np.diff(nodes) > 0.0)
    z_nodes = _nodes(mesh, "z")
    assert 0.0 in z_nodes
    assert -300.0 in z_nodes

    bounds = mesh["metadata"]["bounds_m"]
    for point in (
        song_case.source_start_down,
        song_case.source_end_down,
        song_case.receiver_down,
    ):
        internal_point = (point[0], point[1], -point[2])
        for coordinate, axis in zip(internal_point, "xyz"):
            assert bounds[axis][0] < coordinate < bounds[axis][1]


def test_padding_widths_grow_monotonically_away_from_horizontal_anchors(song_case):
    mesh = build_benchmark_config(
        song_case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )["mesh"]
    x_nodes = _nodes(mesh, "x")
    hx = np.asarray(mesh["hx"])
    left_source_node = int(np.flatnonzero(np.isclose(x_nodes, -500.0, atol=1.0e-10))[0])
    right_source_node = int(np.flatnonzero(np.isclose(x_nodes, 500.0, atol=1.0e-10))[0])

    assert np.all(np.diff(hx[:left_source_node]) <= 1.0e-10)
    assert np.all(np.diff(hx[right_source_node:]) >= -1.0e-10)
    assert np.max(hx[1:] / hx[:-1]) <= 1.4 + 1.0e-12
    assert np.max(hx[:-1] / hx[1:]) <= 1.4 + 1.0e-12


def test_spatial_and_boundary_levels_have_ordered_counts_extents_and_unique_hashes(song_case):
    spatial_configs = [
        build_benchmark_config(
            song_case,
            variant="noip",
            spatial_level=level,
            boundary_level="B0",
            substeps=1,
        )
        for level in ("S0", "S1", "S2")
    ]
    boundary_configs = [
        build_benchmark_config(
            song_case,
            variant="noip",
            spatial_level="S0",
            boundary_level=level,
            substeps=1,
        )
        for level in ("B0", "B1", "B2")
    ]

    spatial_counts = [config["mesh"]["metadata"]["n_cells"] for config in spatial_configs]
    boundary_counts = [config["mesh"]["metadata"]["n_cells"] for config in boundary_configs]
    assert spatial_counts == sorted(spatial_counts) and len(set(spatial_counts)) == 3
    assert boundary_counts == sorted(boundary_counts) and len(set(boundary_counts)) == 3
    assert [
        config["mesh"]["metadata"]["nominal_far_extent_m"]
        for config in boundary_configs
    ] == [25_000.0, 50_000.0, 100_000.0]
    assert len({config["mesh_hash"] for config in spatial_configs + boundary_configs}) == 5
    assert spatial_counts[0] == 93_600
    assert max(spatial_counts + boundary_counts) < 300_000


def test_mesh_and_time_hashes_are_deterministic_and_time_hash_tracks_substeps(song_case):
    first = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S2",
        boundary_level="B2",
        substeps=2,
    )
    second = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S2",
        boundary_level="B2",
        substeps=2,
    )
    changed_time = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S2",
        boundary_level="B2",
        substeps=3,
    )

    assert first["mesh_hash"] == second["mesh_hash"] == changed_time["mesh_hash"]
    assert first["adapter_metadata"]["time_hash"] == second["adapter_metadata"]["time_hash"]
    assert first["adapter_metadata"]["time_hash"] != changed_time["adapter_metadata"]["time_hash"]


def test_song_material_fit_passes_frequency_accuracy_dc_and_positivity_gates(song_case):
    config = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    fit = config["adapter_metadata"]["material_fit"]
    terms = config["model"]["layers"][1]["debye_terms"]

    assert fit["fit_frequency_min_hz"] == pytest.approx(1.0e-3)
    assert fit["fit_frequency_max_hz"] == pytest.approx(1.0e4)
    assert fit["fit_frequency_count"] == 81
    assert fit["fit_term_count"] == 16
    assert fit["relative_l2"] <= 0.01
    assert fit["sigma_dc"] == pytest.approx(0.01)
    assert fit["sigma_infinity"] == pytest.approx(1.0 / 70.0)
    assert abs(fit["dc_residual"]) <= 1.0e-14
    assert fit["material_gate_pass"] is True
    assert len(terms) == 16
    assert all(term["tau"] > 0.0 and term["delta_sigma"] > 0.0 for term in terms)
    assert fit["debye_terms"] == terms


def test_zero_chargeability_ip_degenerates_to_noip_system(song_case):
    polarization = dict(song_case.polarization)
    polarization["m"] = 0.0
    zero_m_case = replace(song_case, polarization=MappingProxyType(polarization))

    noip, ip = paired_model_dicts(
        zero_m_case,
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )

    assert noip["model"] == ip["model"]
    assert ip["adapter_metadata"]["material_fit"]["fit_term_count"] == 0
    assert ip["adapter_metadata"]["material_fit"]["material_gate_pass"] is True


def test_ip_variant_is_rejected_for_lei_and_pair_requires_polarization(lei_case):
    with pytest.raises(ValueError, match="polarizable"):
        build_benchmark_config(
            lei_case,
            variant="ip",
            spatial_level="S0",
            boundary_level="B0",
            substeps=1,
        )
    with pytest.raises(ValueError, match="polarizable"):
        paired_model_dicts(
            lei_case,
            spatial_level="S0",
            boundary_level="B0",
            substeps=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("variant", "IP"), ("spatial_level", "S3"), ("boundary_level", "B3")],
)
def test_config_rejects_unknown_variant_or_levels(lei_case, field, value):
    kwargs = {
        "variant": "noip",
        "spatial_level": "S0",
        "boundary_level": "B0",
        "substeps": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        build_benchmark_config(lei_case, **kwargs)


def _fake_solver_for_config(config, *, nonfinite=False):
    steps = np.asarray(config["time_steps"], dtype=float)
    times = np.r_[0.0, np.cumsum(steps)]
    data = np.arange(times.size * 4, dtype=float).reshape(times.size, 4)
    if nonfinite:
        data[-1, -1] = np.nan
    metadata = config["mesh"]["metadata"]
    result = SimpleNamespace(times=times, data=data, memories=[])
    mesh = SimpleNamespace(n_cells=metadata["n_cells"], n_edges=metadata["n_edges"])
    diagnostics = [
        {
            "step_index": index,
            "dt_s": float(dt),
            "solver": "petsc_ksp_hypre_ams",
            "solve_mode": "petsc_ksp",
            "ksp_type": "gmres",
            "pc_type": "hypre_ams",
            "backend_reason": 2,
            "backend_reported_converged": True,
            "backend_iterations": 3,
            "external_true_relative_residual": 0.0,
            "external_tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "residual_replacement_steps": 0,
        }
        for index, dt in enumerate(config["time_steps"])
    ]
    initialization_diagnostics = [
        {
            "phase": phase,
            "solver": solver,
            "solve_mode": "petsc_ksp",
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "backend_reason": 2,
            "backend_reported_converged": True,
            "backend_iterations": 3,
            "external_true_relative_residual": 1.0e-12,
            "external_tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "residual_replacement_steps": 0,
            "balance_name": balance_name,
            "balance_relative_residual": 1.0e-12,
            "balance_tolerance": 1.0e-8,
        }
        for phase, solver, ksp_type, pc_type, balance_name in (
            (
                "dc_electric",
                "petsc_ksp_hypre_boomeramg",
                "cg",
                "hypre_boomeramg",
                "discrete_current_divergence",
            ),
            (
                "ampere_magnetic",
                "petsc_ksp_hypre_ams",
                "gmres",
                "hypre_ams",
                "static_ampere",
            ),
        )
    ]
    initialization_diagnostics[1].update(
        gauge_stabilization_weight=5.0e14,
        stiffness_operator_max_abs=2.0e12,
        gauge_operator_max_abs=4.0e-3,
    )
    return SimpleNamespace(
        mesh=mesh,
        run_data_only=lambda: result,
        linear_solver_diagnostics=diagnostics,
        initialization_solver_diagnostics=initialization_diagnostics,
    ), data


def test_run_selects_only_canonical_outputs_and_reports_honest_metadata(monkeypatch, song_case):
    captured = {}

    def fake_build(config):
        captured["config"] = config
        simulation, all_data = _fake_solver_for_config(config)
        captured["all_data"] = all_data
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)
    result = run_simpeg_benchmark(
        song_case,
        variant="ip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=4,
    )

    indices = np.asarray(captured["config"]["adapter_metadata"]["output_indices"])
    np.testing.assert_array_equal(result["data"], captured["all_data"][1:][indices])
    np.testing.assert_allclose(result["times"], song_case.observation_times)
    assert result["components"] == ["Ex", "Ey", "Hz", "dBzdt"]
    assert result["solver_id"] == "atem3d_simpeg_discretize_debye"
    assert result["variant"] == "ip"
    assert result["mesh_hash"] == result["mesh_stats"]["mesh_hash"]
    assert result["material_fit"]["material_gate_pass"] is True
    assert result["coordinate_system"] == "z_down"
    assert result["coordinate_transform"]["output_component_signs"]["Hz"] == 1.0
    assert [
        item["phase"] for item in result["initialization_solver_diagnostics"]
    ] == ["dc_electric", "ampere_magnetic"]


def test_run_rejects_missing_per_step_external_solver_diagnostics(monkeypatch, lei_case):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        simulation.linear_solver_diagnostics.pop()
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="linear solver diagnostics"):
        run_simpeg_benchmark(
            lei_case,
            variant="noip",
            spatial_level="S0",
            boundary_level="B0",
            substeps=1,
        )


def test_run_accepts_exact_zero_transient_diagnostics(monkeypatch, lei_case):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        for record in simulation.linear_solver_diagnostics:
            record.update(
                solve_mode="exact_zero_rhs",
                backend_reason=0,
                backend_reported_converged=False,
                backend_iterations=0,
                external_true_relative_residual=0.0,
                residual_replacement_steps=0,
            )
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    result = run_simpeg_benchmark(lei_case, variant="noip")

    assert result["linear_solver_diagnostics"]
    assert all(
        record["solve_mode"] == "exact_zero_rhs"
        for record in result["linear_solver_diagnostics"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("external_true_relative_residual", -1.0e-12),
        ("backend_reason", 0),
        ("backend_reported_converged", False),
    ],
)
def test_run_rejects_invalid_petsc_transient_diagnostics(
    monkeypatch,
    lei_case,
    field,
    value,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        simulation.linear_solver_diagnostics[0][field] = value
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="linear solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_reason", 1),
        ("backend_reported_converged", True),
        ("backend_iterations", 1),
        ("external_true_relative_residual", 1.0e-12),
        ("residual_replacement_steps", 1),
    ],
)
def test_run_rejects_incoherent_exact_zero_transient_diagnostics(
    monkeypatch,
    lei_case,
    field,
    value,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        for record in simulation.linear_solver_diagnostics:
            record.update(
                solve_mode="exact_zero_rhs",
                backend_reason=0,
                backend_reported_converged=False,
                backend_iterations=0,
                external_true_relative_residual=0.0,
                residual_replacement_steps=0,
            )
        simulation.linear_solver_diagnostics[0][field] = value
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="linear solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_reported_converged", False),
        ("backend_reason", -3),
        ("backend_iterations", -1),
        ("solve_mode", "unknown"),
        ("ksp_type", "wrong"),
        ("pc_type", "wrong"),
        ("balance_name", "wrong"),
        ("internal_tolerance", 2.0e-11),
        ("residual_replacement_steps", 3),
        ("external_true_relative_residual", np.nan),
        ("external_true_relative_residual", -1.0e-12),
        ("external_true_relative_residual", 1.01e-8),
        ("balance_relative_residual", np.inf),
        ("balance_relative_residual", -1.0e-12),
        ("balance_relative_residual", 1.01e-8),
    ],
)
def test_run_rejects_failed_initialization_diagnostic(
    monkeypatch,
    lei_case,
    field,
    value,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        simulation.initialization_solver_diagnostics[0][field] = value
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="initialization solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gauge_stabilization_weight", None),
        ("gauge_stabilization_weight", np.inf),
        ("gauge_stabilization_weight", -1.0),
        ("stiffness_operator_max_abs", np.nan),
        ("gauge_operator_max_abs", 0.0),
        ("gauge_stabilization_weight", 4.0e14),
    ],
)
def test_run_rejects_invalid_ampere_gauge_stabilization_diagnostic(
    monkeypatch,
    lei_case,
    field,
    value,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        simulation.initialization_solver_diagnostics[1][field] = value
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="initialization solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


def test_run_accepts_truthful_exact_zero_initialization_diagnostics(
    monkeypatch,
    lei_case,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        for record in simulation.initialization_solver_diagnostics:
            record.update(
                solve_mode="exact_zero_rhs",
                backend_reason=0,
                backend_reported_converged=False,
                backend_iterations=0,
                external_true_relative_residual=0.0,
                residual_replacement_steps=0,
                balance_relative_residual=0.0,
            )
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    result = run_simpeg_benchmark(lei_case, variant="noip")

    assert all(
        item["solve_mode"] == "exact_zero_rhs"
        for item in result["initialization_solver_diagnostics"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_reason", 1),
        ("backend_reported_converged", True),
        ("backend_iterations", 1),
        ("external_true_relative_residual", 1.0e-12),
        ("residual_replacement_steps", 1),
        ("balance_relative_residual", 1.0e-12),
    ],
)
def test_run_rejects_incoherent_exact_zero_initialization_diagnostics(
    monkeypatch,
    lei_case,
    field,
    value,
):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        for record in simulation.initialization_solver_diagnostics:
            record.update(
                solve_mode="exact_zero_rhs",
                backend_reason=0,
                backend_reported_converged=False,
                backend_iterations=0,
                external_true_relative_residual=0.0,
                residual_replacement_steps=0,
                balance_relative_residual=0.0,
            )
        simulation.initialization_solver_diagnostics[0][field] = value
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="initialization solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


def test_run_rejects_missing_initialization_phase(monkeypatch, lei_case):
    def fake_build(config):
        simulation, _data = _fake_solver_for_config(config)
        simulation.initialization_solver_diagnostics.pop()
        return simulation

    monkeypatch.setattr(adapter, "build_simulation", fake_build)

    with pytest.raises(RuntimeError, match="initialization solver diagnostics"):
        run_simpeg_benchmark(lei_case, variant="noip")


def test_resource_metadata_discloses_petsc_initialization_and_full_scale_rerun_gate(
    song_case,
):
    config = build_benchmark_config(
        song_case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )

    note = config["adapter_metadata"]["resource_note"].lower()
    assert "petsc/hypre initialization" in note
    assert "external residual" in note
    assert "prior sparse-direct oom" in note
    assert "full-scale s0" in note
    assert "not yet rerun" in note


def test_run_rejects_nonfinite_solver_output(monkeypatch, lei_case):
    monkeypatch.setattr(
        adapter,
        "build_simulation",
        lambda config: _fake_solver_for_config(config, nonfinite=True)[0],
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        run_simpeg_benchmark(lei_case, variant="noip")


def test_adapter_never_imports_empymod(monkeypatch, lei_case):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "empymod" or name.startswith("empymod."):
            raise AssertionError("adapter must not import empymod")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = build_benchmark_config(
        lei_case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    monkeypatch.setattr(adapter, "build_simulation", lambda cfg: _fake_solver_for_config(cfg)[0])
    result = run_simpeg_benchmark(lei_case, variant="noip")
    assert config["adapter_metadata"]["material_fit"] is None
    assert result["material_fit"] is None


def test_generated_noip_schema_runs_the_real_solver_on_a_tiny_mesh(lei_case):
    tiny_case = replace(
        lei_case,
        source_start_down=(-0.5, 0.0, 0.1),
        source_end_down=(0.5, 0.0, 0.1),
        receiver_down=(0.2, 0.2, 0.1),
        observation_times=np.array([0.01]),
    )
    config = build_benchmark_config(
        tiny_case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    config["mesh"] = {
        "hx": [1.0, 1.0, 1.0],
        "hy": [1.0, 1.0, 1.0],
        "hz": [1.0, 1.0, 1.0],
        "origin": [-1.5, -1.5, -1.0],
    }
    # Keep this cross-platform schema smoke independent of the WSL-only PETSc
    # production backend.  PETSc/AMS has dedicated real-solver tests.
    config["solver"] = {"type": "direct"}
    config["initialization_solver"] = {"type": "direct"}

    result = build_simulation(config).run_data_only()

    assert result.times.tolist() == pytest.approx([0.0, 0.01])
    assert result.data.shape == (2, 4)
    assert np.all(np.isfinite(result.data))


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ("strong", "strong relative error"),
        ("weak", "weak peak-normalized error"),
    ],
)
def test_strong_weak_parity_metric_rejects_bias_above_limit(region, message):
    reference = np.array([[1.0, 0.5, 1.0e-8]])
    biased = reference.copy()
    if region == "strong":
        biased[0, :2] *= 1.0 + 2.1e-4
    else:
        biased[0, 2] += 2.1e-4

    with pytest.raises(AssertionError, match=message):
        _assert_strong_weak_parity(
            biased,
            reference,
            strong_floor_fraction=1.0e-3,
            limit=2.0e-4,
        )


def test_adapter_tiny_layered_z_up_matches_upstream_e_and_faraday_b_path(song_case):
    tiny_case = replace(
        song_case,
        source_start_down=(-0.5, 0.0, 0.1),
        source_end_down=(0.5, 0.0, 0.1),
        receiver_down=(0.2, 0.2, 0.1),
        observation_times=np.array([1.0e-5, 1.0e-2, 2.0e-2]),
    )
    config = build_benchmark_config(
        tiny_case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    mapped_song_boundary = config["model"]["layers"][1]["bottom"]
    config["model"]["layers"][1]["bottom"] = -1.0
    config["model"]["layers"][2]["top"] = -1.0
    config["mesh"] = {
        "hx": [1.0, 1.0, 1.0],
        "hy": [1.0, 1.0, 1.0],
        "hz": [1.0, 1.0, 1.0],
        "origin": [-1.5, -1.5, -2.0],
    }
    config["solver"] = {"type": "direct"}
    config["initialization_solver"] = {"type": "direct"}

    ours_simulation = build_simulation(config)
    ours = ours_simulation.run()
    mesh = ours_simulation.mesh
    source = config["source"]
    upstream_source = tdem.sources.LineCurrent(
        [],
        location=np.array([source["start"], source["end"]]),
        current=source["current"],
        waveform=tdem.sources.StepOffWaveform(off_time=0.0),
    )
    upstream = tdem.Simulation3DElectricField(
        mesh,
        survey=tdem.Survey([upstream_source]),
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
        time_steps=config["time_steps"],
    )
    fields = upstream.fields(ours_simulation.ip_model.sigma_infinity)
    upstream_e = np.column_stack(
        [fields[upstream_source, "e", index] for index in range(ours.times.size)]
    ).T
    # Upstream's grounded LineCurrent has no EB B-form initialization
    # (_getAmmr/jInitial are NotImplemented). Its E-form does expose the legal
    # face dbdt=-curl(E) path, which independently verifies our Faraday B update.
    upstream_dbdt = np.column_stack(
        [fields[upstream_source, "dbdt", index] for index in range(ours.times.size)]
    ).T
    reconstructed_b = np.empty_like(ours.b)
    reconstructed_b[0] = ours.b[0]
    for index, dt in enumerate(config["time_steps"], start=1):
        reconstructed_b[index] = reconstructed_b[index - 1] + dt * upstream_dbdt[index]

    source_vector = ours_simulation.sources[0].initial_edge_vector(mesh)
    initial_current = (
        mesh.get_edge_inner_product(ours_simulation.ip_model.low_frequency_sigma())
        @ ours.e[0]
        + source_vector
    )
    ampere_current = (
        mesh.edge_curl.T @ ours_simulation.face_mu_inverse_matrix @ ours.b[0]
    )

    assert config["coordinate_system"] == "z_up"
    assert config["source"]["start"][2] == -0.1
    assert config["model"]["layers"][1]["top"] == 0.0
    assert mapped_song_boundary == -300.0
    assert config["model"]["layers"][1]["bottom"] == -1.0
    dc_difference = ours.e[0] - upstream_e[0]
    dc_relative_l2 = float(np.linalg.norm(dc_difference) / np.linalg.norm(upstream_e[0]))
    dc_peak_normalized = float(
        np.max(np.abs(dc_difference)) / np.max(np.abs(upstream_e[0]))
    )
    assert dc_relative_l2 <= 1.0e-9
    assert dc_peak_normalized <= 1.0e-9
    # The late-time reference contains entries near its numerical floor. Test
    # strong samples relatively and weak samples against the off-time peak so
    # those floor values cannot either dominate or be hidden by a large atol.
    off_time_errors = _assert_strong_weak_parity(
        ours.e[1:],
        upstream_e[1:],
        strong_floor_fraction=1.0e-3,
        limit=2.0e-4,
    )
    assert off_time_errors["strong_relative_error"] >= 0.0
    assert off_time_errors["weak_peak_normalized_error"] >= 0.0
    np.testing.assert_allclose(
        np.diff(ours.b, axis=0) / np.asarray(config["time_steps"])[:, None],
        upstream_dbdt[1:],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(ours.b, reconstructed_b, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(ampere_current, initial_current, rtol=1.0e-8, atol=1.0e-10)
