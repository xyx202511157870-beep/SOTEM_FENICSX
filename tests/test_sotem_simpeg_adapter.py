import builtins
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

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
    assert noip["source"]["start"] == list(song_case.source_start_down)
    assert noip["receivers"][0]["location"] == list(song_case.receiver_down)
    assert noip["time_steps"][0] != 123.0


def test_canonical_source_receiver_and_boundary_are_not_shifted(song_case):
    config = build_benchmark_config(
        song_case,
        variant="ip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )

    assert config["coordinate_system"] == "depth_down"
    assert config["initial_magnetic_field"] == "ampere"
    assert config["solver"] == {
        "type": "cg",
        "tolerance": 1.0e-8,
        "maxiter": 2000,
        "preconditioner": "jacobi",
    }
    assert config["adapter_metadata"]["initial_magnetic_field"] == "ampere"
    assert config["adapter_metadata"]["initialization_solver"] == "scipy_sparse_direct"
    assert config["source"] == {
        "start": list(song_case.source_start_down),
        "end": list(song_case.source_end_down),
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
        receiver["location"] == list(song_case.receiver_down)
        for receiver in config["receivers"]
    )
    assert config["boundary"] == {"kind": "none", "thickness_cells": 0}


def _nodes(mesh, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    return mesh["origin"][axis_index] + np.r_[0.0, np.cumsum(mesh[f"h{axis}"])]


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
    assert np.any(np.isclose(z_nodes, 0.0, rtol=0.0, atol=1.0e-10))
    assert np.any(np.isclose(z_nodes, 300.0, rtol=0.0, atol=1.0e-10))

    bounds = mesh["metadata"]["bounds_m"]
    for point in (
        song_case.source_start_down,
        song_case.source_end_down,
        song_case.receiver_down,
    ):
        for coordinate, axis in zip(point, "xyz"):
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
    return SimpleNamespace(mesh=mesh, run_data_only=lambda: result), data


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

    result = build_simulation(config).run_data_only()

    assert result.times.tolist() == pytest.approx([0.0, 0.01])
    assert result.data.shape == (2, 4)
    assert np.all(np.isfinite(result.data))
