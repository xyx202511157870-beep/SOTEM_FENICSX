from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_model_consistency_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_pipeline_execution_bootstraps_checkout_src(tmp_path):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    expected_package = (root / "src" / "atem3d").resolve()
    code = (
        "import runpy\n"
        "from pathlib import Path\n"
        f"runpy.run_path({str(module_path)!r})\n"
        "import atem3d.sotem_acceptance as acceptance\n"
        "print(Path(acceptance.__file__).resolve())\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).is_relative_to(expected_package)


def test_default_model_consistency_reports_empymod_matching_depths_and_resistivity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["reference_mode"] == "noip"
    assert diagnostics["source_depth_start"] == pytest.approx(0.1)
    assert diagnostics["source_depth_end"] == pytest.approx(0.1)
    assert diagnostics["receiver_depth"] == pytest.approx(0.1)
    assert diagnostics["rho_air"] == pytest.approx(1.0e8)
    assert diagnostics["rho_earth"] == pytest.approx(100.0)
    assert diagnostics["layer_depths"] == []
    assert diagnostics["layer_resistivities"] == [pytest.approx(100.0)]


def test_layered_model_consistency_builds_shared_fem_and_empymod_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    diagnostics = sp.validate_model_consistency(config)
    depth, res = sp._empymod_depth_res(config)

    assert diagnostics["layer_depths"] == [pytest.approx(2000.0), pytest.approx(2200.0)]
    assert diagnostics["layer_resistivities"] == [pytest.approx(200.0), pytest.approx(20.0), pytest.approx(200.0)]
    assert diagnostics["rho_earth"] == pytest.approx(200.0)
    assert diagnostics["sigma_earth"] == pytest.approx(1.0 / 200.0)
    assert depth == [pytest.approx(0.0), pytest.approx(2000.0), pytest.approx(2200.0)]
    assert res == [pytest.approx(config.rho_air), pytest.approx(200.0), pytest.approx(20.0), pytest.approx(200.0)]
    assert sp._earth_resistivity_at_depth(1999.9, config) == pytest.approx(200.0)
    assert sp._earth_resistivity_at_depth(2000.0, config) == pytest.approx(20.0)
    assert sp._earth_resistivity_at_depth(2199.9, config) == pytest.approx(20.0)
    assert sp._earth_resistivity_at_depth(2200.0, config) == pytest.approx(200.0)


def test_zhou_numerical_sublayers_preserve_one_physical_pelton_material():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 0.0, -0.1),
        source_end=(500.0, 0.0, -0.1),
        receiver=(0.0, 1000.0, -0.1),
        expected_parallel_offset=1000.0,
        layer_depths=(500.0, 505.0, 510.0, 515.0, 520.0),
        layer_resistivities=(100.0, 10.0, 10.0, 10.0, 10.0, 200.0),
        polarization="cole-cole",
        cole_layer_top=500.0,
        cole_layer_bottom=520.0,
        cole_rho0=10.0,
        cole_m=0.1,
        cole_tau=1.0,
        cole_c=0.3,
        observation_times=tuple(np.geomspace(1.0e-4, 3.0, 101)),
        canonical_surface_z=0.0,
        numerical_surface_offset=0.1,
    )

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["layer_depths"] == pytest.approx([500.0, 505.0, 510.0, 515.0, 520.0])
    assert diagnostics["cole_layer_top"] == pytest.approx(500.0)
    assert diagnostics["cole_layer_bottom"] == pytest.approx(520.0)
    assert diagnostics["sigma_dc_polarizable"] == pytest.approx(0.1)
    assert diagnostics["sigma_infinity_polarizable"] == pytest.approx(1.0 / 9.0)
    assert diagnostics["canonical_surface_z"] == pytest.approx(0.0)
    assert diagnostics["numerical_surface_offset"] == pytest.approx(0.1)
    assert diagnostics["effective_t_max"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_depths": (2000.0,), "layer_resistivities": (200.0,)},
        {"layer_depths": (2200.0, 2000.0), "layer_resistivities": (200.0, 20.0, 200.0)},
        {"layer_depths": (0.0,), "layer_resistivities": (200.0, 20.0)},
        {"layer_depths": (2000.0,), "layer_resistivities": (200.0, -20.0)},
    ],
)
def test_model_consistency_rejects_invalid_layered_models(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="layer"):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_model_consistency_rejects_cole_cole_window_spanning_multiple_resistivity_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    with pytest.raises(ValueError, match="spans multiple resistivity layers"):
        sp.validate_model_consistency(config)


def test_model_consistency_rejects_analytic_halfspace_initial_for_nonuniform_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        initial_dc_mode="analytic_halfspace",
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    with pytest.raises(ValueError, match="analytic_halfspace"):
        sp.validate_model_consistency(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rho_air", 0.0),
        ("rho_earth", -100.0),
        ("mu_r_air", 0.0),
        ("mu_r_earth", -1.0),
    ],
)
def test_model_consistency_rejects_nonpositive_material_parameters(field, value):
    sp = _load_pipeline_module()
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_start": (-50.0, 0.0, 0.0)},
        {"source_end": (50.0, 0.0, 0.1)},
        {"receiver": (500.0, 50.0, 0.0)},
    ],
)
def test_model_consistency_rejects_air_or_interface_electrodes(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="below the z=0 earth surface"):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_current", 0.0),
        ("t_min", 0.0),
        ("time_growth", 1.0),
        ("time_theta", 0.49),
        ("source_mesh_size", 0.0),
        ("receiver_mesh_size", -1.0),
        ("receiver_anchor_mesh_size", -1.0),
        ("source_rhs_sign", 0.0),
        ("source_quadrature_points", -1),
        ("stop_after_outputs", -1),
        ("memory_limit_gb", -1.0),
        ("memory_safety_fraction", 0.0),
        ("min_steps_before_first_observation", 0),
    ],
)
def test_model_consistency_rejects_nonpositive_runtime_parameters(field, value):
    sp = _load_pipeline_module()
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_model_consistency_accepts_ideal_step_off():
    sp = _load_pipeline_module()

    result = sp.validate_model_consistency(sp.PipelineConfig(ramp_off_time=0.0))

    assert result["ramp_off_time"] == 0.0


def test_model_consistency_rejects_negative_ramp_off_time():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError) as exc_info:
        sp.validate_model_consistency(sp.PipelineConfig(ramp_off_time=-1.0e-5))

    assert str(exc_info.value) == "ramp_off_time must be finite and nonnegative"


def test_source_consistency_diagnostics_define_ideal_step_off_current_jump():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(source_current=12.5, ramp_off_time=0.0)

    diagnostics = sp.diagnose_source_consistency(config)

    assert diagnostics["current_initial"] == pytest.approx(12.5)
    assert diagnostics["current_final"] == pytest.approx(0.0)
    assert diagnostics["integral_didt_dt"] == pytest.approx(-12.5)
    assert diagnostics["expected_current_change"] == pytest.approx(-12.5)
    assert diagnostics["waveform_integral_residual"] == pytest.approx(0.0)


def test_source_consistency_diagnostics_preserve_finite_ramp_integral():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(source_current=12.5, ramp_off_time=2.0e-5)

    diagnostics = sp.diagnose_source_consistency(config)

    assert diagnostics["current_initial"] == pytest.approx(12.5)
    assert diagnostics["current_final"] == pytest.approx(0.0)
    assert diagnostics["integral_didt_dt"] == pytest.approx(-12.5)
    assert diagnostics["expected_current_change"] == pytest.approx(-12.5)
    assert diagnostics["waveform_integral_residual"] == pytest.approx(0.0)


def test_ideal_step_off_time_stepping_uses_a_strictly_positive_first_interval():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(source_current=10.0, ramp_off_time=0.0)

    assert sp._source_interval_average_didt(0.0, 2.0e-6, config) == pytest.approx(
        -5.0e6
    )
    with pytest.raises(ValueError, match="t1 must be greater than t0"):
        sp._source_interval_average_didt(0.0, 0.0, config)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_model_consistency_rejects_invalid_divergence_cleaning_strength(value):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning_strength"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning_strength=value))


def test_model_consistency_reports_divergence_cleaning_strength():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(divergence_cleaning="conductivity", divergence_cleaning_strength=0.25)
    )

    assert diagnostics["divergence_cleaning"] == "conductivity"
    assert diagnostics["divergence_cleaning_strength"] == pytest.approx(0.25)


def test_model_consistency_rejects_negative_divergence_cleaning_t_obs_min():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning_t_obs_min"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning_t_obs_min=-1.0e-3))


def test_model_consistency_reports_divergence_cleaning_t_obs_min():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(divergence_cleaning="conductivity", divergence_cleaning_t_obs_min=0.05)
    )

    assert diagnostics["divergence_cleaning_t_obs_min"] == pytest.approx(0.05)


def test_should_apply_divergence_cleaning_respects_observation_time_gate():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_cleaning_t_obs_min=0.05)

    assert sp._should_apply_divergence_cleaning(0.04999 + 1.0e-5, config) is False
    assert sp._should_apply_divergence_cleaning(0.05 + 1.0e-5, config) is True


def test_model_consistency_rejects_negative_divergence_control_weight():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_control_weight"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_control_weight=-1.0))


def test_model_consistency_reports_divergence_control_settings():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            divergence_control_weight=1.0e-8,
            divergence_control_t_obs_min=0.02,
            divergence_control_scale="lhs",
        )
    )

    assert diagnostics["divergence_control_weight"] == pytest.approx(1.0e-8)
    assert diagnostics["divergence_control_t_obs_min"] == pytest.approx(0.02)
    assert diagnostics["divergence_control_scale"] == "lhs"


def test_model_consistency_rejects_unknown_divergence_control_scale():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_control_scale"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_control_scale="bad-scale"))


def test_divergence_control_lhs_scale_uses_left_hand_operator_norm():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(divergence_control_weight=0.25, divergence_control_scale="lhs")

    stats = sp._divergence_control_step_stats(
        config,
        dt=2.0,
        lhs_mass=3.0,
        lhs_stiffness=4.0,
        mass_norm=5.0,
        stiffness_norm=7.0,
        control_norm=10.0,
    )

    assert stats["divergence_control_scale"] == "lhs"
    assert stats["divergence_control_weight"] == pytest.approx(0.25)
    assert stats["divergence_control_reference_norm"] == pytest.approx(43.0)
    assert stats["divergence_control_matrix_norm"] == pytest.approx(10.0)
    assert stats["divergence_control_applied_weight"] == pytest.approx(1.075)
    assert stats["divergence_control_relative_weight"] == pytest.approx(0.25)


def test_should_apply_divergence_control_respects_weight_and_observation_time_gate():
    sp = _load_pipeline_module()
    off = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_control_weight=0.0, divergence_control_t_obs_min=0.02)
    gated = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_control_weight=1.0e-8, divergence_control_t_obs_min=0.02)

    assert sp._should_apply_divergence_control(0.03, off) is False
    assert sp._should_apply_divergence_control(0.01999 + 1.0e-5, gated) is False
    assert sp._should_apply_divergence_control(0.02 + 1.0e-5, gated) is True


def test_model_consistency_accepts_positive_source_rhs_sign_for_diagnostics():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_rhs_sign=1.0))

    assert diagnostics["source_rhs_sign"] == pytest.approx(1.0)


def test_model_consistency_reports_manual_source_quadrature_points():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_quadrature_points=201))

    assert diagnostics["source_quadrature_points"] == 201


def test_manual_line_auto_quadrature_resolves_source_mesh_scale():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        source_mesh_size=40.0,
    )

    assert sp._manual_line_source_quadrature_count(1000.0, config) >= 5001
    assert sp._manual_line_source_quadrature_count(1000.0, sp.PipelineConfig(source_quadrature_points=2001)) == 2001


def test_source_line_segments_from_meshio_blocks_selects_and_sorts_physical_source_lines():
    sp = _load_pipeline_module()
    points = np.asarray(
        [
            [0.0, 1.0, -0.1],
            [1.0, 1.0, -0.1],
            [2.0, 1.0, -0.1],
            [3.0, 1.0, -0.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    line_cells = np.asarray([[2, 3], [0, 1], [1, 2], [0, 4]], dtype=np.int64)
    physical_tags = np.asarray([sp.PHYS_SOURCE_LINE, sp.PHYS_SOURCE_LINE, sp.PHYS_SOURCE_LINE, 999], dtype=np.int64)

    segments = sp._source_line_segments_from_meshio_blocks(
        points,
        [("line", line_cells)],
        [physical_tags],
        sp.PipelineConfig(source_start=(0.0, 1.0, -0.1), source_end=(3.0, 1.0, -0.1)),
    )

    assert segments is not None
    np.testing.assert_allclose(
        segments["segments"],
        np.asarray(
            [
                [[0.0, 1.0, -0.1], [1.0, 1.0, -0.1]],
                [[1.0, 1.0, -0.1], [2.0, 1.0, -0.1]],
                [[2.0, 1.0, -0.1], [3.0, 1.0, -0.1]],
            ]
        ),
    )
    assert segments["segment_count"] == 3
    assert segments["total_length"] == pytest.approx(3.0)
    assert segments["max_segment_length"] == pytest.approx(1.0)


def test_manual_line_integration_points_use_mesh_segments_when_available():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(0.0, 1.0, -0.1),
        source_end=(2.0, 1.0, -0.1),
        source_mesh_size=1.0,
    )
    mesh_segments = {
        "segments": np.asarray(
            [
                [[0.0, 1.0, -0.1], [1.0, 1.0, -0.1]],
                [[1.0, 1.0, -0.1], [2.0, 1.0, -0.1]],
            ],
            dtype=float,
        ),
        "segment_count": 2,
        "total_length": 2.0,
        "min_segment_length": 1.0,
        "max_segment_length": 1.0,
        "mean_segment_length": 1.0,
    }

    points, weights, svals, diagnostics = sp._manual_line_source_integration_points(
        config,
        mesh_segments=mesh_segments,
    )

    assert diagnostics["integration_mode"] == "mesh_segments"
    assert diagnostics["segment_count"] == 2
    assert diagnostics["quadrature_points_per_segment_min"] >= 2
    assert diagnostics["quadrature_points_per_segment_max"] > 2
    assert points.shape[0] > 4
    assert svals.tolist() == pytest.approx(sorted(svals.tolist()))
    assert float(np.sum(weights)) == pytest.approx(2.0)


def test_manual_line_integration_points_report_line_orientation_diagnostics():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        source_mesh_size=100.0,
    )

    _points, _weights, _svals, diagnostics = sp._manual_line_source_integration_points(config)

    orientation = diagnostics["line_orientation"]
    assert orientation["source_length_m"] == pytest.approx(1000.0)
    assert orientation["expected_displacement_m"] == pytest.approx([1000.0, 0.0, 0.0])
    assert orientation["integrated_displacement_m"] == pytest.approx([1000.0, 0.0, 0.0])
    assert orientation["quadrature_weight_sum_m"] == pytest.approx(1000.0)
    assert orientation["signed_parallel_projection_m"] == pytest.approx(1000.0)
    assert orientation["relative_parallel_length_error"] < 1.0e-12
    assert orientation["transverse_residual_m"] < 1.0e-12
    assert orientation["orientation_cosine"] == pytest.approx(1.0)
    assert orientation["s_parameter_min"] >= 0.0
    assert orientation["s_parameter_max"] <= 1.0
    assert orientation["s_parameter_monotonic"] is True
    assert orientation["reversed_orientation"] is False


def test_transient_source_projection_uses_unit_current_shape():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(source_current=10.0)

    assert sp._endpoint_load_current(config, use_unit_current=False) == pytest.approx(10.0)
    assert sp._endpoint_load_current(config, use_unit_current=True) == pytest.approx(1.0)


def test_source_projection_mode_defaults_to_charge_conserving_and_accepts_raw_diagnostic_mode():
    sp = _load_pipeline_module()

    default_diagnostics = sp.validate_model_consistency(sp.PipelineConfig())
    raw_diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_projection_mode="raw"))

    assert sp.PipelineConfig().source_projection_mode == "charge_conserving"
    assert default_diagnostics["source_projection_mode"] == "charge_conserving"
    assert raw_diagnostics["source_projection_mode"] == "raw"


def test_collision_sampling_source_is_explicitly_ineligible_for_formal_acceptance():
    sp = _load_pipeline_module()
    source_info = {
        "mode": "manual_line_collision_diagnostic",
        "projection_diagnostics": {
            "endpoint_norm": 1.0,
            "before_residual": 0.0,
            "after_residual": 0.0,
            "correction_l2_over_raw": 0.0,
        },
        "boundary_elimination_diagnostics": {"relative_residual": 0.0},
        "local_projection_diagnostics": {
            "integration_mode": "collision_sampling_diagnostic/global_gauss",
            "formal_acceptance_eligible": False,
        },
    }

    diagnostics = sp._source_preflight_gate_diagnostics(source_info)

    assert diagnostics["passed"] is False
    assert diagnostics["failed_gates"] == ["source_integration"]
    assert diagnostics["gates"]["source_integration"]["failed_metrics"] == [
        "formal_acceptance_eligible"
    ]
    with pytest.raises(RuntimeError, match="source_integration.formal_acceptance_eligible"):
        sp._require_source_preflight_gates(source_info)


def _failed_exact_interval_source_info():
    return {
        "mode": "manual_line",
        "vector": object(),
        "projection_diagnostics": {
            "endpoint_norm": 1.0,
            "before_residual": 0.0,
            "after_residual": 0.0,
            "correction_l2_over_raw": 0.0,
        },
        "boundary_elimination_diagnostics": {"relative_residual": 0.0},
        "local_projection_diagnostics": {
            "integration_mode": "exact_tetra_intervals",
            "formal_acceptance_eligible": True,
            "assembly_complete": False,
            "interval_gate": {
                "passed": False,
                "serial": True,
                "affine_tetrahedra": True,
                "union_coverage_fraction": 0.9,
                "gap_length": 100.0,
                "overlap_length": 0.0,
                "start_endpoint_covered": True,
                "end_endpoint_covered": True,
                "positive_intervals": True,
                "source_length": 1000.0,
                "interval_total_length": 900.0,
                "parameter_tolerance": 1.0e-12,
            },
        },
    }


def test_formal_exact_manual_line_integration_gate_checks_each_interval_invariant():
    sp = _load_pipeline_module()

    diagnostics = sp._source_preflight_gate_diagnostics(
        _failed_exact_interval_source_info()
    )

    assert diagnostics["passed"] is False
    assert diagnostics["failed_gates"] == ["source_integration"]
    gate = diagnostics["gates"]["source_integration"]
    assert gate["failed_metrics"] == [
        "union_coverage_fraction",
        "gap_length",
        "assembly_complete",
        "interval_total_length",
        "interval_gate_passed",
    ]
    assert gate["metrics"]["union_coverage_fraction"] == pytest.approx(0.9)
    assert gate["metrics"]["gap_length"] == pytest.approx(100.0)


def test_formal_exact_manual_line_gate_accepts_collective_distributed_ownership():
    sp = _load_pipeline_module()
    source = _failed_exact_interval_source_info()
    local = source["local_projection_diagnostics"]
    local["assembly_complete"] = True
    interval = local["interval_gate"]
    interval.update(
        {
            "passed": True,
            "serial": False,
            "distributed_ownership_complete": True,
            "union_coverage_fraction": 1.0,
            "gap_length": 0.0,
            "overlap_length": 0.0,
            "interval_total_length": 1000.0,
        }
    )

    gate = sp._source_integration_gate_diagnostics(source)

    assert gate["passed"] is True
    assert gate["metrics"]["serial"] is False
    assert gate["metrics"]["distributed_ownership_complete"] is True


def test_formal_exact_manual_line_gate_rejects_uncoordinated_nonserial_assembly():
    sp = _load_pipeline_module()
    source = _failed_exact_interval_source_info()
    local = source["local_projection_diagnostics"]
    local["assembly_complete"] = True
    interval = local["interval_gate"]
    interval.update(
        {
            "passed": True,
            "serial": False,
            "distributed_ownership_complete": False,
            "union_coverage_fraction": 1.0,
            "gap_length": 0.0,
            "overlap_length": 0.0,
            "interval_total_length": 1000.0,
        }
    )

    gate = sp._source_integration_gate_diagnostics(source)

    assert gate["passed"] is False
    assert gate["failed_metrics"] == ["distributed_ownership_complete"]


def _source_projection_info(*, raw_relative, correction_relative, after_relative):
    return {
        "projection_diagnostics": {
            "before_residual": float(raw_relative) * 2.0,
            "after_residual": float(after_relative) * 2.0,
            "endpoint_norm": 2.0,
            "correction_l2_over_raw": float(correction_relative),
        }
    }


def test_source_projection_gate_accepts_all_three_metrics_at_fixed_limits():
    sp = _load_pipeline_module()
    source = _source_projection_info(
        raw_relative=sp.SOURCE_PROJECTION_MAX_RAW_RELATIVE_RESIDUAL,
        correction_relative=sp.SOURCE_PROJECTION_MAX_CORRECTION_L2_OVER_RAW,
        after_relative=sp.SOURCE_PROJECTION_MAX_AFTER_RELATIVE_RESIDUAL,
    )

    gate = sp._require_source_projection_gate(source)

    assert gate["passed"] is True
    assert gate["failed_metrics"] == []
    assert gate["thresholds"] == {
        "raw_endpoint_relative_residual": 0.05,
        "correction_l2_over_raw": 0.01,
        "projected_endpoint_relative_residual": 1.0e-8,
    }


@pytest.mark.parametrize(
    "metric,overrides",
    [
        ("raw_endpoint_relative_residual", {"raw_relative": 0.0500001}),
        ("correction_l2_over_raw", {"correction_relative": 0.0100001}),
        ("projected_endpoint_relative_residual", {"after_relative": 1.00001e-8}),
    ],
)
def test_source_projection_gate_fails_closed_above_each_fixed_limit(metric, overrides):
    sp = _load_pipeline_module()
    values = {
        "raw_relative": 0.05,
        "correction_relative": 0.01,
        "after_relative": 1.0e-8,
    }
    values.update(overrides)

    with pytest.raises(RuntimeError, match=metric):
        sp._require_source_projection_gate(_source_projection_info(**values))


def test_source_projection_gate_fails_closed_when_diagnostics_are_missing():
    sp = _load_pipeline_module()

    with pytest.raises(RuntimeError, match="missing_projection_diagnostics"):
        sp._require_source_projection_gate({"mode": "manual_line"})


def test_model_consistency_rejects_unknown_source_projection_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="source_projection_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(source_projection_mode="bad"))


def test_empymod_receiver_mapping_supports_hz_and_dbdt():
    sp = _load_pipeline_module()

    hz_rec, hz_mrec, hz_signal, hz_factor = sp._empymod_rec_mapping((1.0, 2.0, -0.1), "Hz")
    dbdt_rec, dbdt_mrec, dbdt_signal, dbdt_factor = sp._empymod_rec_mapping((1.0, 2.0, -0.1), "dBzdt")

    assert hz_rec == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(0.1), pytest.approx(0.0), pytest.approx(90.0)]
    assert hz_mrec is True
    assert hz_signal == -1
    assert hz_factor == pytest.approx(1.0)
    assert dbdt_rec == hz_rec
    assert dbdt_mrec is True
    assert dbdt_signal == 0
    assert dbdt_factor < 0.0


def test_forward_components_include_hz_for_formal_default_and_biot_receiver():
    sp = _load_pipeline_module()

    assert sp._forward_components(sp.PipelineConfig()) == ["Ex", "Ey", "Hz", "dBzdt"]
    assert sp._forward_components(sp.PipelineConfig(magnetic_receiver_mode="curl")) == [
        "Ex",
        "Ey",
        "dBzdt",
    ]
    assert sp._forward_components(sp.PipelineConfig(magnetic_receiver_mode="biot_current")) == [
        "Ex",
        "Ey",
        "Hz",
        "dBzdt",
    ]


def test_model_consistency_rejects_biot_rate_dbdt_without_biot_h_receiver():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="biot_rate"):
        sp.validate_model_consistency(sp.PipelineConfig(magnetic_receiver_mode="curl", magnetic_dbdt_mode="biot_rate"))

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(magnetic_receiver_mode="biot_current", magnetic_dbdt_mode="biot_rate")
    )
    assert diagnostics["magnetic_dbdt_mode"] == "biot_rate"


def test_model_consistency_reports_nedelec_order():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(nedelec_order=2))

    assert diagnostics["nedelec_order"] == 2


def test_cli_defaults_to_required_nedelec_order(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config, *args, **kwargs):
        captured["nedelec_order"] = config.nedelec_order
        return validate_model_consistency(config, *args, **kwargs)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    assert sp.main(["--check-env-only", "--no-install"]) == 0
    assert captured["nedelec_order"] == sp.REQUIRED_NEDELEC_ORDER == 2


@pytest.mark.parametrize("nedelec_order", [0, 3])
def test_model_consistency_rejects_unsupported_nedelec_order(nedelec_order):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="nedelec_order"):
        sp.validate_model_consistency(sp.PipelineConfig(nedelec_order=nedelec_order))


def test_bdf2_step_coefficients_reduce_to_constant_step_formula():
    sp = _load_pipeline_module()

    coeffs = sp._bdf2_step_coefficients(dt=2.0, previous_dt=2.0)

    assert coeffs["lhs"] == pytest.approx(0.75)
    assert coeffs["old"] == pytest.approx(1.0)
    assert coeffs["older"] == pytest.approx(-0.25)


def test_bdf2_starts_after_first_output_step():
    sp = _load_pipeline_module()

    assert not sp._should_use_bdf2_step(time_method="bdf2", step=0, first_output_step=15)
    assert not sp._should_use_bdf2_step(time_method="bdf2", step=15, first_output_step=15)
    assert sp._should_use_bdf2_step(time_method="bdf2", step=16, first_output_step=15)
    assert not sp._should_use_bdf2_step(time_method="theta", step=16, first_output_step=15)


def test_model_consistency_reports_bdf2_time_method():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(time_method="bdf2"))

    assert diagnostics["time_method"] == "bdf2"


def test_model_consistency_reports_conductivity_divergence_cleaning():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning="conductivity"))

    assert diagnostics["divergence_cleaning"] == "conductivity"


def test_model_consistency_rejects_unknown_divergence_cleaning_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning="bad-mode"))


def test_model_consistency_accepts_ohmic_biot_magnetic_receiver_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(magnetic_receiver_mode="biot_ohmic")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["magnetic_receiver_mode"] == "biot_ohmic"


def test_model_consistency_accepts_natural_outer_boundary_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(outer_boundary_mode="natural")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["outer_boundary_mode"] == "natural"


def test_model_consistency_accepts_robin_outer_boundary_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(outer_boundary_mode="robin", outer_boundary_robin_scale=0.5)
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["outer_boundary_mode"] == "robin"
    assert diagnostics["outer_boundary_robin_scale"] == pytest.approx(0.5)


def test_model_consistency_rejects_unknown_outer_boundary_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="outer_boundary_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(outer_boundary_mode="bad"))


def test_model_consistency_rejects_negative_robin_boundary_scale():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="outer_boundary_robin_scale"):
        sp.validate_model_consistency(sp.PipelineConfig(outer_boundary_robin_scale=-1.0))


def test_model_consistency_accepts_mean_receiver_evaluation_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(receiver_evaluation_mode="mean")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["receiver_evaluation_mode"] == "mean"


def test_qwe_reference_identity_remains_available_for_function_level_diagnostics():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        empymod_srcpts=33,
        empymod_ht="qwe",
        empymod_ht_qwe_rtol=2.0e-11,
        empymod_ft="qwe",
        empymod_ft_qwe_pts_per_dec=13,
        empymod_ft_qwe_rtol=3.0e-9,
    )
    identity = sp._empymod_reference_identity(config)
    kwargs = sp._empymod_call_kwargs(config)

    assert identity["hankel_transform"]["parameters"]["rtol"] == 2.0e-11
    assert identity["fourier_transform"]["parameters"]["rtol"] == 3.0e-9
    assert kwargs["htarg"] == identity["hankel_transform"]["parameters"]
    assert kwargs["ftarg"] == identity["fourier_transform"]["parameters"]


def test_model_consistency_reports_resolved_quasistatic_reference_identity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["empymod_reference_identity"] == sp._empymod_reference_identity(
        sp.PipelineConfig()
    )
    assert diagnostics["empymod_reference_identity"]["equation"] == "quasistatic"
    assert diagnostics["empymod_reference_identity"]["fourier_transform"] == {
        "method": "dlf",
        "parameters": {
            "filter": "key_201_2012",
            "pts_per_dec": 0,
        },
    }


def test_pipeline_defaults_to_verified_source_quadrature_pair():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig()

    assert config.empymod_srcpts == 9
    assert config.reference_audit_srcpts == 17


def test_default_formal_configuration_produces_all_four_real_forward_components():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig()

    diagnostics = sp.validate_model_consistency(config)

    assert config.magnetic_receiver_mode == "faraday_integrated"
    assert diagnostics["magnetic_receiver_mode"] == "faraday_integrated"
    assert sp._forward_components(config) == ["Ex", "Ey", "Hz", "dBzdt"]


@pytest.mark.parametrize("audit_srcpts", [0, 9])
def test_acceptance_configuration_cannot_skip_higher_order_reference_audit(
    audit_srcpts
):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="reference_audit_srcpts.*(positive|greater)"):
        sp.validate_model_consistency(
            sp.PipelineConfig(empymod_srcpts=9, reference_audit_srcpts=audit_srcpts)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"empymod_ht": "qwe"},
        {"empymod_ht": "quad"},
        {"empymod_ft": "qwe", "empymod_ft_qwe_pts_per_dec": 20},
        {"empymod_ht_filter": "key_101_2009"},
        {"empymod_ft_filter": "key_101_2012"},
        {"empymod_ht_pts_per_dec": 1},
        {"empymod_ft_pts_per_dec": -1},
        {"empymod_eperm_h": 1.0},
        {"empymod_eperm_v": 1.0},
        {"empymod_mperm_h": 2.0},
        {"empymod_mperm_v": 2.0},
    ],
)
def test_formal_model_validation_rejects_diagnostic_only_reference_identity(changes):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="diagnostic-only|approved"):
        sp.validate_model_consistency(sp.PipelineConfig(**changes))


def test_formal_model_validation_accepts_exact_approved_reference_identity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["empymod_reference_identity"] == sp._approved_empymod_reference_identity()


def test_acceptance_artifact_writer_rejects_diagnostic_identity_before_writing(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path / "acceptance", empymod_ht="qwe")
    times = np.asarray([1.0e-4])
    data = np.asarray([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="diagnostic-only|approved"):
        sp.write_validation_artifacts(
            times,
            data,
            data.copy(),
            ("Ex", "Ey", "dBzdt"),
            config,
            case_type="noip",
            reference_type="empymod",
        )

    assert not config.workdir.exists()


def test_resolved_config_yaml_records_empymod_reference_identity_fields():
    sp = _load_pipeline_module()

    resolved = sp._resolved_config_yaml(sp.PipelineConfig())

    for line in (
        "empymod_srcpts: 9\n",
        "reference_audit_srcpts: 17\n",
        "empymod_equation: quasistatic\n",
        "empymod_eperm_h: 0\n",
        "empymod_eperm_v: 0\n",
        "empymod_mperm_h: 1\n",
        "empymod_mperm_v: 1\n",
        "empymod_ht: dlf\n",
        "empymod_ht_filter: key_201_2009\n",
        "empymod_ht_pts_per_dec: 0\n",
        "empymod_ft: dlf\n",
        "empymod_ft_filter: key_201_2012\n",
        "empymod_ft_pts_per_dec: 0\n",
        "empymod_ft_qwe_pts_per_dec: 30\n",
    ):
        assert line in resolved


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empymod_srcpts": 0},
        {"empymod_equation": "full-wave"},
        {"empymod_eperm_h": 1.0},
        {"empymod_eperm_v": 1.0},
        {"empymod_mperm_h": 2.0},
        {"empymod_mperm_v": 2.0},
        {"empymod_ht": "bad"},
        {"empymod_ht_filter": ""},
        {"empymod_ht_pts_per_dec": 0.5},
        {"empymod_ht_pts_per_dec": True},
        {"empymod_ft": "bad"},
        {"empymod_ft_filter": ""},
        {"empymod_ft_pts_per_dec": 0.5},
        {"empymod_ft_pts_per_dec": False},
        {"empymod_ft_qwe_pts_per_dec": 0},
        {"empymod_ft_qwe_pts_per_dec": False},
        {"empymod_mperm_h": True},
        {"reference_audit_srcpts": -1},
    ],
)
def test_model_consistency_rejects_invalid_empymod_reference_settings(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


@pytest.mark.parametrize("ft", ["sin", "cos"])
def test_acceptance_rejects_unsupported_empymod_fourier_alias_before_writing(
    tmp_path, ft
):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path / ft, empymod_ft=ft)
    data = np.asarray([[1.0, 2.0, 3.0]])

    with pytest.raises(
        ValueError,
        match="empymod_ft must be 'dlf', 'qwe', 'fftlog', or 'fft'",
    ):
        sp.write_validation_artifacts(
            np.asarray([1.0e-4]),
            data,
            data.copy(),
            ("Ex", "Ey", "dBzdt"),
            config,
            case_type="noip",
            reference_type="empymod",
        )

    assert not config.workdir.exists()


@pytest.mark.parametrize("ft", ["sin", "cos"])
def test_pipeline_cli_parser_rejects_unsupported_empymod_fourier_alias(
    tmp_path, ft
):
    sp = _load_pipeline_module()
    workdir = tmp_path / ft

    with pytest.raises(SystemExit) as exc_info:
        sp.main(
            [
                "--workdir",
                str(workdir),
                "--check-env-only",
                "--no-install",
                "--empymod-ft",
                ft,
            ]
        )

    assert exc_info.value.code == 2
    assert not workdir.exists()


def test_pipeline_cli_resolves_independent_qwe_sampling_density(tmp_path, monkeypatch):
    sp = _load_pipeline_module()
    captured = {}

    def fake_validate(config):
        captured["ft"] = config.empymod_ft
        captured["qwe_pts_per_dec"] = config.empymod_ft_qwe_pts_per_dec
        captured["dlf_pts_per_dec"] = config.empymod_ft_pts_per_dec
        captured["empymod_srcpts"] = config.empymod_srcpts
        captured["reference_audit_srcpts"] = config.reference_audit_srcpts
        return {
            "source_length": 1000.0,
            "inline_distance_from_source_start": 1500.0,
            "parallel_offset": 500.0,
            "reference_mode": "noip",
            "time_origin": "after_ramp",
            "source_depth_start": 0.1,
            "source_depth_end": 0.1,
            "receiver_depth": 0.1,
            "sponge": {"enabled": False},
        }

    monkeypatch.setattr(sp, "validate_model_consistency", fake_validate)
    monkeypatch.setattr(sp, "validate_formulation", lambda config: config.formulation)
    monkeypatch.setattr(sp, "check_environment", lambda **kwargs: {})
    monkeypatch.setattr(
        sp,
        "get_empymod_reference",
        lambda *_args, **_kwargs: pytest.fail(
            "check-env-only must not compute references"
        ),
    )
    monkeypatch.setattr(
        sp,
        "run_fetd_forward",
        lambda *_args, **_kwargs: pytest.fail("check-env-only must not run E forward"),
    )
    monkeypatch.setattr(
        sp,
        "run_h_forward",
        lambda *_args, **_kwargs: pytest.fail("check-env-only must not run H forward"),
    )

    assert (
        sp.main(
            [
                "--workdir",
                str(tmp_path / "qwe-cli"),
                "--check-env-only",
                "--no-install",
                "--empymod-ft",
                "qwe",
                "--empymod-ft-qwe-pts-per-dec",
                "47",
            ]
        )
        == 0
    )
    assert captured == {
        "ft": "qwe",
        "qwe_pts_per_dec": 47,
        "dlf_pts_per_dec": 0,
        "empymod_srcpts": 9,
        "reference_audit_srcpts": 17,
    }


def test_pipeline_cli_default_resolves_real_four_component_formal_contract(
    tmp_path, monkeypatch
):
    sp = _load_pipeline_module()
    captured = {}
    real_validate = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return real_validate(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    assert (
        sp.main(
            [
                "--workdir",
                str(tmp_path / "four-component-default"),
                "--check-env-only",
                "--no-install",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.magnetic_receiver_mode == "faraday_integrated"
    assert sp._forward_components(config) == ["Ex", "Ey", "Hz", "dBzdt"]


def test_tracked_smoke_scripts_use_explicit_approved_quadrature_pairs_and_hz_mode():
    root = Path(__file__).resolve().parents[1]
    contracts = {
        "run_analyticdc_small_smoke.sh": (17, 33),
        "run_sponge_uniform200_receiver2p5_t1e4.sh": (65, 129),
    }

    for name, (primary, audit) in contracts.items():
        text = (root / "dolfinx" / name).read_text(encoding="utf-8")
        assert f"--empymod-srcpts {primary}" in text
        assert f"--reference-audit-srcpts {audit}" in text
        assert "--magnetic-receiver-mode faraday_integrated" in text


def test_song_full_window_runner_uses_faraday_hz_with_curl_dbdt():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "benchmarks" / "sotem" / "run_song2025_fenicsx_p2_t4_full.sh"
    ).read_text(encoding="utf-8")

    assert "--magnetic-receiver-mode faraday_integrated" in text
    assert "--magnetic-dbdt-mode curl" in text
    assert "--magnetic-recovery-quadrature-degree 8" in text
    assert "--magnetic-recovery-quadrature-audit-degrees 2,4,6,8,10" in text


def test_song_full_window_runner_refines_turnoff_to_first_observation():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "benchmarks" / "sotem" / "run_song2025_fenicsx_p2_t4_full.sh"
    ).read_text(encoding="utf-8")

    assert "OUTPUT_INTERVAL_SUBSTEPS=${OUTPUT_INTERVAL_SUBSTEPS:-16}" in text
    assert "MIN_STEPS_BEFORE_FIRST_OBSERVATION=${MIN_STEPS_BEFORE_FIRST_OBSERVATION:-16}" in text


def test_song_full_window_runner_allows_explicit_time_method_without_changing_default():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "benchmarks" / "sotem" / "run_song2025_fenicsx_p2_t4_full.sh"
    ).read_text(encoding="utf-8")

    assert "TIME_METHOD=${TIME_METHOD:-theta}" in text
    assert "TIME_THETA=${TIME_THETA:-1}" in text
    assert '--time-method "$TIME_METHOD"' in text
    assert '--time-theta "$TIME_THETA"' in text


def test_receiver_evaluation_mode_defaults_to_median():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().receiver_evaluation_mode == "median"


def test_time_origin_defaults_to_after_ramp_and_is_reported():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert sp.PipelineConfig().time_origin == "after_ramp"
    assert diagnostics["time_origin"] == "after_ramp"
    assert diagnostics["reference_ramp_window"] == "after_ramp"


def test_time_theta_defaults_to_backward_euler_and_is_reported():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert sp.PipelineConfig().time_theta == pytest.approx(1.0)
    assert diagnostics["time_theta"] == pytest.approx(1.0)


def test_time_theta_coefficients_support_crank_nicolson():
    sp = _load_pipeline_module()

    coeffs = sp._theta_step_coefficients(dt=2.0, theta=0.5)

    assert coeffs["mass_lhs"] == pytest.approx(0.5)
    assert coeffs["stiffness_lhs"] == pytest.approx(0.5)
    assert coeffs["mass_rhs"] == pytest.approx(0.5)
    assert coeffs["stiffness_rhs"] == pytest.approx(-0.5)


def test_model_consistency_rejects_theta_polarized_model_until_debye_theta_is_supported():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(polarization="cole-cole", time_theta=0.5)
    with pytest.raises(ValueError, match="time_theta"):
        sp.validate_model_consistency(config)


def test_late_time_diffusion_audit_flags_default_box_for_uniform_200_late_window():
    sp = _load_pipeline_module()

    audit = sp._diffusion_refinement_audit(sp.PipelineConfig(rho_earth=200.0, t_max=1.0e-3))

    assert audit["diffusion_length"] == pytest.approx(564.1895, rel=1.0e-4)
    assert audit["recommended_radius"] == pytest.approx(1128.379, rel=1.0e-4)
    assert audit["box_radius"] == pytest.approx(1000.0)
    assert audit["box_depth"] == pytest.approx(500.0)
    assert audit["underresolved"] is True


def test_late_time_diffusion_audit_accepts_explicit_diffusion_refinement_factor():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(rho_earth=200.0, t_max=1.0e-3, diffusion_refinement_factor=2.0)
    audit = sp._diffusion_refinement_audit(config)

    assert audit["box_radius"] == pytest.approx(audit["recommended_radius"])
    assert audit["box_depth"] == pytest.approx(audit["recommended_radius"])
    assert audit["underresolved"] is False


def test_explicit_observation_times_define_effective_diffusion_window():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        rho_earth=100.0,
        t_max=1.0,
        observation_times=(1.0e-5, 1.0e-3, 10.0),
        diffusion_refinement_factor=2.0,
    )

    expected_length = sp._max_earth_diffusion_length(config, time=10.0)
    raw_t_max_length = sp._max_earth_diffusion_length(config, time=1.0)
    audit = sp._diffusion_refinement_audit(config)
    diagnostics = sp.validate_model_consistency(config)

    assert sp._effective_t_max(config) == pytest.approx(10.0)
    assert audit["diffusion_length"] == pytest.approx(expected_length)
    assert audit["diffusion_length"] != pytest.approx(raw_t_max_length)
    assert audit["box_radius"] == pytest.approx(2.0 * expected_length)
    assert audit["box_depth"] == pytest.approx(2.0 * expected_length)
    assert diagnostics["configured_t_max"] == pytest.approx(1.0)
    assert diagnostics["effective_t_max"] == pytest.approx(10.0)
    assert diagnostics["t_max"] == pytest.approx(10.0)


def test_h_static_initial_dt_ignores_explicit_observation_window():
    sp = _load_pipeline_module()
    short_window = sp.PipelineConfig(
        t_max=2.0,
        ramp_off_time=3.0,
        observation_times=(1.0e-5, 10.0),
    )
    long_window = sp.PipelineConfig(
        t_max=2.0,
        ramp_off_time=3.0,
        observation_times=(1.0e-5, 100.0),
    )
    expected = max(float(short_window.t_max), float(short_window.ramp_off_time), 1.0) * 1.0e9

    assert sp._h_static_initial_dt(short_window) == pytest.approx(expected)
    assert sp._h_static_initial_dt(long_window) == pytest.approx(expected)


def test_late_time_diffusion_audit_reports_finite_domain_separately_from_refinement_box():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        rho_earth=100.0,
        t_max=1.0,
        x_extent=30000.0,
        y_extent=30000.0,
        earth_depth=30000.0,
        diffusion_refinement_factor=0.0,
    )
    audit = sp._diffusion_refinement_audit(config)

    assert audit["box_radius"] == pytest.approx(1000.0)
    assert audit["box_depth"] == pytest.approx(500.0)
    assert audit["underresolved"] is True
    assert audit["domain_horizontal_radius"] == pytest.approx(30000.0)
    assert audit["domain_depth"] == pytest.approx(30000.0)
    assert audit["domain_underresolved"] is False


def test_ramp_start_time_origin_reports_ramp_start_reference_window():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(time_origin="ramp_start"))

    assert diagnostics["time_origin"] == "ramp_start"
    assert diagnostics["reference_ramp_window"] == "ramp_start"


def test_model_consistency_rejects_invalid_time_origin():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="time_origin"):
        sp.validate_model_consistency(sp.PipelineConfig(time_origin="bad"))


def test_explicit_observation_times_override_growth_grid():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=(1.0e-5, 1.0e-4, 1.0e-3))

    assert sp.generate_time_array(config).tolist() == pytest.approx(config.observation_times)


@pytest.mark.parametrize("observation_times", [(), [], np.asarray([], dtype=float)])
def test_empty_one_dimensional_observation_times_use_legacy_growth_grid(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0e-3,
        time_growth=10.0,
        observation_times=observation_times,
    )

    assert sp.generate_time_array(config).tolist() == pytest.approx([1.0e-5, 1.0e-4, 1.0e-3])


def test_explicit_observation_times_return_independent_array_copy():
    sp = _load_pipeline_module()
    supplied = np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float)

    generated = sp.generate_time_array(sp.PipelineConfig(observation_times=supplied))

    assert not np.shares_memory(generated, supplied)
    supplied[0] = 2.0e-5
    assert generated[0] == pytest.approx(1.0e-5)
    generated[1] = 2.0e-4
    assert supplied[1] == pytest.approx(1.0e-4)


@pytest.mark.parametrize(
    "observation_times",
    [
        (1.0e-5, float("nan"), 1.0e-3),
        (1.0e-5, float("inf"), 1.0e-3),
        (1.0e-5, float("-inf"), 1.0e-3),
        (0.0, 1.0e-4, 1.0e-3),
        (-1.0e-5, 1.0e-4, 1.0e-3),
        (1.0e-5, 1.0e-5, 1.0e-3),
        (1.0e-4, 1.0e-5, 1.0e-3),
    ],
)
def test_explicit_observation_times_reject_invalid_values(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=observation_times)

    with pytest.raises(ValueError) as exc_info:
        sp.generate_time_array(config)

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


@pytest.mark.parametrize(
    "observation_times",
    [
        np.asarray(1.0e-5),
        np.empty((0, 1)),
    ],
)
def test_explicit_observation_times_reject_non_one_dimensional_inputs(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=observation_times)

    with pytest.raises(ValueError) as exc_info:
        sp.generate_time_array(config)

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


def test_model_consistency_rejects_invalid_explicit_observation_times():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError) as exc_info:
        sp.validate_model_consistency(sp.PipelineConfig(observation_times=(1.0e-5, float("nan"))))

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


def test_ideal_step_off_schedule_has_no_synthetic_ramp_steps():
    sp = _load_pipeline_module()
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(
        times, sp.PipelineConfig(ramp_off_time=0.0, time_origin="after_ramp")
    )

    assert schedule["step_times"].tolist() == pytest.approx(times)
    assert schedule["output_step_indices"] == [0, 1, 2]


def test_ideal_step_off_schedule_subdivides_every_output_interval():
    sp = _load_pipeline_module()
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(
        times,
        sp.PipelineConfig(
            ramp_off_time=0.0,
            time_origin="after_ramp",
            output_interval_substeps=4,
        ),
    )

    assert schedule["step_times"].tolist() == pytest.approx(
        [
            2.5e-6,
            5.0e-6,
            7.5e-6,
            1.0e-5,
            3.25e-5,
            5.5e-5,
            7.75e-5,
            1.0e-4,
            3.25e-4,
            5.5e-4,
            7.75e-4,
            1.0e-3,
        ]
    )
    assert schedule["output_internal_times"].tolist() == pytest.approx(times)
    assert schedule["return_times"].tolist() == pytest.approx(times)
    assert schedule["output_step_indices"] == [3, 7, 11]


@pytest.mark.parametrize("invalid_substeps", [True, 1.9, "2"])
def test_model_consistency_rejects_non_integer_output_interval_substeps(invalid_substeps):
    sp = _load_pipeline_module()

    with pytest.raises(
        ValueError,
        match="output_interval_substeps must be a non-bool positive integer",
    ):
        sp.validate_model_consistency(
            sp.PipelineConfig(output_interval_substeps=invalid_substeps)
        )


@pytest.mark.parametrize("invalid_substeps", [True, 1.9, "2", 0, -1])
def test_forward_schedule_rejects_invalid_output_interval_substeps(invalid_substeps):
    sp = _load_pipeline_module()

    with pytest.raises(
        ValueError,
        match="output_interval_substeps must be a non-bool positive integer",
    ):
        sp._forward_observation_schedule(
            [1.0e-5, 1.0e-4],
            sp.PipelineConfig(output_interval_substeps=invalid_substeps),
        )


def test_explicit_observation_times_round_trip_through_cli_and_resolved_yaml(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return validate_model_consistency(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--observation-times",
            "1e-5,1e-4,1e-3",
            "--output-interval-substeps",
            "4",
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].observation_times == pytest.approx((1.0e-5, 1.0e-4, 1.0e-3))
    assert captured["config"].output_interval_substeps == 4
    resolved = sp._resolved_config_yaml(captured["config"])
    assert "observation_times: [1e-05, 0.0001, 0.001]\n" in resolved
    assert "configured_t_max: 1\n" in resolved
    assert "effective_t_max: 0.001\n" in resolved
    assert "t_max: 0.001\n" in resolved


def test_invalid_explicit_observation_times_fail_before_check_env_only(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    calls = {"environment": 0}

    def check_environment(**_kwargs):
        calls["environment"] += 1
        return {}

    monkeypatch.setattr(sp, "check_environment", check_environment)

    with pytest.raises(SystemExit) as exc_info:
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--observation-times",
                "1e-5,nan",
                "--check-env-only",
                "--no-install",
            ]
        )

    assert str(exc_info.value) == "[model] observation_times must be finite, positive, and strictly increasing"
    assert calls["environment"] == 0


def test_invalid_explicit_observation_times_cannot_reach_mesh_only(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    calls = {"environment": 0, "mesh": 0}

    def check_environment(**_kwargs):
        calls["environment"] += 1
        return {}

    def generate_verification_mesh(_config):
        calls["mesh"] += 1
        return tmp_path / "unused.msh"

    monkeypatch.setattr(sp, "check_environment", check_environment)
    monkeypatch.setattr(sp, "generate_verification_mesh", generate_verification_mesh)
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(
                COMM_WORLD=SimpleNamespace(
                    size=1,
                    rank=0,
                    bcast=lambda value, root=0: value,
                    barrier=lambda: None,
                )
            )
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--observation-times",
                "1e-5,nan",
                "--mesh-only",
                "--no-install",
            ]
        )

    assert str(exc_info.value) == "[model] observation_times must be finite, positive, and strictly increasing"
    assert calls == {"environment": 0, "mesh": 0}


@pytest.mark.parametrize(
    "checkpoint_args",
    [
        ("--checkpoint-forward",),
        ("--resume-forward",),
        ("--stop-after-outputs", "1"),
    ],
)
def test_main_rejects_mpi_checkpoint_modes_before_mesh_generation(
    monkeypatch, tmp_path, checkpoint_args
):
    sp = _load_pipeline_module()
    calls = {"mesh": 0}
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        sp,
        "generate_verification_mesh",
        lambda _config: calls.__setitem__("mesh", calls["mesh"] + 1),
    )
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(
                COMM_WORLD=SimpleNamespace(
                    size=2,
                    rank=0,
                    bcast=lambda value, root=0: value,
                    barrier=lambda: None,
                )
            )
        ),
    )

    with pytest.raises(SystemExit, match="MPI.*checkpoint"):
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--mesh-only",
                *checkpoint_args,
                "--no-install",
            ]
        )

    assert calls["mesh"] == 0


def test_main_allows_no_checkpoint_mpi_mesh_only(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    calls = {"mesh": 0}
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        sp,
        "generate_verification_mesh",
        lambda _config: calls.__setitem__("mesh", calls["mesh"] + 1),
    )
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(
                COMM_WORLD=SimpleNamespace(
                    size=2,
                    rank=0,
                    bcast=lambda value, root=0: value,
                    barrier=lambda: None,
                )
            )
        ),
    )

    assert sp.main(["--workdir", str(tmp_path), "--mesh-only", "--no-install"]) == 0
    assert calls["mesh"] == 1


def test_malformed_observation_times_csv_is_reported_by_argparse(capsys):
    sp = _load_pipeline_module()

    with pytest.raises(SystemExit) as exc_info:
        sp.main(["--observation-times", "1e-5,,1e-3"])

    assert exc_info.value.code == 2
    assert "argument --observation-times" in capsys.readouterr().err


def test_after_ramp_observation_schedule_solves_through_ramp_then_returns_observation_times():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=5.0,
        min_steps_during_turnoff=10,
    )
    observation_times = [1.0e-6, 2.0e-6, 1.0e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_internal_times"].tolist() == pytest.approx([1.1e-5, 1.2e-5, 2.0e-5])
    assert schedule["step_times"][0] == pytest.approx(1.0e-6)
    assert np.any(np.isclose(schedule["step_times"], 1.0e-5))
    assert schedule["step_times"][-1] == pytest.approx(2.0e-5)
    assert schedule["output_step_indices"] == [10, 11, 12]


def test_after_ramp_observation_schedule_keeps_ramp_grid_when_observations_end_before_ramp():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=2.0,
        min_steps_during_turnoff=10,
    )
    observation_times = [1.0e-6, 2.0e-6]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"][:10].tolist() == pytest.approx(np.linspace(1.0e-6, 1.0e-5, 10))
    assert schedule["step_times"][-2:].tolist() == pytest.approx([1.1e-5, 1.2e-5])
    assert schedule["output_step_indices"] == [10, 11]


def test_after_ramp_observation_schedule_uses_ramp_solver_t_min_before_later_observation_start():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=2.0,
        ramp_solver_t_min=1.0e-6,
        min_steps_during_turnoff=10,
    )
    observation_times = [2.0e-6, 4.0e-6]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"][:10].tolist() == pytest.approx(np.linspace(1.0e-6, 1.0e-5, 10))
    assert schedule["step_times"][-2:].tolist() == pytest.approx([1.2e-5, 1.4e-5])
    assert schedule["output_step_indices"] == [10, 11]


def test_after_ramp_observation_schedule_substeps_before_first_observation():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        ramp_solver_t_min=1.0e-6,
        min_steps_during_turnoff=10,
        min_steps_before_first_observation=4,
    )
    observation_times = [1.0e-5, 1.25e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert np.any(np.isclose(schedule["step_times"], 1.25e-5))
    assert np.any(np.isclose(schedule["step_times"], 1.50e-5))
    assert np.any(np.isclose(schedule["step_times"], 1.75e-5))
    assert schedule["output_internal_times"].tolist() == pytest.approx([2.0e-5, 2.25e-5])
    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_step_indices"] == [13, 14]


def test_ramp_start_observation_schedule_is_identity():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(ramp_off_time=1.0e-5, time_origin="ramp_start")
    observation_times = [1.0e-6, 2.0e-6, 1.0e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_internal_times"].tolist() == pytest.approx(observation_times)
    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_step_indices"] == [0, 1, 2]


def test_model_consistency_rejects_invalid_time_window():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="t_max"):
        sp.validate_model_consistency(sp.PipelineConfig(t_min=1.0e-3, t_max=1.0e-3))


def test_model_consistency_requires_cole_rho0_to_match_fem_earth_resistivity():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(polarization="cole-cole", rho_earth=50.0, cole_rho0=100.0)
    with pytest.raises(ValueError, match="cole_rho0"):
        sp.validate_model_consistency(config)


def test_layered_cole_cole_model_accepts_explicit_polarizable_depth_window():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(350.0, 650.0),
        layer_resistivities=(100.0, 100.0, 100.0),
        cole_rho0=100.0,
        cole_layer_top=350.0,
        cole_layer_bottom=650.0,
        earth_depth=1000.0,
    )

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["reference_mode"] == "cole-cole-exact"
    assert diagnostics["cole_layer_top"] == pytest.approx(350.0)
    assert diagnostics["cole_layer_bottom"] == pytest.approx(650.0)


def test_layered_cole_cole_model_requires_valid_polarizable_depth_window():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(350.0, 650.0),
        layer_resistivities=(100.0, 100.0, 100.0),
        cole_rho0=100.0,
        cole_layer_top=650.0,
        cole_layer_bottom=350.0,
        earth_depth=1000.0,
    )

    with pytest.raises(ValueError, match="cole_layer"):
        sp.validate_model_consistency(config)


def test_empymod_polarizable_layer_indices_select_middle_layer_only():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_layer_top=350.0, cole_layer_bottom=650.0)

    indices = sp._empymod_polarizable_layer_indices(
        [0.0, 350.0, 650.0],
        [1.0e6, 100.0, 100.0, 100.0],
        config,
    )

    assert indices == [2]


@pytest.mark.parametrize(
    "builder_name",
    ["_exact_cole_cole_empymod_material", "_debye_cole_cole_empymod_material"],
)
def test_cole_cole_empymod_callbacks_flatten_standard_dlf_frequency_grid(
    builder_name, monkeypatch
):
    sp = _load_pipeline_module()
    if builder_name == "_debye_cole_cole_empymod_material":
        monkeypatch.setattr(
            sp,
            "fit_cole_cole_to_debye",
            lambda _config: SimpleNamespace(
                sigma_infinity=0.02,
                terms=(SimpleNamespace(delta_sigma=0.01, tau=0.1),),
            ),
        )
    _depth, material = getattr(sp, builder_name)(sp.PipelineConfig())

    eta_h, eta_v = material["func_eta"](
        material,
        {"freq": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
    )

    assert eta_h.shape == (4, 2)
    assert eta_v.shape == (4, 2)
    assert np.isfinite(eta_h).all()
    np.testing.assert_allclose(eta_h, eta_v)


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
def test_model_consistency_rejects_invalid_cole_fit_tolerance(value):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_fit_tolerance=value)

    with pytest.raises(ValueError, match="cole_fit_tolerance"):
        sp.validate_model_consistency(config)


def test_cole_fit_tolerance_round_trips_through_cli_and_resolved_yaml(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return validate_model_consistency(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--cole-fit-tolerance",
            "0.0075",
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].cole_fit_tolerance == pytest.approx(0.0075)
    resolved = sp._resolved_config_yaml(captured["config"])
    assert "cole_fit_tolerance: 0.0075\n" in resolved


def test_postprocess_cli_selects_exact_cole_cole_reference(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}

    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    def postprocess_saved_forward(config, env, *, ref_mode, runtime, **_kwargs):
        captured["config"] = config
        captured["ref_mode"] = ref_mode
        return {}

    monkeypatch.setattr(sp, "postprocess_saved_forward", postprocess_saved_forward)

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--polarization",
            "cole-cole",
            "--postprocess-partial",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].polarization == "cole-cole"
    assert captured["ref_mode"] == "cole-cole-exact"


@pytest.mark.parametrize("ref_mode", ["noip", "cole-cole-exact", "cole-cole-debye"])
def test_postprocess_records_actual_empymod_reference_mode(monkeypatch, tmp_path, ref_mode):
    sp = _load_pipeline_module()
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "Hz", "dBzdt")
    data = np.ones((times.size, len(components)))
    fem_result = {"times": times, "data": data, "components": components, "solver_log": []}
    captured = {}

    monkeypatch.setattr(sp, "_load_forward_partial", lambda _config: fem_result)
    monkeypatch.setattr(
        sp,
        "get_empymod_reference",
        lambda requested_times, _config, *, mode, **_kwargs: {
            "times": requested_times,
            "data": data.copy(),
            "components": components,
            "reference_mode": mode,
        },
    )
    monkeypatch.setattr(sp, "compute_error", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sp, "_save_npz", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sp, "plot_verification", lambda *_args, **_kwargs: None)

    def write_validation_artifacts(*_args, reference_mode=None, **_kwargs):
        captured["artifact_reference_mode"] = reference_mode
        return {}

    def write_report(_config, _env, _fem_result, ref_result, *_args, **_kwargs):
        captured["report_reference_mode"] = ref_result.get("reference_mode")

    monkeypatch.setattr(sp, "write_validation_artifacts", write_validation_artifacts)
    monkeypatch.setattr(sp, "write_report", write_report)
    config = sp.PipelineConfig(workdir=tmp_path, polarization="cole-cole" if ref_mode != "noip" else "none")

    sp.postprocess_saved_forward(config, {}, ref_mode=ref_mode)

    assert captured["artifact_reference_mode"] == ref_mode
    assert captured["report_reference_mode"] == ref_mode


def test_postprocess_reference_quadrature_failure_blocks_acceptance_writes(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    times = np.asarray([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "Hz", "dBzdt")
    primary = np.asarray([[1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    audit = np.asarray([[1.02, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    calls = []
    monkeypatch.setattr(
        sp,
        "_load_forward_partial",
        lambda _config: {
            "times": times,
            "data": primary.copy(),
            "components": components,
            "solver_log": [],
        },
    )

    def fake_reference(requested_times, config, *, mode, srcpts=None):
        calls.append(srcpts)
        data = audit if srcpts == 17 else primary
        return {
            "times": requested_times,
            "data": data.copy(),
            "components": components,
            "reference_mode": mode,
        }

    monkeypatch.setattr(sp, "get_empymod_reference", fake_reference)
    monkeypatch.setattr(sp, "compute_error", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sp,
        "_save_npz",
        lambda *_args, **_kwargs: pytest.fail(
            "failed 9-to-17 reference audit must block acceptance writes"
        ),
    )

    config = sp.PipelineConfig(workdir=tmp_path)
    with pytest.raises(RuntimeError, match="source-quadrature audit failed.*Ex"):
        sp.postprocess_saved_forward(config, {})

    assert calls == [None, 17]
    assert not config.reference_source_quadrature_audit_json().exists()
    assert not any(tmp_path.iterdir())


def test_reference_source_quadrature_audit_requires_ex_hz_and_dbdzdt():
    sp = _load_pipeline_module()
    data = np.ones((2, 2), dtype=float)

    with pytest.raises(ValueError, match="required components.*Hz"):
        sp._reference_source_quadrature_audit(
            data,
            data.copy(),
            ("Ex", "dBzdt"),
        )


def test_reference_source_quadrature_audit_rejects_hz_above_fixed_tolerance():
    sp = _load_pipeline_module()
    primary = np.ones((2, 3), dtype=float)
    audit = primary.copy()
    audit[0, 1] = 1.02

    summary = sp._reference_source_quadrature_audit(
        primary,
        audit,
        ("Ex", "Hz", "dBzdt"),
    )

    assert summary["threshold"] == pytest.approx(0.005)
    assert summary["passed"] is False
    assert summary["failed_components"] == ["Hz"]


def test_reference_source_quadrature_audit_accepts_all_three_required_components():
    sp = _load_pipeline_module()
    audit = np.asarray([[2.0, 3.0, 4.0], [1.0, 1.5, 2.0]])
    primary = audit * (1.0 + 1.0e-4)

    summary = sp._reference_source_quadrature_audit(
        primary,
        audit,
        ("Ex", "Hz", "dBzdt"),
    )

    assert summary["passed"] is True
    assert summary["failed_components"] == []
    assert set(summary["components"]) == {"Ex", "Hz", "dBzdt"}


@pytest.mark.parametrize(
    "primary_times,audit_times,error",
    [
        ([], [], "at least one"),
        ([[1.0e-4, 1.0e-3]], [[1.0e-4, 1.0e-3]], "one-dimensional"),
        ([1.0e-4], [1.0e-4, 1.0e-3], "same shape"),
        ([1.0e-4, 1.0e-3], [1.0e-4, 2.0e-3], "exactly equal"),
        ([1.0e-4, float("nan")], [1.0e-4, float("nan")], "finite"),
        ([1.0e-4, 1.0e-3], [0.0, 1.0e-3], "positive"),
        ([1.0e-4, 1.0e-3], [1.0e-3, 1.0e-4], "strictly increasing"),
    ],
)
def test_required_reference_source_quadrature_rejects_invalid_or_unequal_time_axes(
    primary_times, audit_times, error
):
    sp = _load_pipeline_module()
    components = ("Ex", "Hz", "dBzdt")
    primary = {
        "times": np.asarray(primary_times),
        "data": np.ones((len(primary_times), len(components))),
        "components": components,
    }
    audit = {
        "times": np.asarray(audit_times),
        "data": np.ones((len(audit_times), len(components))),
        "components": components,
    }

    with pytest.raises(ValueError, match=error):
        sp._require_reference_source_quadrature_audit(
            sp.PipelineConfig(), primary, audit
        )


def test_required_reference_source_quadrature_rejects_time_axis_without_data_row():
    sp = _load_pipeline_module()
    components = ("Ex", "Hz", "dBzdt")
    times = np.asarray([1.0e-4, 1.0e-3])
    result = {
        "times": times,
        "data": np.ones((1, len(components))),
        "components": components,
    }

    with pytest.raises(ValueError, match="one data row per time"):
        sp._require_reference_source_quadrature_audit(
            sp.PipelineConfig(), result, dict(result)
        )


def _patch_formal_forward_seams(monkeypatch, sp, events, *, audit_scale):
    fake_mesh = SimpleNamespace(comm=SimpleNamespace(rank=0))
    times = np.asarray([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "Hz", "dBzdt")
    primary = np.ones((times.size, len(components)), dtype=float)
    audit = primary.copy()
    audit[:, 0] *= audit_scale

    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(
                COMM_WORLD=SimpleNamespace(
                    size=1,
                    rank=0,
                    bcast=lambda value, root=0: value,
                    barrier=lambda: None,
                )
            )
        ),
    )
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})
    monkeypatch.setattr(sp, "generate_verification_mesh", lambda _config: events.append("mesh"))
    monkeypatch.setattr(
        sp,
        "load_mesh",
        lambda _config: (fake_mesh, object(), object()),
    )
    monkeypatch.setattr(sp, "build_function_spaces", lambda *_args: {})
    monkeypatch.setattr(sp, "assign_materials", lambda *_args: {})
    monkeypatch.setattr(sp, "apply_transient_sponge", lambda *_args: None)
    monkeypatch.setattr(
        sp,
        "build_source",
        lambda *_args: {
            "mode": "test",
            "vector": object(),
            "projection_diagnostics": {
                "before_residual": 0.04,
                "after_residual": 1.0e-9,
                "endpoint_norm": 1.0,
                "correction_l2_over_raw": 0.005,
            },
        },
    )
    monkeypatch.setattr(
        sp,
        "_make_zero_tangential_bc",
        lambda *_args: (None, [], []),
    )
    monkeypatch.setattr(
        sp,
        "_validate_source_after_boundary_elimination",
        lambda *_args: {
            "passed": True,
            "relative_residual": 1.0e-12,
            "relative_tolerance": 1.0e-8,
        },
    )
    passing_quality = {
        "passed": True,
        "failed_selections": [],
        "thresholds": {
            "min_quality_3r_over_R": 0.01,
            "max_aspect_R_over_3r": 100.0,
        },
        "selections": {},
        "receiver": {"colliding_cell_count": 1, "selection_mode": "colliding"},
    }
    monkeypatch.setattr(
        sp, "diagnose_local_mesh_quality", lambda *_args: passing_quality
    )
    monkeypatch.setattr(
        sp,
        "_pre_forward_diagnostics",
        lambda *_args, **_kwargs: {
            "mesh_sha256": "test",
            "global_cells": 1,
            "global_nedelec_dofs": 1,
            "memory": {"estimated_gb": 0.0, "ok": True},
            "receiver": passing_quality["receiver"],
            "polarization": {"mode": "none"},
        },
    )
    monkeypatch.setattr(sp, "_write_pre_forward_diagnostics", lambda *_args: None)
    monkeypatch.setattr(
        sp,
        "_build_regularized_current_density",
        lambda *_args: object(),
    )
    monkeypatch.setattr(sp, "generate_time_array", lambda _config: times.copy())

    def fake_reference(requested_times, _config, *, mode, srcpts=None):
        events.append(f"reference:{srcpts}")
        return {
            "times": np.asarray(requested_times),
            "data": (audit if srcpts == 17 else primary).copy(),
            "components": components,
            "reference_mode": mode,
        }

    monkeypatch.setattr(sp, "get_empymod_reference", fake_reference)
    return times, primary, components


@pytest.mark.parametrize("formulation, selected_solver", [("e", "run_fetd_forward"), ("h", "run_h_forward")])
def test_formal_forward_quadrature_failure_precedes_both_solvers_and_all_acceptance_writes(
    monkeypatch, tmp_path, formulation, selected_solver
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.02)

    for solver_name in ("run_fetd_forward", "run_h_forward"):
        monkeypatch.setattr(
            sp,
            solver_name,
            lambda *_args, _solver_name=solver_name, **_kwargs: events.append(
                f"solver:{_solver_name}"
            ),
        )
    for writer_name in (
        "_save_forward_partial",
        "_save_forward_checkpoint",
        "_save_npz",
        "plot_verification",
        "write_validation_artifacts",
        "write_report",
    ):
        monkeypatch.setattr(
            sp,
            writer_name,
            lambda *_args, _writer_name=writer_name, **_kwargs: events.append(
                f"write:{_writer_name}"
            ),
        )

    with pytest.raises(RuntimeError, match="source-quadrature audit failed.*Ex"):
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--formulation",
                formulation,
                "--checkpoint-forward",
                "--no-install",
            ]
        )

    assert events == ["mesh", "reference:None", "reference:17"]
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.png"))
    assert f"solver:{selected_solver}" not in events


@pytest.mark.parametrize("formulation, selected_solver", [("e", "run_fetd_forward"), ("h", "run_h_forward")])
def test_formal_forward_passes_reference_gate_before_solver_and_reuses_precomputed_reference(
    monkeypatch, tmp_path, formulation, selected_solver
):
    sp = _load_pipeline_module()
    events = []
    times, primary, components = _patch_formal_forward_seams(
        monkeypatch, sp, events, audit_scale=1.0001
    )

    def fake_forward(*_args, **_kwargs):
        events.append(f"solver:{selected_solver}")
        return {
            "times": times[:1].copy(),
            "data": primary[:1].copy(),
            "components": components,
            "solver_log": [],
        }

    monkeypatch.setattr(sp, selected_solver, fake_forward)
    other_solver = "run_h_forward" if selected_solver == "run_fetd_forward" else "run_fetd_forward"
    monkeypatch.setattr(
        sp,
        other_solver,
        lambda *_args, **_kwargs: pytest.fail("unselected formulation solver called"),
    )
    def fake_save_npz(_config, fem_result, ref_result, _errors):
        np.testing.assert_array_equal(ref_result["times"], fem_result["times"])
        assert ref_result["data"].shape == fem_result["data"].shape
        events.append("write:npz")

    monkeypatch.setattr(sp, "_save_npz", fake_save_npz)
    monkeypatch.setattr(
        sp, "plot_verification", lambda *_args, **_kwargs: events.append("write:plot")
    )
    evidence_chain = {}

    def capture_validation(*_args, reference_source_quadrature_audit=None, **_kwargs):
        evidence_chain["validation"] = reference_source_quadrature_audit
        events.append("write:acceptance")

    def capture_report(*_args, reference_source_quadrature_audit=None, **_kwargs):
        evidence_chain["report"] = reference_source_quadrature_audit
        events.append("write:report")

    monkeypatch.setattr(sp, "write_validation_artifacts", capture_validation)
    monkeypatch.setattr(sp, "write_report", capture_report)

    assert (
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--formulation",
                formulation,
                "--no-install",
            ]
        )
        == 0
    )

    assert events[:4] == [
        "mesh",
        "reference:None",
        "reference:17",
        f"solver:{selected_solver}",
    ]
    assert events.count("reference:None") == 1
    assert events.count("reference:17") == 1
    evidence = json.loads(
        (tmp_path / "reference_source_quadrature_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence == evidence_chain["validation"] == evidence_chain["report"]
    assert evidence["artifact_schema"] == "atem3d.reference_source_quadrature_audit.v1"
    assert evidence["artifact_scope"] == "reference-only"
    assert evidence["final_acceptance_eligible"] is False
    assert evidence["final_acceptance_status"] == "blocked"
    assert evidence["approved_reference_identity"] == sp._approved_empymod_reference_identity()
    assert evidence["primary_srcpts"] == 9
    assert evidence["audit_srcpts"] == 17
    assert evidence["primary_times"] == times.tolist()
    assert evidence["audit_times"] == times.tolist()
    assert evidence["time_axes_equal"] is True
    assert evidence["threshold"] == pytest.approx(0.005)
    assert evidence["floor_rule"] == "0.01 * peak(abs(higher-order reference)) per component"
    assert evidence["global_pass"] is True
    assert set(evidence["components"]) == {"Ex", "Hz", "dBzdt"}
    for component in evidence["components"].values():
        assert set(component) >= {
            "max_acceptance_error",
            "max_index",
            "time_at_max",
            "passed",
        }


@pytest.mark.parametrize(
    "artifact_name,extra_args",
    [
        ("error_summary.json", []),
        ("verification_result.png", ["--resume-forward"]),
        ("verification_report.txt", ["--postprocess-partial"]),
        ("run_config_resolved.yaml", ["--source-only"]),
        ("error_summary.json", ["--mesh-only"]),
    ],
)
def test_formal_main_rejects_stale_acceptance_artifacts_before_environment_reference_or_solver(
    monkeypatch, tmp_path, artifact_name, extra_args
):
    sp = _load_pipeline_module()
    workdir = tmp_path / "stale-formal-run"
    workdir.mkdir()
    stale_path = workdir / artifact_name
    stale_bytes = b"old-formal-artifact\x00must-not-change"
    stale_path.write_bytes(stale_bytes)
    audit_path = workdir / "reference_source_quadrature_audit.json"
    audit_bytes = b'{"artifact_scope":"reference-only","old":true}\n'
    audit_path.write_bytes(audit_bytes)
    mesh_path = workdir / "verification_mesh.msh"
    mesh_bytes = b"old-mesh-must-not-change"
    mesh_path.write_bytes(mesh_bytes)

    for seam in (
        "check_environment",
        "generate_verification_mesh",
        "get_empymod_reference",
        "run_fetd_forward",
        "run_h_forward",
        "_save_npz",
        "plot_verification",
        "write_validation_artifacts",
        "write_report",
    ):
        monkeypatch.setattr(
            sp,
            seam,
            lambda *_args, _seam=seam, **_kwargs: pytest.fail(
                f"stale formal run reached forbidden seam: {_seam}"
            ),
        )

    with pytest.raises(RuntimeError, match="stale formal acceptance artifact.*refusing"):
        sp.main(
            [
                "--workdir",
                str(workdir),
                *extra_args,
                "--no-install",
            ]
        )

    assert stale_path.read_bytes() == stale_bytes
    assert audit_path.read_bytes() == audit_bytes
    assert mesh_path.read_bytes() == mesh_bytes


def test_direct_postprocess_rejects_stale_acceptance_artifact_before_loading_partial_or_reference(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    partial_bytes = b"legal-stage-only-partial"
    stale_bytes = b"old-report-must-not-change"
    config.forward_partial_npz().write_bytes(partial_bytes)
    config.output_report().write_bytes(stale_bytes)

    for seam in ("_load_forward_partial", "get_empymod_reference", "_save_npz"):
        monkeypatch.setattr(
            sp,
            seam,
            lambda *_args, _seam=seam, **_kwargs: pytest.fail(
                f"stale postprocess reached forbidden seam: {_seam}"
            ),
        )

    with pytest.raises(RuntimeError, match="stale formal acceptance artifact.*refusing"):
        sp.postprocess_saved_forward(config, {})

    assert config.forward_partial_npz().read_bytes() == partial_bytes
    assert config.output_report().read_bytes() == stale_bytes


def test_direct_postprocess_has_no_caller_controlled_lock_bypass(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)

    with pytest.raises(TypeError, match="_formal_lock_held"):
        sp.postprocess_saved_forward(config, {}, _formal_lock_held=True)


def test_formal_artifact_guard_preserves_legal_reference_and_resume_stage_files(tmp_path):
    sp = _load_pipeline_module()
    from atem3d.final_acceptance import (
        REQUIRED_CASE_ARTIFACTS,
        REQUIRED_POLARIZATION_EFFECT_ARTIFACTS,
    )

    config = sp.PipelineConfig(workdir=tmp_path)
    stage_files = {
        config.reference_source_quadrature_audit_json(): b"reference-stage-audit",
        config.forward_partial_npz(): b"partial-stage",
        config.forward_checkpoint_npz(): b"resume-stage",
        config.mesh_path(): b"source-stage-mesh",
        tmp_path / "source_diagnostics.json": b"source-stage-diagnostics",
        tmp_path / "source_diagnostics_report.txt": b"source-stage-report",
        tmp_path / "source_run_config_resolved.yaml": b"source-stage-config",
    }
    for path, payload in stage_files.items():
        path.write_bytes(payload)

    formal_names = set(sp._formal_acceptance_artifact_names())
    assert set(REQUIRED_CASE_ARTIFACTS) <= formal_names
    assert set(REQUIRED_POLARIZATION_EFFECT_ARTIFACTS) <= formal_names
    assert {path.name for path in stage_files}.isdisjoint(formal_names)
    assert sp._existing_formal_acceptance_artifacts(config) == ()
    sp._require_fresh_formal_output_directory(config)

    assert {path: path.read_bytes() for path in stage_files} == stage_files


def test_formal_artifact_names_ignore_unrelated_installed_atem3d_contract(monkeypatch):
    sp = _load_pipeline_module()
    fake_contract = SimpleNamespace(
        REQUIRED_CASE_ARTIFACTS=("fake_old_case_artifact.txt",),
        REQUIRED_POLARIZATION_EFFECT_ARTIFACTS=("fake_old_effect_artifact.txt",),
    )
    monkeypatch.setitem(sys.modules, "atem3d.final_acceptance", fake_contract)

    names = set(sp._formal_acceptance_artifact_names())

    assert "error_summary.json" in names
    assert "polarization_effect_summary.json" in names
    assert "fake_old_case_artifact.txt" not in names
    assert "fake_old_effect_artifact.txt" not in names


def test_direct_subprocess_uses_checkout_artifact_contract_over_fake_pythonpath(
    tmp_path
):
    root = Path(__file__).resolve().parents[1]
    pipeline_path = root / "dolfinx" / "sotem_pipeline.py"
    fake_root = tmp_path / "fake-site"
    fake_package = fake_root / "atem3d"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "artifact_contract.py").write_text(
        "REQUIRED_CASE_ARTIFACTS=('fake_old_case.txt',)\n"
        "REQUIRED_POLARIZATION_EFFECT_ARTIFACTS=('fake_old_effect.txt',)\n",
        encoding="utf-8",
    )
    script = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "path=Path(sys.argv[1]); "
        "spec=importlib.util.spec_from_file_location('direct_pipeline',path); "
        "module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; "
        "spec.loader.exec_module(module); "
        "print(json.dumps(module._formal_acceptance_artifact_names()))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_root)

    completed = subprocess.run(
        [sys.executable, "-c", script, str(pipeline_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    names = set(json.loads(completed.stdout))

    assert "error_summary.json" in names
    assert "polarization_effect_summary.json" in names
    assert "fake_old_case.txt" not in names
    assert "fake_old_effect.txt" not in names


def test_formal_run_lock_is_reentrant_releases_after_exception_and_rejects_process(
    tmp_path
):
    sp = _load_pipeline_module()
    run_dir = tmp_path / "locked-run"
    marker = tmp_path / "locked.marker"
    release = tmp_path / "release.marker"
    run_lock_path = (
        Path(__file__).resolve().parents[1] / "src" / "atem3d" / "run_lock.py"
    )
    script = (
        "import importlib.util,sys,time; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('child_run_lock',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "run=Path(sys.argv[2]); marker=Path(sys.argv[3]); release=Path(sys.argv[4]); "
        "\nwith module.run_lock(run):\n marker.write_text('locked')\n"
        " while not release.exists(): time.sleep(0.01)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(run_lock_path),
            str(run_dir),
            str(marker),
            str(release),
        ]
    )
    try:
        deadline = time.monotonic() + 10.0
        while not marker.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("child process did not acquire formal run lock")
            time.sleep(0.01)
        assert process.poll() is None
        with pytest.raises(RuntimeError, match="another writer holds the run lock"):
            with sp._formal_run_lock(run_dir):
                pytest.fail("competing process acquired formal run lock")
    finally:
        release.write_text("release", encoding="utf-8")
        process.wait(timeout=10)

    with pytest.raises(RuntimeError, match="synthetic exception"):
        with sp._formal_run_lock(run_dir):
            with sp._formal_run_lock(run_dir):
                raise RuntimeError("synthetic exception")
    with sp._formal_run_lock(run_dir):
        pass


class _RecordingComm:
    def __init__(self, *, rank, size, events, root_payload=None):
        self.rank = int(rank)
        self.size = int(size)
        self.events = events
        self.root_payload = root_payload

    def bcast(self, value, root=0):
        self.events.append(f"bcast:{root}")
        if self.rank == root:
            return value
        return self.root_payload

    def barrier(self):
        self.events.append("barrier")


class _RecordingContext:
    def __init__(self, events, label, *, enter_error=None):
        self.events = events
        self.label = label
        self.enter_error = enter_error

    def __enter__(self):
        self.events.append(f"{self.label}:enter")
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append(f"{self.label}:exit")


def test_coordinated_formal_run_lock_only_root_touches_lock_and_releases_after_barrier(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    comm = _RecordingComm(rank=0, size=8, events=events)
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda path: _RecordingContext(events, f"lock:{path}"),
    )

    with sp._coordinated_formal_run_lock(tmp_path, comm):
        events.append("body")

    assert events == [
        f"lock:{tmp_path}:enter",
        "bcast:0",
        "body",
        "barrier",
        f"lock:{tmp_path}:exit",
    ]


def test_coordinated_formal_run_lock_peer_never_touches_filesystem_lock(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    comm = _RecordingComm(
        rank=3,
        size=8,
        events=events,
        root_payload=(True, ""),
    )
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda _path: pytest.fail("peer rank attempted to acquire filesystem lock"),
    )

    with sp._coordinated_formal_run_lock(tmp_path, comm):
        events.append("body")

    assert events == ["bcast:0", "body", "barrier"]


def test_coordinated_formal_run_lock_broadcasts_root_acquisition_failure(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    message = f"another writer holds the run lock: {tmp_path}"
    root_events = []
    root_comm = _RecordingComm(rank=0, size=8, events=root_events)
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda path: _RecordingContext(
            root_events,
            f"lock:{path}",
            enter_error=RuntimeError(message),
        ),
    )

    with pytest.raises(RuntimeError, match="another writer holds the run lock"):
        with sp._coordinated_formal_run_lock(tmp_path, root_comm):
            pytest.fail("root entered body after failed lock acquisition")

    assert root_events == [f"lock:{tmp_path}:enter", "bcast:0"]

    peer_events = []
    peer_comm = _RecordingComm(
        rank=6,
        size=8,
        events=peer_events,
        root_payload=(False, message),
    )
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda _path: pytest.fail("peer rank attempted to acquire filesystem lock"),
    )

    with pytest.raises(RuntimeError, match="another writer holds the run lock"):
        with sp._coordinated_formal_run_lock(tmp_path, peer_comm):
            pytest.fail("peer entered body after root lock failure")

    assert peer_events == ["bcast:0"]


def test_collective_root_call_runs_only_on_root_and_barriers_before_return():
    sp = _load_pipeline_module()
    root_events = []
    root_comm = _RecordingComm(rank=0, size=8, events=root_events)

    result = sp._collective_root_call(
        root_comm,
        lambda: root_events.append("action") or "root-result",
    )

    assert result == "root-result"
    assert root_events == ["action", "bcast:0", "barrier"]

    peer_events = []
    peer_comm = _RecordingComm(
        rank=5,
        size=8,
        events=peer_events,
        root_payload=(True, ""),
    )
    result = sp._collective_root_call(
        peer_comm,
        lambda: pytest.fail("peer rank executed root action"),
    )

    assert result is None
    assert peer_events == ["bcast:0", "barrier"]


def test_collective_root_call_propagates_root_failure_to_peer():
    sp = _load_pipeline_module()
    message = "synthetic root-stage failure"
    peer_events = []
    peer_comm = _RecordingComm(
        rank=2,
        size=8,
        events=peer_events,
        root_payload=(False, message),
    )

    with pytest.raises(RuntimeError, match=message):
        sp._collective_root_call(
            peer_comm,
            lambda: pytest.fail("peer rank executed failed root action"),
        )

    assert peer_events == ["bcast:0"]


def test_gather_receiver_sample_candidates_combines_nonempty_ranks_in_rank_order():
    sp = _load_pipeline_module()
    local = (
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([[10.0, 20.0, 30.0]]),
        np.asarray([[0.0, 0.0, -1.0]]),
    )
    empty = (
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
    )
    remote = (
        np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        np.asarray([[40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]),
        np.asarray([[1.0, 0.0, -2.0], [2.0, 0.0, -3.0]]),
    )
    comm = SimpleNamespace(size=3, allgather=lambda _payload: [local, empty, remote])

    electric, magnetic_rate, centers = sp._gather_receiver_sample_candidates(
        comm,
        *local,
    )

    np.testing.assert_array_equal(
        electric,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    np.testing.assert_array_equal(
        magnetic_rate,
        np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]),
    )
    np.testing.assert_array_equal(
        centers,
        np.asarray([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [2.0, 0.0, -3.0]]),
    )


def test_gather_receiver_sample_candidates_serial_path_does_not_call_allgather():
    sp = _load_pipeline_module()
    electric = np.asarray([[1.0, 2.0, 3.0]])
    magnetic_rate = np.asarray([[4.0, 5.0, 6.0]])
    comm = SimpleNamespace(
        size=1,
        allgather=lambda _payload: pytest.fail("serial receiver path called allgather"),
    )

    gathered = sp._gather_receiver_sample_candidates(
        comm,
        electric,
        magnetic_rate,
        None,
    )

    np.testing.assert_array_equal(gathered[0], electric)
    np.testing.assert_array_equal(gathered[1], magnetic_rate)
    assert gathered[2] is None


def test_receiver_excludes_ghost_cells_before_candidate_collapse(monkeypatch):
    sp = _load_pipeline_module()
    evaluated_cells = []

    class FakeField:
        def __init__(self, scale):
            self.scale = float(scale)

        def eval(self, _points, cells):
            cell_array = np.asarray(cells, dtype=int)
            evaluated_cells.append(cell_array.tolist())
            return np.column_stack(
                [
                    self.scale * (cell_array + 1),
                    self.scale * (cell_array + 2),
                    self.scale * (cell_array + 3),
                ]
            )

    index_map = SimpleNamespace(size_local=2)
    msh = SimpleNamespace(
        comm=SimpleNamespace(size=1, rank=0),
        topology=SimpleNamespace(dim=3, index_map=lambda _dim: index_map),
    )
    monkeypatch.setattr(
        sp,
        "_find_cells_for_point",
        lambda _msh, _point: np.asarray([0, 2], dtype=np.int32),
    )

    record = sp.evaluate_receivers(
        FakeField(1.0),
        FakeField(10.0),
        msh,
        sp.PipelineConfig(receiver_evaluation_mode="median"),
    )

    assert evaluated_cells == [[0], [0]]
    assert record["Ex"] == pytest.approx(1.0)
    assert record["Ey"] == pytest.approx(2.0)
    assert record["dBzdt"] == pytest.approx(30.0)
    assert record["candidate_count_max"] == 1


def test_cell_geometry_helpers_use_geometry_dofmap_after_mpi_reordering():
    sp = _load_pipeline_module()
    wrong_topology_vertices = np.asarray([0, 1, 2, 3], dtype=np.int32)
    correct_geometry_dofs = np.asarray([[4, 5, 6, 7]], dtype=np.int32)
    coordinates = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 11.0],
            [0.0, 0.0, -2.0],
            [1.0, 0.0, -2.0],
            [0.0, 1.0, -2.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    connectivity = SimpleNamespace(links=lambda _cell: wrong_topology_vertices)
    topology = SimpleNamespace(
        dim=3,
        create_connectivity=lambda _from_dim, _to_dim: None,
        connectivity=lambda _from_dim, _to_dim: connectivity,
        index_map=lambda _dim: SimpleNamespace(size_local=1),
    )
    msh = SimpleNamespace(
        topology=topology,
        geometry=SimpleNamespace(x=coordinates, dofmap=correct_geometry_dofs),
    )

    centers = sp._cell_centers(msh)
    centers_radii_volumes = sp._cell_centers_radii_volumes(msh)
    cell_geometry = sp._cell_geometry(msh, 0)

    np.testing.assert_allclose(centers, [[0.25, 0.25, -1.75]])
    np.testing.assert_allclose(centers_radii_volumes[0], centers)
    assert centers_radii_volumes[2][0] == pytest.approx(1.0 / 6.0)
    np.testing.assert_array_equal(cell_geometry, coordinates[4:8])


def test_mesh_source_and_formal_cli_writers_reject_competing_process(tmp_path):
    sp = _load_pipeline_module()
    pipeline_path = Path(__file__).resolve().parents[1] / "dolfinx" / "sotem_pipeline.py"
    workdir = tmp_path / "contended-writer"

    with sp._formal_run_lock(workdir):
        for mode_args in (("--mesh-only",), ("--source-only",), ()):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(pipeline_path),
                    "--workdir",
                    str(workdir),
                    *mode_args,
                    "--no-install",
                ],
                capture_output=True,
                text=True,
            )
            assert completed.returncode != 0
            assert "another writer holds the run lock" in completed.stderr

    assert not workdir.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX os.fork")
def test_forked_child_cannot_reenter_parent_formal_run_lock(tmp_path):
    sp = _load_pipeline_module()
    run_dir = tmp_path / "fork-locked-run"
    read_fd, write_fd = os.pipe()

    with sp._formal_run_lock(run_dir):
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                try:
                    with sp._formal_run_lock(run_dir):
                        outcome = b"acquired"
                except RuntimeError:
                    outcome = b"blocked"
                os.write(write_fd, outcome)
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        outcome = os.read(read_fd, 64)
        os.close(read_fd)
        waited_pid, status = os.waitpid(child_pid, 0)

    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0
    assert outcome == b"blocked"


def test_formal_artifact_guard_rejects_dangling_link_without_following_target(tmp_path):
    sp = _load_pipeline_module()
    workdir = tmp_path / "link-workdir"
    workdir.mkdir()
    missing_target = tmp_path / "outside" / "must-remain-missing.json"
    stale_link = workdir / "error_summary.json"
    try:
        stale_link.symlink_to(missing_target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(RuntimeError, match="error_summary.json"):
        sp._require_fresh_formal_output_directory(
            sp.PipelineConfig(workdir=workdir)
        )

    assert stale_link.is_symlink()
    assert not missing_target.exists()


def test_formal_main_holds_writer_lock_through_reference_solver_and_failure(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)

    class RecordingLock:
        def __enter__(self):
            events.append("lock:enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("lock:exit")

    monkeypatch.setattr(
        sp, "_formal_run_lock", lambda _workdir: RecordingLock()
    )

    def failing_forward(*_args, **_kwargs):
        assert events[0] == "lock:enter"
        assert "reference:None" in events
        assert "reference:17" in events
        events.append("solver:inside-lock")
        raise RuntimeError("synthetic locked forward failure")

    monkeypatch.setattr(sp, "run_fetd_forward", failing_forward)
    monkeypatch.setattr(
        sp,
        "run_h_forward",
        lambda *_args, **_kwargs: pytest.fail("unselected H solver called"),
    )

    with pytest.raises(RuntimeError, match="synthetic locked forward failure"):
        sp.main(["--workdir", str(tmp_path), "--no-install"])

    assert events[-2:] == ["solver:inside-lock", "lock:exit"]


@pytest.mark.parametrize("stage_flag", ["--mesh-only", "--source-only"])
def test_stage_writer_modes_hold_lock_through_last_stage_action(
    monkeypatch, tmp_path, stage_flag
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)

    class RecordingLock:
        def __enter__(self):
            events.append("lock:enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("lock:exit")

    monkeypatch.setattr(sp, "_formal_run_lock", lambda _workdir: RecordingLock())
    monkeypatch.setattr(
        sp,
        "write_source_only_diagnostics",
        lambda *_args, **_kwargs: events.append("write:source-only"),
    )

    assert sp.main(["--workdir", str(tmp_path), stage_flag, "--no-install"]) == 0

    assert events[0] == "lock:enter"
    assert "mesh" in events
    if stage_flag == "--source-only":
        assert "write:source-only" in events
    assert events[-1] == "lock:exit"


def test_source_only_writes_both_failed_source_gates_before_raising(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)
    monkeypatch.setattr(
        sp,
        "build_source",
        lambda *_args: {
            "mode": "test",
            "vector": object(),
            "projection_diagnostics": {
                "before_residual": 0.087624,
                "after_residual": 1.0e-9,
                "endpoint_norm": 1.0,
                "correction_l2_over_raw": 0.005,
            },
        },
    )
    monkeypatch.setattr(
        sp,
        "_validate_source_after_boundary_elimination",
        lambda *_args: {
            "passed": False,
            "relative_residual": 2.6606e-8,
            "relative_tolerance": 1.0e-8,
            "absolute_tolerance": 1.0e-8,
            "endpoint_norm": 1.0,
            "residual_norm": 2.6606e-8,
            "eliminated_l2_over_source": 1.0e-8,
        },
    )

    with pytest.raises(RuntimeError) as exc_info:
        sp.main(["--workdir", str(tmp_path), "--source-only", "--no-install"])

    payload = json.loads(
        (tmp_path / "source_diagnostics.json").read_text(encoding="utf-8")
    )
    report = (tmp_path / "source_diagnostics_report.txt").read_text(encoding="utf-8")
    assert payload["source_projection_gate"]["passed"] is False
    assert payload["source_projection_gate"]["failed_metrics"] == [
        "raw_endpoint_relative_residual"
    ]
    assert payload["source_boundary_elimination_gate"]["passed"] is False
    assert payload["source_boundary_elimination_gate"]["failed_metrics"] == [
        "relative_residual"
    ]
    assert payload["source_boundary_elimination_gate"]["thresholds"] == {
        "relative_residual": 1.0e-8
    }
    assert payload["source_preflight_gates"]["failed_gates"] == [
        "source_projection",
        "source_boundary_elimination",
    ]
    assert "source projection gate: FAIL" in report
    assert "source boundary elimination gate: FAIL" in report
    message = str(exc_info.value)
    assert "raw_endpoint_relative_residual" in message
    assert "source_boundary_elimination.relative_residual" in message
    assert message.index("raw_endpoint_relative_residual") < message.index(
        "source_boundary_elimination.relative_residual"
    )
    assert events == ["mesh"]
    assert not list(tmp_path.glob("*.tmp"))


def test_source_only_writes_failed_exact_interval_gate_before_raising(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)
    source_info = _failed_exact_interval_source_info()
    source_info.pop("boundary_elimination_diagnostics")
    monkeypatch.setattr(sp, "build_source", lambda *_args: source_info)
    monkeypatch.setattr(
        sp,
        "_validate_source_after_boundary_elimination",
        lambda *_args: pytest.fail("invalid interval source must not reach boundary assembly"),
    )
    monkeypatch.setattr(
        sp,
        "run_fetd_forward",
        lambda *_args, **_kwargs: pytest.fail("invalid interval source must not reach forward"),
    )

    with pytest.raises(RuntimeError, match="source_integration.union_coverage_fraction"):
        sp.main(["--workdir", str(tmp_path), "--source-only", "--no-install"])

    payload = json.loads(
        (tmp_path / "source_diagnostics.json").read_text(encoding="utf-8")
    )
    report = (tmp_path / "source_diagnostics_report.txt").read_text(encoding="utf-8")
    integration = payload["source_preflight_gates"]["gates"]["source_integration"]
    assert integration["passed"] is False
    assert integration["metrics"]["union_coverage_fraction"] == pytest.approx(0.9)
    assert integration["metrics"]["gap_length"] == pytest.approx(100.0)
    assert "source integration gate: FAIL" in report
    assert events == ["mesh"]
    assert not list(tmp_path.glob("*.tmp"))


def test_formal_forward_projection_gate_failure_precedes_reference_and_solver(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)
    monkeypatch.setattr(
        sp,
        "build_source",
        lambda *_args: {
            "mode": "test",
            "vector": object(),
            "projection_diagnostics": {
                "before_residual": 0.050001,
                "after_residual": 1.0e-9,
                "endpoint_norm": 1.0,
                "correction_l2_over_raw": 0.005,
            },
        },
    )
    monkeypatch.setattr(
        sp,
        "run_fetd_forward",
        lambda *_args, **_kwargs: pytest.fail("solver must not run after gate failure"),
    )

    with pytest.raises(RuntimeError, match="raw_endpoint_relative_residual"):
        sp.main(["--workdir", str(tmp_path), "--no-install"])

    assert events == ["mesh"]


def test_direct_fetd_call_requires_both_source_gates_before_any_field_solve(monkeypatch):
    sp = _load_pipeline_module()
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(fem=object()))
    monkeypatch.setitem(
        sys.modules,
        "petsc4py",
        SimpleNamespace(PETSc=object()),
    )
    monkeypatch.setattr(
        sp,
        "assemble_operators",
        lambda *_args, **_kwargs: {"bc_global": []},
    )
    monkeypatch.setattr(
        sp,
        "_validate_source_after_boundary_elimination",
        lambda *_args: {
            "relative_residual": 2.6606e-8,
            "relative_tolerance": 1.0e-8,
            "passed": False,
        },
    )
    monkeypatch.setattr(
        sp,
        "_solve_initial_dc_field",
        lambda *_args, **_kwargs: pytest.fail("field solve reached after source gate failure"),
    )
    source = {
        "mode": "test",
        "vector": object(),
        "projection_diagnostics": {
            "before_residual": 0.087624,
            "after_residual": 1.0e-9,
            "endpoint_norm": 1.0,
            "correction_l2_over_raw": 0.005,
        },
    }

    with pytest.raises(RuntimeError) as exc_info:
        sp.run_fetd_forward(
            object(),
            object(),
            object(),
            {},
            {},
            source,
            sp.PipelineConfig(ramp_off_time=0.0),
            times=[1.0e-5],
        )
    message = str(exc_info.value)
    assert "source_projection.raw_endpoint_relative_residual" in message
    assert "source_boundary_elimination.relative_residual" in message


def test_direct_fetd_call_blocks_failed_exact_interval_gate_before_field_solve(monkeypatch):
    sp = _load_pipeline_module()
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(fem=object()))
    monkeypatch.setitem(sys.modules, "petsc4py", SimpleNamespace(PETSc=object()))
    monkeypatch.setattr(
        sp,
        "assemble_operators",
        lambda *_args, **_kwargs: {"bc_global": []},
    )
    monkeypatch.setattr(
        sp,
        "_validate_source_after_boundary_elimination",
        lambda *_args: {"relative_residual": 0.0, "passed": True},
    )
    monkeypatch.setattr(
        sp,
        "_solve_initial_dc_field",
        lambda *_args, **_kwargs: pytest.fail("field solve reached after interval gate failure"),
    )

    with pytest.raises(RuntimeError, match="source_integration.union_coverage_fraction"):
        sp.run_fetd_forward(
            object(),
            object(),
            object(),
            {},
            {},
            _failed_exact_interval_source_info(),
            sp.PipelineConfig(ramp_off_time=0.0),
            times=[1.0e-5],
        )


def test_direct_h_call_blocks_failed_exact_interval_gate_before_operator_assembly(monkeypatch):
    sp = _load_pipeline_module()
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(fem=object()))
    monkeypatch.setattr(
        sp,
        "assemble_h_operators",
        lambda *_args, **_kwargs: pytest.fail("H operators reached after interval gate failure"),
    )

    with pytest.raises(RuntimeError, match="source_integration.union_coverage_fraction"):
        sp.run_h_forward(
            object(),
            object(),
            object(),
            {},
            {},
            _failed_exact_interval_source_info(),
            sp.PipelineConfig(formulation="h", ramp_off_time=0.0),
            times=[1.0e-5],
        )


def test_check_env_only_neither_locks_nor_creates_workdir(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    workdir = tmp_path / "check-only"
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda _workdir: pytest.fail("check-env-only must not acquire writer lock"),
    )
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    assert (
        sp.main(
            ["--workdir", str(workdir), "--check-env-only", "--no-install"]
        )
        == 0
    )

    assert not workdir.exists()


def test_help_does_not_acquire_formal_writer_lock(monkeypatch):
    sp = _load_pipeline_module()
    monkeypatch.setattr(
        sp,
        "_formal_run_lock",
        lambda _workdir: pytest.fail("--help must not acquire formal writer lock"),
    )

    with pytest.raises(SystemExit) as exc:
        sp.main(["--help"])

    assert exc.value.code == 0


def test_fresh_forward_failure_keeps_reference_only_audit_and_blocks_final_acceptance(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    events = []
    _patch_formal_forward_seams(monkeypatch, sp, events, audit_scale=1.0001)
    monkeypatch.setattr(
        sp,
        "run_fetd_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic forward failure")
        ),
    )
    monkeypatch.setattr(
        sp,
        "run_h_forward",
        lambda *_args, **_kwargs: pytest.fail("unselected H solver called"),
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        sp.main(["--workdir", str(tmp_path), "--no-install"])

    audit = json.loads(
        sp.PipelineConfig(workdir=tmp_path)
        .reference_source_quadrature_audit_json()
        .read_text(encoding="utf-8")
    )
    assert audit["artifact_scope"] == "reference-only"
    assert audit["final_acceptance_eligible"] is False
    assert audit["final_acceptance_status"] == "blocked"
    assert audit["global_pass"] is True
    assert sp._existing_formal_acceptance_artifacts(
        sp.PipelineConfig(workdir=tmp_path)
    ) == ()


def test_h_form_resume_fails_before_environment_mesh_reference_forward_or_writes(
    monkeypatch, tmp_path
):
    sp = _load_pipeline_module()
    workdir = tmp_path / "h-resume"

    for seam in (
        "check_environment",
        "generate_verification_mesh",
        "get_empymod_reference",
        "run_h_forward",
        "_save_forward_partial",
        "_save_forward_checkpoint",
        "_save_npz",
        "plot_verification",
        "write_validation_artifacts",
        "write_report",
    ):
        monkeypatch.setattr(
            sp,
            seam,
            lambda *_args, _seam=seam, **_kwargs: pytest.fail(
                f"H resume reached forbidden seam: {_seam}"
            ),
        )

    with pytest.raises(SystemExit, match="formulation.*h.*resume_forward"):
        sp.main(
            [
                "--workdir",
                str(workdir),
                "--formulation",
                "h",
                "--resume-forward",
                "--check-env-only",
                "--no-install",
            ]
        )

    assert not workdir.exists()


def test_report_uses_reference_mode_from_reference_result(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, polarization="cole-cole")
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "dBzdt")
    data = np.array([[1.0, 1.0e-4, 1.0e-6], [0.5, 5.0e-5, 5.0e-7]])
    fem_result = {
        "times": times,
        "data": data,
        "components": components,
        "solver_log": [],
    }
    ref_result = {
        "times": times,
        "data": data.copy(),
        "components": components,
        "reference_mode": "cole-cole-debye",
    }
    errors = sp.compute_error(data, data, components)

    sp.write_report(config, {}, fem_result, ref_result, errors, {"mode": "test"})

    report = config.output_report().read_text(encoding="utf-8")
    assert "reference_mode=cole-cole-debye" in report
    assert "reference_mode=cole-cole-exact" not in report


def test_validation_artifacts_record_actual_reference_mode(tmp_path):
    sp = _load_pipeline_module()
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "dBzdt")
    data = np.array([[1.0, 1.0e-4, 1.0e-6], [0.5, 5.0e-5, 5.0e-7]])
    config = sp.PipelineConfig(workdir=tmp_path)

    summary = sp.write_validation_artifacts(
        times,
        data,
        data.copy(),
        components,
        config,
        case_type="ip",
        reference_type="empymod",
        reference_mode="cole-cole-debye",
    )

    assert summary["reference_mode"] == "cole-cole-debye"


def test_receiver_depth_profile_depths_are_validated():
    sp = _load_pipeline_module()

    assert sp._validated_receiver_depth_profile_depths(
        sp.PipelineConfig(receiver_depth_profile_depths=(300.0, 400.0, 500.0, 600.0))
    ) == (300.0, 400.0, 500.0, 600.0)
    for values in (
        (0.0,),
        (-1.0,),
        (300.0, 300.0),
        (400.0, 300.0),
        (float("nan"),),
    ):
        with pytest.raises(ValueError, match="receiver_depth_profile_depths"):
            sp._validated_receiver_depth_profile_depths(
                sp.PipelineConfig(receiver_depth_profile_depths=values)
            )


def test_receiver_depth_profile_default_is_disabled():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig()

    assert config.receiver_depth_profile_depths == ()
    assert sp._validated_receiver_depth_profile_depths(config) == ()


def test_receiver_depth_profile_round_trips_through_cli(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return validate_model_consistency(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--receiver-depth-profile-depths",
            "300,400,500,600",
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].receiver_depth_profile_depths == pytest.approx(
        (300.0, 400.0, 500.0, 600.0)
    )
