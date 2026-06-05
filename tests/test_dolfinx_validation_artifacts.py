from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_validation_artifacts_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_validation_artifacts_generates_required_p2_outputs(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.array([1.0e-5, 1.0e-4])
    ref = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]])
    pred = np.array([[1.01, 1.0e-16, 2.02], [2.02, 2.0e-16, 1.01]])
    components = ["Ex", "Ey", "dBzdt"]

    summary = sp.write_validation_artifacts(
        times,
        pred,
        ref,
        components,
        config,
        case_type="noip",
        reference_type="empymod",
        source_info={
            "mode": "manual_line",
            "projection_diagnostics": {
                "applied": True,
                "before_residual": 2.0,
                "after_residual": 1.0e-9,
                "endpoint_norm": 1.4,
                "raw_source_l2_norm": 10.0,
                "projected_source_l2_norm": 11.0,
                "correction_l2_norm": 1.5,
                "correction_l2_over_raw": 0.15,
                "raw_source_l1_norm": 20.0,
                "projected_source_l1_norm": 21.0,
                "correction_l1_norm": 2.5,
                "correction_l1_over_raw": 0.125,
                "divergence_residual_reduction": 0.9999999995,
            },
        },
        receiver_diagnostic_rows=[
            {
                "time_obs": 1.0e-5,
                "receiver_type": "point",
                "radius": 0.0,
                "Ex": 1.0,
                "Ey": 0.0,
                "Hz": np.nan,
                "dBzdt": 1.0,
                "sample_count": 1,
                "candidate_count_min": 2,
                "candidate_count_max": 2,
                "candidate_count_mean": 2.0,
            },
            {
                "time_obs": 1.0e-5,
                "receiver_type": "disk_average",
                "radius": 2.0,
                "Ex": 1.2,
                "Ey": 0.0,
                "Hz": np.nan,
                "dBzdt": 1.8,
                "sample_count": 5,
                "candidate_count_min": 1,
                "candidate_count_max": 3,
                "candidate_count_mean": 1.8,
            },
        ],
    )

    required = [
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "error_summary.json",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
    ]
    for name in required:
        assert (tmp_path / name).is_file(), name

    with (tmp_path / "errors.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) >= {
        "time_obs",
        "component",
        "pred",
        "ref",
        "abs_error",
        "ordinary_relative_error",
        "relative_error_with_floor",
        "peak_normalized_error",
        "pass_5pct",
    }

    report = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert report["case_type"] == "noip"
    assert report["reference_type"] == "empymod"
    assert report["magnetic_quantity"] == "dBzdt"
    assert report["pass_all_components"] is True
    assert summary["pass_all_components"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["source_consistency"]["source_endpoint_balance_residual"] == 1.0e-9
    assert diagnostics["source_projection"]["before_residual"] == 2.0
    assert diagnostics["source_projection"]["after_residual"] == 1.0e-9
    assert diagnostics["source_projection"]["correction_l2_over_raw"] == pytest.approx(0.15)
    assert diagnostics["source_projection"]["correction_l1_over_raw"] == pytest.approx(0.125)
    assert diagnostics["source_projection"]["divergence_residual_reduction"] == pytest.approx(0.9999999995)
    assert diagnostics["receiver_sampling"]["enabled"] is True
    assert diagnostics["receiver_sampling"]["comparisons"]["disk_average"]["dBzdt"]["max_relative_difference"] == 0.8
    assert diagnostics["receiver_vs_reference"]["enabled"] is True
    assert diagnostics["receiver_vs_reference"]["comparisons"]["disk_average"]["dBzdt"]["improves_over_baseline"] is True
    assert (tmp_path / "receiver_reference_errors.csv").is_file()
    assert (tmp_path / "receiver_reference_error_curves.png").is_file()
    receiver_rows = list(csv.DictReader((tmp_path / "receiver_reference_errors.csv").open("r", encoding="utf-8", newline="")))
    assert receiver_rows[0]["receiver_type"] == "point"
    assert {"time_obs", "receiver_type", "component", "pred", "ref", "relative_error_with_floor", "pass_5pct"} <= set(receiver_rows[0])
    diagnostic_rows = list(csv.DictReader((tmp_path / "receiver_diagnostics.csv").open("r", encoding="utf-8", newline="")))
    assert {"sample_count", "candidate_count_min", "candidate_count_max", "candidate_count_mean"} <= set(diagnostic_rows[0])
    assert diagnostic_rows[0]["sample_count"] == "1"
    assert diagnostic_rows[1]["candidate_count_max"] == "3"


def test_validation_artifacts_report_physical_pass_for_weak_horizontal_component(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.array([1.0e-5, 1.0e-4])
    ref = np.array(
        [
            [10.0, 1.0e-12, 2.0e-9],
            [5.0, 2.0e-12, 1.0e-9],
        ]
    )
    pred = ref.copy()
    pred[:, 1] += np.array([0.1, -0.2])

    summary = sp.write_validation_artifacts(
        times,
        pred,
        ref,
        ["Ex", "Ey", "dBzdt"],
        config,
        case_type="noip",
        reference_type="empymod",
    )

    report = json.loads((tmp_path / "error_summary.json").read_text(encoding="utf-8"))
    assert report["pass_all_components"] is False
    assert report["physical_pass_all_components"] is True
    assert report["weak_component_passed"] is True
    assert report["weak_components"] == ["Ey"]
    assert report["physical_failed_components"] == []
    assert summary["physical_pass_all_components"] is True

    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["validation_failure"]["failed"] is False
    assert diagnostics["validation_failure"]["strict_failed"] is True


def test_manual_line_source_local_projection_diagnostics_summarize_cell_and_endpoint_support():
    sp = _load_pipeline_module()

    diagnostics = sp._summarize_manual_line_source_local_diagnostics(
        npts=6,
        added=5,
        missed=1,
        hit_cell_ids=[10, 10, None, 11, 12, 12],
        svals=[0.02, 0.08, 0.2, 0.55, 0.93, 0.98],
        cell_l1_contributions={10: 4.0, 11: 2.0, 12: 4.0},
        dof_l1_contributions={1: 5.0, 2: 3.0, 3: 2.0},
        endpoint_window_fraction=0.1,
    )

    assert diagnostics["quadrature_points"] == 6
    assert diagnostics["added_points"] == 5
    assert diagnostics["missed_points"] == 1
    assert diagnostics["missed_fraction"] == pytest.approx(1.0 / 6.0)
    assert diagnostics["unique_hit_cells"] == 3
    assert diagnostics["cell_hit_count_min"] == 1
    assert diagnostics["cell_hit_count_max"] == 2
    assert diagnostics["cell_hit_count_mean"] == pytest.approx(5.0 / 3.0)
    assert diagnostics["cell_hit_top_fraction"] == pytest.approx(2.0 / 5.0)
    assert diagnostics["cell_sequence_changes"] == 2
    assert diagnostics["cell_contribution_top_fraction"] == pytest.approx(0.4)
    assert diagnostics["cell_contribution_top_cell"] == 10
    assert diagnostics["active_dof_count"] == 3
    assert diagnostics["dof_contribution_top_fraction"] == pytest.approx(0.5)
    assert diagnostics["start_window_points"] == 2
    assert diagnostics["start_window_unique_cells"] == 1
    assert diagnostics["start_window_missed"] == 0
    assert diagnostics["end_window_points"] == 2
    assert diagnostics["end_window_unique_cells"] == 1
    assert diagnostics["end_window_missed"] == 0


def test_validation_artifacts_include_source_local_projection_diagnostics(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.array([1.0e-5])
    ref = np.array([[1.0, 0.0, 2.0]])
    pred = np.array([[1.0, 0.0, 2.0]])

    sp.write_validation_artifacts(
        times,
        pred,
        ref,
        ["Ex", "Ey", "dBzdt"],
        config,
        case_type="noip",
        reference_type="empymod",
        source_info={
            "mode": "manual_line",
            "projection_diagnostics": {
                "applied": False,
                "before_residual": 0.0,
                "after_residual": 0.0,
                "endpoint_norm": 1.0,
            },
            "local_projection_diagnostics": {
                "quadrature_points": 6,
                "unique_hit_cells": 3,
                "cell_contribution_top_fraction": 0.4,
            },
        },
    )

    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["source_local_projection"]["quadrature_points"] == 6
    assert diagnostics["source_local_projection"]["unique_hit_cells"] == 3
    assert diagnostics["source_local_projection"]["cell_contribution_top_fraction"] == pytest.approx(0.4)


def test_scalar_source_balance_vector_diagnostics_reports_residual_distribution():
    sp = _load_pipeline_module()

    diagnostics = sp._scalar_source_balance_vector_diagnostics(
        endpoint=np.asarray([1.0, -1.0, 0.0, 0.0]),
        current_div=np.asarray([0.8, -0.4, 0.3, 0.0]),
        residual=np.asarray([0.2, -0.6, -0.3, 0.0]),
    )

    assert diagnostics["endpoint_active_dofs"] == 2
    assert diagnostics["current_div_active_dofs"] == 3
    assert diagnostics["residual_active_dofs"] == 3
    assert diagnostics["residual_l1_norm"] == pytest.approx(1.1)
    assert diagnostics["residual_linf_norm"] == pytest.approx(0.6)
    assert diagnostics["residual_l2_over_endpoint_l2"] == pytest.approx(np.linalg.norm([0.2, -0.6, -0.3]) / np.sqrt(2.0))
    assert diagnostics["residual_top_abs_fraction"] == pytest.approx(0.6 / 1.1)
    assert diagnostics["current_div_endpoint_alignment"] == pytest.approx(1.2 / (np.linalg.norm([0.8, -0.4, 0.3]) * np.sqrt(2.0)))


def test_write_source_only_diagnostics_generates_source_artifacts(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, source_only=True)

    sp.write_source_only_diagnostics(
        config,
        env={"python": "test-python"},
        source_info={
            "mode": "manual_line",
            "projection_diagnostics": {
                "projection_mode": "raw",
                "applied": False,
                "before_residual": 6.0,
                "after_residual": 6.0,
                "endpoint_norm": 1.0,
                "scalar_balance": {
                    "residual_active_dofs": 12,
                    "residual_l2_over_endpoint_l2": 6.0,
                    "residual_top_abs_fraction": 0.25,
                    "current_div_endpoint_alignment": 0.1,
                },
            },
            "local_projection_diagnostics": {
                "quadrature_points": 101,
                "missed_points": 0,
                "unique_hit_cells": 20,
            },
        },
        runtime={"mesh_seconds": 1.0, "setup_seconds": 2.0},
    )

    diagnostics = json.loads((tmp_path / "source_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["source_mode"] == "manual_line"
    assert diagnostics["source_projection"]["projection_mode"] == "raw"
    assert diagnostics["source_projection"]["scalar_balance"]["residual_active_dofs"] == 12
    assert diagnostics["source_local_projection"]["quadrature_points"] == 101
    assert (tmp_path / "source_diagnostics_report.txt").is_file()
    assert (tmp_path / "run_config_resolved.yaml").is_file()


def test_faraday_integrated_hz_trace_uses_trapezoid_dbdt():
    sp = _load_pipeline_module()
    times = np.asarray([1.0, 3.0, 6.0])
    dbzdt = np.asarray([2.0, 4.0, 8.0])

    hz = sp._faraday_integrated_hz_trace(times, dbzdt, initial_hz=10.0, mu=2.0)

    np.testing.assert_allclose(hz, np.asarray([10.0, 13.0, 22.0]))


def test_validation_artifacts_include_faraday_magnetic_recovery_summary(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path)
    times = np.asarray([1.0, 3.0, 6.0])
    pred = np.asarray(
        [
            [1.0, 0.0, 10.0, 2.0],
            [1.0, 0.0, 13.0, 4.0],
            [1.0, 0.0, 23.0, 8.0],
        ]
    )
    ref = pred.copy()
    components = ["Ex", "Ey", "Hz", "dBzdt"]

    sp.write_validation_artifacts(
        times,
        pred,
        ref,
        components,
        config,
        case_type="noip",
        reference_type="empymod",
    )

    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["magnetic_recovery"]["enabled"] is True
    assert diagnostics["magnetic_recovery"]["method"] == "faraday_integrated_dBzdt"
    assert diagnostics["magnetic_recovery"]["max_absolute_hz_difference"] > 0.0


def test_magnetic_recovery_summary_checks_hz_rate_against_dbdt():
    sp = _load_pipeline_module()
    mu0 = 1.2566370614359173e-6
    times = np.asarray([0.0, 1.0, 2.0])
    dbzdt = np.asarray([2.0, 4.0, 8.0])
    hz = np.asarray([0.0, 3.0 / mu0, 9.0 / mu0])
    pred = np.column_stack([np.ones_like(times), np.zeros_like(times), hz, dbzdt])

    summary = sp._magnetic_recovery_summary(times, pred, ["Ex", "Ey", "Hz", "dBzdt"])

    assert summary["rate_consistency"]["enabled"] is True
    assert summary["rate_consistency"]["method"] == "mu_dHzdt_vs_trapezoid_dBzdt"
    assert summary["rate_consistency"]["sample_count"] == 2
    assert summary["rate_consistency"]["max_relative_difference"] < 1.0e-12


def test_biot_receiver_dbdt_from_h_uses_interval_rate():
    sp = _load_pipeline_module()

    dbdt = sp._biot_receiver_dbdt_from_h([0.0, 1.0, 5.0], [0.0, -1.0, 1.0], dt=2.0, mu=4.0)

    np.testing.assert_allclose(dbdt, np.asarray([0.0, 4.0, 8.0]))


def test_receiver_candidate_collapse_supports_geometric_selection_modes():
    sp = _load_pipeline_module()
    values = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )
    centers = np.asarray(
        [
            [0.0, 0.0, -10.0],
            [1.0, 0.0, -0.05],
            [0.0, 0.0, -0.02],
        ]
    )

    np.testing.assert_allclose(
        sp._collapse_receiver_cell_candidates(values, "nearest_center", centers=centers, point=[0.9, 0.0, -0.1]),
        values[1],
    )
    np.testing.assert_allclose(
        sp._collapse_receiver_cell_candidates(values, "shallowest", centers=centers, point=[0.0, 0.0, -0.1]),
        values[2],
    )
