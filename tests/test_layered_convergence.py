from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atem3d.layered_convergence import (
    ConvergenceResponse,
    build_convergence_levels,
    build_pipeline_command_arguments,
    compare_responses,
    load_response,
)


def _option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_convergence_levels_match_approved_three_level_design(tmp_path):
    levels = build_convergence_levels(tmp_path / "layered", tmp_path / "convergence")

    assert [
        (level.level_id, level.max_internal_dt, level.max_internal_dt_fraction)
        for level in levels["time"]
    ] == [
        ("coarse", 5.0e-5, 0.02),
        ("standard", 2.5e-5, 0.01),
        ("fine", 1.25e-5, 0.005),
    ]
    assert [
        (level.source_mesh_size, level.receiver_mesh_size)
        for level in levels["mesh"]
    ] == [(12.0, 9.0), (8.0, 6.0), (6.0, 4.5)]
    assert [
        (level.x_extent, level.far_field_mesh_size)
        for level in levels["domain"]
    ] == [(3000.0, 750.0), (6000.0, 750.0), (12000.0, 750.0)]


def test_time_level_changes_time_controls_and_reuses_baseline_mesh(tmp_path):
    level = build_convergence_levels(
        tmp_path / "layered", tmp_path / "convergence"
    )["time"][0]

    arguments = build_pipeline_command_arguments(level)

    assert _option_value(arguments, "--max-internal-dt") == "5e-05"
    assert _option_value(arguments, "--max-internal-dt-fraction") == "0.02"
    assert _option_value(arguments, "--source-mesh-size") == "8"
    assert _option_value(arguments, "--receiver-mesh-size") == "6"
    assert _option_value(arguments, "--stop-after-outputs") == "25"
    assert _option_value(arguments, "--reuse-mesh") == str(level.reuse_mesh_path)


def test_mesh_level_changes_only_local_mesh_targets(tmp_path):
    level = build_convergence_levels(
        tmp_path / "layered", tmp_path / "convergence"
    )["mesh"][0]

    arguments = build_pipeline_command_arguments(level)

    assert _option_value(arguments, "--source-mesh-size") == "12"
    assert _option_value(arguments, "--receiver-mesh-size") == "9"
    assert _option_value(arguments, "--max-internal-dt") == "2.5e-05"
    assert _option_value(arguments, "--max-internal-dt-fraction") == "0.01"
    assert _option_value(arguments, "--stop-after-outputs") == "25"
    assert "--reuse-mesh" not in arguments


def test_existing_levels_point_to_completed_publication_runs(tmp_path):
    layered_root = tmp_path / "layered"
    levels = build_convergence_levels(layered_root, tmp_path / "convergence")

    baseline = (
        layered_root
        / "domain6000"
        / "resistive_basement_rho1000_offset100"
    )
    large = (
        layered_root
        / "domain12000"
        / "resistive_basement_rho1000_offset100"
    )
    assert levels["time"][1].existing_run_dir == baseline
    assert levels["mesh"][1].existing_run_dir == baseline
    assert levels["domain"][1].existing_run_dir == baseline
    assert levels["domain"][2].existing_run_dir == large
    assert levels["time"][0].reuse_mesh_path == baseline / "verification_mesh.msh"
    assert levels["time"][2].reuse_mesh_path == baseline / "verification_mesh.msh"


def test_domain_small_keeps_far_field_mesh_size_fixed(tmp_path):
    level = build_convergence_levels(
        tmp_path / "layered", tmp_path / "convergence"
    )["domain"][0]

    arguments = build_pipeline_command_arguments(level)

    assert _option_value(arguments, "--x-extent") == "3000"
    assert _option_value(arguments, "--earth-depth") == "3000"
    assert _option_value(arguments, "--far-field-mesh-size") == "750"
    assert _option_value(arguments, "--expected-parallel-offset") == "100"


def test_generated_level_workdir_matches_axis_and_level(tmp_path):
    output_root = tmp_path / "convergence"
    level = build_convergence_levels(tmp_path / "layered", output_root)["time"][2]

    assert level.workdir == output_root / "time" / "fine"
    assert Path(_option_value(build_pipeline_command_arguments(level), "--workdir")) == level.workdir


def _response(values) -> ConvergenceResponse:
    return ConvergenceResponse(
        times=np.array([1.0e-5, 1.0e-4, 1.0e-3]),
        dbzdt=np.asarray(values, dtype=float),
        reference=np.array([1.0, 0.1, 0.01]),
    )


def test_compare_responses_reports_median_rms_and_max_percent():
    result = compare_responses(
        _response([1.1, 0.11, 0.011]),
        _response([1.0, 0.1, 0.01]),
    )

    assert result["sample_count"] == 3
    assert result["excluded_below_floor_count"] == 0
    assert result["median_percent"] == pytest.approx(10.0)
    assert result["rms_percent"] == pytest.approx(10.0)
    assert result["max_percent"] == pytest.approx(10.0)
    assert result["relative"] == pytest.approx([0.1, 0.1, 0.1])


def test_compare_responses_rejects_mismatched_observation_grids():
    fine = _response([1.0, 0.1, 0.01])
    coarse = ConvergenceResponse(
        times=fine.times * 1.01,
        dbzdt=fine.dbzdt,
        reference=fine.reference,
    )

    with pytest.raises(ValueError, match="observation grids"):
        compare_responses(coarse, fine)


def test_compare_responses_excludes_below_reference_floor():
    fine = ConvergenceResponse(
        times=np.array([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]),
        dbzdt=np.array([1.0, 0.1, 0.01, 1.0e-7]),
        reference=np.array([1.0, 0.1, 0.01, 1.0e-7]),
    )
    coarse = ConvergenceResponse(
        times=fine.times,
        dbzdt=np.array([1.01, 0.102, 0.0101, 1.0]),
        reference=fine.reference,
    )

    result = compare_responses(coarse, fine)

    assert result["sample_count"] == 3
    assert result["excluded_below_floor_count"] == 1
    assert result["amplitude_floor"] == pytest.approx(1.0e-6)
    assert result["max_percent"] == pytest.approx(2.0)


def test_compare_responses_rejects_near_duplicate_observation_times():
    times = np.array([1.0e-5, 1.0e-4, 0.009999999999, 0.01])
    response = ConvergenceResponse(
        times=times,
        dbzdt=np.ones(times.size),
        reference=np.ones(times.size),
    )

    with pytest.raises(ValueError, match="near-duplicate"):
        compare_responses(response, response)


def test_load_response_selects_dbdzdt_component(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez(
        run_dir / "verification_data.npz",
        times=np.array([1.0e-5, 1.0e-4, 1.0e-3]),
        fem=np.array([[10.0, -1.0], [20.0, -0.1], [30.0, -0.01]]),
        empymod=np.array([[11.0, -1.1], [21.0, -0.11], [31.0, -0.011]]),
        components=np.array(["Ex", "dBzdt"]),
    )

    result = load_response(run_dir)

    assert result.times == pytest.approx([1.0e-5, 1.0e-4, 1.0e-3])
    assert result.dbzdt == pytest.approx([-1.0, -0.1, -0.01])
    assert result.reference == pytest.approx([-1.1, -0.11, -0.011])


def test_load_response_rejects_missing_required_arrays(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez(run_dir / "verification_data.npz", times=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="missing keys"):
        load_response(run_dir)
