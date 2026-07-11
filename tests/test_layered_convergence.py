from __future__ import annotations

from pathlib import Path

from atem3d.layered_convergence import (
    build_convergence_levels,
    build_pipeline_command_arguments,
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
