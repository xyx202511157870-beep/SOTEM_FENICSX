from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import meshio
import numpy as np
import pytest

import atem3d.layered_convergence as layered_convergence
from atem3d.layered_convergence import (
    ConvergenceResponse,
    build_convergence_levels,
    build_pipeline_command_arguments,
    compare_responses,
    evaluate_axis_metrics,
    load_response,
    read_run_metadata,
    write_convergence_reports,
)


def _option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def _load_convergence_runner_module():
    path = Path("dolfinx/run_layered_convergence_study.py").resolve()
    spec = importlib.util.spec_from_file_location("convergence_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_memory_contract_defaults_to_20_6_14():
    contract = layered_convergence.PublicationMemoryContract()

    assert contract.total_memory_gb == 20.0
    assert contract.reserve_memory_gb == 6.0
    assert contract.solver_memory_limit_gb == 14.0
    assert contract.as_dict() == {
        "total_memory_gb": 20.0,
        "reserve_memory_gb": 6.0,
        "solver_memory_limit_gb": 14.0,
    }


@pytest.mark.parametrize(
    ("total", "reserve"),
    [
        (float("nan"), 6.0),
        (20.0, float("inf")),
        (0.0, 0.0),
        (20.0, -1.0),
        (20.0, 20.0),
        (20.0, 21.0),
    ],
)
def test_publication_memory_contract_rejects_invalid_values(total, reserve):
    with pytest.raises(ValueError, match="memory contract"):
        layered_convergence.PublicationMemoryContract(total, reserve)


def test_live_resource_gate_passes_exact_available_memory():
    result = layered_convergence.evaluate_publication_live_resources(
        estimated_memory_gb=13.25,
        available_memory_gb=13.25,
        comsol_processes=[],
    )

    assert result["passed"] is True
    assert result["blocking_reasons"] == []


@pytest.mark.parametrize(
    ("available", "processes", "reason"),
    [
        (13.24, [], "insufficient_available_memory"),
        (20.0, ["comsolmphserver.exe"], "comsol_process_running"),
        (float("nan"), [], "invalid_available_memory"),
    ],
)
def test_live_resource_gate_rejects_unsafe_launch(available, processes, reason):
    result = layered_convergence.evaluate_publication_live_resources(
        estimated_memory_gb=13.25,
        available_memory_gb=available,
        comsol_processes=processes,
    )

    assert result["passed"] is False
    assert reason in result["blocking_reasons"]


def test_runner_collects_a_valid_live_resource_snapshot():
    runner = _load_convergence_runner_module()

    available = runner._available_physical_memory_gb()
    process_names = runner._comsol_process_names()

    assert np.isfinite(available)
    assert available > 0.0
    assert process_names == sorted(set(process_names))
    assert all("comsol" in name.lower() for name in process_names)


def test_paper_baseline_levels_match_approved_stage_two_design(tmp_path):
    levels = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )

    assert [
        (level.level_id, level.max_internal_dt, level.max_internal_dt_fraction)
        for level in levels["time"]
    ] == [
        ("coarse", 2.5e-5, 0.01),
        ("standard", 1.25e-5, 0.005),
        ("fine", 6.25e-6, 0.0025),
    ]
    assert [
        (level.level_id, level.x_extent, level.earth_depth, level.air_height)
        for level in levels["domain"]
    ] == [
        ("small", 6000.0, 6000.0, 600.0),
        ("standard", 12000.0, 12000.0, 1200.0),
        ("large", 18000.0, 18000.0, 1800.0),
    ]
    assert [
        (level.source_mesh_size, level.receiver_mesh_size)
        for level in levels["mesh"]
    ] == [(12.0, 9.0), (8.0, 6.0), (6.0, 4.5)]
    assert {
        level.run_id
        for axis_levels in levels.values()
        for level in axis_levels
        if level.existing_run_dir is None
    } == {
        "baseline_12km_dt005_mesh8_6",
        "time_fine_12km_dt0025_mesh8_6",
        "domain_large_18km_dt005_mesh8_6",
        "mesh_coarse_12km_dt005_mesh12_9",
        "mesh_fine_12km_dt005_mesh6_4p5",
    }


def test_stage_two_axes_change_only_the_intended_parameters(tmp_path):
    levels = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )

    def physical(level):
        return (
            level.x_extent,
            level.y_extent,
            level.earth_depth,
            level.air_height,
            level.far_field_mesh_size,
            level.source_mesh_size,
            level.receiver_mesh_size,
            level.max_internal_dt,
            level.max_internal_dt_fraction,
        )

    time = [physical(level) for level in levels["time"]]
    mesh = [physical(level) for level in levels["mesh"]]
    domain = [physical(level) for level in levels["domain"]]
    assert all(row[:7] == time[1][:7] for row in time)
    assert all(row[:5] + row[7:] == mesh[1][:5] + mesh[1][7:] for row in mesh)
    assert all(row[4:] == domain[1][4:] for row in domain)


def test_stage_two_pipeline_arguments_lock_solver_and_observation_contract(tmp_path):
    baseline = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )["time"][1]

    contract = layered_convergence.PublicationMemoryContract()
    arguments = build_pipeline_command_arguments(
        baseline,
        memory_limit_gb=contract.solver_memory_limit_gb,
    )

    assert _option_value(arguments, "--t-min") == "1e-05"
    assert _option_value(arguments, "--max-internal-dt") == "1.25e-05"
    assert _option_value(arguments, "--max-internal-dt-fraction") == "0.005"
    assert _option_value(arguments, "--rtol") == "1e-07"
    assert _option_value(arguments, "--atol") == "1e-12"
    assert _option_value(arguments, "--memory-limit-gb") == "14"
    assert _option_value(arguments, "--stop-after-outputs") == "25"


def test_manifest_records_locked_mesh_path_and_sha256(tmp_path):
    layered_root = tmp_path / "layered"
    locked_mesh = (
        layered_root
        / "domain12000"
        / "resistive_basement_rho1000_offset100"
        / "verification_mesh.msh"
    )
    locked_mesh.parent.mkdir(parents=True)
    locked_mesh.write_bytes(b"publication-mesh")
    baseline = layered_convergence.build_paper_baseline_convergence_levels(
        layered_root,
        tmp_path / "stage2",
        tmp_path / "stage1",
    )["time"][1]

    manifest = layered_convergence.convergence_level_manifest(baseline)

    assert manifest["run_id"] == "baseline_12km_dt005_mesh8_6"
    assert manifest["reuse_mesh"]["path"] == str(locked_mesh)
    assert manifest["reuse_mesh"]["sha256"] == hashlib.sha256(
        b"publication-mesh"
    ).hexdigest()


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (
            {
                "estimated_memory_gb": 10.0,
                "receiver_found": True,
                "source_divergence_passed": True,
            },
            "source coverage",
        ),
        (
            {
                "estimated_memory_gb": 10.0,
                "source_coverage_passed": True,
                "source_divergence_passed": True,
            },
            "receiver location",
        ),
        (
            {
                "estimated_memory_gb": 10.0,
                "source_coverage_passed": True,
                "receiver_found": True,
            },
            "source divergence",
        ),
        (
            {
                "estimated_memory_gb": 14.1,
                "source_coverage_passed": True,
                "receiver_found": True,
                "source_divergence_passed": True,
            },
            "14 GB",
        ),
    ],
)
def test_preflight_rejects_invalid_publication_mesh(tmp_path, diagnostics, message):
    mesh = tmp_path / "verification_mesh.msh"
    mesh.write_bytes(b"mesh")

    with pytest.raises(ValueError, match=message):
        layered_convergence.validate_publication_preflight(
            mesh_path=mesh,
            diagnostics=diagnostics,
            memory_limit_gb=14.0,
        )


def test_preflight_accepts_complete_publication_mesh_evidence(tmp_path):
    mesh = tmp_path / "verification_mesh.msh"
    mesh.write_bytes(b"mesh")

    result = layered_convergence.validate_publication_preflight(
        mesh_path=mesh,
        diagnostics={
            "estimated_memory_gb": 14.0,
            "source_coverage_passed": True,
            "receiver_found": True,
            "source_divergence_passed": True,
        },
        memory_limit_gb=14.0,
    )

    assert result == {
        "passed": True,
        "mesh_path": str(mesh),
        "mesh_sha256": hashlib.sha256(b"mesh").hexdigest(),
        "estimated_memory_gb": 14.0,
        "memory_limit_gb": 14.0,
    }


def test_runner_rejects_preflight_from_a_different_memory_contract(tmp_path):
    runner = _load_convergence_runner_module()
    level = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )["mesh"][0]
    level.workdir.mkdir(parents=True)
    mesh = level.workdir / "verification_mesh.msh"
    mesh.write_bytes(b"mesh")
    (level.workdir / "preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "mesh_sha256": hashlib.sha256(b"mesh").hexdigest(),
                "estimated_memory_gb": 10.0,
                "memory_limit_gb": 18.0,
                "total_memory_gb": 24.0,
                "reserve_memory_gb": 6.0,
                "solver_memory_limit_gb": 18.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="memory contract changed"):
        runner._require_publication_preflight(
            level,
            layered_convergence.PublicationMemoryContract(),
        )


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


def test_time_axis_passes_when_refined_difference_is_small_and_ordered():
    result = evaluate_axis_metrics(
        "time",
        {
            "coarse_to_standard": {"rms_percent": 3.0},
            "standard_to_fine": {
                "median_percent": 0.5,
                "rms_percent": 1.0,
                "max_percent": 2.0,
            },
        },
    )

    assert result == {"axis": "time", "passed": True, "blocking_reasons": []}


def test_mesh_axis_rejects_non_decreasing_rms():
    result = evaluate_axis_metrics(
        "mesh",
        {
            "coarse_to_standard": {"rms_percent": 0.5},
            "standard_to_fine": {
                "median_percent": 0.5,
                "rms_percent": 1.0,
                "max_percent": 2.0,
            },
        },
    )

    assert result["passed"] is False
    assert result["blocking_reasons"] == ["mesh_rms_not_decreasing"]


def test_time_axis_reports_every_refined_threshold_failure():
    result = evaluate_axis_metrics(
        "time",
        {
            "coarse_to_standard": {"rms_percent": 8.0},
            "standard_to_fine": {
                "median_percent": 1.1,
                "rms_percent": 2.1,
                "max_percent": 5.1,
            },
        },
    )

    assert result["blocking_reasons"] == [
        "time_median_above_1pct",
        "time_rms_above_2pct",
        "time_max_above_5pct",
    ]


def test_domain_axis_requires_decreasing_change_and_large_external_pass():
    passed = evaluate_axis_metrics(
        "domain",
        {
            "small_to_standard": {"rms_percent": 12.0},
            "standard_to_large": {"rms_percent": 8.0},
        },
        large_external_passed=True,
    )
    failed = evaluate_axis_metrics(
        "domain",
        {
            "small_to_standard": {"rms_percent": 7.0},
            "standard_to_large": {"rms_percent": 8.0},
        },
        large_external_passed=False,
    )

    assert passed["passed"] is True
    assert failed["blocking_reasons"] == [
        "domain_rms_not_decreasing",
        "domain_large_external_gate_failed",
    ]


def test_read_run_metadata_uses_real_mesh_npz_and_timing_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meshio.write(
        run_dir / "verification_mesh.msh",
        meshio.Mesh(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=int))],
        ),
        file_format="gmsh22",
        binary=False,
    )
    np.savez(
        run_dir / "verification_data.npz",
        times=np.array([1.0e-5], dtype=float),
    )
    np.savez(
        run_dir / "forward_partial.npz",
        internal_solver_steps=np.array([0, 1, 2], dtype=int),
    )
    events = [
        {"event": "forward_done", "seconds": 5.0},
        {"event": "setup_done", "seconds": 99.0},
        {"event": "forward_done", "seconds": 7.5},
    ]
    (run_dir / "timing_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    result = read_run_metadata(run_dir)

    assert result["nodes"] == 4
    assert result["tetrahedra"] == 1
    assert result["cells_blocks"] == 1
    assert result["nedelec_dofs"] == 6
    assert result["internal_step_count"] == 3
    assert result["forward_runtime_seconds"] == pytest.approx(12.5)
    assert result["estimated_memory_gb"] == pytest.approx(
        1 * 2.85e-5 + 4 * 1.5e-6
    )


def _write_metadata_mesh_and_timing(run_dir: Path) -> None:
    run_dir.mkdir()
    meshio.write(
        run_dir / "verification_mesh.msh",
        meshio.Mesh(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=int))],
        ),
        file_format="gmsh22",
        binary=False,
    )
    (run_dir / "timing_events.jsonl").write_text(
        '{"event":"forward_done","seconds":120.75}\n',
        encoding="utf-8",
    )


def test_read_run_metadata_reports_output_ksp_statistics(tmp_path):
    run_dir = tmp_path / "run"
    _write_metadata_mesh_and_timing(run_dir)
    np.savez(
        run_dir / "forward_partial.npz",
        internal_solver_steps=np.arange(9),
        solver_iterations=np.array([12, 15, 9]),
        solver_reasons=np.array([2, 2, 2]),
        solver_residuals=np.array([1.0e-9, 2.0e-9, 8.0e-10]),
    )

    result = read_run_metadata(run_dir)

    assert result["ksp_output_solve_count"] == 3
    assert result["ksp_iterations_median"] == 12.0
    assert result["ksp_iterations_max"] == 15
    assert result["ksp_residual_max"] == pytest.approx(2.0e-9)
    assert result["ksp_all_converged"] is True
    assert result["forward_runtime_seconds"] == pytest.approx(120.75)


@pytest.mark.parametrize(
    ("iterations", "reasons", "residuals", "message"),
    [
        ([12, 15], [2], [1.0e-9, 2.0e-9], "equal lengths"),
        ([12], [0], [1.0e-9], "convergence reasons"),
        ([-1], [2], [1.0e-9], "negative iterations"),
        ([12], [2], [np.nan], "finite residuals"),
    ],
)
def test_read_run_metadata_rejects_invalid_ksp_evidence(
    tmp_path,
    iterations,
    reasons,
    residuals,
    message,
):
    run_dir = tmp_path / "run"
    _write_metadata_mesh_and_timing(run_dir)
    np.savez(
        run_dir / "forward_partial.npz",
        internal_solver_steps=np.arange(3),
        solver_iterations=np.asarray(iterations),
        solver_reasons=np.asarray(reasons),
        solver_residuals=np.asarray(residuals),
    )

    with pytest.raises(ValueError, match=message):
        read_run_metadata(run_dir)


def _write_complete_convergence_run(
    run_dir: Path,
    *,
    response_scale: float,
    external_error: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    meshio.write(
        run_dir / "verification_mesh.msh",
        meshio.Mesh(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=int))],
        ),
        file_format="gmsh22",
        binary=False,
    )
    times = np.geomspace(1.0e-5, 2.5118864315e-3, 25)
    reference = -np.geomspace(1.0, 1.0e-4, 25)
    fem = np.zeros((25, 4), dtype=float)
    empymod = np.zeros((25, 4), dtype=float)
    fem[:, 3] = reference * response_scale
    empymod[:, 3] = reference
    np.savez(
        run_dir / "verification_data.npz",
        times=times,
        fem=fem,
        empymod=empymod,
        components=np.array(["Ex", "Ey", "Hz", "dBzdt"]),
    )
    np.savez(
        run_dir / "forward_partial.npz",
        internal_solver_steps=np.arange(100),
        solver_iterations=np.full(25, 12, dtype=int),
        solver_reasons=np.full(25, 2, dtype=int),
        solver_residuals=np.full(25, 1.0e-9),
    )
    (run_dir / "timing_events.jsonl").write_text(
        '{"event":"forward_done","seconds":60.0}\n',
        encoding="utf-8",
    )
    (run_dir / "preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "estimated_memory_gb": 1.0,
                "memory_limit_gb": 14.0,
                "total_memory_gb": 20.0,
                "reserve_memory_gb": 6.0,
                "solver_memory_limit_gb": 14.0,
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "time_obs",
                "ref",
                "ordinary_relative_error",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for time, value in zip(times, reference):
            writer.writerow(
                {
                    "component": "dBzdt",
                    "time_obs": time,
                    "ref": value,
                    "ordinary_relative_error": external_error,
                }
            )


def _make_complete_stage_two_fixture(tmp_path, *, passing: bool):
    levels = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )
    scales = {
        "existing_12km_dt01": 1.02,
        "baseline_12km_dt005_mesh8_6": 1.005,
        "time_fine_12km_dt0025_mesh8_6": 1.0 if passing else 0.9,
        "mesh_coarse_12km_dt005_mesh12_9": 1.02,
        "mesh_fine_12km_dt005_mesh6_4p5": 1.0,
        "existing_6km_dt005": 1.04,
        "domain_large_18km_dt005_mesh8_6": 1.0,
    }
    written: set[str] = set()
    for axis_levels in levels.values():
        for level in axis_levels:
            if level.run_id in written:
                continue
            written.add(level.run_id)
            _write_complete_convergence_run(
                layered_convergence.resolved_run_dir(level),
                response_scale=scales[level.run_id],
                external_error=(
                    0.008
                    if level.run_id == "domain_large_18km_dt005_mesh8_6"
                    else 0.005
                ),
            )
    return levels


def test_stage_two_summary_reports_baseline_and_large_empymod_gates(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=True)

    summary = layered_convergence.evaluate_convergence_study(
        levels,
        study_id="layered_resistive_offset100_stage2",
        resource_contract=layered_convergence.PublicationMemoryContract(),
    )

    assert summary["study_passed"] is True
    assert summary["resource_contract"] == {
        "total_memory_gb": 20.0,
        "reserve_memory_gb": 6.0,
        "solver_memory_limit_gb": 14.0,
    }
    assert summary["candidate_baseline"]["run_id"] == (
        "baseline_12km_dt005_mesh8_6"
    )
    assert summary["candidate_baseline"]["accepted_for_paper_figures"] is True
    assert summary["candidate_baseline"]["external_reference_gate"][
        "publication_gate_passed"
    ] is True
    domain = next(axis for axis in summary["axes"] if axis["axis"] == "domain")
    assert domain["large_external_reference_gate"][
        "publication_gate_passed"
    ] is True


def test_baseline_is_rejected_when_any_stage_two_axis_fails(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=False)

    summary = layered_convergence.evaluate_convergence_study(
        levels,
        study_id="layered_resistive_offset100_stage2",
        resource_contract=layered_convergence.PublicationMemoryContract(),
    )

    assert summary["candidate_baseline"]["accepted_for_paper_figures"] is False
    assert summary["study_passed"] is False
    time_axis = next(axis for axis in summary["axes"] if axis["axis"] == "time")
    assert time_axis["passed"] is False


def test_stage_two_report_writes_baseline_acceptance_record(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=True)
    summary = layered_convergence.evaluate_convergence_study(
        levels,
        study_id="layered_resistive_offset100_stage2",
        resource_contract=layered_convergence.PublicationMemoryContract(),
    )
    report_dir = tmp_path / "report"

    write_convergence_reports(report_dir, summary)

    acceptance = json.loads(
        (report_dir / "baseline_acceptance.json").read_text(encoding="utf-8")
    )
    assert acceptance["accepted_for_paper_figures"] is True
    assert acceptance["resource_contract"]["solver_memory_limit_gb"] == 14.0
    assert acceptance["candidate_baseline"]["mesh_sha256"]
    assert acceptance["candidate_baseline"]["ksp_output_solve_count"] == 25
    assert acceptance["candidate_baseline"]["internal_step_count"] == 100
    assert acceptance["candidate_baseline"]["forward_runtime_seconds"] == 60.0
    assert [gate["axis"] for gate in acceptance["axis_gates"]] == [
        "time",
        "mesh",
        "domain",
    ]


def test_independent_audit_recomputes_stage_two_reports_from_disk(tmp_path):
    levels = _make_complete_stage_two_fixture(tmp_path, passing=True)
    summary = layered_convergence.evaluate_convergence_study(
        levels,
        study_id="layered_resistive_offset100_stage2",
        resource_contract=layered_convergence.PublicationMemoryContract(),
    )
    report_dir = tmp_path / "report"
    write_convergence_reports(report_dir, summary)
    audit_path = report_dir / "independent_audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "dolfinx/audit_layered_convergence.py",
            "--summary",
            str(report_dir / "convergence_summary.json"),
            "--output",
            str(audit_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "INDEPENDENT_RECOMPUTE_OK" in result.stdout
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["verified"] is True
    assert audit["comparison_count"] == 6
    assert audit["external_gate_count"] == 2
    assert audit["resource_contract_verified"] is True
    assert audit["resource_preflight_count"] == 2


def _synthetic_report_summary() -> dict:
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])
    reference = np.array([-1.0, -0.1, -0.01])
    standard = ConvergenceResponse(times, reference * 1.01, reference)
    fine = ConvergenceResponse(times, reference, reference)
    comparison = compare_responses(standard, fine)
    return {
        "study_id": "layered_resistive_offset100",
        "study_passed": True,
        "coordinate_convention": "z=0 ground; underground positive; air negative",
        "axes": [
            {
                "axis": "time",
                "passed": True,
                "blocking_reasons": [],
                "levels": [
                    {"level_id": "standard", "nedelec_dofs": 100},
                    {"level_id": "fine", "nedelec_dofs": 100},
                ],
                "comparisons": [
                    {
                        "comparison_id": "standard_to_fine",
                        **comparison,
                    }
                ],
                "_responses": {"standard": standard, "fine": fine},
            }
        ],
    }


def test_report_writer_emits_all_publication_artifacts(tmp_path):
    write_convergence_reports(tmp_path, _synthetic_report_summary())

    expected = {
        "convergence_summary.json",
        "convergence_summary.csv",
        "convergence_report.md",
        "convergence_curves.png",
        "convergence_differences.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    payload = json.loads((tmp_path / "convergence_summary.json").read_text())
    assert payload["study_passed"] is True
    assert "_responses" not in payload["axes"][0]
    with (tmp_path / "convergence_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["axis"] == "time"
    assert rows[0]["comparison_id"] == "standard_to_fine"
    assert float(rows[0]["rms_percent"]) == pytest.approx(1.0)
    report = (tmp_path / "convergence_report.md").read_text(encoding="utf-8")
    assert "RMS <= 2%" in report
    assert "maximum <= 5%" in report


def test_report_writer_text_outputs_are_deterministic(tmp_path):
    summary = _synthetic_report_summary()
    write_convergence_reports(tmp_path, summary)
    first = {
        name: (tmp_path / name).read_bytes()
        for name in (
            "convergence_summary.json",
            "convergence_summary.csv",
            "convergence_report.md",
        )
    }

    write_convergence_reports(tmp_path, summary)

    assert first == {name: (tmp_path / name).read_bytes() for name in first}


def test_report_figures_have_publication_resolution(tmp_path):
    import matplotlib.image as mpimg

    write_convergence_reports(tmp_path, _synthetic_report_summary())

    for name in ("convergence_curves.png", "convergence_differences.png"):
        image = mpimg.imread(tmp_path / name)
        assert image.shape[1] >= 1200
        assert image.shape[0] >= 800


def _run_study_cli(*arguments: str, check: bool = True):
    return subprocess.run(
        [sys.executable, "dolfinx/run_layered_convergence_study.py", *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def test_stage_two_dry_run_emits_exactly_five_canonical_commands(tmp_path):
    output_root = tmp_path / "stage2"

    result = _run_study_cli(
        "--study",
        "paper-baseline",
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--prior-convergence-root",
        str(tmp_path / "stage1"),
        "--mode",
        "full",
        "--dry-run",
    )

    commands = list((output_root / "runs").glob("*/command.txt"))
    lines = result.stdout.splitlines()
    assert len(commands) == 5
    assert sum(line.startswith("RUN_ID=") for line in lines) == 5
    assert sum(line.startswith("REUSE_RUN_ID=") for line in lines) == 2
    pointers = list((output_root / "axis_levels").glob("*/*/level_pointer.json"))
    assert len(pointers) == 9


def test_stage_two_custom_memory_contract_reaches_command(tmp_path):
    output_root = tmp_path / "stage2"

    _run_study_cli(
        "--study",
        "paper-baseline",
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--prior-convergence-root",
        str(tmp_path / "stage1"),
        "--axis",
        "mesh",
        "--level",
        "coarse",
        "--mode",
        "full",
        "--total-memory-gb",
        "19",
        "--memory-reserve-gb",
        "5",
        "--dry-run",
    )

    command = (
        output_root
        / "runs"
        / "mesh_coarse_12km_dt005_mesh12_9"
        / "command.txt"
    ).read_text(encoding="utf-8")
    assert "--memory-limit-gb 14" in command


def test_stage_two_shared_baseline_is_processed_once(tmp_path):
    output_root = tmp_path / "stage2"

    result = _run_study_cli(
        "--study",
        "paper-baseline",
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--prior-convergence-root",
        str(tmp_path / "stage1"),
        "--axis",
        "time",
        "--axis",
        "mesh",
        "--axis",
        "domain",
        "--mode",
        "full",
        "--dry-run",
    )

    assert result.stdout.count("RUN_ID=baseline_12km_dt005_mesh8_6") == 1
    assert result.stdout.count("SHARED_RUN=baseline_12km_dt005_mesh8_6") == 2


def test_stage_two_mesh_mode_writes_strict_preflight_from_source_diagnostics(tmp_path):
    output_root = tmp_path / "stage2"
    layered_root = tmp_path / "layered"
    locked_mesh = (
        layered_root
        / "domain12000"
        / "resistive_basement_rho1000_offset100"
        / "verification_mesh.msh"
    )
    locked_mesh.parent.mkdir(parents=True)
    meshio.write(
        locked_mesh,
        meshio.Mesh(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=int))],
        ),
        file_format="gmsh22",
    )
    workdir = output_root / "runs" / "baseline_12km_dt005_mesh8_6"
    workdir.mkdir(parents=True)
    (workdir / "verification_mesh.msh").write_bytes(locked_mesh.read_bytes())
    (workdir / "source_diagnostics.json").write_text(
        json.dumps(
            {
                "source_local_projection": {
                    "quadrature_points": 101,
                    "missed_points": 0,
                    "unique_hit_cells": 20,
                },
                "source_projection": {
                    "after_residual": 2.0e-10,
                    "endpoint_norm": 1.0,
                },
                "receiver_location": {"found": True},
            }
        ),
        encoding="utf-8",
    )

    _run_study_cli(
        "--study",
        "paper-baseline",
        "--output-root",
        str(output_root),
        "--layered-root",
        str(layered_root),
        "--prior-convergence-root",
        str(tmp_path / "stage1"),
        "--axis",
        "time",
        "--level",
        "standard",
        "--mode",
        "mesh",
        "--dry-run",
    )

    command = (workdir / "command.txt").read_text(encoding="utf-8")
    assert "--source-only" in command
    assert "--mesh-only" not in command
    preflight = json.loads((workdir / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["passed"] is True
    assert preflight["source_coverage_passed"] is True
    assert preflight["receiver_found"] is True
    assert preflight["source_divergence_passed"] is True
    assert preflight["total_memory_gb"] == 20.0
    assert preflight["reserve_memory_gb"] == 6.0
    assert preflight["solver_memory_limit_gb"] == 14.0
    assert preflight["memory_limit_gb"] == 14.0
    assert preflight["mesh_sha256"] == hashlib.sha256(
        locked_mesh.read_bytes()
    ).hexdigest()


def test_runner_dry_run_writes_generated_level_manifest_and_command(tmp_path):
    output_root = tmp_path / "convergence"

    result = _run_study_cli(
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--axis",
        "time",
        "--level",
        "coarse",
        "--mode",
        "full",
        "--dry-run",
    )

    assert "RUN_AXIS=time" in result.stdout
    assert "RUN_LEVEL=coarse" in result.stdout
    workdir = output_root / "time" / "coarse"
    assert (workdir / "case_spec.json").is_file()
    command = (workdir / "command.txt").read_text(encoding="utf-8")
    assert "--checkpoint-forward" in command
    assert "--checkpoint-interval-steps 10" in command
    assert "--stop-after-outputs 25" in command


def test_runner_resume_requests_only_outputs_missing_from_checkpoint(tmp_path):
    output_root = tmp_path / "convergence"
    common = (
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--axis",
        "time",
        "--level",
        "coarse",
        "--mode",
        "full",
        "--dry-run",
    )
    _run_study_cli(*common)
    workdir = output_root / "time" / "coarse"
    np.savez(workdir / "forward_checkpoint.npz", rows=np.zeros((10, 4)))

    _run_study_cli(*common)

    command = (workdir / "command.txt").read_text(encoding="utf-8")
    assert "--resume-forward" in command
    assert "--stop-after-outputs 15" in command
    assert "--stop-after-outputs 25" not in command


def test_runner_postprocesses_checkpoint_that_already_has_target_outputs(tmp_path):
    output_root = tmp_path / "convergence"
    common = (
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--axis",
        "time",
        "--level",
        "coarse",
        "--mode",
        "full",
        "--dry-run",
    )
    _run_study_cli(*common)
    workdir = output_root / "time" / "coarse"
    np.savez(workdir / "forward_checkpoint.npz", rows=np.zeros((25, 4)))
    np.savez(workdir / "forward_partial.npz", times=np.arange(25), fem=np.zeros((25, 4)))

    _run_study_cli(*common)

    command = (workdir / "command.txt").read_text(encoding="utf-8")
    assert "--postprocess-partial" in command
    assert "--resume-forward" not in command
    assert "--stop-after-outputs" not in command


def test_runner_existing_level_writes_pointer_without_solver_command(tmp_path):
    output_root = tmp_path / "convergence"
    layered_root = tmp_path / "layered"

    result = _run_study_cli(
        "--output-root",
        str(output_root),
        "--layered-root",
        str(layered_root),
        "--axis",
        "time",
        "--level",
        "standard",
        "--mode",
        "full",
        "--dry-run",
    )

    assert "REUSE_LEVEL=standard" in result.stdout
    workdir = output_root / "time" / "standard"
    pointer = json.loads((workdir / "existing_run.json").read_text(encoding="utf-8"))
    assert Path(pointer["existing_run_dir"]) == (
        layered_root
        / "domain6000"
        / "resistive_basement_rho1000_offset100"
    )
    assert not (workdir / "command.txt").exists()


def test_runner_evaluate_reports_incomplete_axes_with_nonzero_exit(tmp_path):
    output_root = tmp_path / "convergence"

    result = _run_study_cli(
        "--output-root",
        str(output_root),
        "--layered-root",
        str(tmp_path / "layered"),
        "--mode",
        "evaluate",
        check=False,
    )

    assert result.returncode == 1
    assert "CONVERGENCE_COMPLETE=0" in result.stdout
    assert "CONVERGENCE_PASSED=0" in result.stdout
    summary = json.loads(
        (output_root / "convergence_summary.json").read_text(encoding="utf-8")
    )
    assert summary["study_passed"] is False
    assert [axis["status"] for axis in summary["axes"]] == [
        "incomplete",
        "incomplete",
        "incomplete",
    ]
