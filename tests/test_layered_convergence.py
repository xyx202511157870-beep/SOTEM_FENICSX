from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import meshio
import numpy as np
import pytest

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
    assert "--stop-after-outputs 25" in command


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
