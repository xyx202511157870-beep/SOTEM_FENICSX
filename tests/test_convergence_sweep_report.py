import csv
import json

from atem3d.cli import main
from atem3d.convergence_sweep import write_convergence_sweep_report


def test_write_convergence_sweep_report_summarizes_runs(tmp_path):
    run_a = _write_run(
        tmp_path,
        "domain_a",
        max_error_ex=0.62,
        max_error_dbdt=0.13,
        failed_band="late_time",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=24.0,
        reference_runtime=164.0,
    )
    run_b = _write_run(
        tmp_path,
        "domain_b",
        max_error_ex=0.21,
        max_error_dbdt=0.04,
        failed_band="late_time",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=27.0,
        reference_runtime=170.0,
    )

    summary = write_convergence_sweep_report(
        [run_a, run_b],
        tmp_path / "sweep",
        labels=["domain_a", "domain_b"],
    )

    assert summary["run_count"] == 2
    assert summary["best_by_max_physical_error"]["label"] == "domain_b"
    assert summary["best_by_max_physical_error"]["max_physical_error"] == 0.21
    json_payload = json.loads((tmp_path / "sweep" / "convergence_sweep_summary.json").read_text(encoding="utf-8"))
    assert json_payload["runs"][0]["failed_time_band"] == "late_time"
    assert json_payload["runs"][1]["reference_cells"] == [4, 4, 2]
    assert json_payload["runs"][0]["prediction_leakage_cell_count"] == 1
    assert json_payload["runs"][0]["reference_leakage_cell_count"] == 8
    assert json_payload["runs"][0]["prediction_initial_leakage_cell_count"] == 0
    assert json_payload["runs"][0]["prediction_marker_fallback_used"] is True
    assert json_payload["runs"][0]["prediction_marker_fallback_added_cell_count"] == 1
    assert json_payload["runs"][0]["prediction_min_marked_cells"] == 1
    assert json_payload["runs"][0]["leakage_cell_count_ratio"] == 0.125
    assert json_payload["runs"][0]["leakage_marker_issue"] == "fallback_used"
    assert json_payload["runs"][0]["prediction_secondary_effect_nonzero"] is True
    assert json_payload["runs"][0]["reference_secondary_effect_nonzero"] is True
    assert json_payload["runs"][0]["secondary_effect_nonzero"] is True
    assert json_payload["runs"][0]["max_prediction_secondary_effect_Ex"] == 0.2
    assert json_payload["runs"][0]["max_reference_secondary_effect_dBzdt"] == 0.18

    with (tmp_path / "sweep" / "convergence_sweep_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["label"] == "domain_a"
    assert rows[0]["physical_failed_components"] == "Ex;dBzdt"
    assert rows[0]["prediction_leakage_cell_count"] == "1"
    assert rows[0]["reference_leakage_cell_count"] == "8"
    assert rows[0]["prediction_marker_fallback_used"] == "True"
    assert rows[0]["leakage_cell_count_ratio"] == "0.125"
    assert rows[0]["leakage_marker_issue"] == "fallback_used"
    assert rows[0]["prediction_secondary_effect_nonzero"] == "True"
    assert rows[0]["reference_secondary_effect_nonzero"] == "True"
    assert rows[0]["secondary_effect_nonzero"] == "True"
    assert rows[0]["max_prediction_secondary_effect_Ex"] == "0.2"
    assert rows[0]["max_reference_secondary_effect_dBzdt"] == "0.18"
    assert rows[1]["max_physical_error"] == "0.21"


def test_convergence_sweep_report_cli_writes_summary(tmp_path):
    run_a = _write_run(
        tmp_path,
        "domain_a",
        max_error_ex=0.62,
        max_error_dbdt=0.13,
        failed_band="late_time",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=24.0,
        reference_runtime=164.0,
    )
    run_b = _write_run(
        tmp_path,
        "domain_b",
        max_error_ex=0.21,
        max_error_dbdt=0.04,
        failed_band="late_time",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=27.0,
        reference_runtime=170.0,
    )

    exit_code = main(
        [
            "convergence-sweep-report",
            str(run_a),
            str(run_b),
            "--output-dir",
            str(tmp_path / "cli_sweep"),
            "--labels",
            "domain_a,domain_b",
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "cli_sweep" / "convergence_sweep_summary.json").read_text(encoding="utf-8"))
    assert payload["best_by_max_physical_error"]["label"] == "domain_b"


def test_convergence_sweep_report_selects_best_nonzero_secondary_run(tmp_path):
    primary_only = _write_run(
        tmp_path,
        "primary_only",
        max_error_ex=0.0,
        max_error_dbdt=0.0,
        failed_band="none",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=10.0,
        reference_runtime=11.0,
        secondary_nonzero=False,
    )
    nonzero_secondary = _write_run(
        tmp_path,
        "nonzero_secondary",
        max_error_ex=0.21,
        max_error_dbdt=0.04,
        failed_band="late_time",
        prediction_cells=[2, 2, 1],
        reference_cells=[4, 4, 2],
        forward_runtime=27.0,
        reference_runtime=170.0,
        secondary_nonzero=True,
    )

    summary = write_convergence_sweep_report(
        [primary_only, nonzero_secondary],
        tmp_path / "sweep",
        labels=["primary_only", "nonzero_secondary"],
    )

    assert summary["best_by_max_physical_error"]["label"] == "primary_only"
    assert summary["usable_nonzero_secondary_run_count"] == 1
    assert summary["best_usable_by_max_physical_error"]["label"] == "nonzero_secondary"
    assert summary["runs"][0]["nonzero_secondary_validation_usable"] is False
    assert summary["runs"][1]["nonzero_secondary_validation_usable"] is True


def _write_run(
    root,
    name,
    *,
    max_error_ex,
    max_error_dbdt,
    failed_band,
    prediction_cells,
    reference_cells,
    forward_runtime,
    reference_runtime,
    secondary_nonzero=True,
):
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "error_summary.json").write_text(
        json.dumps(
            {
                "case_type": "noip",
                "reference_type": "dolfinx_refined",
                "max_error_Ex": max_error_ex,
                "max_error_Ey": 1.0,
                "max_error_dBzdt": max_error_dbdt,
                "physical_failed_components": ["Ex", "dBzdt"],
                "physical_pass_all_components": False,
                "final_acceptance_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "runtime_seconds": {
                    "forward": forward_runtime,
                    "reference": reference_runtime,
                },
                "validation_failure": {
                    "convergence_diagnostic": {
                        "failed_time_band": failed_band,
                        "prediction_cells": prediction_cells,
                        "reference_cells": reference_cells,
                        "physical_failed_components": ["Ex", "dBzdt"],
                    }
                },
                "leakage_marker_preflight": {
                    "prediction": {
                        "initial_leakage_cell_count": 0,
                        "leakage_cell_count": 1,
                        "nearest_channel_distance_m": 670.0,
                        "fallback_used": True,
                        "fallback_added_cell_count": 1,
                        "min_marked_cells": 1,
                        "marked": True,
                    },
                    "reference": {
                        "initial_leakage_cell_count": 8,
                        "leakage_cell_count": 8,
                        "nearest_channel_distance_m": 335.0,
                        "fallback_used": False,
                        "fallback_added_cell_count": 0,
                        "min_marked_cells": 1,
                        "marked": True,
                    },
                },
                "secondary_effect_diagnostic": {
                    "reference_type": "dolfinx_refined",
                    "component_names": ["Ex", "Ey", "dBzdt"],
                    "prediction_secondary_effect_nonzero": secondary_nonzero,
                    "reference_secondary_effect_nonzero": secondary_nonzero,
                    "secondary_effect_nonzero": secondary_nonzero,
                    "max_abs_prediction_minus_primary_by_component": {
                        "Ex": 0.2,
                        "Ey": 0.0,
                        "dBzdt": 0.12,
                    },
                    "max_abs_reference_minus_primary_by_component": {
                        "Ex": 0.3,
                        "Ey": 0.0,
                        "dBzdt": 0.18,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir
