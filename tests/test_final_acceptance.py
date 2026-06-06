import json

from atem3d.final_acceptance import (
    summarize_final_acceptance,
    write_final_acceptance_report,
)


def test_final_acceptance_summary_requires_noip_and_ip_final_pass():
    noip = _summary("noip", True, internal_time_grid_verified=True)
    ip = _summary("ip", False, reasons=["physical_error_gate_failed"])

    summary = summarize_final_acceptance({"noip": noip, "ip": ip})

    assert summary["final_acceptance_passed"] is False
    assert summary["required_cases"] == ["noip", "ip"]
    assert summary["passed_cases"] == ["noip"]
    assert summary["failed_cases"] == ["ip"]
    assert summary["missing_cases"] == []
    assert summary["blocking_reasons_by_case"]["ip"] == ["physical_error_gate_failed"]


def test_final_acceptance_summary_passes_when_both_cases_pass():
    summary = summarize_final_acceptance(
        {
            "noip": _summary("noip", True, internal_time_grid_verified=True),
            "ip": _summary("ip", True, internal_time_grid_verified=True),
        }
    )

    assert summary["final_acceptance_passed"] is True
    assert summary["failed_cases"] == []
    assert summary["missing_cases"] == []


def test_final_acceptance_summary_requires_internal_grid_evidence_for_each_case():
    summary = summarize_final_acceptance(
        {
            "noip": _summary("noip", True, internal_time_grid_verified=True),
            "ip": _summary("ip", True),
        }
    )

    assert summary["final_acceptance_passed"] is False
    assert summary["passed_cases"] == ["noip"]
    assert summary["failed_cases"] == ["ip"]
    assert summary["blocking_reasons_by_case"]["ip"] == ["internal_time_grid_not_verified"]


def test_final_acceptance_can_use_diagnostics_internal_grid_evidence():
    diagnostics = {
        "noip": _diagnostics_with_internal_grid(),
        "ip": _diagnostics_with_internal_grid(),
    }

    summary = summarize_final_acceptance(
        {
            "noip": _summary("noip", True),
            "ip": _summary("ip", True),
        },
        case_diagnostics=diagnostics,
    )

    assert summary["final_acceptance_passed"] is True
    assert summary["cases"]["noip"]["internal_time_grid_verified"] is True
    assert summary["cases"]["ip"]["internal_time_grid_verified"] is True


def test_write_final_acceptance_report_reads_summary_paths(tmp_path):
    noip_path = tmp_path / "noip" / "error_summary.json"
    ip_path = tmp_path / "ip" / "error_summary.json"
    noip_path.parent.mkdir()
    ip_path.parent.mkdir()
    noip_path.write_text(json.dumps(_summary("noip", True, internal_time_grid_verified=True)), encoding="utf-8")
    _write_required_case_artifacts(noip_path.parent)
    ip_path.write_text(
        json.dumps(_summary("ip", False, reasons=["validation_scope_not_corrected_model_full"])),
        encoding="utf-8",
    )

    summary = write_final_acceptance_report(
        noip_summary_json=noip_path,
        ip_summary_json=ip_path,
        output_dir=tmp_path / "acceptance",
    )

    assert summary["final_acceptance_passed"] is False
    payload = json.loads((tmp_path / "acceptance" / "final_acceptance_summary.json").read_text(encoding="utf-8"))
    assert payload["failed_cases"] == ["ip"]
    text = (tmp_path / "acceptance" / "final_acceptance_report.txt").read_text(encoding="utf-8")
    assert "FINAL_ACCEPTANCE_PASSED=false" in text
    assert "ip: validation_scope_not_corrected_model_full" in text


def test_write_final_acceptance_report_requires_case_artifact_set(tmp_path):
    noip_path = tmp_path / "noip" / "error_summary.json"
    ip_path = tmp_path / "ip" / "error_summary.json"
    noip_path.parent.mkdir()
    ip_path.parent.mkdir()
    noip_path.write_text(json.dumps(_summary("noip", True, internal_time_grid_verified=True)), encoding="utf-8")
    ip_path.write_text(json.dumps(_summary("ip", True, internal_time_grid_verified=True)), encoding="utf-8")
    _write_required_case_artifacts(noip_path.parent, omit={"comparison_3comp.png"})
    _write_required_case_artifacts(ip_path.parent)

    summary = write_final_acceptance_report(
        noip_summary_json=noip_path,
        ip_summary_json=ip_path,
        output_dir=tmp_path / "acceptance",
    )

    assert summary["final_acceptance_passed"] is False
    assert summary["failed_cases"] == ["noip"]
    assert summary["blocking_reasons_by_case"]["noip"] == ["validation_artifacts_missing"]
    assert summary["cases"]["noip"]["artifact_status"]["missing"] == ["comparison_3comp.png"]


def test_write_final_acceptance_report_summarizes_case_diagnostics(tmp_path):
    noip_path = tmp_path / "noip" / "error_summary.json"
    ip_path = tmp_path / "ip" / "error_summary.json"
    noip_diag_path = tmp_path / "noip" / "diagnostics.json"
    ip_diag_path = tmp_path / "ip" / "diagnostics.json"
    noip_path.parent.mkdir()
    ip_path.parent.mkdir()
    noip_path.write_text(json.dumps(_summary("noip", True, internal_time_grid_verified=True)), encoding="utf-8")
    ip_path.write_text(json.dumps(_summary("ip", False, reasons=["physical_error_gate_failed"])), encoding="utf-8")
    noip_diag_path.write_text(json.dumps({"validation_failure": {"failed": False}}), encoding="utf-8")
    ip_diag_path.write_text(
        json.dumps(
            {
                "validation_failure": {
                    "failed": True,
                    "reason_codes": ["physical_error_gate_failed"],
                    "checks": {
                        "time_step_error": {
                            "status": "needs_evaluation",
                            "evidence": {"failed_times": [1.0]},
                            "recommended_action": "rerun with denser internal time steps",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    summary = write_final_acceptance_report(
        noip_summary_json=noip_path,
        ip_summary_json=ip_path,
        noip_diagnostics_json=noip_diag_path,
        ip_diagnostics_json=ip_diag_path,
        output_dir=tmp_path / "acceptance",
    )

    ip_diagnostics = summary["failure_diagnostics_by_case"]["ip"]
    assert ip_diagnostics["reason_codes"] == ["physical_error_gate_failed"]
    assert ip_diagnostics["checks"]["time_step_error"]["status"] == "needs_evaluation"
    text = (tmp_path / "acceptance" / "final_acceptance_report.txt").read_text(encoding="utf-8")
    assert "diagnostic_reason_codes:" in text
    assert "ip: physical_error_gate_failed" in text


def _summary(case_type: str, passed: bool, *, reasons=None, internal_time_grid_verified=False):
    reasons = [] if reasons is None else list(reasons)
    return {
        "case_type": case_type,
        "reference_type": "empymod",
        "magnetic_quantity": "dBzdt",
        "final_acceptance_passed": passed,
        "acceptance_status": {
            "final_acceptance_passed": passed,
            "blocking_reasons": reasons,
            "internal_time_grid_verified": bool(internal_time_grid_verified),
        },
    }


def _diagnostics_with_internal_grid():
    return {
        "primary_secondary_internal_time_grid": {
            "contains_turnoff_start": True,
            "contains_turnoff_end": True,
            "contains_all_observation_outputs": True,
            "last_output_internal_time_s": 1.00001,
        }
    }


def _write_required_case_artifacts(directory, *, omit=None):
    omit = set() if omit is None else set(omit)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "predictions.csv",
        "reference_empymod_or_1d.csv",
        "errors.csv",
        "comparison_3comp.png",
        "error_curves_3comp.png",
        "diagnostics.json",
        "run_config_resolved.yaml",
    ):
        if name in omit:
            continue
        (directory / name).write_text("placeholder", encoding="utf-8")
