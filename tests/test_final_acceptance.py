import json

from atem3d.final_acceptance import (
    summarize_final_acceptance,
    write_final_acceptance_report,
)


def test_final_acceptance_summary_requires_noip_and_ip_final_pass():
    noip = _summary("noip", True)
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
            "noip": _summary("noip", True),
            "ip": _summary("ip", True),
        }
    )

    assert summary["final_acceptance_passed"] is True
    assert summary["failed_cases"] == []
    assert summary["missing_cases"] == []


def test_write_final_acceptance_report_reads_summary_paths(tmp_path):
    noip_path = tmp_path / "noip" / "error_summary.json"
    ip_path = tmp_path / "ip" / "error_summary.json"
    noip_path.parent.mkdir()
    ip_path.parent.mkdir()
    noip_path.write_text(json.dumps(_summary("noip", True)), encoding="utf-8")
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


def _summary(case_type: str, passed: bool, *, reasons=None):
    reasons = [] if reasons is None else list(reasons)
    return {
        "case_type": case_type,
        "reference_type": "empymod",
        "magnetic_quantity": "dBzdt",
        "final_acceptance_passed": passed,
        "acceptance_status": {
            "final_acceptance_passed": passed,
            "blocking_reasons": reasons,
        },
    }
